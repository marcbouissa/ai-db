"""
ai_db.parser.ts_graph
Tree-sitter based extraction of symbols, cross-references, and syntax errors.
Supports all languages with .scm query files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node, Query, QueryCursor
from tree_sitter_language_pack import get_language, get_parser

SUPPORTED_LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    # ".go": "go",      # disabled: tree-sitter query issues
    # ".rs": "rust",    # disabled: tree-sitter query issues
    ".c": "c",
    ".h": "c",
    # ".cpp": "cpp",    # disabled: tree-sitter query issues
    # ".hpp": "cpp",    # disabled: tree-sitter query issues
    # ".cc": "cpp",     # disabled: tree-sitter query issues
    # ".java": "java",  # disabled: tree-sitter query issues
}


@dataclass
class Symbol:
    """Extracted symbol from source code."""
    name: str
    symbol_type: str  # function, method, class, interface, type, etc.
    filepath: str
    line: int
    signature: str | None = None


@dataclass
class Ref:
    """Extracted cross-reference from source code."""
    caller_name: str
    caller_line: int
    callee_name: str
    ref_type: str  # call, import, inherit


@dataclass
class SyntaxErrorInfo:
    """Syntax error information."""
    line: int
    col: int
    message: str


def language_for(filepath: str) -> str | None:
    """Return the tree-sitter language name for a file path, or None if unsupported."""
    ext = os.path.splitext(filepath)[1].lower()
    return SUPPORTED_LANGUAGES.get(ext)


def _read_query(lang: str) -> str:
    """Read the .scm query file for a language."""
    query_path = Path(__file__).parent / "queries" / f"{lang}.scm"
    if not query_path.exists():
        raise ValueError(f"No query file for language: {lang}")
    return query_path.read_text(encoding="utf-8")


def _get_qualified_name(base_name: str, node: Node, source: bytes, lang: str, symbols_by_line: dict[int, str]) -> str:
    """Build a qualified name for a symbol node by prepending containing class names."""
    # Find the containing class/struct/interface
    parent = node.parent
    parts = []
    while parent:
        if parent.type in ("class_definition", "class_declaration", "class_specifier",
                           "struct_item", "struct_specifier", "interface_declaration",
                           "interface_declaration", "type_declaration", "enum_declaration",
                           "impl_item", "namespace_definition"):
            name_node = parent.child_by_field_name("name")
            if name_node:
                parts.append(source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace"))
        parent = parent.parent
    parts.reverse()
    if parts:
        return "module." + ".".join(parts) + "." + base_name
    # Module-level symbols get "module." prefix
    return "module." + base_name


# Helper functions for call analysis
_call_seq_counter: dict[str, int] = {}

def _get_call_seq(caller_name: str, line: int) -> int:
    """Get sequence number for a call within a caller."""
    key = f"{caller_name}:{line}"
    _call_seq_counter[key] = _call_seq_counter.get(key, 0) + 1
    return _call_seq_counter[key]

SPAWN_FUNCS = frozenset({
    "create_task", "ensure_future", "run_in_executor", "run_until_complete",
    "Thread", "submit", "map", "spawn", "spawn_blocking", "spawn_local",
    "Task", "Process",
})

CALLBACK_FUNCS = frozenset({
    "add_done_callback", "add_callback", "then", "connect", "subscribe",
    "on", "register", "listen", "watch", "observe", "hook", "emit", "dispatch",
})

HANDLER_DECORATORS = frozenset({
    "on_event", "on_message", "on_connect", "on_disconnect", "on_ready",
    "listener", "handler", "callback", "event", "subscribe", "route",
})

def _is_spawn_call(callee_name: str) -> bool:
    """Check if a call is a spawn/fire-and-forget."""
    return callee_name in SPAWN_FUNCS

def _is_callback_call(node: Node) -> bool:
    """Check if a call is passing a callback function."""
    parent = node.parent
    if parent and parent.type in ("call", "argument_list"):
        # Check if this call is an argument to a callback function
        grandparent = parent.parent
        if grandparent and grandparent.type == "call":
            func_node = grandparent.child_by_field_name("function")
            if func_node and hasattr(func_node, 'text') and func_node.text is not None:
                func_name = func_node.text.decode("utf-8", errors="replace")
                if any(cb in func_name for cb in CALLBACK_FUNCS):
                    return True
    return False

def _is_deferred_context(node: Node) -> bool:
    """Check if a call is inside a deferred/handler context."""
    parent = node.parent
    while parent:
        if parent.type in ("function_definition", "function_declaration", "function_item",
                           "method_declaration", "function_definition", "async_item"):
            # Check for handler decorators
            for child in parent.children:
                if child.type == "decorator" and hasattr(child, 'text') and child.text is not None:
                    deco_name = child.text.decode("utf-8", errors="replace")
                    if any(hd in deco_name for hd in HANDLER_DECORATORS):
                        return True
        parent = parent.parent
    return False

def _get_guard(node: Node) -> str | None:
    """Get the guard condition for a call."""
    parent = node.parent
    while parent:
        if parent.type in ("if_statement", "while_statement", "for_statement"):
            # Get the condition
            condition_node = parent.child_by_field_name("condition")
            if condition_node and hasattr(condition_node, 'text') and condition_node.text is not None:
                text = condition_node.text.decode("utf-8", errors="replace")
                return text[:50]
            return "condition"
        if parent.type in ("try_statement", "with_statement"):
            return "try/with block"
        parent = parent.parent
    return None

def _get_receiver(node: Node) -> str | None:
    """Get the receiver for a method call."""
    if node.type == "call":
        func = node.child_by_field_name("function")
        if func and func.type == "attribute":
            parts = []
            attr_node = func
            while attr_node.type == "attribute":
                attr = attr_node.child_by_field_name("attribute")
                if attr and hasattr(attr, 'text') and attr.text is not None:
                    parts.append(attr.text.decode("utf-8", errors="replace"))
                obj = attr_node.child_by_field_name("object")
                if obj is None:
                    break
                attr_node = obj
            if attr_node and attr_node.type == "identifier" and hasattr(attr_node, 'text') and attr_node.text is not None:
                parts.append(attr_node.text.decode("utf-8", errors="replace"))
            if parts:
                return ".".join(reversed(parts))
    return None


def extract_graph(filepath: str, content: str) -> tuple[list[dict], list[dict], list[tuple[int, int, str]]]:
    """
    Extract symbols, cross-references, and syntax errors from a source file.

    Returns:
        (symbols, refs, syntax_errors) where:
        - symbols: list of dicts with keys: name, symbol_type, filepath, line, signature
        - refs: list of dicts with keys: caller_name, caller_line, callee_name, ref_type
        - syntax_errors: list of tuples (line, col, message)
    """
    lang = language_for(filepath)
    if lang is None:
        return [], [], []

    parser = get_parser(lang)
    tree = parser.parse(content.encode("utf-8"))
    source = content.encode("utf-8")

    # Load queries
    query_text = _read_query(lang)
    query = Query(get_language(lang), query_text)
    cursor = QueryCursor(query)

    symbols: list[dict] = []
    refs: list[dict] = []
    syntax_errors: list[tuple[int, int, str]] = []
    seen_symbols: set[tuple[str, str, str, int]] = set()

    # First pass: collect all symbols with their line numbers
    symbols_by_line: dict[int, str] = {}
    for pattern_idx, captures in cursor.matches(tree.root_node):
        # Get the captured name nodes for this pattern
        name_nodes = captures.get("name", [])
        for capture_name, nodes in captures.items():
            if capture_name.startswith("def."):
                for i, node in enumerate(nodes):
                    line = node.start_point[0] + 1
                    # Use captured name node if available, otherwise fall back to field lookup
                    if i < len(name_nodes):
                        def_name_node = name_nodes[i]
                        name = source[def_name_node.start_byte:def_name_node.end_byte].decode("utf-8", errors="replace")
                    else:
                        fallback_name_node: Node | None = node.child_by_field_name("name")
                        if fallback_name_node:
                            name = source[fallback_name_node.start_byte:fallback_name_node.end_byte].decode("utf-8", errors="replace")
                        else:
                            name = ""
                    sym_type = capture_name[4:]  # remove "def."
                    signature = None
                    if node.type in ("function_definition", "function_declaration", "function_item",
                                     "method_declaration", "constructor_declaration", "async_item"):
                        # Get first line of function as signature
                        decl_node = node.child_by_field_name("declarator") or node
                        sig_text = source[decl_node.start_byte:decl_node.end_byte].decode("utf-8", errors="replace")
                        signature = sig_text.split("\n")[0][:200]
                    elif node.type in ("class_definition", "class_declaration", "class_specifier",
                                       "interface_declaration", "struct_item", "struct_specifier",
                                       "enum_specifier", "enum_declaration", "type_declaration"):
                        # Get first line of class/struct/interface/enum declaration as signature
                        signature = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace").split("\n")[0][:200]
                    if name:
                        # Deduplicate symbols by (name, symbol_type, filepath, line)
                        key = (name, sym_type, filepath, line)
                        if key not in seen_symbols:
                            seen_symbols.add(key)
                            symbols_by_line[line] = name
                            symbols.append({
                                "name": name,
                                "symbol_type": sym_type,
                                "filepath": filepath,
                                "line": line,
                                "signature": signature,
                            })

    # Second pass: collect cross-references
    for pattern_idx, captures in cursor.matches(tree.root_node):
        # Collect base class names for inherit refs (extends and implements)
        base_names = []
        for cap_name in ("base", "impl"):
            if cap_name in captures:
                for node in captures.get(cap_name, []):
                    base_names.append(source[node.start_byte:node.end_byte].decode("utf-8", errors="replace"))

        # Get callee names for this pattern
        callee_nodes = captures.get("callee", [])

        for capture_name, nodes in captures.items():
            if capture_name.startswith("ref."):
                for i, node in enumerate(nodes):
                    line = node.start_point[0] + 1
                    ref_type = capture_name[4:]  # remove "ref."
                    if ref_type == "call":
                        # Find the caller (enclosing function/method)
                        caller_name = "module"
                        caller_line = line
                        parent = node.parent
                        while parent:
                            if parent.type in ("function_definition", "function_declaration", "function_item",
                                               "method_declaration", "function_definition", "async_item"):
                                caller_name_node: Node | None = parent.child_by_field_name("name")
                                if caller_name_node:
                                    caller_name = source[caller_name_node.start_byte:caller_name_node.end_byte].decode("utf-8", errors="replace")
                                caller_line = parent.start_point[0] + 1
                                # Qualify with class if inside one
                                caller_name = _get_qualified_name(caller_name, parent, source, lang, symbols_by_line)
                                break
# Handle case where call is inside a block that's a method body
                            # (block's parent is function_definition, not class_definition)
                            if parent.type == "block":
                                # Check if this block is the body of a function_definition
                                # by checking if parent.parent is a function_definition with this block as body
                                func_def = None
                                if parent.parent and parent.parent.type in ("function_definition", "function_declaration", "function_item",
                                                                          "method_declaration", "function_definition", "async_item"):
                                    # Check if this function's body is our block
                                    body = parent.parent.child_by_field_name("body")
                                    if body and body == parent:
                                        func_def = parent.parent
                                
                                # If not found, check previous sibling (for nested functions)
                                if not func_def:
                                    prev_sibling = parent.prev_sibling
                                    while prev_sibling:
                                        if prev_sibling.type in ("function_definition", "function_declaration", "function_item",
                                                                 "method_declaration", "function_definition", "async_item"):
                                            func_def = prev_sibling
                                            break
                                        prev_sibling = prev_sibling.prev_sibling
                                
                                # If not found, check parent's children (for class body case)
                                if not func_def and parent.parent:
                                    for child in parent.parent.children:
                                        if child.type in ("function_definition", "function_declaration", "function_item",
                                                          "method_declaration", "function_definition", "async_item"):
                                            # Check if this function's body is our block
                                            body = child.child_by_field_name("body")
                                            if body and body == parent:
                                                func_def = child
                                                break
                                
                                if func_def:
                                    func_name_node: Node | None = func_def.child_by_field_name("name")
                                    if func_name_node:
                                        caller_name = source[func_name_node.start_byte:func_name_node.end_byte].decode("utf-8", errors="replace")
                                    caller_line = func_def.start_point[0] + 1
                                    # Qualify with class if inside one
                                    caller_name = _get_qualified_name(caller_name, func_def, source, lang, symbols_by_line)
                                    break
                            parent = parent.parent
                        callee_name = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                        # Use callee capture for cleaner name if available
                        if callee_nodes and i < len(callee_nodes):
                            callee_node = callee_nodes[i]
                            callee_name = source[callee_node.start_byte:callee_node.end_byte].decode("utf-8", errors="replace")
                        
                        # Determine await_kind
                        await_kind = "sync"
                        await_node = node.parent
                        if await_node and await_node.type == "await":
                            await_kind = "await"
                        elif _is_spawn_call(callee_name):
                            await_kind = "spawn"
                        elif _is_callback_call(node):
                            await_kind = "callback"
                        elif _is_deferred_context(node):
                            await_kind = "deferred"
                        
                        # Determine guard condition
                        guard = _get_guard(node)
                        
                        # Determine receiver
                        receiver = _get_receiver(node)
                        
                        # Determine sequence number
                        seq = _get_call_seq(caller_name, line)
                        
                        if caller_name and callee_name:
                            refs.append({
                                "caller_name": caller_name,
                                "caller_line": caller_line,
                                "call_col": node.start_point[1] + 1,
                                "seq": seq,
                                "callee_name": callee_name,
                                "ref_type": "call",
                                "await_kind": await_kind,
                                "guard": guard,
                                "receiver": receiver,
                            })
                    elif ref_type == "import":
                        # Get module name from captures (for import_statement, require calls, etc.)
                        module_nodes = captures.get("module", [])
                        if module_nodes and i < len(module_nodes):
                            module_node = module_nodes[i]
                            module_name = source[module_node.start_byte:module_node.end_byte].decode("utf-8", errors="replace")
                        else:
                            module_node = node.child_by_field_name("module") or node
                            module_name = source[module_node.start_byte:module_node.end_byte].decode("utf-8", errors="replace")
                        # Clean up module name (remove quotes for string literals)
                        if module_name.startswith('"') and module_name.endswith('"') or module_name.startswith("'") and module_name.endswith("'"):
                            module_name = module_name[1:-1]
                        refs.append({
                            "caller_name": "module",
                            "caller_line": line,
                            "callee_name": module_name,
                            "ref_type": "import",
                        })
                    elif ref_type == "inherit":
                        # Get base class names from the corresponding base/impl captures
                        # The node might be the class declaration or a child (like argument_list)
                        # Find the class/struct/interface declaration that inherits
                        caller_name = "module"
                        class_node: Node | None = node
                        # Walk up to find the class/struct/interface declaration
                        while class_node and class_node.type not in ("class_definition", "class_declaration", "class_specifier",
                                                                   "struct_item", "struct_specifier", "interface_declaration",
                                                                   "type_declaration", "enum_declaration", "impl_item"):
                            class_node = class_node.parent
                        if class_node:
                            class_name_node: Node | None = class_node.child_by_field_name("name")
                            if class_name_node:
                                caller_name = source[class_name_node.start_byte:class_name_node.end_byte].decode("utf-8", errors="replace")
                        # Qualify the caller (derived class) name with its containing classes
                        if caller_name != "module" and class_node is not None:
                            caller_name = _get_qualified_name(caller_name, class_node, source, lang, symbols_by_line)
                        # Add inherit ref for each base class/interface
                        for base_name in base_names:
                            if caller_name and base_name:
                                refs.append({
                                    "caller_name": caller_name,
                                    "caller_line": line,
                                    "callee_name": base_name,
                                    "ref_type": "inherit",
                                })

    # Third pass: collect syntax errors
    def collect_errors(node: Node):
        if node.type == "ERROR" or node.type == "MISSING":
            line = node.start_point[0] + 1
            col = node.start_point[1] + 1
            text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            syntax_errors.append((line, col, f"syntax error near '{text[:50]}'"))
        for child in node.children:
            collect_errors(child)

    collect_errors(tree.root_node)

    return symbols, refs, syntax_errors