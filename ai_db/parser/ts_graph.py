"""
ai_db.parser.ts_graph
Tree-sitter based extraction of symbols, cross-references, and syntax errors.
Supports all languages with .scm query files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ai_db.parser.languages import declared_name, language_for
from ai_db.parser.naming import bound_variable, definition_name, qualified_name

if TYPE_CHECKING:
    # Annotation-only: `from __future__ import annotations` makes every `Node`
    # in a signature a string, so this costs nothing at runtime.
    from tree_sitter import Node



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


def _read_query(lang: str) -> str:
    """Read the .scm query file for a language."""
    query_path = Path(__file__).parent / "queries" / f"{lang}.scm"
    if not query_path.exists():
        raise ValueError(f"No query file for language: {lang}")
    return query_path.read_text(encoding="utf-8")


def _python_in_class_body(func: Node) -> bool:
    """True when a Python function_definition is a direct member of a class body."""
    parent = func.parent
    if parent is not None and parent.type == "decorated_definition":
        parent = parent.parent
    return (parent is not None and parent.type == "block"
            and parent.parent is not None and parent.parent.type == "class_definition")


_CLASS_NODES = ("class_definition", "class_declaration", "class_specifier", "struct_item",
                "struct_specifier", "interface_declaration", "enum_declaration", "impl_item",
                "struct_declaration", "record_declaration", "clazz")
_FUNCTION_NODES = ("function_definition", "function_declaration", "function_item",
                   "method_declaration", "constructor_declaration", "arrow_function",
                   "function_expression", "lambda", "destructor_declaration",
                   "local_function_statement")
# Nodes whose name qualifies the symbols inside them (classes, impls, namespaces).
_QUALIFYING_NODES = _CLASS_NODES + ("type_declaration", "namespace_definition",
                                    "namespace_declaration")
# Nodes a call is attributed to (the enclosing "caller").
_CALLER_NODES = ("function_definition", "function_declaration", "function_item",
                 "method_declaration", "async_item", "constructor_declaration",
                 "destructor_declaration", "local_function_statement",
                 # JS/TS class methods and generators
                 "method_definition", "generator_function_declaration",
                 # Pkl: methods, and properties (a property's value is an expression,
                 # so calls in it belong to the property).
                 "classMethod", "classProperty")


def _scope_of(node: Node) -> tuple[int, bool]:
    """(number of enclosing classes, defined inside a function body)."""
    depth, local = 0, False
    parent = node.parent
    while parent is not None:
        if parent.type in _CLASS_NODES:
            depth += 1
        elif parent.type in _FUNCTION_NODES:
            local = True
        parent = parent.parent
    return depth, local


def _bound_top_function(node: Node) -> bool:
    """A JS/TS function value bound to a module-level (or exported) variable: the chunker
    makes it a chunk named after the variable, so it is a caller scope too."""
    var = bound_variable(node)
    decl = var.parent if var is not None else None
    return decl is not None and decl.parent is not None and decl.parent.type in ("program", "export_statement")


def _caller_scope(node: Node, source: bytes, lang: str) -> tuple[str, int] | None:
    """(``module.<qualified name>``, def line) of the definition a call belongs to, or None
    for module-level code. Unnamed functions (lambdas, callbacks) belong to their enclosing
    definition. Names come from ``ai_db.parser.naming``, like the chunks' names."""
    parent = node.parent
    while parent is not None:
        if parent.type in _CALLER_NODES or _bound_top_function(parent):
            name = definition_name(parent, source, lang)
            if name:
                return "module." + qualified_name(parent, source, lang, own=name), parent.start_point[0] + 1
        parent = parent.parent
    return None


def _first_line(source: bytes, node: Node) -> str:
    """The node's first source line, at most 200 characters, decoding only that line."""
    end = source.find(b"\n", node.start_byte, node.end_byte)
    end = node.end_byte if end < 0 else end
    return source[node.start_byte:min(end, node.start_byte + 800)].decode("utf-8", errors="replace")[:200]


# Helper functions for call analysis
def _get_call_seq(counter: dict[str, int], caller_name: str, line: int) -> int:
    """Sequence number of a call within its caller and line (1, 2, ... in match order).

    ``counter`` is per ``extract_graph`` call: it used to be module-global and never
    reset, so parsing the same file twice gave different numbers."""
    key = f"{caller_name}:{line}"
    counter[key] = counter.get(key, 0) + 1
    return counter[key]

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

    # Imported here, not at module scope: tree-sitter costs ~34 ms to import and
    # five modules (including ai_db/__init__.py) import this one eagerly, so a
    # module-level import made every `ai-db` command pay for a parser that
    # retrieval never uses -- a query is served entirely from SQLite + FTS.
    from tree_sitter import Query, QueryCursor
    from tree_sitter_language_pack import get_language, get_parser

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
                        fallback_name_node: Node | None = declared_name(node)
                        if fallback_name_node:
                            name = source[fallback_name_node.start_byte:fallback_name_node.end_byte].decode("utf-8", errors="replace")
                        else:
                            name = ""
                    sym_type = capture_name[4:]  # remove "def."
                    if lang == "python" and sym_type == "function" and _python_in_class_body(node):
                        continue  # emitted by the method pattern; one symbol per def
                    class_depth, is_local = _scope_of(node)
                    if lang == "python" and sym_type == "type" and is_local:
                        continue  # function-local annotated/typed assignment, not a type alias
                    signature = None
                    if node.type in _CALLER_NODES:
                        # Get first line of function as signature
                        decl_node = node.child_by_field_name("declarator") or node
                        signature = _first_line(source, decl_node)
                    elif node.type in _CLASS_NODES + ("enum_specifier", "type_declaration"):
                        # Get first line of class/struct/interface/enum declaration as signature
                        signature = _first_line(source, node)
                    if name:
                        # Deduplicate symbols by (name, symbol_type, filepath, line)
                        key = (name, sym_type, filepath, line)
                        if key not in seen_symbols:
                            seen_symbols.add(key)
                            symbols.append({
                                "name": name,
                                "symbol_type": sym_type,
                                "filepath": filepath,
                                "line": line,
                                "signature": signature,
                                "class_depth": class_depth,
                                "local": is_local,
                            })

    # Second pass: collect cross-references
    from ai_db.parser.receiver_types import ReceiverResolver
    receivers = ReceiverResolver(source)  # per-scope tables, built once per file
    call_seq: dict[str, int] = {}
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
                        scope = _caller_scope(node, source, lang)
                        caller_name, caller_line = scope if scope is not None else ("module", line)
                        # The callee capture's text; the whole call expression only when the
                        # pattern has none (decoding it first was quadratic in nesting).
                        named = callee_nodes[i] if callee_nodes and i < len(callee_nodes) else node
                        callee_name = source[named.start_byte:named.end_byte].decode("utf-8", errors="replace")
                        
                        # Determine await_kind
                        await_kind = "sync"
                        await_node = node.parent
                        if await_node and await_node.type in ("await", "await_expression"):
                            await_kind = "await"
                        elif _is_spawn_call(callee_name):
                            await_kind = "spawn"
                        elif _is_callback_call(node):
                            await_kind = "callback"
                        elif _is_deferred_context(node):
                            await_kind = "deferred"
                        
                        # Determine guard condition
                        guard = _get_guard(node)
                        
                        # Determine receiver and (python only) the class it denotes
                        receiver = _get_receiver(node)
                        # `receiver` is the dotted call target ("self.bar"); the qualifier is
                        # the object part alone ("self").
                        qualifier = receiver.rsplit(".", 1)[0] if receiver and "." in receiver else None
                        callee_resolved = None
                        if lang == "python" and qualifier:
                            recv_cls = receivers.resolve(node, qualifier)
                            if recv_cls:
                                callee_resolved = f"{recv_cls}.{callee_name}"

                        # Call-site position: the callee identifier (unique per call, even
                        # for chained ``a.f().g()`` where both calls start at the same byte).
                        site = callee_nodes[i] if callee_nodes and i < len(callee_nodes) else node
                        
                        # Determine sequence number
                        seq = _get_call_seq(call_seq, caller_name, line)
                        
                        if caller_name and callee_name:
                            refs.append({
                                "caller_name": caller_name,
                                "caller_line": caller_line,
                                "call_line": site.start_point[0] + 1,
                                "call_col": site.start_point[1] + 1,
                                "seq": seq,
                                "callee_name": callee_name,
                                "ref_type": "call",
                                "await_kind": await_kind,
                                "guard": guard,
                                "receiver": receiver,
                                "callee_qualifier": qualifier,
                                "callee_resolved": callee_resolved,
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
                        while class_node and class_node.type not in _QUALIFYING_NODES:
                            class_node = class_node.parent
                        class_name = definition_name(class_node, source, lang) if class_node else None
                        if class_node is not None and class_name:
                            # The derived class, qualified by its containers
                            caller_name = "module." + qualified_name(class_node, source, lang, own=class_name)
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
        # `is_error`/`is_missing` are the flags to test, not the node type. A
        # missing node carries the *expected token's* type -- `def broken(:`
        # yields a node typed ")" with is_missing set -- so comparing
        # node.type against the literal strings "ERROR"/"MISSING" never matches
        # and these errors went unreported.
        if node.is_error or node.is_missing:
            line = node.start_point[0] + 1
            col = node.start_point[1] + 1
            text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            kind = "error: missing" if node.is_missing else "error near"
            detail = node.type if node.is_missing else f"'{text[:50]}'"
            syntax_errors.append((line, col, f"syntax {kind} {detail}"))
        for child in node.children:
            collect_errors(child)

    collect_errors(tree.root_node)

    return symbols, refs, syntax_errors