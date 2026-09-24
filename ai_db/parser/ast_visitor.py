import ast
import os
import re
from typing import Any


def extract_symbols(filepath: str, content: str) -> list[dict[str, Any]]:
    """Extracts code symbols (classes, functions, methods, types) from source code."""
    ext = os.path.splitext(filepath)[1].lower()
    symbols = []

    if ext in (".py", ".pyi"):
        try:
            tree = ast.parse(content, filename=filepath)
            lines = content.splitlines()

            class SymbolVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.scope_prefix = []

                def visit_ClassDef(self, node: ast.ClassDef):
                    sig = f"class {node.name}"
                    bases = []
                    for b in node.bases:
                        if isinstance(b, ast.Name):
                            bases.append(b.id)
                        elif isinstance(b, ast.Attribute):
                            bases.append(f"{getattr(b.value, 'id', '')}.{b.attr}")
                    if bases:
                        sig += f"({', '.join(bases)})"

                    symbols.append({
                        "name": node.name,
                        "symbol_type": "class",
                        "filepath": filepath,
                        "line": node.lineno,
                        "signature": sig
                    })
                    self.scope_prefix.append(node.name)
                    self.generic_visit(node)
                    self.scope_prefix.pop()

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    self._handle_func(node, is_async=False)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                    self._handle_func(node, is_async=True)

                def _handle_func(self, node, is_async: bool):
                    prefix = "async def " if is_async else "def "
                    func_line = lines[node.lineno - 1].strip() if 0 <= node.lineno - 1 < len(lines) else f"{prefix}{node.name}(...)"
                    sig = func_line[:120].rstrip(":")
                    if not sig.startswith("def ") and not sig.startswith("async def "):
                        sig = f"{prefix}{node.name}(...)"

                    sym_type = "def"
                    symbols.append({
                        "name": node.name,
                        "symbol_type": sym_type,
                        "filepath": filepath,
                        "line": node.lineno,
                        "signature": sig
                    })
                    self.scope_prefix.append(node.name)
                    self.generic_visit(node)
                    self.scope_prefix.pop()

            SymbolVisitor().visit(tree)
            return symbols
        except Exception:
            pass

    # Generic regex-based symbol extractor for other languages or unparseable Python
    symbol_pattern = re.compile(
        r"^(?:\s*(?:export\s+|public\s+|private\s+|protected\s+|static\s+|async\s+)*)"
        r"(?:(class|interface|type|struct|enum)\s+([A-Za-z0-9_$]+)"
        r"|(?:def|function|fn)\s+([A-Za-z0-9_$]+)"
        r"|const\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)"
    )

    for lineno, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        m = symbol_pattern.match(stripped)
        if m:
            kind = m.group(1)
            name = m.group(2) or m.group(3) or m.group(4)
            if name:
                sym_type = kind if kind else "def"
                sig = stripped[:120].rstrip("{:;")
                symbols.append({
                    "name": name,
                    "symbol_type": sym_type,
                    "filepath": filepath,
                    "line": lineno,
                    "signature": sig
                })

    return symbols


def extract_file_outline(filepath: str, content: str) -> list[tuple[int, str]]:
    """Extracts file outline lines (line number and label) for fast inspection."""
    ext = os.path.splitext(filepath)[1].lower()
    lines = content.splitlines()
    outline = []

    if ext in (".py", ".pyi"):
        try:
            tree = ast.parse(content, filename=filepath)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    args_list = [a.arg for a in node.args.args]
                    arg_str = ", ".join(args_list)
                    outline.append((node.lineno, f"{prefix} {node.name}({arg_str})"))
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            sub_prefix = "async def" if isinstance(sub, ast.AsyncFunctionDef) else "def"
                            args_list = [a.arg for a in sub.args.args]
                            arg_str = ", ".join(args_list)
                            outline.append((sub.lineno, f"  {sub_prefix} {sub.name}({arg_str})"))
                elif isinstance(node, ast.ClassDef):
                    bases = [getattr(b, 'id', getattr(b, 'attr', '')) for b in node.bases]
                    base_str = f"({', '.join(b for b in bases if b)})" if bases else ""
                    outline.append((node.lineno, f"class {node.name}{base_str}"))
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            sub_prefix = "async def" if isinstance(sub, ast.AsyncFunctionDef) else "def"
                            args_list = [a.arg for a in sub.args.args]
                            arg_str = ", ".join(args_list)
                            outline.append((sub.lineno, f"  {sub_prefix} {sub.name}({arg_str})"))
                        elif isinstance(sub, ast.ClassDef):
                            outline.append((sub.lineno, f"  class {sub.name}"))
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and (target.id.isupper() or target.id.startswith("_")):
                            val_type = type(node.value).__name__.lower()
                            outline.append((node.lineno, f"{target.id} ({val_type})"))
            outline.sort(key=lambda x: x[0])
            return outline
        except Exception:
            pass

    # Generic outline fallback using regex
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "/*", "*")):
            continue
        m = re.match(
            r"^(?:export\s+|public\s+|private\s+|protected\s+|static\s+|async\s+)*"
            r"(class\s+\w+|interface\s+\w+|type\s+\w+|def\s+\w+\([^)]*\)|function\s+\w+\([^)]*\)|fn\s+\w+|[A-Z0-9_]{3,}\s*=)",
            stripped
        )
        if m:
            outline.append((i, stripped[:100]))

    return outline

