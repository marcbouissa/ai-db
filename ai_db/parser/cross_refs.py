"""
ai_db.parser.cross_refs
Extracts symbol cross-references (calls, imports, inheritance) from source code.
Phase 2: Python-only. JS/TS/Rust support planned for Phase 3.
"""
import ast
import os
from typing import Any


def extract_cross_refs(filepath: str, content: str) -> list[dict[str, Any]]:
    """
    Extracts call sites, imports, and class inheritance from a source file.
    Returns a list of dicts: {caller_name, caller_line, callee_name, ref_type}
    ref_type in: 'call', 'import', 'inherit'
    """
    ext = os.path.splitext(filepath)[1].lower()
    refs: list[dict[str, Any]] = []

    if ext not in (".py", ".pyi"):
        return refs  # Phase 2: Python only

    try:
        tree = ast.parse(content, filename=filepath)
    except Exception:
        return refs

    current_scope = ["module"]

    class RefVisitor(ast.NodeVisitor):

        def _scope(self) -> str:
            return ".".join(current_scope)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            current_scope.append(node.name)
            self.generic_visit(node)
            current_scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef):
            # Collect inheritance references
            for base in node.bases:
                name = None
                if isinstance(base, ast.Name):
                    name = base.id
                elif isinstance(base, ast.Attribute):
                    name = base.attr
                if name:
                    refs.append({
                        "caller_name": node.name,
                        "caller_line": node.lineno,
                        "callee_name": name,
                        "ref_type": "inherit",
                    })
            current_scope.append(node.name)
            self.generic_visit(node)
            current_scope.pop()

        def visit_Call(self, node: ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name:
                refs.append({
                    "caller_name": self._scope(),
                    "caller_line": node.lineno,
                    "callee_name": name,
                    "ref_type": "call",
                })
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                refs.append({
                    "caller_name": self._scope(),
                    "caller_line": node.lineno,
                    "callee_name": alias.name,
                    "ref_type": "import",
                })

        def visit_ImportFrom(self, node: ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                callee = f"{module}.{alias.name}" if module else alias.name
                refs.append({
                    "caller_name": self._scope(),
                    "caller_line": node.lineno,
                    "callee_name": callee,
                    "ref_type": "import",
                })

    RefVisitor().visit(tree)
    return refs
