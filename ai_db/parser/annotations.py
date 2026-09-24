"""
ai_db.parser.annotations
Extracts TODO/FIXME/HACK/NOTE/XXX comment tags and Python docstrings from source files.
"""
import ast
import os
import re
from typing import Any

# Pattern matches: # TODO: message  or  // FIXME message  or  /* HACK: ... */
_TAG_PATTERN = re.compile(
    r"(?:#|//|/\*)\s*(TODO|FIXME|HACK|NOTE|XXX)\s*:?\s*(.+?)(?:\*/)?$",
    re.IGNORECASE
)


def extract_annotations(filepath: str, content: str) -> list[dict[str, Any]]:
    """
    Returns a list of annotation dicts:
    {line, kind, symbol (or None), content}
    """
    results: list[dict[str, Any]] = []
    lines = content.splitlines()

    # 1. Scan all lines for tag comments (works for all languages)
    for i, line in enumerate(lines, 1):
        m = _TAG_PATTERN.search(line)
        if m:
            kind = m.group(1).lower()
            text = m.group(2).strip()
            if text:
                results.append({
                    "line": i,
                    "kind": kind,
                    "symbol": None,
                    "content": text[:500],
                })

    # 2. Python docstrings (module, class, function level)
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in (".py", ".pyi"):
        return results

    try:
        tree = ast.parse(content, filename=filepath)
    except (SyntaxError, ValueError):
        return results  # unparseable file: syntax error is recorded separately

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            doc = ast.get_docstring(node, clean=True)
            if doc and doc.strip():
                symbol = getattr(node, "name", None)
                doc_line = node.lineno if not isinstance(node, ast.Module) else 1
                results.append({
                    "line": doc_line,
                    "kind": "docstring",
                    "symbol": symbol,
                    "content": doc.strip()[:800],
                })

    return results
