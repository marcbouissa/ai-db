"""File outline extraction using tree-sitter queries."""
from __future__ import annotations

import os
from typing import Any

from ai_db.parser.ts_graph import extract_graph, language_for


def extract_file_outline(filepath: str, content: str) -> list[tuple[int, str]]:
    """
    Extracts file outline lines (line number and label) for fast inspection.
    Uses tree-sitter queries for supported languages.
    """
    lang = language_for(filepath)
    if lang is None:
        # Fallback for unsupported languages
        return _regex_outline(content)

    symbols, _, _ = extract_graph(filepath, content)
    outline = []
    for s in symbols:
        label = f"{s['symbol_type']} {s['name']}"
        if s.get("signature"):
            sig = s["signature"]
            if sig.startswith("async "):
                label = f"async {label}"
            elif not label.startswith(("class ", "interface ", "struct ", "enum ", "type ")):
                label = f"def {s['name']}"
        outline.append((s["line"], label))

    outline.sort(key=lambda x: x[0])
    return outline


def _regex_outline(content: str) -> list[tuple[int, str]]:
    """Generic regex-based outline fallback."""
    import re
    outline = []
    for i, line in enumerate(content.splitlines(), 1):
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