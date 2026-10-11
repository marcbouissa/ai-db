"""File outline extraction using tree-sitter queries."""
from __future__ import annotations

from collections.abc import Collection

from ai_db.parser.languages import SUPPORTED_LANGUAGES, enabled_language_for
from ai_db.parser.ts_graph import extract_graph


def extract_file_outline(filepath: str, content: str,
                         languages: Collection[str] = SUPPORTED_LANGUAGES) -> list[tuple[int, str]]:
    """
    Extracts file outline lines (line number and label) for fast inspection:
    tree-sitter symbols for a selected language, section headers for markdown,
    nothing for other text.
    """
    lang = enabled_language_for(filepath, languages)
    if lang is None:
        # No grammar: markdown has its section tree; other text has no structure to list.
        if not filepath.lower().endswith(".md"):
            return []
        from ai_db.parser.chunker import chunk_file
        return [(c["start_line"], "  " * c["qualified_name"].count(" > ") + c["name"])
                for c in chunk_file(filepath, content, languages)]

    symbols, _, _ = extract_graph(filepath, content)
    outline = []
    seen: set[tuple[int, str]] = set()
    for s in symbols:
        if s.get("local"):
            continue  # defs/assignments inside function bodies are noise in an outline
        label = f"{s['symbol_type']} {s['name']}"
        if s.get("signature"):
            sig = s["signature"]
            if sig.startswith("async "):
                label = f"async {label}"
            elif not label.startswith(("class ", "interface ", "struct ", "enum ", "type ", "property ")):
                label = f"def {s['name']}"
        label = "  " * int(s.get("class_depth", 0)) + label
        if (s["line"], label) not in seen:
            seen.add((s["line"], label))
            outline.append((s["line"], label))

    outline.sort(key=lambda x: x[0])
    return outline
