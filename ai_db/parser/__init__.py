"""Parsing: tree-sitter chunks, symbols, refs, outlines and syntax checks.

Exports resolve lazily (``ai_db._lazy``): importing one submodule of this package does
not import the others.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_db._lazy import lazy_exports

_EXPORTS = {
    "chunk_file": "ai_db.parser.chunker",
    "extract_file_outline": "ai_db.parser.ast_visitor",
    "extract_graph": "ai_db.parser.ts_graph",
    "language_for": "ai_db.parser.languages",
    "validate_python_syntax": "ai_db.parser.linters",
}

__all__ = [
    "chunk_file",
    "extract_file_outline",
    "extract_graph",
    "language_for",
    "validate_python_syntax",
]

__getattr__, __dir__ = lazy_exports(__name__, globals(), _EXPORTS)

if TYPE_CHECKING:
    from ai_db.parser.ast_visitor import extract_file_outline
    from ai_db.parser.chunker import chunk_file
    from ai_db.parser.languages import language_for
    from ai_db.parser.linters import validate_python_syntax
    from ai_db.parser.ts_graph import extract_graph
