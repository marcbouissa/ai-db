"""Per-file token-optimised analysis (analyze, expand, diff).

Exports resolve lazily (``ai_db._lazy``): importing one submodule of this package does
not import the others.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_db._lazy import lazy_exports

_EXPORTS = {
    "AnalyzerEngine": "ai_db.analyzer.engine",
    "format_as_sexp": "ai_db.analyzer.formatters",
    "format_as_stub": "ai_db.analyzer.formatters",
    "format_pack": "ai_db.analyzer.formatters",
    "Formatters": "ai_db.analyzer.formatters",
    "ReferenceStore": "ai_db.analyzer.references",
}

__all__ = [
    "AnalyzerEngine",
    "Formatters",
    "ReferenceStore",
    "format_as_sexp",
    "format_as_stub",
    "format_pack",
]

__getattr__, __dir__ = lazy_exports(__name__, globals(), _EXPORTS)

if TYPE_CHECKING:
    from ai_db.analyzer.engine import AnalyzerEngine
    from ai_db.analyzer.formatters import Formatters, format_as_sexp, format_as_stub, format_pack
    from ai_db.analyzer.references import ReferenceStore
