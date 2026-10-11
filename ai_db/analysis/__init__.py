"""Code analysis: ``investigate`` (one call returns a ranked evidence pack), per-file
``analyze``/``expand`` (``analyze``, ``refs``), the diff engine (``diff``), call-flow
``trace`` and the pack/trace renderers (``formatters``, ``trace_format``).

Exports resolve lazily (``ai_db._lazy``): importing one submodule of this package does
not import the others.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_db._lazy import lazy_exports

_EXPORTS = {
    "AnalyzerEngine": "ai_db.analysis.analyze",
    "Formatters": "ai_db.analysis.formatters",
    "InvestigationPack": "ai_db.analysis.pack",
    "Investigator": "ai_db.analysis.investigate",
    "ReferenceStore": "ai_db.analysis.refs",
    "format_as_sexp": "ai_db.analysis.formatters",
    "format_as_stub": "ai_db.analysis.formatters",
    "format_pack": "ai_db.analysis.formatters",
}

__all__ = [
    "AnalyzerEngine",
    "Formatters",
    "InvestigationPack",
    "Investigator",
    "ReferenceStore",
    "format_as_sexp",
    "format_as_stub",
    "format_pack",
]

__getattr__, __dir__ = lazy_exports(__name__, globals(), _EXPORTS)

if TYPE_CHECKING:
    from ai_db.analysis.analyze import AnalyzerEngine
    from ai_db.analysis.formatters import Formatters, format_as_sexp, format_as_stub, format_pack
    from ai_db.analysis.investigate import Investigator
    from ai_db.analysis.pack import InvestigationPack
    from ai_db.analysis.refs import ReferenceStore
