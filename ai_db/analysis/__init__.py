"""Deterministic investigation: one call returns a ranked evidence pack.

Exports resolve lazily (``ai_db._lazy``): importing one submodule of this package does
not import the others.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_db._lazy import lazy_exports

_EXPORTS = {
    "InvestigationPack": "ai_db.analysis.pack",
    "Investigator": "ai_db.analysis.investigate",
}

__all__ = [
    "InvestigationPack",
    "Investigator",
]

__getattr__, __dir__ = lazy_exports(__name__, globals(), _EXPORTS)

if TYPE_CHECKING:
    from ai_db.analysis.investigate import Investigator
    from ai_db.analysis.pack import InvestigationPack
