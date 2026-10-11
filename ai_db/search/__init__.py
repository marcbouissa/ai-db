"""Indexing, retrieval and skill routing.

Exports resolve lazily (``ai_db._lazy``): importing one submodule of this package does
not import the others.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_db._lazy import lazy_exports

_EXPORTS = {
    "Indexer": "ai_db.search.indexer",
    "QueryEngine": "ai_db.search.query",
    "SkillRouter": "ai_db.search.skills",
}

__all__ = [
    "Indexer",
    "QueryEngine",
    "SkillRouter",
]

__getattr__, __dir__ = lazy_exports(__name__, globals(), _EXPORTS)

if TYPE_CHECKING:
    from ai_db.search.indexer import Indexer
    from ai_db.search.query import QueryEngine
    from ai_db.search.skills import SkillRouter
