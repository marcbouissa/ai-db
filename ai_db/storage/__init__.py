"""Pluggable storage layer for ai-db.

Exports resolve lazily (``ai_db._lazy``): importing one submodule of this package does
not import the others.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_db._lazy import lazy_exports

_EXPORTS = {
    "AnalysisRefRecord": "ai_db.storage.models",
    "AnnotationRecord": "ai_db.storage.models",
    "ChunkRecord": "ai_db.storage.models",
    "ContextRecord": "ai_db.storage.models",
    "Database": "ai_db.storage.database",
    "FileRecord": "ai_db.storage.models",
    "SearchResult": "ai_db.storage.models",
    "SkillRecord": "ai_db.storage.models",
    "SQLiteBackend": "ai_db.storage.sqlite_backend",
    "StorageBackend": "ai_db.storage.backend",
    "StorageBackendFactory": "ai_db.storage.factory",
    "SymbolRecord": "ai_db.storage.models",
    "SymbolRefRecord": "ai_db.storage.models",
    "SyntaxErrorRecord": "ai_db.storage.models",
    "VectorCapable": "ai_db.storage.backend",
}

__all__ = [
    "AnalysisRefRecord",
    "AnnotationRecord",
    "ChunkRecord",
    "ContextRecord",
    "Database",
    "FileRecord",
    "SQLiteBackend",
    "SearchResult",
    "SkillRecord",
    "StorageBackend",
    "StorageBackendFactory",
    "SymbolRecord",
    "SymbolRefRecord",
    "SyntaxErrorRecord",
    "VectorCapable",
]

__getattr__, __dir__ = lazy_exports(__name__, globals(), _EXPORTS)

if TYPE_CHECKING:
    from ai_db.storage.backend import StorageBackend, VectorCapable
    from ai_db.storage.database import Database
    from ai_db.storage.factory import StorageBackendFactory
    from ai_db.storage.models import (
        AnalysisRefRecord,
        AnnotationRecord,
        ChunkRecord,
        ContextRecord,
        FileRecord,
        SearchResult,
        SkillRecord,
        SymbolRecord,
        SymbolRefRecord,
        SyntaxErrorRecord,
    )
    from ai_db.storage.sqlite_backend import SQLiteBackend
