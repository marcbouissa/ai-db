"""Pluggable storage layer for ai-db."""

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
from ai_db.storage.state import (
    get_session_state,
    get_telemetry_state,
    get_telemetry_table_counts,
    get_telemetry_weak_points,
    set_session_state,
    set_telemetry_state,
)

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
    "get_session_state",
    "get_telemetry_state",
    "get_telemetry_table_counts",
    "get_telemetry_weak_points",
    "set_session_state",
    "set_telemetry_state",
]
