"""Pluggable storage layer for ai-db."""

from ai_db.storage.models import (
    FileRecord, ChunkRecord, SymbolRecord, SymbolRefRecord,
    AnnotationRecord, SyntaxErrorRecord, SkillRecord,
    ContextRecord, AnalysisRefRecord, SearchResult
)
from ai_db.storage.backend import StorageBackend, VectorCapable
from ai_db.storage.sqlite_backend import SQLiteBackend
from ai_db.storage.factory import StorageBackendFactory
from ai_db.storage.database import Database
from ai_db.storage.state import (
    get_session_state,
    set_session_state,
    get_telemetry_state,
    set_telemetry_state,
    get_telemetry_table_counts,
    get_telemetry_weak_points,
)

__all__ = [
    "FileRecord",
    "ChunkRecord",
    "SymbolRecord",
    "SymbolRefRecord",
    "AnnotationRecord",
    "SyntaxErrorRecord",
    "SkillRecord",
    "ContextRecord",
    "AnalysisRefRecord",
    "SearchResult",
    "StorageBackend",
    "VectorCapable",
    "SQLiteBackend",
    "StorageBackendFactory",
    "Database",
    "get_session_state",
    "set_session_state",
    "get_telemetry_state",
    "set_telemetry_state",
    "get_telemetry_table_counts",
    "get_telemetry_weak_points",
]
