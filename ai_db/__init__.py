"""
ai_db: High-Performance Vector & Code Intelligence Database.
Code intelligence and retrieval for AI agents.

The package exports resolve on first access (PEP 562 ``__getattr__``): importing one
submodule -- ``ai_db.constants``, ``ai_db.errors``, ``ai_db.cli`` -- no longer imports the
storage, parser, search and analysis stack through this file. ``from ai_db import
VectorDB`` still works and imports exactly what ``VectorDB`` needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_db._lazy import lazy_exports

__version__ = "0.1.0"

# name -> module it lives in
_EXPORTS: dict[str, str] = {
    "AiDbConfigError": "ai_db.errors",
    "AiDbError": "ai_db.errors",
    "AiDbQueryError": "ai_db.errors",
    "AnalyzerEngine": "ai_db.analyzer.engine",
    "AppConfig": "ai_db.config",
    "chunk_file": "ai_db.parser.chunker",
    "compute_sha256": "ai_db.utils",
    "config_path": "ai_db.config",
    "ContextMemory": "ai_db.memory.context",
    "Database": "ai_db.storage.database",
    "DEFAULT_CONFIG_FILE": "ai_db.constants",
    "DEFAULT_DB_FILE": "ai_db.constants",
    "DEFAULT_SKILL_DIRS": "ai_db.constants",
    "detect_project_name": "ai_db.utils",
    "extract_file_outline": "ai_db.parser.ast_visitor",
    "extract_graph": "ai_db.parser.ts_graph",
    "format_as_sexp": "ai_db.analyzer.formatters",
    "format_as_stub": "ai_db.analyzer.formatters",
    "Formatters": "ai_db.analyzer.formatters",
    "get_allowed_projects": "ai_db.utils",
    "HARD_IGNORE_DIRS": "ai_db.constants",
    "INDEXABLE_EXTENSIONS": "ai_db.constants",
    "Indexer": "ai_db.search.indexer",
    "language_for": "ai_db.parser.languages",
    "load_config": "ai_db.config",
    "QueryEngine": "ai_db.search.query",
    "ReferenceStore": "ai_db.analyzer.references",
    "run_watch": "ai_db.facade",
    "should_index_path": "ai_db.utils",
    "SkillRouter": "ai_db.search.skills",
    "StorageBackend": "ai_db.storage.backend",
    "StorageBackendFactory": "ai_db.storage.factory",
    "strip_code_bloat": "ai_db.utils",
    "tokenize": "ai_db.utils",
    "validate_python_syntax": "ai_db.parser.linters",
    "VectorDB": "ai_db.facade",
    "VENDOR_NOISE_EXTENSIONS": "ai_db.constants",
}

__all__ = [
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_DB_FILE",
    "DEFAULT_SKILL_DIRS",
    "HARD_IGNORE_DIRS",
    "INDEXABLE_EXTENSIONS",
    "VENDOR_NOISE_EXTENSIONS",
    "AiDbConfigError",
    "AiDbError",
    "AiDbQueryError",
    "AnalyzerEngine",
    "AppConfig",
    "ContextMemory",
    "Database",
    "Formatters",
    "Indexer",
    "QueryEngine",
    "ReferenceStore",
    "SkillRouter",
    "StorageBackend",
    "StorageBackendFactory",
    "VectorDB",
    "__version__",
    "chunk_file",
    "compute_sha256",
    "config_path",
    "detect_project_name",
    "extract_file_outline",
    "extract_graph",
    "format_as_sexp",
    "format_as_stub",
    "get_allowed_projects",
    "language_for",
    "load_config",
    "run_watch",
    "should_index_path",
    "strip_code_bloat",
    "tokenize",
    "validate_python_syntax",
]

__getattr__, __dir__ = lazy_exports(__name__, globals(), _EXPORTS)

if TYPE_CHECKING:
    from ai_db.analyzer.engine import AnalyzerEngine
    from ai_db.analyzer.formatters import Formatters, format_as_sexp, format_as_stub
    from ai_db.analyzer.references import ReferenceStore
    from ai_db.config import AppConfig, config_path, load_config
    from ai_db.constants import (
        DEFAULT_CONFIG_FILE,
        DEFAULT_DB_FILE,
        DEFAULT_SKILL_DIRS,
        HARD_IGNORE_DIRS,
        INDEXABLE_EXTENSIONS,
        VENDOR_NOISE_EXTENSIONS,
    )
    from ai_db.errors import AiDbConfigError, AiDbError, AiDbQueryError
    from ai_db.facade import VectorDB, run_watch
    from ai_db.memory.context import ContextMemory
    from ai_db.parser.ast_visitor import extract_file_outline
    from ai_db.parser.chunker import chunk_file
    from ai_db.parser.languages import language_for
    from ai_db.parser.linters import validate_python_syntax
    from ai_db.parser.ts_graph import extract_graph
    from ai_db.search.indexer import Indexer
    from ai_db.search.query import QueryEngine
    from ai_db.search.skills import SkillRouter
    from ai_db.storage.backend import StorageBackend
    from ai_db.storage.database import Database
    from ai_db.storage.factory import StorageBackendFactory
    from ai_db.utils import (
        compute_sha256,
        detect_project_name,
        get_allowed_projects,
        should_index_path,
        strip_code_bloat,
        tokenize,
    )
