#!/usr/bin/env python3
"""
vectordb.py: backward-compatible facade shim for ai_db.

Re-exports the public ``ai_db`` surface so older imports (``import vectordb``,
``from vectordb import VectorDB``) keep working, and forwards CLI invocations
to :func:`ai_db.cli.main`. Declared as a root ``py-module`` in ``pyproject.toml``
and executed directly as ``python vectordb.py ...``, so this file must stay
valid Python.
"""

import os
import sys
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    # Allow running straight from a source checkout, before installation.
    sys.path.insert(0, CURRENT_DIR)

from ai_db import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_DB_FILE,
    DEFAULT_SKILL_DIRS,
    HARD_IGNORE_DIRS,
    INDEXABLE_EXTENSIONS,
    VENDOR_NOISE_EXTENSIONS,
    AiDbConfigError,
    AiDbError,
    AiDbQueryError,
    AnalyzerEngine,
    AppConfig,
    ContextMemory,
    Database,
    Formatters,
    Indexer,
    QueryEngine,
    ReferenceStore,
    SkillRouter,
    StorageBackend,
    StorageBackendFactory,
    VectorDB,
    __version__,
    chunk_file,
    compute_sha256,
    config_path,
    detect_project_name,
    extract_file_outline,
    extract_graph,
    format_as_sexp,
    format_as_stub,
    get_allowed_projects,
    get_session_state,
    language_for,
    load_config,
    run_watch,
    set_session_state,
    should_index_path,
    strip_code_bloat,
    tokenize,
    validate_python_syntax,
)
from ai_db.cli import main


def extract_symbols(filepath: str, content: str) -> list[dict[str, Any]]:
    """Thin backward-compat wrapper: symbols only, via the tree-sitter extractor.

    ``cross_refs.extract_symbols`` was replaced by ``ts_graph.extract_graph``,
    which returns ``(symbols, refs, syntax_errors)``. Legacy callers that only
    want symbols are served from element ``[0]``.
    """
    return extract_graph(filepath, content)[0]


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
    "extract_symbols",
    "format_as_sexp",
    "format_as_stub",
    "get_allowed_projects",
    "get_session_state",
    "language_for",
    "load_config",
    "main",
    "run_watch",
    "set_session_state",
    "should_index_path",
    "strip_code_bloat",
    "tokenize",
    "validate_python_syntax",
]


if __name__ == "__main__":
    sys.exit(main())
