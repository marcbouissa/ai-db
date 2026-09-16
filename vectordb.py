#!/usr/bin/env python3
"""
vectordb.py: Backward-compatible facade shim for ai_db.
Preserves existing CLI commands, MCP server imports, and symlinks.
"""

import sys
from ai_db import (
    DEFAULT_DB_FILE,
    DEFAULT_CONFIG_FILE,
    DEFAULT_SKILL_DIRS,
    INDEXABLE_EXTENSIONS,
    HARD_IGNORE_DIRS,
    VENDOR_NOISE_EXTENSIONS,
    VectorDB,
    load_config,
    detect_project_name,
    get_allowed_projects,
    compute_sha256,
    tokenize,
    strip_code_bloat,
    should_index_path,
    validate_python_syntax,
    extract_symbols,
    extract_file_outline,
    chunk_file,
    run_watch
)
from ai_db.cli import main

if __name__ == "__main__":
    main()
