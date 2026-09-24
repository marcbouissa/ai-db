#!/usr/bin/env python3
"""
vectordb.py: Backward-compatible facade shim for ai_db.
Preserves existing CLI commands, MCP server imports, and symlinks.
"""

from ai_db.cli import main

if __name__ == "__main__":
    main()
