#!/usr/bin/env bash
# Keep the ai-db index in sync with a directory (event-driven, re-indexes changed files only).
TARGET_DIR="${1:-.}"
if [ -n "$2" ]; then
    exec ai-db watch "$TARGET_DIR" --db "$2"
fi
exec ai-db watch "$TARGET_DIR"
