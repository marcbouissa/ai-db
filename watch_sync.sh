#!/usr/bin/env bash
# Background file watcher that automatically synchronizes ai-db when files change
TARGET_DIR="${1:-.}"
DB_PATH="${2:-$HOME/GitRepos/ai-db/codebase_knowledge.db}"

echo "[ai-db watcher] Monitoring '$TARGET_DIR' -> DB: '$DB_PATH'"

if command -v inotifywait >/dev/null 2>&1; then
    inotifywait -m -r -e modify,create,delete,move \
        --exclude '(\.git|\.venv|node_modules|__pycache__|\.db|\.sqlite)' \
        "$TARGET_DIR" 2>/dev/null | while read -r directory events filename; do
            ai-db sync "$TARGET_DIR" --db "$DB_PATH" >/dev/null 2>&1
        done
else
    # Fallback polling loop every 10 seconds if inotifywait is not installed
    while true; do
        sleep 10
        ai-db sync "$TARGET_DIR" --db "$DB_PATH" >/dev/null 2>&1
    done
fi
