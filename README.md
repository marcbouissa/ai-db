# ai-db: Zero-Dependency Code Knowledge & Vector Index Engine

A high-speed, zero-dependency SQLite FTS5 + BM25 ranking vector database and knowledge index designed specifically for AI coding assistants (like Antigravity / Gemini / Claude) to **instantly recall codebase architecture, symbols, and files without slow repetitive codebase re-analysis on every new chat**.

## Features
- **Zero External Dependencies**: Powered entirely by the Python 3 standard library (`sqlite3`, `math`, `re`, `hashlib`, `json`).
- **Sub-Millisecond Queries**: Instant search across thousands of code chunks using SQLite FTS5 with BM25 ranking and identifier-aware tokenization.
- **Incremental Synchronization**: SHA-256 hash detection indexes only new or modified files, pruning deleted files automatically.
- **Smart Chunking**: Syntax-aware splitting for classes, functions, markdown headers, and modules.

## CLI Usage

### 1. Synchronize / Index a Codebase
```bash
ai-db sync /path/to/project
```

### 2. Semantic & Keyword Search
```bash
ai-db query "sqlite fts5 ranking" --top 5
```

### 3. Check Index Health
```bash
ai-db status
```

### 4. Continuous Background Sync
```bash
~/GitRepos/ai-db/watch_sync.sh /path/to/project &
```

## AI Agent Policy Integration
Antigravity automatically checks `ai-db` first on new sessions to retrieve architecture and file symbols without performing full-codebase directory crawls.
