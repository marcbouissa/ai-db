# Original User Request

> **As-built note (2026-09-26).** This file is a verbatim record of what was asked
> for on 2026-09-17. It is kept unedited so the original scope stays auditable, which
> means it still describes things that were **not** built. Where that happens, the
> requirement is marked below rather than rewritten. Two items did not ship:
>
> - **The MySQL adapter / `ai-db[mysql]` extra (R1, R2).** SQLite is the only backend
>   implemented. The `StorageBackend` abstraction and its conformance suite
>   (`ai_db/storage/conformance.py`) are what a second backend would have to satisfy;
>   no second backend exists.
> - **"Zero-dependency" core.** The core does depend on tree-sitter,
>   tree-sitter-language-pack, sqlite-vec, tiktoken, watchfiles and numpy. What *is*
>   true is that heavy ML (torch, sentence-transformers) stays in the optional
>   `ai-db[local-embed]` extra, so a bare install stays small and the test suite runs
>   without torch.

## Initial Request — 2026-09-17T20:50:14Z

Convert `ai-db` into an extensible, production-ready, open-source code intelligence and vector index platform following SOLID (Open/Closed) and KISS principles, featuring pluggable storage backends (SQLite default with FTS5, MySQL-ready interface), pluggable communication interfaces (CLI, stdio MCP, HTTP API), comprehensive performance and token telemetry, standard Python packaging (virtual environment, `pyproject.toml`, `requirements.txt`), automated test verification, and complete sanitization for public release.

Working directory: `/path/to/user/GitRepos/ai-db`
Integrity mode: demo

## Requirements

### R1. Modern Python Standard Packaging & Environment
- Configure standard virtual environment (`venv`) and dependency specifications (`requirements.txt` and `pyproject.toml` with `pip install -e .` editable support).
- Provide modular dependency extras (e.g. `ai-db[mysql]`, `ai-db[dev]`) while keeping default installation minimal and lightweight.
  - **As-built:** the extras are `ai-db[local-embed]`, `ai-db[dev]` and `ai-db[all]`. There is
    no `[mysql]` extra, because no MySQL backend was written.
- Enforce clean Python 3.10+ typing, standard project metadata, license, and `.gitignore` preventing binaries, `.pyc`, and personal databases from git tracking.

### R2. Pluggable Storage Layer (Open/Closed Principle)
- Define a storage abstraction / interface (`StorageBackend` protocol / abstract base class) covering files, chunks, symbols, skills, contexts, state, and search indexing.
- Maintain the high-performance SQLite backend with WAL mode, FTS5 full-text indexing, BM25 ranking, and zlib compression.
- Implement a pluggable backend factory and a modular relational/MySQL adapter interface so new databases can be added without modifying core search, analysis, or parser logic.
  - **As-built:** the factory and the `StorageBackend` interface are done, and
    `ai_db/storage/conformance.py` is a backend-agnostic contract suite that any new
    backend must pass. The MySQL adapter itself was scoped out.

### R3. Pluggable Communication & Transport Interfaces
- Decouple communication layers from core indexing and search domain services.
- Provide clean adapters for:
  1. CLI interface (`ai-db <command>`)
  2. Model Context Protocol (stdio JSON-RPC MCP server)
  3. Lightweight HTTP REST/JSON server (`ai-db serve --port <port>`)
- All interfaces interact with core services through uniform tool/service dispatching.

### R4. Performance & Token Telemetry
- Build a telemetry tracking module to measure:
  - Query latency and throughput across search backends
  - Token savings across serializations (Stub vs. S-Exp vs. JSON vs. raw source)
  - Semantic cache hit rates and memory/disk space trends
  - Codebase weak points (syntax error density, high-complexity files, unindexed areas)
- Expose actionable telemetry metrics via CLI command (`ai-db telemetry`), MCP tool, and HTTP `/telemetry` endpoint.

### R5. Open-Source Sanitization & Architectural Documentation
- Completely sanitize the repository of personal data: purge hardcoded user paths (`/path/to/user`), private credentials, personal emails, or machine-specific configs from all source files, README, and configs.
- Provide comprehensive architectural documentation (`ARCHITECTURE.md` and updated `README.md`) detailing:
  - Component decoupling and SOLID rationale
  - Storage & transport extension guides
  - Directory structure breakdown
  - Benchmarks, telemetry guide, and public contribution workflow.

## Acceptance Criteria

### Packaging & Environment
- [ ] Working virtual environment (`.venv`) created with dependencies captured in `requirements.txt` and `pyproject.toml`.
- [ ] Package installs cleanly via `pip install -e .` and CLI entry points (`ai-db`, `vectordb`) execute as expected.
- [ ] `.gitignore` properly excludes `__pycache__`, `.venv`, `.pytest_cache`, and local SQLite database files.

### Storage Extensibility
- [ ] Storage interface abstraction clearly separates domain logic from storage implementation.
- [ ] SQLite backend passes all operations (file indexing, symbol extraction, BM25 querying, context persistence).
- [ ] Backend factory supports selecting storage via configuration or connection string.

### Transport Agnostic Interfaces
- [ ] CLI commands (`sync`, `query`, `symbol`, `check`, `analyze`, `status`, `context`, `telemetry`) run reliably.
- [ ] MCP server (`mcp_server.py` / `ai-db mcp`) responds to tools/list and tools/call over stdio.
- [ ] HTTP server (`ai-db serve`) exposes health, status, telemetry, and tool execution endpoints.

### Telemetry & Insights
- [ ] Telemetry subsystem captures latency, token efficiency, and cache hit metrics.
- [ ] `ai-db telemetry` reports concrete quantitative token savings and identifies index bottlenecks.

### Verification & Sanitization
- [ ] Automated test suite (`pytest`) verifies parser, storage, telemetry, and transports.
- [ ] `git grep` reveals zero instances of personal paths (`/path/to/user`) or personal credentials across tracked repository files.
- [ ] `ARCHITECTURE.md` and `README.md` clearly document design decisions, extension patterns, and usage instructions.
