# Test Infrastructure Specification: ai-db Test Harness

## 1. Overview & Principles

The `ai-db` test suite provides automated, requirement-driven verification of all features defined in `PROJECT.md` and `ORIGINAL_REQUEST.md`. The test infrastructure adheres to the following core principles:

1. **Opaque-Box & Requirement-Driven**: Tests exercise public CLI commands (`ai-db`, `vectordb`), standard transports (stdio MCP, HTTP REST), and public interface contracts (`StorageBackend`, `ServiceDispatcher`, `TelemetryTracker`) without depending on private internal functions or implementation details.
2. **Complete Feature Coverage**: Every feature across Milestones 1 through 5 (Features 1–23 in `PROJECT.md`) is systematically tested across 4 progressive tiers.
3. **Hermetic Test Isolation**: Tests execute in completely isolated temporary sandboxes (`isolated_env`, `temp_workspace`, `temp_db`). Tests never touch live user databases, `$HOME/.gemini`, or personal configurations.
4. **Progressive Testability**: Tests run cleanly and give clear pass/fail signals. Optional components (such as the `torch` / `sentence-transformers` stack behind the `[local-embed]` extra, or a GPU for the device-resolution paths) gracefully skip when dependencies are not present, while fully verifying satisfied features.
5. **No Heavy ML for Core Verification**: Core test execution requires the declared runtime dependencies (tree-sitter, sqlite-vec, tiktoken, numpy) plus `pytest`. Torch and sentence-transformers are never needed to run the suite; the tests that touch them skip.

---

## 2. Test Hierarchy (4 Tiers)

Each test module implements test cases across four structured tiers:

| Tier | Name | Purpose | Minimum Coverage |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Feature Coverage** | Happy-path isolation testing for each specified capability and public interface contract. | $\ge 5$ tests per feature |
| **Tier 2** | **Boundary & Corner Cases** | Extreme limits, malformed inputs, empty queries, invalid parameters, special characters, and crash recovery. | $\ge 5$ tests per feature |
| **Tier 3** | **Cross-Feature Combinations** | Pairwise interaction tests verifying compatibility between decoupled subsystems (e.g. Storage + Transports, Packaging + Gitignore). | Major subsystem pairs |
| **Tier 4** | **Real-World Workflows** | End-to-end integration workflows simulating realistic developer and AI agent usage scenarios. | $\ge 5$ multi-step workflows |

---

## 3. Directory Structure & Module Ownership

```
ai-db/
├── TEST_INFRA.md                  # Test harness architecture & execution specification (this document)
├── TEST_READY.md                  # Test suite readiness certification & coverage report
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Hermetic isolation fixtures and CLI runner
│   ├── fake_providers.py          # Deterministic stand-ins for embedding/rerank providers
│   ├── test_packaging.py          # Features 1–5: pyproject.toml, CLI scripts, requirements, gitignore
│   ├── test_sanitization.py       # Features 5 & 21: zero /path/to/user, credentials, tracked binaries
│   ├── test_storage.py            # StorageBackend ABC, DTOs, SQLite WAL, Factory, transactions
│   ├── test_search.py             # QueryEngine, BM25 ranking, project boundary filtering, symbols
│   ├── test_parser.py             # Parsing, outline generation, syntax validation, chunking
│   ├── test_ts_graph.py           # Tree-sitter symbol/cross-ref queries for all 9 languages
│   ├── test_chunker_ts.py         # Tree-sitter-backed chunking
│   ├── test_storage_conformance.py # Backend-agnostic StorageBackend contract suite
│   ├── test_storage_adversarial_m2.py # Adversarial storage cases
│   ├── test_investigate.py         # Investigation packs, modes, budgets
│   ├── test_config.py              # Config v2 validation and migration
│   ├── test_embeddings.py          # Embedding providers and vector search
│   ├── test_rerank.py              # Rerank providers
│   ├── test_lexical.py             # Lexical-only retrieval paths
│   ├── test_observability.py       # Health, telemetry surfaces
│   ├── test_cache_watch.py         # Query cache and file watcher
│   ├── test_eval_metrics.py        # Eval harness and baseline comparison
│   ├── test_phase11.py             # Phase 11: config migration, skills, context vectors
│   ├── test_phase13.py             # Phase 13: device resolution, write lock, incremental centrality
│   ├── test_phase14.py             # Phase 14: MCP progress, diff mode, cache isolation
│   ├── test_bench_index.py         # Indexer benchmark (marked `bench`)
│   ├── test_bench_vectors.py       # exact vs vec0 vector benchmark (marked `bench`)
│   └── test_bench_gpu.py           # CPU vs CUDA embedding benchmark (marked `bench`)
│   ├── test_transports.py         # Features 12–15: ServiceDispatcher, CLI, stdio MCP, HTTP server
│   └── test_telemetry.py          # Features 16–20: Latency, 4-format token savings, cache hit metrics
```

### Module Responsibilities

- **`tests/conftest.py`**: Declares shared pytest fixtures ensuring strict environment isolation, virtualized environment variables, temporary database generation, sample project workspaces, and in-process/subprocess CLI execution runners. Defined fixtures: `isolated_env`, `temp_db`, `temp_db_path`, `temp_workspace`, `sample_code_dir`, `cli_runner`, `sqlite_backend`, `memory_sqlite_backend`, `sample_records`.
- **`tests/test_packaging.py`**: Verifies PEP 517/518/621 packaging standards, dual console script entry points (`ai-db` and `vectordb`), that the core runtime dependency set is the parsing/index stack and excludes heavy ML, modular dependency extras (`[local-embed]`, `[dev]`, `[all]`), `.scm` package-data inclusion, and `.gitignore` hygiene.
- **`tests/test_sanitization.py`**: Verifies 100% absence of personal paths (`/path/to/user`), personal usernames, credentials (AWS, GitHub, OpenAI), private keys, tracked database binaries, and compiled `.pyc` files across git-tracked files.
- **`tests/test_storage.py`**: Verifies the `StorageBackend` abstract base class, domain DTOs (`FileRecord`, `ChunkRecord`, `SymbolRecord`, `ContextRecord`, `SearchResult`), the SQLite backend (WAL mode, FTS5 BM25, zlib compression, transaction rollback), `StorageBackendFactory`, and SQL call decoupling.
- **`tests/test_search.py` & `tests/test_parser.py`**: Verifies AST symbol extraction, outline analysis, syntax checking, and BM25 full-text indexing.
- **`tests/test_transports.py`**: Verifies unified `ServiceDispatcher`, CLI command execution, stdio JSON-RPC 2.0 MCP server, and `ThreadingHTTPServer` REST endpoints.
- **`tests/test_telemetry.py`**: Verifies latency tracking (p50/p95/p99), token compression savings across 4 formats (Stub vs S-Exp vs JSON vs Raw), semantic cache hit rates, and codebase weak points diagnostics.

---

## 4. Fixture Catalog (`tests/conftest.py`)

| Fixture Name | Scope | Description |
| :--- | :--- | :--- |
| `isolated_env` | `function` | Virtualizes `AI_DB_PATH`, `AI_DB_CONFIG`, `AI_DB_SKILL_DIRS`, `XDG_DATA_HOME`, `XDG_CONFIG_HOME`, and `AI_DB_CONNECTION_STRING` into a dedicated temporary directory. Restores original environment on teardown. |
| `temp_db` | `function` | Yields the filesystem path to a clean SQLite database file within `isolated_env`. Automatically removes file on teardown. |
| `temp_db_path` | `function` | Generates a distinct temporary SQLite database path within `tmp_path`. |
| `temp_workspace` | `function` | Constructs a multi-file, multi-language workspace with realistic Python source (`service.py`, `utils.py`), syntax error cases (`broken_syntax.py`), JavaScript, and Markdown documentation. |
| `sample_code_dir` | `function` | Convenience alias providing a populated source code directory for indexing tests. |
| `cli_runner` | `function` | Helper object offering `run(*args, use_subprocess=False)` to execute CLI commands in-process with captured `stdout`/`stderr` or out-of-process via `subprocess`. |
| `sqlite_backend` | `function` | Instantiates and initializes a concrete `SQLiteBackend` targeting a temporary database; guarantees `.close()` on test completion. |
| `memory_sqlite_backend` | `function` | Instantiates an in-memory `SQLiteBackend("sqlite:///:memory:")` with schema initialized; guarantees `.close()` on teardown. |
| `sample_records` | `function` | Factory producing standardized domain DTOs (`FileRecord`, `ChunkRecord`, `SymbolRecord`, `ContextRecord`, `SearchResult`). |

---

## 5. Execution Instructions

### Complete Test Suite
```bash
pytest tests/ -v
```

### Module-Specific Invocations
```bash
# Packaging & Environment tests (Features 1–5)
pytest tests/test_packaging.py -v

# Sanitization & Privacy tests (Features 5, 21)
pytest tests/test_sanitization.py -v

# Storage Layer tests (Features 6–11)
pytest tests/test_storage.py -v

# Transport & Communication tests (Features 12–15)
pytest tests/test_transports.py -v

# Performance & Telemetry tests (Features 16–20)
pytest tests/test_telemetry.py -v
```

### Tier-Specific Invocations
```bash
# Run only Tier 1 Feature Coverage tests
pytest tests/ -k "Tier1" -v

# Run only Tier 2 Boundary & Corner tests
pytest tests/ -k "Tier2" -v

# Run only Tier 3 Pairwise tests
pytest tests/ -k "Pairwise or Tier3" -v

# Run only Tier 4 Workflow tests
pytest tests/ -k "Workflow or Tier4" -v
```

### Coverage & Static Analysis
```bash
# Execute with statement and branch coverage
pytest --cov=ai_db --cov-report=term-missing --cov-report=html

# Syntax health check
ai-db check .
```

---

## 6. Safety & Hermetic Guarantees

1. **Host Isolation**: `isolated_env` intercepts all database paths and configuration lookups. Even if commands are executed with default arguments, operations target temporary directory locations.
2. **No Git Workspace Mutation**: Tests that generate build artifacts, temporary databases, or bytecode write exclusively to pytest `tmp_path`. No untracked files are left in the git working tree.
3. **Graceful Degradation for Optional Extras**:
   - Tests needing the `[local-embed]` stack (torch, sentence-transformers) skip with an informative message when it is absent, so the core suite runs on a bare install.
   - GPU-dependent device tests and the `bench`-marked benchmarks skip when no CUDA device is present.
