# Test Infrastructure Specification: ai-db Test Harness

## 1. Overview & Principles

The `ai-db` test suite provides automated, requirement-driven verification of all features defined in `PROJECT.md` and `ORIGINAL_REQUEST.md`. The test infrastructure adheres to the following core principles:

1. **Opaque-Box & Requirement-Driven**: Tests exercise public CLI commands (`ai-db`, `vectordb`), standard transports (stdio MCP, HTTP REST), and public interface contracts (`StorageBackend`, `ServiceDispatcher`, `TelemetryTracker`) without depending on private internal functions or implementation details.
2. **Complete Feature Coverage**: Every feature across Milestones 1 through 5 (Features 1–23 in `PROJECT.md`) is systematically tested across 4 progressive tiers.
3. **Hermetic Test Isolation**: Tests execute in completely isolated temporary sandboxes (`isolated_env`, `temp_workspace`, `temp_db`). Tests never touch live user databases, `$HOME/.gemini`, or personal configurations.
4. **Progressive Testability**: Tests run cleanly and give clear pass/fail signals. Optional components (such as the MySQL driver `pymysql`) or uncompleted milestone dependencies gracefully skip when dependencies are not present, while fully verifying satisfied features.
5. **Zero External Dependencies for Core Verification**: Core test execution requires only Python standard library capabilities and `pytest`.

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
│   ├── conftest.py                # Hermetic isolation fixtures, CLI runner, mock drivers
│   ├── test_packaging.py          # Features 1–5: pyproject.toml, CLI scripts, requirements, gitignore
│   ├── test_sanitization.py       # Features 5 & 21: zero /path/to/user, credentials, tracked binaries
│   ├── test_storage.py            # Features 6–11: StorageBackend ABC, DTOs, SQLite WAL, Factory, MySQL
│   ├── test_search.py             # QueryEngine, BM25 ranking, project boundary filtering, symbols
│   ├── test_parser.py             # AST parsing, outline generation, syntax validation, chunking
│   ├── test_transports.py         # Features 12–15: ServiceDispatcher, CLI, stdio MCP, HTTP server
│   └── test_telemetry.py          # Features 16–20: Latency, 4-format token savings, cache hit metrics
```

### Module Responsibilities

- **`tests/conftest.py`**: Declares shared pytest fixtures ensuring strict environment isolation, virtualized environment variables, temporary database generation, sample project workspaces, in-process/subprocess CLI execution runners, and mock MySQL database drivers.
- **`tests/test_packaging.py`**: Verifies PEP 517/518/621 packaging standards, dual console script entry points (`ai-db` and `vectordb`), zero-dependency core `requirements.txt`, modular dependency extras (`[mysql]`, `[dev]`, `[all]`), and `.gitignore` hygiene.
- **`tests/test_sanitization.py`**: Verifies 100% absence of personal paths (`/path/to/user`), personal usernames, credentials (AWS, GitHub, OpenAI), private keys, tracked database binaries, and compiled `.pyc` files across git-tracked files.
- **`tests/test_storage.py`**: Verifies the `StorageBackend` abstract base class, domain DTOs (`FileRecord`, `ChunkRecord`, `SymbolRecord`, `ContextRecord`, `SearchResult`), high-performance SQLite backend (WAL mode, FTS5 BM25, zlib level 9 compression, transaction rollback), `StorageBackendFactory`, MySQL 8.0+ adapter interface, and SQL call decoupling.
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
| `mock_mysql_connection` | `function` | Mocks `pymysql.connect` and cursor execution using `monkeypatch`, enabling MySQL adapter testing without requiring a live MySQL server. |
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
   - Optional MySQL tests gracefully skip if `pymysql` is not installed, while mock-based adapter query generation tests execute everywhere.
   - Milestone dependencies that are planned for future iterations cleanly skip with informative pytest messages until implemented.
