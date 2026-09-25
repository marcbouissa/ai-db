# Project: ai-db Open-Source Conversion

## Architecture

ai-db is an extensible, high-performance local code intelligence and vector indexing platform designed following SOLID (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion) and KISS principles.

The system is decoupled into four primary layers:
1. **Domain & Core Engine Layer**: AST code parsing, symbol extraction, chunking, and semantic/lexical search algorithms.
2. **Pluggable Storage Layer**: `StorageBackend` abstract protocol separating data persistence from core logic. SQLite with WAL mode, FTS5 BM25, and zlib compression as the only shipped backend; a pluggable factory resolves a backend by connection string, so a second backend can be added without touching core search, analysis or parser logic. **SQLite is the only implemented backend** — the MySQL adapter described in the original request was scoped out and does not exist.
3. **Transport & Dispatch Layer**: Unified `ServiceDispatcher` registering tools with JSON Schema and callable handlers. Transport adapters for CLI (`ai-db`, `vectordb`), Model Context Protocol (stdio JSON-RPC), and HTTP REST API (`ThreadingHTTPServer`).
4. **Telemetry Subsystem**: Cross-cutting performance and token telemetry measuring query latency, token compression efficiency across 4 serialization formats (Stub vs S-Exp vs JSON vs Raw), semantic cache hit rates, and codebase weak points diagnostics.

```
                  ┌────────────────────────────────────────────────┐
                  │                 Clients / AI                   │
                  └──────────────┬──────────────────┬──────────────┘
                                 │                  │
                         ┌───────▼────────┐  ┌──────▼───────┐
                         │   CLI / stdio  │  │   HTTP API   │
                         │ (ai-db, MCP)   │  │ (REST / JSON)│
                         └───────┬────────┘  └──────┬───────┘
                                 │                  │
                     ┌───────────▼──────────────────▼───────────┐
                     │         Unified ServiceDispatcher        │
                     │          (ai_db/dispatcher.py)           │
                     └─────────────────────┬────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         │                                 │                                 │
┌────────▼────────┐               ┌────────▼────────┐               ┌────────▼────────┐
│  Parser & AST   │               │ Search & Query  │               │    Telemetry    │
│ (ai_db/parser/) │               │ (ai_db/search/) │               │(ai_db/telemetry)│
└────────┬────────┘               └────────┬────────┘               └────────┬────────┘
         │                                 │                                 │
         └─────────────────────────────────┼─────────────────────────────────┘
                                           │
                     ┌─────────────────────▼────────────────────┐
                     │          StorageBackend (ABC)            │
                     │         (ai_db/storage/backend.py)       │
                     └──────────────┬──────────────────┬────────┘
                                    │                  │
                            ┌───────▼────────┐  ┌──────▼───────┐
                            │ SQLiteBackend  │  │  (not yet     │
                            │ (WAL, FTS5)    │  │   built)      │
                            └────────────────┘  └──────────────┘
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Standard Packaging | `pyproject.toml` with setuptools build-backend and editable install support | M1 | R1, spec_miner |
| 2 | Dual CLI Console Scripts | Console script entry points for both `ai-db` and `vectordb` executing `ai_db.cli:main` | M1 | R1, spec_miner |
| 3 | Requirements Specifications | `requirements.txt` (core runtime) and `requirements-dev.txt` (pytest, ruff, mypy) | M1 | R1, spec_miner |
| 4 | Modular Dependency Extras | Extras `ai-db[local-embed]`, `ai-db[dev]`, and `ai-db[all]` in `pyproject.toml` | M1 | R1, spec_miner |
| 5 | Clean Environment & Gitignore | Standard `.gitignore` excluding `.venv`, `__pycache__`, `.pytest_cache`, `.db` files; untrack git binaries | M1 | R1, spec_miner |
| 6 | StorageBackend Abstraction | Protocol / Abstract Base Class defining contracts for files, chunks, symbols, skills, contexts, state, search | M2 | R2, survey_1 |
| 7 | Storage Domain DTOs | Typed domain data objects in `ai_db/storage/models.py` separating storage records from sqlite3.Row | M2 | R2, survey_1 |
| 8 | High-Performance SQLite Backend | SQLite implementation with WAL mode (`PRAGMA journal_mode=WAL`), FTS5, BM25 ranking, zlib level 9 compression | M2 | R2, survey_1 |
| 9 | StorageBackend Factory | Pluggable factory resolving a backend by connection string (`sqlite://`) or environment configuration | M2 | R2, survey_1 |
| 10 | Backend Conformance Suite | `ai_db/storage/conformance.py` — a backend-agnostic contract suite every `StorageBackend` must pass, so a second backend can be validated when one is written | M2 | R2, survey_1 |
| 11 | Decouple Leaked SQL Calls | Refactor direct SQL queries in indexer, query, skills, context, engine to use `StorageBackend` methods | M2 | R2, survey_1 |
| 12 | Unified ServiceDispatcher | Centralized tool registry with JSON Schema and callable handlers for uniform cross-transport dispatch | M3 | R3, survey_2 |
| 13 | Agnostic CLI Dispatch | CLI command suite executing via `ServiceDispatcher` with uniform error handling and JSON/text formatting | M3 | R3, survey_2 |
| 14 | Dynamic MCP Server | stdio JSON-RPC 2.0 MCP server dynamically exposing all registered tools from `ServiceDispatcher` | M3 | R3, survey_2 |
| 15 | Threaded HTTP Server | `ThreadingHTTPServer` (`ai-db serve --port`) exposing `/health`, `/status`, `/telemetry`, `/tools`, `POST /tools/{name}` | M3 | R3, survey_2 |
| 16 | Latency & Throughput Telemetry | Tracking p50/p95/p99 query latency and throughput across storage backends | M4 | R4, survey_2 |
| 17 | Token Compression Telemetry | Measuring token consumption and savings across 4 serialization formats (Stub vs S-Exp vs JSON vs Raw) | M4 | R4, survey_2 |
| 18 | Cache Hit Rate & Resource Trends | Tracking semantic cache hits/misses, ratio %, disk size, and memory usage trends | M4 | R4, survey_2 |
| 19 | Codebase Weak Points Diagnostic | Detecting syntax error density, cyclomatic complexity hotspots, and unindexed code areas | M4 | R4, survey_2 |
| 20 | Telemetry Exposure Interfaces | Exposing telemetry metrics via CLI (`ai-db telemetry`), MCP tool (`telemetry`), and HTTP (`GET /telemetry`) | M4 | R4, survey_2 |
| 21 | Repository Sanitization | Purging all occurrences of `/path/to/user`, personal emails, and local machine configs from all tracked files | M5 | R5, spec_miner |
| 22 | Architectural Documentation | Comprehensive `ARCHITECTURE.md` explaining SOLID rationale, component decoupling, and extension guides | M5 | R5, spec_miner |
| 23 | Open-Source README & License | Clean, public `README.md` with quickstart, MCP configuration for Claude/Cursor/Antigravity, and MIT License | M5 | R5, spec_miner |
| 24 | E2E Opaque-Box Test Suite | Comprehensive 4-tier test suite verifying packaging, storage, transports, and telemetry independently | E2E Track | ORIGINAL_REQUEST |
| 25 | Final Verification & Hardening | 100% pass of E2E tests + Tier 5 Adversarial Coverage Hardening | M6 | ORIGINAL_REQUEST |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Packaging & Environment | `pyproject.toml`, `requirements.txt`, `.gitignore`, untrack `.pyc`/`.db`, setup `.venv` | none | DONE |
| 2 | M2: Pluggable Storage Layer | `StorageBackend` ABC, DTOs, SQLite WAL refactor, Factory, conformance suite, decouple SQL calls. (MySQL adapter was scoped out — see item 10.) | M1 | DONE |
| 3 | M3: Pluggable Transports | `ServiceDispatcher`, CLI dispatch, dynamic stdio MCP server, Threaded HTTP Server | M2 | DONE |
| 4 | M4: Performance & Telemetry | Telemetry subsystem (latency, 4-format token savings, cache hit rate, weak points), CLI/MCP/HTTP integration | M2, M3 | IN_PROGRESS (conv: bd8c6eb1-b5e8-4f12-9bc0-4712a6e53664) |
| 5 | M5: Sanitization & Documentation | Purge `/path/to/user` across repo, write `ARCHITECTURE.md`, update `README.md`, add `LICENSE` | M1 | IN_PROGRESS (conv: 770b9db7-07a5-4a4a-a1b7-5cd7858bf6e0) |
| - | E2E Testing Track | Independent opaque-box test harness and test cases (Tiers 1-4), publish `TEST_READY.md` | M1 (runs parallel to M2-M5) | IN_PROGRESS (conv: 3c52e57a-26a5-4325-8a55-54525ff1064a) |
| 6 | M6: Final Verification | Pass 100% E2E tests + Phase 2 Adversarial Coverage Hardening (Tier 5) | M2, M3, M4, M5, TEST_READY.md | PLANNED |

## Interface Contracts

### Domain & Storage Layer (`ai_db/storage/backend.py`)
```python
class StorageBackend(abc.ABC):
    @property
    @abc.abstractmethod
    def backend_name(self) -> str: ...
    @abc.abstractmethod
    def initialize(self) -> None: ...
    @abc.abstractmethod
    def close(self) -> None: ...
    @abc.abstractmethod
    def transaction(self) -> ContextManager[None]: ...
    # Files
    @abc.abstractmethod
    def upsert_file(self, record: FileRecord) -> None: ...
    @abc.abstractmethod
    def get_file(self, path: str) -> Optional[FileRecord]: ...
    # Symbols & Chunks
    @abc.abstractmethod
    def upsert_symbols(self, symbols: List[SymbolRecord]) -> None: ...
    @abc.abstractmethod
    def search_symbols(self, query: str, limit: int = 10) -> List[SymbolRecord]: ...
    # Full-Text Search (BM25)
    @abc.abstractmethod
    def search_chunks_bm25(self, query: str, limit: int = 10) -> List[ChunkRecord]: ...
    # State & Context
    @abc.abstractmethod
    def get_state(self, key: str) -> Optional[Dict[str, Any]]: ...
    @abc.abstractmethod
    def set_state(self, key: str, value: Dict[str, Any]) -> None: ...
```

### Transport & Dispatch Layer (`ai_db/dispatcher.py`)
```python
class ServiceDispatcher:
    def register_tool(self, name: str, description: str, parameters_schema: Dict[str, Any], handler: Callable[..., Any]) -> None: ...
    def list_tools(self) -> List[Dict[str, Any]]: ...
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any: ...
```

### Telemetry Subsystem (`ai_db/telemetry/`)
```python
class TelemetryTracker:
    def record_query(self, backend: str, latency_ms: float, results_count: int) -> None: ...
    def record_token_compression(self, raw_tokens: int, stub_tokens: int, sexp_tokens: int, json_tokens: int) -> None: ...
    def record_cache_access(self, hit: bool, tokens_saved: int = 0) -> None: ...
    def compute_weak_points(self) -> Dict[str, Any]: ...
    def get_summary(self) -> Dict[str, Any]: ...
```

## Code Layout
```
ai-db/
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
├── README.md
├── ARCHITECTURE.md
├── TEST_INFRA.md
├── TEST_READY.md
├── vectordb.py                 # Facade entry point
├── mcp_server.py               # Facade MCP server entry point
├── ai_db/
│   ├── __init__.py
│   ├── cli.py                  # CLI argument parsing & output formatting
│   ├── dispatcher.py           # Unified ServiceDispatcher
│   ├── constants.py            # Environment-aware paths & defaults
│   ├── analyzer/               # AST parsing, code analysis
│   ├── parser/                 # Tree/token parsing & linters
│   ├── search/                 # QueryEngine, ranking, bm25
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── backend.py          # StorageBackend ABC & protocol
│   │   ├── models.py           # Domain DTOs
│   │   ├── factory.py          # StorageBackendFactory
│   │   ├── conformance.py      # backend-agnostic contract suite
│   │   ├── sqlite_backend.py   # SQLite backend (WAL, FTS5, BM25, zlib)
│   ├── server/
│   │   ├── __init__.py
│   │   └── http_server.py      # ThreadingHTTPServer REST/JSON server
│   └── telemetry/
│       ├── __init__.py
│       ├── tracker.py          # TelemetryTracker
│       ├── metrics.py          # Latency, token savings, cache metrics
│       └── diagnostics.py      # Codebase weak points analyzer
└── tests/                      # Opaque-box & unit automated test suite
    ├── conftest.py
    ├── test_packaging.py
    ├── test_storage.py
    ├── test_search.py
    ├── test_parser.py
    ├── test_transports.py
    ├── test_telemetry.py
    └── test_sanitization.py
```
