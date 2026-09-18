# ai-db Architecture Specification

> **Version**: 0.2.0  
> **Status**: Authoritative Architectural Blueprint  
> **Scope**: Component decoupling, SOLID design rationale, pluggable storage & transport guides, performance telemetry, repository layout, and public contribution workflow.

---

## 1. Architectural Philosophy & Overview

`ai-db` is an extensible, high-performance local code intelligence and vector indexing platform engineered specifically for autonomous AI agents, coding assistants (Claude Desktop, Cursor, Antigravity, OpenCodeInterpreter), and developer workflows.

Core tenets of the architecture:
- **Zero External Runtime Dependencies**: The core package relies exclusively on the Python 3.10+ Standard Library (`ast`, `sqlite3`, `zlib`, `http.server`, `argparse`, `typing`, `json`, `hashlib`, `threading`).
- **Token Optimization First**: AI agents pay heavily for context window consumption. `ai-db` optimizes token consumption through progressive disclosure (`summary` -> `structure` -> `targeted` -> `full`) and alternative serialization formats (Stub skeletons and S-Expressions), achieving 50% to 70% token reductions.
- **SOLID and KISS Design**: Clear layer boundaries, protocol-based abstractions, interchangeable storage engines, uniform cross-transport dispatching, and dependency injection.
- **Sub-Millisecond Retrieval**: High-throughput SQLite WAL mode with FTS5 BM25 ranking, zlib level 9 compression for code chunk payloads, and in-memory semantic caching.

### 1.1 High-Level Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Clients & AI Agent IDEs                            │
│           (Claude Desktop, Cursor, Antigravity, Terminal CLI, Scripts)      │
└───────────────┬─────────────────────────────┬───────────────────────────────┘
                │                             │
    ┌───────────▼───────────┐     ┌───────────▼───────────┐     ┌─────────────▼───────────┐
    │     CLI Interface     │     │    Stdio MCP Server   │     │    HTTP REST / JSON     │
    │   (ai-db / vectordb)  │     │   (JSON-RPC 2.0 stdio)│     │  (ThreadingHTTPServer)  │
    └───────────┬───────────┘     └───────────┬───────────┘     └─────────────┬───────────┘
                │                             │                               │
                └─────────────────────────────┼───────────────────────────────┘
                                              │
                              ┌───────────────▼───────────────┐
                              │   Unified ServiceDispatcher   │
                              │     (ai_db/dispatcher.py)     │
                              └───────────────┬───────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         │                                    │                                    │
┌────────▼────────┐                  ┌────────▼────────┐                  ┌────────▼────────┐
│  Parser Subsys  │                  │  Search Engine  │                  │ Analyzer Engine │
│ (ai_db/parser/) │                  │ (ai_db/search/) │                  │(ai_db/analyzer/)│
│ AST & Chunking  │                  │ Indexer & BM25  │                  │Token Progressive│
└────────┬────────┘                  └────────┬────────┘                  └────────┬────────┘
         │                                    │                                    │
         └────────────────────────────────────┼────────────────────────────────────┘
                                              │
                              ┌───────────────▼───────────────┐
                              │   VectorDB Facade / Domain    │
                              │       (ai_db/__init__.py)     │
                              └───────────────┬───────────────┘
                                              │
                              ┌───────────────▼───────────────┐
                              │      StorageBackend (ABC)     │
                              │   (ai_db/storage/backend.py)  │
                              └───────┬───────────────┬───────┘
                                      │               │
                      ┌───────────────▼──┐         ┌──▼────────────────┐
                      │  SQLiteBackend   │         │   MySQLBackend    │
                      │ (WAL, FTS5, zlib)│         │(Relational Adapter│
                      └──────────────────┘         └───────────────────┘
```

---

## 2. SOLID & KISS Design Principles

`ai-db` strictly adheres to SOLID object-oriented design and KISS (Keep It Simple, Stupid) engineering principles:

| Principle | Architectural Implementation in `ai-db` | Primary File Locations |
|-----------|------------------------------------------|------------------------|
| **Single Responsibility Principle (SRP)** | Every module has one, and only one, reason to change. Parsing ASTs does not query databases; storage backends do not format CLI output; transports do not parse ASTs. | `ai_db/parser/` (AST parsing only)<br>`ai_db/storage/` (persistence only)<br>`ai_db/search/` (indexing/ranking only)<br>`ai_db/transports/` (protocol handling only) |
| **Open/Closed Principle (OCP)** | Core indexing, querying, and analysis services are closed for modification but open for extension. New storage engines (e.g. DuckDB, PostgreSQL) and new transport adapters (e.g. WebSocket, gRPC) can be plugged in without modifying existing domain code. | `ai_db/storage/backend.py`<br>`ai_db/storage/factory.py`<br>`ai_db/dispatcher.py` |
| **Liskov Substitution Principle (LSP)** | Any storage backend implementing `StorageBackend` can be substituted into `VectorDB` or `Indexer` without altering the correctness of the system. Implementations communicate strictly through typed `slots=True` domain DTOs. | `ai_db/storage/models.py`<br>`ai_db/storage/sqlite_backend.py`<br>`ai_db/storage/mysql_backend.py` |
| **Interface Segregation Principle (ISP)** | The storage contract is cleanly segregated into focused functional areas (Files, Chunks, Symbols, Cross-Refs, Diagnostics, Skills, Context Memory, Analysis Refs, State/Cache). Domain services invoke only the interface subsets they need. | `ai_db/storage/backend.py` |
| **Dependency Inversion Principle (DIP)** | High-level business logic (`VectorDB`, `Indexer`, `QueryEngine`, `SkillRouter`, `ContextMemory`, `AnalyzerEngine`) depends on the abstract `StorageBackend` contract, never on concrete database connection drivers (`sqlite3.Connection`, PyMySQL). Dependencies are injected via constructors. | `ai_db/__init__.py`<br>`ai_db/search/indexer.py`<br>`ai_db/search/query.py` |

---

## 3. Pluggable Storage Layer

### 3.1 Domain Data Transfer Objects (DTOs)

To satisfy LSP and eliminate database row coupling (`sqlite3.Row`, raw tuples, or dictionary cursors), all storage interactions use strongly typed, memory-efficient `slots=True` dataclasses located in `ai_db/storage/models.py`:

- `FileRecord`: File path, SHA-256 content hash, last modified timestamp, chunk count, project scope.
- `ChunkRecord`: Code block type (function, class, module), name, line span `[start_line, end_line]`, and uncompressed code content.
- `SymbolRecord`: Extracted code symbols (name, symbol_type, file path, line number, signature).
- `SymbolRefRecord`: Symbol cross-references and call sites (caller, callee, line number, reference type: `call`, `import`, `inherit`).
- `AnnotationRecord`: Inline annotations (TODO, FIXME, HACK, note, docstring).
- `SyntaxErrorRecord`: AST parsing errors with precise `(line, col, message)` coordinates.
- `SkillRecord`: Assistant skill definitions, markdown prompt triggers, and metadata.
- `ContextRecord`: Session snapshots, active files, open pending tasks, and conversation notes.
- `AnalysisRefRecord`: Opaque progressive-disclosure handles (`ref:<hash>`) with cached body text.
- `SearchResult`: Ranked chunk matches with BM25 score, snippet, and line coordinates.

### 3.2 StorageBackend Abstract Base Class

`StorageBackend` (`ai_db/storage/backend.py`) defines the contract that every storage engine must implement:

```python
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple, ContextManager
from ai_db.storage.models import (
    FileRecord, ChunkRecord, SymbolRecord, SymbolRefRecord,
    AnnotationRecord, SyntaxErrorRecord, SkillRecord, ContextRecord,
    AnalysisRefRecord, SearchResult
)

class StorageBackend(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def transaction(self) -> ContextManager[None]: ...

    # File Operations
    @abstractmethod
    def upsert_file(self, record: FileRecord) -> None: ...
    @abstractmethod
    def get_file(self, filepath: str) -> Optional[FileRecord]: ...
    @abstractmethod
    def delete_file(self, filepath: str) -> None: ...
    @abstractmethod
    def get_files_by_prefix(self, prefix: str) -> Dict[str, str]: ...
    @abstractmethod
    def get_all_filepaths(self) -> List[str]: ...

    # Chunk & Code Search Operations
    @abstractmethod
    def insert_chunks(self, chunks: List[ChunkRecord]) -> None: ...
    @abstractmethod
    def get_chunks_for_file(self, filepath: str) -> List[ChunkRecord]: ...
    @abstractmethod
    def search_chunks(
        self, query_tokens: List[str], allowed_projects: Optional[List[str]] = None,
        top_k: int = 5, path_prefix: Optional[str] = None
    ) -> List[SearchResult]: ...

    # Symbol & Cross-Reference Operations
    @abstractmethod
    def insert_symbols(self, symbols: List[SymbolRecord]) -> None: ...
    @abstractmethod
    def query_symbols(self, name: str, allowed_projects: Optional[List[str]] = None, limit: int = 50) -> List[SymbolRecord]: ...
    @abstractmethod
    def insert_symbol_refs(self, refs: List[SymbolRefRecord]) -> None: ...
    @abstractmethod
    def query_symbol_callers(self, callee_name: str, allowed_projects: Optional[List[str]] = None, limit: int = 100) -> List[SymbolRefRecord]: ...

    # Diagnostics, Annotations, Skills, Context, State & Cache...
```

### 3.3 Default SQLite Backend (`SQLiteBackend`)
The default engine (`ai_db/storage/sqlite_backend.py`) provides zero-dependency persistence:
- **Write-Ahead Logging (WAL)**: `PRAGMA journal_mode=WAL` enables concurrent readers alongside writers without mutual locking.
- **FTS5 Full-Text Indexing**: Fast BM25 keyword matching across code tokens and skill documentation.
- **zlib Level 9 Compression**: Code chunks are compressed into binary blobs (`zcontent`) upon insertion and decompressed transparently on retrieval, saving 60–80% disk space.
- **Cascading Deletions**: Deleting a file cascades to associated chunks, symbols, and cross-references.

### 3.4 Step-by-Step Guide: Implementing a Custom Storage Backend

To implement a new backend (e.g. DuckDB, PostgreSQL, or an In-Memory engine):

#### Step 1: Subclass `StorageBackend`
Implement the abstract methods using your database driver, mapping results to the domain DTOs.

```python
# custom_backend.py
import contextlib
from typing import Optional, List, Dict, Any, Tuple, ContextManager
from ai_db.storage.backend import StorageBackend
from ai_db.storage.models import (
    FileRecord, ChunkRecord, SymbolRecord, SymbolRefRecord,
    AnnotationRecord, SyntaxErrorRecord, SkillRecord, ContextRecord,
    AnalysisRefRecord, SearchResult
)

class InMemoryStorageBackend(StorageBackend):
    """Minimal in-memory storage backend demonstration."""

    def __init__(self, uri: str = "memory://"):
        self.uri = uri
        self._files: Dict[str, FileRecord] = {}
        self._chunks: Dict[str, List[ChunkRecord]] = {}
        self._symbols: List[SymbolRecord] = []
        self._state: Dict[str, Any] = {}

    @property
    def backend_name(self) -> str:
        return "in_memory"

    def initialize(self) -> None:
        pass  # Initialize memory structures

    def close(self) -> None:
        self._files.clear()
        self._chunks.clear()

    @contextlib.contextmanager
    def transaction(self) -> ContextManager[None]:
        # Atomic transaction context
        yield

    def upsert_file(self, record: FileRecord) -> None:
        self._files[record.filepath] = record

    def get_file(self, filepath: str) -> Optional[FileRecord]:
        return self._files.get(filepath)

    def delete_file(self, filepath: str) -> None:
        self._files.pop(filepath, None)
        self._chunks.pop(filepath, None)

    def get_files_by_prefix(self, prefix: str) -> Dict[str, str]:
        return {fp: rec.sha256 for fp, rec in self._files.items() if fp.startswith(prefix)}

    def get_all_filepaths(self) -> List[str]:
        return list(self._files.keys())

    def insert_chunks(self, chunks: List[ChunkRecord]) -> None:
        for chunk in chunks:
            self._chunks.setdefault(chunk.filepath, []).append(chunk)

    def get_chunks_for_file(self, filepath: str) -> List[ChunkRecord]:
        return self._chunks.get(filepath, [])

    def search_chunks(
        self, query_tokens: List[str], allowed_projects: Optional[List[str]] = None,
        top_k: int = 5, path_prefix: Optional[str] = None
    ) -> List[SearchResult]:
        hits = []
        tokens_lower = [t.lower() for t in query_tokens]
        for fp, chunks in self._chunks.items():
            for c in chunks:
                if any(t in c.content.lower() for t in tokens_lower):
                    hits.append(SearchResult(
                        chunk_id=c.id or 1, filepath=c.filepath, name=c.name,
                        chunk_type=c.chunk_type, project=c.project,
                        start_line=c.start_line, end_line=c.end_line,
                        score=1.0, snippet=c.content[:100]
                    ))
        return hits[:top_k]

    # Implement remaining abstract methods (symbols, diagnostics, state, etc.)...
```

#### Step 2: Register with `StorageBackendFactory`
Register the custom scheme in `StorageBackendFactory`:

```python
from ai_db.storage.factory import StorageBackendFactory

# Register custom URI scheme
StorageBackendFactory.register_backend("memory", InMemoryStorageBackend)

# Instantiate via URI
backend = StorageBackendFactory.create("memory://")
```

#### Step 3: Pass into VectorDB
Inject the backend into `VectorDB` without modifying any indexing or analysis logic:

```python
from ai_db import VectorDB

vdb = VectorDB(backend)
vdb.sync("/path/to/project")
results = vdb.query("MyClass")
```

---

## 4. Pluggable Transport & Communication Layer

`ai-db` decouples communication interfaces (CLI, MCP, HTTP REST) from business logic through a centralized `ServiceDispatcher`.

### 4.1 Unified ServiceDispatcher (`ai_db/dispatcher.py`)

`ServiceDispatcher` maintains a tool registry where tools declare:
- `name`: Unique tool identifier (`locate`, `analyze`, `sync`, `check`, etc.).
- `description`: LLM-readable description of what the tool does.
- `parameters_schema`: Standard JSON Schema defining inputs, types, and required fields.
- `handler`: Callable taking an argument dictionary and returning structured output.

```python
class ServiceDispatcher:
    def register_tool(self, name: str, description: str, parameters_schema: Dict[str, Any], handler: Callable[..., Any]) -> None:
        """Register a tool with JSON Schema validation and a callable handler."""
        ...

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return registered tools and schemas for MCP tools/list and HTTP GET /tools."""
        ...

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Validate input arguments against schema and execute tool handler."""
        ...
```

### 4.2 Transport Adapters
- **CLI (`ai_db/cli.py`)**: Translates argparse commands to dispatcher tool calls. Formats output for terminal consumption (colored tables, text outlines) or JSON (`--format json`).
- **MCP Server (`mcp_server.py`)**: Exposes registered tools over stdio JSON-RPC 2.0. Dynamically exports all tools in `tools/list` and executes calls via `dispatcher.execute()`.
- **HTTP Server (`ai_db/server/http_server.py`)**: Threaded stdlib HTTP server (`ai-db serve --port 8765`). Endpoints:
  * `GET /health`: Health probe (`{"ok": true}`).
  * `GET /status`: Database size and indexing metrics.
  * `GET /tools`: Tool inventory with schemas.
  * `POST /tools/{name}` & `POST /`: Execute tool with JSON payload.
  * `GET /telemetry`: Performance and token savings metrics.

### 4.3 Step-by-Step Guide: Creating a New Transport Adapter

To add a new transport (e.g. WebSocket, gRPC, or Unix Socket daemon):

```python
# websocket_transport.py
import json
from ai_db.dispatcher import ServiceDispatcher

class WebSocketTransport:
    """Demonstration of a custom transport wrapping ServiceDispatcher."""

    def __init__(self, dispatcher: ServiceDispatcher):
        self.dispatcher = dispatcher

    def on_message(self, raw_message: str) -> str:
        try:
            payload = json.loads(raw_message)
            tool_name = payload["tool"]
            arguments = payload.get("args", {})

            # Execute via unified dispatcher
            result = self.dispatcher.execute(tool_name, arguments)
            return json.dumps({"status": "success", "data": result})
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})
```

---

## 5. Token Optimization & Progressive Disclosure

LLM context windows are constrained and expensive. `ai-db` avoids dumping whole files into context by implementing a progressive disclosure pipeline:

```
                      ┌────────────────────────┐
                      │  ai-db locate <query>  │
                      └───────────┬────────────┘
                                  │ (top-k files & symbols)
                                  ▼
                      ┌────────────────────────┐
                      │  ai-db outline <file>  │
                      └───────────┬────────────┘
                                  │ (signatures & lines)
                                  ▼
                      ┌────────────────────────┐
                      │  ai-db analyze <file>  │
                      └───────────┬────────────┘
                                  │ (summary / structure / targeted)
                                  ▼
                      ┌────────────────────────┐
                      │  ai-db expand <ref>    │
                      └────────────────────────┘
                        (pay-per-section body)
```

### 5.1 The 4 Serialization Formats

`ai_db/analyzer/formatters.py` provides 4 distinct serializations:

| Format | Representation | Token Cost (% of Raw) | Best Suited For |
|--------|----------------|----------------------|-----------------|
| **Raw** | Unmodified source code | 100% (baseline) | Final code editing and patching |
| **JSON AST** | Full AST dictionary | 120%–150% | Machine tool parsing and deep inspection |
| **Stub** | Skeletons with signatures, docstrings, and `...` bodies | ~50% (50% savings) | Agent reasoning, planning, and interface verification |
| **S-Exp** | Lisp-style S-expression token-minimal syntax | ~37% (63% savings) | High-volume batch retrieval under tight context limits |

---

## 6. Performance & Telemetry Subsystem

The telemetry subsystem (`ai_db/telemetry/`) provides non-intrusive, zero-overhead observability for search latency, token compression efficiency, cache hit rates, and codebase weak points.

### 6.1 Telemetry Metrics Data Structure

```json
{
  "latency": {
    "sqlite_wal": {
      "count": 420,
      "min_ms": 0.45,
      "max_ms": 12.30,
      "avg_ms": 1.82,
      "p50_ms": 1.40,
      "p95_ms": 3.80,
      "p99_ms": 6.20,
      "qps": 549.45
    }
  },
  "tokens": {
    "raw_tokens": 125000,
    "stub_tokens": 62500,
    "sexp_tokens": 46250,
    "json_tokens": 156250,
    "net_saved_tokens": 62500,
    "savings_pct": 50.0,
    "estimated_cost_saved_usd": 0.1875
  },
  "cache": {
    "lookups": 85,
    "hits": 68,
    "misses": 17,
    "hit_rate_pct": 80.0,
    "tokens_saved": 45200
  },
  "storage": {
    "db_size_kb": 1420.0,
    "table_rows": {
      "files": 45,
      "chunks": 320,
      "symbols": 510
    }
  },
  "weak_points": {
    "syntax_error_density_pct": 2.2,
    "complexity_hotspots": [
      { "filepath": "src/core.py", "chunks": 48, "symbols": 62 }
    ],
    "unindexed_or_stale_files": 0
  }
}
```

### 6.2 Zero-Overhead Guarantees
- **In-Memory Tracking**: Latency measurements use `time.perf_counter()`. Percentiles and metrics are accumulated in lightweight in-memory ring buffers with O(1) appending overhead.
- **Persistence Without Background Threads**: Metrics persist into the database's `session_state` table upon explicit flush or session close, requiring no background monitoring threads.
- **Zero-Division Protection**: All ratios and averages explicitly guard against zero samples, returning clean `0.0` or `1.0` defaults.
- **Memory Database Safe**: Functions safely when running against `:memory:` databases.

---

## 7. Repository Directory Breakdown

```
ai-db/
├── pyproject.toml              # PEP 517/518 build config, metadata, console scripts, extras
├── requirements.txt            # Zero third-party runtime dependencies (-e .)
├── requirements-dev.txt        # Development dependencies (-e .[dev,mysql])
├── LICENSE                     # MIT Open-Source License
├── README.md                   # Public user guide, quickstart, MCP setup, CLI reference
├── ARCHITECTURE.md             # This document: architectural specification and SOLID design
├── vectordb.py                 # Backward-compatible CLI and import facade
├── mcp_server.py               # Model Context Protocol stdio JSON-RPC server
├── watch_sync.sh               # Portable filesystem auto-sync script
│
├── ai_db/                      # Core package
│   ├── __init__.py             # Package version, constants, and VectorDB facade
│   ├── cli.py                  # CLI argument parsing, subcommands, output formatting
│   ├── dispatcher.py           # Unified ServiceDispatcher tool registry
│   ├── constants.py            # Environment-aware paths, defaults, ignore patterns
│   ├── utils.py                # Hashing, tokenization, project detection utilities
│   ├── logger.py               # Standardized logging utilities
│   ├── ignorer.py              # Gitignore and path exclusion rules
│   ├── watcher.py              # File change polling and watch daemon
│   │
│   ├── storage/                # Pluggable storage layer
│   │   ├── __init__.py         # Storage exports
│   │   ├── backend.py          # StorageBackend ABC contract
│   │   ├── models.py           # Strongly typed slots=True domain DTOs
│   │   ├── factory.py          # StorageBackendFactory (URI & connection resolution)
│   │   ├── sqlite_backend.py   # SQLite backend (WAL mode, FTS5, BM25, zlib compression)
│   │   ├── mysql_backend.py    # MySQL 8.0+ relational adapter interface
│   │   ├── database.py         # Backward-compatible Database wrapper
│   │   └── state.py            # Session state persistence helpers
│   │
│   ├── parser/                 # Pure AST traversal & syntax analysis
│   │   ├── __init__.py         # Parser exports
│   │   ├── syntax.py           # Syntax validator via ast.parse
│   │   ├── ast_visitor.py      # Symbol extraction and outline extraction
│   │   ├── chunker.py          # AST-based code block chunker
│   │   ├── annotations.py      # Inline annotation scanner (TODO, FIXME, HACK)
│   │   ├── cross_refs.py       # Function caller and import reference extraction
│   │   └── linters.py          # Multi-language external linter runners
│   │
│   ├── search/                 # Search orchestration & ranking
│   │   ├── __init__.py         # Search exports
│   │   ├── indexer.py          # File discovery, SHA-256 change detection, pruning
│   │   ├── query.py            # BM25 chunk search and exact symbol resolution
│   │   └── skills.py           # Assistant skill routing and BM25 matcher
│   │
│   ├── analyzer/               # Token-optimized progressive disclosure
│   │   ├── __init__.py         # Analyzer exports
│   │   ├── engine.py           # AnalyzerEngine (multi-depth AST inspection)
│   │   ├── formatters.py       # Serializers (stub skeleton, sexp, json)
│   │   └── references.py       # Progressive disclosure handle store (ref:<hash>)
│   │
│   ├── memory/                 # Session context memory
│   │   ├── __init__.py         # Memory exports
│   │   └── context.py          # ContextMemory (session snapshots and recall)
│   │
│   ├── server/                 # Transport servers
│   │   ├── __init__.py         # Server exports
│   │   └── http_server.py      # ThreadingHTTPServer REST/JSON API server
│   │
│   └── telemetry/              # Performance & token telemetry
│       ├── __init__.py         # Telemetry exports
│       ├── tracker.py          # TelemetryTracker collection coordinator
│       ├── metrics.py          # Latency percentiles & token savings aggregators
│       └── diagnostics.py      # Codebase weak points diagnostic analyzer
│
└── tests/                      # 4-Tier automated test suite
    ├── conftest.py             # Shared fixtures (isolated DBs, temp workspaces)
    ├── test_packaging.py       # Features 1-5: Packaging, entry points, extras, gitignore
    ├── test_storage.py         # Features 6-11: StorageBackend, SQLite, MySQL, Factory, DTOs
    ├── test_transports.py      # Features 12-15: CLI, MCP, HTTP REST, ServiceDispatcher
    ├── test_telemetry.py       # Features 16-20: Latency, token compression, cache, weak points
    ├── test_parser.py          # AST parser, chunker, outlines, syntax diagnostics
    ├── test_search.py          # Incremental indexing, BM25 ranking, skill routing
    └── test_sanitization.py    # Features 5, 21: Zero personal paths, hygiene audit
```

---

## 8. Public Contribution & Development Workflow

### 8.1 Prerequisites
- Python 3.10 or higher
- Git
- (Optional) Docker or MySQL 8.0+ if testing the relational adapter

### 8.2 Development Setup
```bash
# 1. Clone repository
git clone https://github.com/ai-db/ai-db.git
cd ai-db

# 2. Create isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install in editable mode with development extras
pip install -e ".[dev]"

# 4. Verify installation
ai-db --version
vectordb --help
```

### 8.3 Quality Standards & Code Formatting
- **PEP 8 Compliance & Linting**: Code formatting and linting are enforced via `ruff`:
  ```bash
  ruff check .
  ruff format --check .
  ```
- **Static Type Checking**: Clean Python 3.10+ typing is enforced via `mypy`:
  ```bash
  mypy ai_db
  ```
- **Zero-Dependency Constraint**: No third-party packages may be added to `project.dependencies` in `pyproject.toml`. Core capabilities must remain pure standard library. Optional dependencies must belong to `[project.optional-dependencies]`.

### 8.4 Running the Test Suite
The automated test suite uses `pytest` and is organized into 4 tiers:
```bash
# Run entire test suite
pytest

# Run specific subsystem tests
pytest tests/test_packaging.py
pytest tests/test_storage.py
pytest tests/test_transports.py
pytest tests/test_telemetry.py
pytest tests/test_sanitization.py

# Run with test coverage
pytest --cov=ai_db --cov-report=term-missing
```

### 8.5 Pull Request & Hygiene Checklist
Before opening a pull request:
1. Ensure all tests pass (`pytest`).
2. Run the sanitization audit to confirm zero leaked machine paths:
   ```bash
   pytest tests/test_sanitization.py
   git grep -n "/home/"
   ```
3. Ensure no database binary files (`*.db`, `*.sqlite`) or `__pycache__` artifacts are tracked in git:
   ```bash
   git status --porcelain
   ```
4. Adhere to conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
