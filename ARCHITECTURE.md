# ai-db Architecture Specification

> **Version**: 0.2.0  
> **Status**: Authoritative Architectural Blueprint  
> **Scope**: Component decoupling, SOLID design rationale, pluggable storage & transport guides, performance telemetry, repository layout, and public contribution workflow.

---

## 1. Architectural Philosophy & Overview

`ai-db` is an extensible, high-performance local code intelligence and vector indexing platform engineered specifically for autonomous AI agents, coding assistants (Claude Desktop, Cursor, Antigravity, OpenCodeInterpreter), and developer workflows.

Core tenets of the architecture:
- **No Heavy Runtime Dependencies**: The core package needs only a small declared stack (tree-sitter, sqlite-vec, tiktoken, watchfiles, numpy) on top of the Python 3.10+ Standard Library (`ast`, `sqlite3`, `zlib`, `argparse`, `typing`, `json`, `hashlib`, `threading`). Torch and sentence-transformers are optional.
- **Token Optimization First**: AI agents pay heavily for context window consumption. `ai-db` optimizes token consumption through progressive disclosure (`summary` -> `structure` -> `targeted` -> `full`) and alternative serialization formats (Stub skeletons and S-Expressions), achieving 50% to 70% token reductions.
- **SOLID and KISS Design**: Clear layer boundaries, protocol-based abstractions, interchangeable storage engines, uniform cross-transport dispatching, and dependency injection.
- **Sub-Millisecond Retrieval**: High-throughput SQLite WAL mode with FTS5 BM25 ranking, zlib level 9 compression for code chunk payloads, and in-memory semantic caching.

### 1.1 High-Level Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Clients & AI Agent IDEs                            │
│           (Claude Desktop, Cursor, Antigravity, Terminal CLI, Scripts)      │
└───────────────────┬───────────────────────────────────────────┘
                    │
    ┌───────────────▼───────────────┐     ┌───────────────────────────────┐
    │       CLI Interface           │     │      Stdio MCP Server         │
    │     (ai-db / vectordb)        │     │    (JSON-RPC 2.0 stdio)       │
    └───────────────┬───────────────┘     └───────────────┬───────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
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
                      │  SQLiteBackend   │         │ Plugin backends   │
                      │(WAL, FTS5, vec0) │         │ (entry points)    │
                      └──────────────────┘         └───────────────────┘
```

---

## 2. SOLID & KISS Design Principles

`ai-db` strictly adheres to SOLID object-oriented design and KISS (Keep It Simple, Stupid) engineering principles:

| Principle | Architectural Implementation in `ai-db` | Primary File Locations |
|-----------|------------------------------------------|------------------------|
| **Single Responsibility Principle (SRP)** | Every module has one, and only one, reason to change. Parsing ASTs does not query databases; storage backends do not format CLI output; transports do not parse ASTs. | `ai_db/parser/` (AST parsing only)<br>`ai_db/storage/` (persistence only)<br>`ai_db/search/` (indexing/ranking only)<br>`ai_db/transports/` (protocol handling only) |
| **Open/Closed Principle (OCP)** | Core indexing, querying, and analysis services are closed for modification but open for extension. New storage engines (e.g. DuckDB, PostgreSQL) and new transport adapters (e.g. WebSocket, gRPC) can be plugged in without modifying existing domain code. | `ai_db/storage/backend.py`<br>`ai_db/storage/factory.py`<br>`ai_db/dispatcher.py` |
| **Liskov Substitution Principle (LSP)** | Any storage backend implementing `StorageBackend` can be substituted into `VectorDB` or `Indexer` without altering the correctness of the system. Implementations communicate strictly through typed `slots=True` domain DTOs. | `ai_db/storage/models.py`<br>`ai_db/storage/sqlite_backend.py`<br>`ai_db/storage/conformance.py` |
| **Interface Segregation Principle (ISP)** | The storage contract is cleanly segregated into focused functional areas (Files, Chunks, Symbols, Cross-Refs, Diagnostics, Skills, Context Memory, Analysis Refs, State/Cache). Domain services invoke only the interface subsets they need. | `ai_db/storage/backend.py` |
| **Dependency Inversion Principle (DIP)** | High-level business logic (`VectorDB`, `Indexer`, `QueryEngine`, `SkillRouter`, `ContextMemory`, `AnalyzerEngine`) depends on the abstract `StorageBackend` contract, never on concrete database connection drivers (`sqlite3.Connection`). Dependencies are injected via constructors. | `ai_db/__init__.py`<br>`ai_db/search/indexer.py`<br>`ai_db/search/query.py` |

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
The default engine (`ai_db/storage/sqlite_backend.py`) provides embedded persistence with no server to run:
- **Write-Ahead Logging (WAL)**: `PRAGMA journal_mode=WAL` enables concurrent readers alongside writers without mutual locking.
- **FTS5 Full-Text Indexing**: Fast BM25 keyword matching across code tokens and skill documentation.
- **zlib Level 9 Compression**: Code chunks are compressed into binary blobs (`zcontent`) upon insertion and decompressed transparently on retrieval, saving 60–80% disk space.
- **Cascading Deletions**: Deleting a file cascades to associated chunks, symbols, and cross-references.

### 3.4 Writing a storage connector

ai-db ships only the SQLite backend. Other databases are separate pip packages that
register an entry point in the `ai_db.storage` group. The built-in backend is
registered the same way (`sqlite = "ai_db.storage.sqlite_backend:SQLiteBackend.from_options"`),
so there is one discovery path for every backend.

**Contract**

1. The entry point loads a callable `(options: dict) -> StorageBackend`. `options` is
   `storage.options` from the config file. Reject unknown keys with `AiDbConfigError`.
2. Subclass `ai_db.storage.backend.StorageBackend` and implement every abstract method.
3. Override `capabilities()` to return the features you support, from
   `{"fts", "vector", "graph"}`. `retrieval.mode = "hybrid"` requires `"vector"`; if you
   declare it, also subclass `ai_db.storage.backend.VectorCapable`.
4. Run the shipped conformance kit (`ai_db.storage.conformance.BackendConformance`)
   in your own test suite.

**Skeleton**

```python
# my_ai_db_postgres/__init__.py
from ai_db.errors import AiDbConfigError
from ai_db.storage.backend import StorageBackend, VectorCapable


class PostgresBackend(StorageBackend, VectorCapable):
    def __init__(self, dsn: str):
        self.dsn = dsn

    @property
    def backend_name(self) -> str:
        return "postgres"

    def capabilities(self) -> frozenset:
        return frozenset({"fts", "vector", "graph"})

    # ... implement every StorageBackend and VectorCapable method ...


def create(options: dict) -> PostgresBackend:
    unknown = set(options) - {"dsn"}
    if unknown or "dsn" not in options:
        raise AiDbConfigError("postgres storage needs exactly {'dsn': str}")
    return PostgresBackend(options["dsn"])
```

```toml
# my_ai_db_postgres/pyproject.toml
[project.entry-points."ai_db.storage"]
postgres = "my_ai_db_postgres:create"
```

```python
# tests/test_conformance.py
import pytest
from ai_db.storage.conformance import BackendConformance
from my_ai_db_postgres import create


class TestPostgres(BackendConformance):
    @pytest.fixture
    def backend(self):
        b = create({"dsn": "postgresql://localhost/aidb_test"})
        b.initialize()
        yield b
        b.close()
```

Select it with `ai-db init --storage postgres`, then set `storage.options` in the config.
If the provider name is not installed, ai-db stops with the list of installed providers.

---

## 4. Pluggable Transport & Communication Layer

`ai-db` decouples communication interfaces (CLI, MCP) from business logic through a centralized `ServiceDispatcher`.

> **Removed: the HTTP REST transport.** `ai-db serve` and `ai_db/server/http_server.py` were
> deleted. The server could only ever analyse code on its own filesystem, in its own
> SQLite file, yet it exposed a network API with wildcard CORS, a tool-discovery endpoint,
> and no authentication -- and `POST /tools/sync` let any unauthenticated caller index an
> arbitrary path on the host. The one genuine capability it had over stdio MCP (N concurrent
> clients on one shared index) was never used and would have needed auth, a read-only mode
> and tenant isolation before it was safe. MCP over SSH covers the single-client case
> without a listening port.

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
        """Return registered tools and schemas for MCP tools/list."""
        ...

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Validate input arguments against schema and execute tool handler."""
        ...
```

### 4.2 Transport Adapters
- **CLI (`ai_db/cli.py`)**: Translates argparse commands to dispatcher tool calls. Formats output for terminal consumption (colored tables, text outlines) or JSON (`--format json`).
- **MCP Server (`mcp_server.py`)**: Exposes registered tools over stdio JSON-RPC 2.0. Dynamically exports all tools in `tools/list` and executes calls via `dispatcher.execute()`. A `tools/call` carrying `params._meta.progressToken` streams `notifications/progress` frames while a long `sync` runs.
- **HTTP REST: removed.** See the note at the top of §4. No HTTP transport ships.

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

### 5.2 Call-Flow Trace Engine (`ai_db/analysis/trace.py`)

The trace engine provides chronological execution order analysis — the missing link between "what calls what" and "in what order does it execute". It reads the enriched `symbol_refs` table (call order `seq`, `await_kind`, `guard`, `receiver`) and walks the call graph depth-first in `seq` order.

**Key capabilities:**
- **Entry points**: Symbol name, `file:line`, or script file (uses `__main__` or module-level code as root)
- **Direction**: Down (callees, default) or Up (callers)
- **Await-kind markers**: `sync`, `await`, `spawn`, `callback`, `deferred`
- **Readiness summary**: For every `spawn`/`callback`/`deferred` edge, identifies the nearest subsequent `wait`/`await` on the same path (or reports "not awaited")
- **Output formats**: Indented tree (default), JSON, Mermaid `sequenceDiagram`
- **Code context**: `--with-code` includes ±N lines around each call site (configurable via `TRACE_CONTEXT_LINES`)

**Wait pattern detection**: Default patterns in `TRACE_WAIT_PATTERNS` (`wait_*`, `*_until*`, `join`, `result`, `gather`, `wait_for`, etc.) plus per-project override via `trace.wait_patterns` / `trace.wait_patterns_extend` in the v2 config.

**Integration:**
- CLI: `ai-db trace <entry> [options]`
- MCP tool: `trace`
- `investigate --mode flow`: Picks best entry-like seed and runs trace from it

**Storage schema extension** (`symbol_refs` table):
```sql
call_col INTEGER,          -- column offset of the call
seq INTEGER,               -- call order within caller body
await_kind TEXT,           -- 'sync' | 'await' | 'spawn' | 'callback' | 'deferred'
guard TEXT,                -- enclosing if/while/try condition text
receiver TEXT              -- e.g., 'self.db', 'asyncio', 'module.func'
```
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
```
ai-db/
├── pyproject.toml              # PEP 517/518 build config, metadata, console scripts, extras
├── requirements.txt            # Core runtime install (-e .)
├── requirements-dev.txt        # Development dependencies (-e .[dev])
├── LICENSE                     # MIT Open-Source License
├── README.md                   # Public user guide, quickstart, MCP setup, CLI reference
├── ARCHITECTURE.md             # This document: architectural specification and SOLID design
├── vectordb.py                 # Backward-compatible CLI and import facade
├── mcp_server.py               # Model Context Protocol stdio JSON-RPC server
├── watch_sync.sh               # Portable filesystem auto-sync script
│
├── ai_db/                      # Core package
│   ├── __init__.py             # VectorDB facade: the object the transports hold
│   ├── cli.py                  # CLI argument parsing, subcommands, output formatting
│   ├── dispatcher.py           # Unified ServiceDispatcher tool registry
│   ├── config.py               # AppConfig loading and validation (CONFIG_VERSION = 2)
│   ├── config_template.py      # Fresh-config builder and v1 -> v2 migration
│   ├── constants.py            # All tunable numbers, per TODO rule 5
│   ├── device.py               # torch device resolution for embed/rerank
│   ├── errors.py               # The project's exception types
│   ├── health.py               # Health and readiness reporting
│   ├── http_client.py          # JSON-over-HTTPS client for hosted model providers
│   ├── ignorer.py              # Gitignore and path exclusion rules
│   ├── logger.py               # Standardized logging utilities
│   ├── utils.py                # Hashing, tokenization, project detection utilities
│   ├── watcher.py              # File change polling and watch daemon
│   │
│   ├── storage/                # Pluggable storage layer
│   │   ├── __init__.py         # Storage exports
│   │   ├── backend.py          # StorageBackend ABC contract
│   │   ├── models.py           # Strongly typed slots=True domain DTOs
│   │   ├── factory.py          # StorageBackendFactory (URI & connection resolution)
│   │   ├── sqlite_backend.py   # SQLite backend (WAL, FTS5, BM25, zlib, vec0)
│   │   ├── conformance.py      # Backend-agnostic contract suite
│   │   ├── database.py         # Backward-compatible Database wrapper
│   │   └── state.py            # Session state persistence helpers
│   │
│   ├── parser/                 # Tree-sitter parsing, symbols and chunking
│   │   ├── __init__.py         # Parser exports
│   │   ├── ts_graph.py         # extract_graph: symbols, cross-refs, syntax errors
│   │   ├── queries/            # One .scm per language (9 languages)
│   │   ├── ast_visitor.py      # Non-tree-sitter outline extraction
│   │   ├── chunker.py          # Tree-sitter-backed code block chunker
│   │   ├── annotations.py      # Inline annotation scanner (TODO, FIXME, HACK)
│   │   └── linters.py          # Multi-language external linter runners
│   │
│   ├── search/                 # Search orchestration & ranking
│   │   ├── __init__.py         # Search exports
│   │   ├── indexer.py          # Incremental indexer, post-sync hooks, progress
│   │   ├── retriever.py        # Retrieval stages and snippet generation
│   │   ├── query.py            # QueryEngine: lexical / hybrid / vector
│   │   ├── query_builder.py    # SQL construction for the FTS and vector paths
│   │   ├── ranking.py          # Score fusion, RRF, exact-symbol boost
│   │   ├── cache.py            # Query cache keys and generation invalidation
│   │   └── skills.py           # SKILL.md parsing and skill routing
│   │
│   ├── analysis/               # Investigation packs and call-flow tracing
│   │   ├── __init__.py         # Analysis exports
│   │   ├── investigate.py      # Investigator: locate/explain/impact/flow/diff packs
│   │   ├── pack.py             # InvestigationPack / EntryPoint / Evidence DTOs
│   │   ├── resolve.py          # Qualified-name resolver
│   │   ├── trace.py            # TraceEngine: call-flow execution order
│   │   └── trace_format.py     # text / json / mermaid trace renderers
│   │
│   ├── embed/                  # Embedding providers
│   │   ├── __init__.py         # Embed exports
│   │   ├── base.py             # EmbeddingProvider protocol
│   │   ├── registry.py         # Provider resolution
│   │   ├── st_provider.py      # sentence-transformers (the [local-embed] extra)
│   │   ├── hosted.py           # Hosted embedding API providers
│   │   ├── indexing.py         # Bulk vector backfill
│   │   └── defaults.py         # Suggested model / weight defaults
│   │
│   ├── rerank/                 # Reranking providers
│   │   ├── __init__.py         # Rerank exports
│   │   ├── base.py             # RerankProvider protocol
│   │   ├── registry.py         # Provider resolution
│   │   └── providers.py        # Cross-encoder rerankers
│   │
│   ├── eval/                   # Offline evaluation harness
│   │   ├── __init__.py         # Eval exports
│   │   ├── harness.py          # Golden-set runners (retrieval, pack, skills)
│   │   └── metrics.py          # recall@k, MRR, nDCG, RRF, LCS order accuracy
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
│   │
│   └── telemetry/              # Performance & token telemetry
│       ├── __init__.py         # Telemetry exports
│       ├── stages.py           # Per-stage timing attribution
│       └── tracker.py          # TelemetryTracker collection coordinator
│
├── tests/                      # Automated test suite
│   ├── conftest.py             # Hermetic isolation fixtures and CLI runner
│   ├── fake_providers.py       # Deterministic embedding/rerank stand-ins
│   ├── test_packaging.py       # Packaging, entry points, extras, gitignore
│   ├── test_storage.py         # StorageBackend, SQLite, Factory, DTOs
│   ├── test_storage_conformance.py  # Backend-agnostic contract suite
│   ├── test_storage_adversarial_m2.py # Adversarial storage cases
│   ├── test_transports.py      # CLI, MCP, ServiceDispatcher parity
│   ├── test_telemetry.py       # Latency, token compression, cache, weak points
│   ├── test_parser.py          # Parsing, outlines, syntax diagnostics, chunking
│   ├── test_ts_graph.py        # Tree-sitter queries for all 9 languages
│   ├── test_chunker_ts.py      # Tree-sitter-backed chunking
│   ├── test_search.py          # Incremental indexing, BM25 ranking, skill routing
│   ├── test_lexical.py         # Lexical-only retrieval paths
│   ├── test_investigate.py     # Investigation packs, modes, budgets
│   ├── test_config.py          # Config v2 validation and migration
│   ├── test_embeddings.py      # Embedding providers and vector search
│   ├── test_rerank.py          # Rerank providers
│   ├── test_cache_watch.py     # Query cache and file watcher
│   ├── test_observability.py   # Health and telemetry surfaces
│   ├── test_eval_metrics.py    # Eval harness and baseline comparison
│   ├── test_sanitization.py    # Zero personal paths, hygiene audit
│   ├── test_phase11.py         # Config migration, skills, context vectors
│   ├── test_phase13.py         # Device resolution, write lock, incremental centrality
│   ├── test_phase14.py         # MCP progress, diff mode, cache isolation
│   ├── test_bench_index.py     # Indexer benchmark (marked `bench`)
│   ├── test_bench_vectors.py   # exact vs vec0 vector benchmark (marked `bench`)
│   └── test_bench_gpu.py       # CPU vs CUDA embedding benchmark (marked `bench`)
│
├── eval/                       # Golden sets, baselines and recorded results
│   ├── golden/                 # Query sets per eval kind
│   ├── baseline*.json          # CI regression gates
│   └── results/                # Recorded measurements
│
└── .github/workflows/ci.yml    # pytest, ruff, mypy and the eval gates
```


---

## 8. Public Contribution & Development Workflow

### 8.1 Prerequisites
- Python 3.10 or higher
- Git

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
- **PEP 8 Compliance & Linting**: Enforced via `ruff`. CI runs exactly this, and
  the scope comes from `[tool.ruff]` in `pyproject.toml` — stray `.py` files at the
  repo root are deliberately out of scope, so a lint run cannot rewrite them:
  ```bash
  uv run ruff check .
  ```
  `ruff format` is **not** run: the repository is not format-clean (80 of 106 files
  would be reformatted), so a `ruff format --check` gate would always fail. If you
  want that gate, run `ruff format` once and commit the result.
- **Static Type Checking**: Clean Python 3.10+ typing is enforced via `mypy`:
  ```bash
  uv run mypy ai_db mcp_server.py
  ```
- **Heavy-ML Dependency Constraint**: The core may depend on the parsing and index
  stack (`tree-sitter`, `tree-sitter-language-pack`, `sqlite-vec`, `tiktoken`,
  `watchfiles`, `numpy`) — the core is *not* zero-dependency. What is forbidden in
  `project.dependencies` is heavy ML (`torch`, `sentence-transformers`): those belong in
  `[project.optional-dependencies].local-embed`, so a bare install stays small and the
  whole test suite runs without torch. Enforced by
  `tests/test_packaging.py::test_core_runtime_dependencies_exclude_heavy_ml`.

### 8.4 Running the Test Suite
The suite uses `pytest` and runs without torch installed; tests needing the
`[local-embed]` stack or a CUDA device skip with an explanatory reason.
```bash
# Entire test suite
uv run pytest

# Specific subsystem
uv run pytest tests/test_storage.py
uv run pytest tests/test_transports.py
uv run pytest tests/test_ts_graph.py

# Benchmarks are marked `bench` and deselected by default
uv run pytest tests/test_bench_vectors.py -m bench -q -s
uv run pytest tests/test_bench_gpu.py -m bench -q -s

# Lint and type-check exactly as CI runs them
uv run ruff check ai_db tests --exclude token_benchmark.py
uv run mypy ai_db mcp_server.py

# Coverage
uv run pytest --cov=ai_db --cov-report=term-missing
```

### 8.5 Pull Request & Hygiene Checklist
Before opening a pull request:
1. Ensure all tests pass (`uv run pytest`), plus `ruff check` and `mypy` — CI gates on all three.
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
