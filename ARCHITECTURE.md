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
- **Sub-Millisecond Retrieval**: High-throughput SQLite WAL mode with FTS5 BM25 ranking, zlib compression (`ZLIB_LEVEL`) for code chunk payloads, and a query-result cache invalidated by index generation.

### 1.1 High-Level Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Clients & AI Agent IDEs                            │
│           (Claude Desktop, Cursor, Antigravity, Terminal CLI, Scripts)      │
└───────────────────┬───────────────────────┬───────────────────┬─────────────┘
                    │                       │                   │
    ┌───────────────▼──────────┐ ┌──────────▼───────────┐ ┌─────▼───────────────────┐
    │      CLI Interface       │ │   Stdio MCP Server   │ │   HTTP daemon (opt-in)  │
    │     (ai_db/cli.py)       │ │(ai_db/transports/    │ │   (ai_db/daemon/)       │
    │                          │ │       mcp.py)        │ │ warm VectorDB / project │
    └───────────────┬──────────┘ └──────────┬───────────┘ └─────┬───────────────────┘
                    └───────────────────────┼───────────────────┘
                                            │
                              ┌─────────────▼─────────────────┐
                              │   Unified ServiceDispatcher   │
                              │     (ai_db/dispatcher.py)     │
                              │  + tool plugins (ai_db.tools) │
                              └───────────────┬───────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         │                                    │                                    │
┌────────▼────────┐                  ┌────────▼────────┐                  ┌────────▼────────┐
│  Parser Subsys  │                  │  Search Engine  │                  │    Analysis     │
│ (ai_db/parser/) │                  │ (ai_db/search/) │                  │(ai_db/analysis/)│
│ AST & Chunking  │                  │ Indexer & BM25  │                  │ investigate,    │
└────────┬────────┘                  └────────┬────────┘                  │ trace, analyze  │
         │                                    │                           └────────┬────────┘
         └────────────────────────────────────┼────────────────────────────────────┘
                                              │
                              ┌───────────────▼───────────────┐
                              │   VectorDB Facade / Domain    │
                              │       (ai_db/facade.py)       │
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
| **Single Responsibility Principle (SRP)** | Every module has one, and only one, reason to change. Parsing ASTs does not query databases; storage backends do not format CLI output; transports do not parse ASTs. | `ai_db/parser/` (AST parsing only)<br>`ai_db/storage/` (persistence only)<br>`ai_db/search/` (indexing/ranking only)<br>`ai_db/transports/mcp.py`, `ai_db/daemon/` (protocol handling only) |
| **Open/Closed Principle (OCP)** | Core indexing, querying, and analysis services are closed for modification but open for extension. New storage engines (e.g. DuckDB, PostgreSQL) and new transport adapters (e.g. WebSocket, gRPC) can be plugged in without modifying existing domain code. | `ai_db/storage/backend.py`<br>`ai_db/storage/factory.py`<br>`ai_db/dispatcher.py` |
| **Liskov Substitution Principle (LSP)** | Any storage backend implementing `StorageBackend` can be substituted into `VectorDB` or `Indexer` without altering the correctness of the system. Implementations communicate strictly through typed `slots=True` domain DTOs. | `ai_db/storage/models.py`<br>`ai_db/storage/sqlite_backend.py`<br>`ai_db/storage/conformance.py` |
| **Interface Segregation Principle (ISP)** | The storage contract is cleanly segregated into focused functional areas (Files, Chunks, Symbols, Cross-Refs, Diagnostics, Skills, Context Memory, Analysis Refs, State/Cache). Domain services invoke only the interface subsets they need. | `ai_db/storage/backend.py` |
| **Dependency Inversion Principle (DIP)** | High-level business logic (`VectorDB`, `Indexer`, `QueryEngine`, `SkillRouter`, `ContextMemory`, `AnalyzerEngine`) depends on the abstract `StorageBackend` contract, never on concrete database connection drivers (`sqlite3.Connection`). Dependencies are injected via constructors. | `ai_db/facade.py`<br>`ai_db/search/indexer.py`<br>`ai_db/search/query.py` |

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

`StorageBackend` (`ai_db/storage/backend.py`) defines the contract that every storage engine must implement. Every method is abstract (there are no `NotImplementedError` defaults); an excerpt:

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
- **zlib Compression** (`ZLIB_LEVEL`): Code chunks are compressed into binary blobs (`zcontent`) upon insertion and decompressed transparently on retrieval, saving 60–80% disk space.
- **Cascading Deletions**: Deleting a file cascades to associated chunks, symbols, and cross-references.
- **One write path**: every public write method runs as one `transaction()` (`@_writes`,
  process-wide `_WRITE_LOCK`, savepoints when nested), so a failed write never leaves rows behind.
- **Repositories**: `SQLiteBackend` (`sqlite_backend.py`) owns the connection, schema,
  transactions, `status`/`optimize`; the data access is split into one mixin per concern in
  `ai_db/storage/sqlite_repos/` (`chunks`: files, chunks, FTS, diagnostics, annotations;
  `vectors`; `graph`: symbols, refs, centrality; `memory`: skills, contexts, recall vectors;
  `cache`: query cache, query log, analysis refs, semantic cache; `state`), all sharing the
  backend's connection and `transaction()`.

### 3.5 Recall extensions (`ai_db/ext/recall.py`)
Skill routing (`route_skills`) and context recall (`query_contexts`) always run keyword
search (BM25, trigger phrases); recall providers add a second signal. A provider is a
`RecallProvider` (`index_skill`, `index_context`, `sync`, `rank_skills`, `rank_contexts`)
built by a factory `(backend, embedder, options)` registered in the `ai_db.recall`
entry-point group; `recall.providers` in the config selects which run. Skill scores add
`SKILL_W_RECALL x` rescaled similarity; context lists are fused with RRF.

The built-in `vectors` provider (`ai_db/ext/vector_recall.py`) is registered the same way,
in ai-db's own pyproject.toml. It embeds skills (`description + triggers`) and contexts
(`title + summary`), records the model id in `session_state`, drops and re-embeds
everything when the model changes, and backfills items it has not seen. Embedding errors
propagate.

### 3.6 Tree-sitter grammars per language (`ai_db/parser/languages.py`)
`EXT_TO_LANG` is the one extension -> grammar map. `index.languages` selects grammars;
the selection is passed explicitly (indexer jobs, `chunk_file`, `parse_source`,
`extract_file_outline`, `AnalyzerEngine`) rather than held globally, because one daemon
process serves projects with different configs. A de-selected language is chunked as text
and never parsed or linted. `VectorDB` stores the selection it indexed with and resets the
hashes of files whose grammar was toggled, so the next sync re-parses exactly those.

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

> **History: the old HTTP REST transport (`ai-db serve`) was removed** because it had no
> authentication and let any caller index an arbitrary host path. The daemon
> (`ai_db/daemon/`, §4.2) is the HTTP transport that replaced it: it is opt-in, binds to
> localhost by default, fails closed once API keys are configured, and gates project
> registration (which takes a DB path) behind the admin key.

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
- **MCP Server (`ai_db/transports/mcp.py`, `ai-db mcp`)**: Exposes registered tools over stdio JSON-RPC 2.0. Dynamically exports all tools in `tools/list` and executes calls via `dispatcher.execute()`. A `tools/call` carrying `params._meta.progressToken` streams `notifications/progress` frames while a long `sync` runs. The root `mcp_server.py` is a shim for existing `python mcp_server.py` client configs.
- **Daemon (`ai_db/daemon/`)**: a FastAPI server holding one warm `VectorDB` +
  `ServiceDispatcher` per project. Tool calls run in worker threads under a per-project
  lock; the client's cwd travels with each call (`ai_db.utils.client_cwd`) so relative
  paths resolve as they would in the CLI. Auth, rate limiting and the request log are one
  middleware (`guard`). Auth fails closed once any key is set: per-project keys
  (`AI_DB_API_KEYS`), per-user keys (`AI_DB_USER_KEYS`, each user's allowed projects in
  `daemon.users`) and the admin key (`AI_DB_ADMIN_KEY`, the only key for management
  routes). `GET /status/html` is a read-only status page (loopback clients or the admin
  key). With `daemon.watch_sync_paths` the daemon watches `auto_sync_paths`
  (`ai_db/watcher.py`) and syncs changed files into their projects. See README "Daemon Mode".

### 4.2a Tool plugins (`ai_db/plugins.py`)
A package adds tools by declaring an entry point in the `ai_db.tools` group whose value is
`register(dispatcher)`. `ServiceDispatcher.__init__` calls every installed plugin, so all
transports expose the same tool set. `register` must be cheap (import heavy dependencies
inside handlers); a plugin that fails to import raises `AiDbConfigError`. Plugin settings
live in the config's `plugins.<name>` section, which core does not interpret. The 3D scene
and video tools are such a plugin: `packages/ai-db-scene`.

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

`ai_db/analysis/formatters.py` provides 4 distinct serializations:

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

Telemetry (`ai_db/telemetry/`) is derived from the query log: every logged tool call
(`query`, `investigate`, `analyze`, ...) is one `query_log` row (the newest
`QUERY_LOG_KEEP`), written by `VectorDB` with its latency, cache hit and per-stage timings.
`TelemetryTracker` only reads: every figure covers the same window, concurrent processes
append rows instead of overwriting a shared blob, and `--reset` clears the log.

### 6.1 Telemetry Summary

```json
{
  "window": {"calls": 420, "oldest": 1760000000.0, "newest": 1760003600.0, "keep": 10000},
  "latency": {
    "count": 420, "min_ms": 0.45, "max_ms": 12.3, "avg_ms": 1.82,
    "p50_ms": 1.4, "p95_ms": 3.8, "p99_ms": 6.2,
    "by_tool": {"query": {"count": 300, "avg_ms": 1.2, "p50_ms": 1.0, "p95_ms": 2.9, "...": "..."}}
  },
  "cache": {"lookups": 420, "hits": 336, "misses": 84, "hit_rate_pct": 80.0},
  "stages": {"total": {"count": 420, "p50_ms": 1.4, "p95_ms": 3.8}, "bm25": {"...": "..."}},
  "storage": {"db_size_kb": 1420.0, "table_rows": {"files": 45, "chunks": 320, "symbols": 510}}
}
```

Weak points (syntax error density, complexity hotspots, unindexed/stale files) come from
`TelemetryTracker.compute_weak_points()` (`StorageBackend.get_weak_points`).

### 6.2 Guarantees
- **One row per call**: the only write is the query-log row the call already produces.
- **Zero-Division Protection**: an empty log reports zeros, not errors.
- **Memory Database Safe**: Functions safely when running against `:memory:` databases.

---

## 7. Repository Directory Breakdown

```
ai-db/
├── pyproject.toml              # PEP 517/518 build config, metadata, console scripts, extras
├── requirements.txt            # Core runtime install (-e .)
├── requirements-dev.txt        # Development dependencies (-e .[dev])
├── LICENSE                     # MIT Open-Source License
├── README.md                   # Public user guide, quickstart, MCP setup, CLI reference
├── ARCHITECTURE.md             # This document: architectural specification and SOLID design
├── vectordb.py                 # Backward-compatible CLI and import facade (root py-module)
├── mcp_server.py               # Shim: `python mcp_server.py` runs ai_db.transports.mcp
├── watch_sync.sh               # Portable filesystem auto-sync script
│
├── ai_db/                      # Core package
│   ├── __init__.py             # Lazy public exports (ai_db/_lazy.py)
│   ├── facade.py               # VectorDB: the object the transports hold
│   ├── cli.py                  # CLI argument parsing, subcommands, output formatting
│   ├── dispatcher.py           # Unified ServiceDispatcher tool registry
│   ├── plugins.py              # `ai_db.tools` entry points (tool plugins)
│   ├── config.py               # AppConfig loading and validation (CONFIG_VERSION = 2)
│   ├── config_template.py      # Fresh-config builder and migrations
│   ├── constants.py            # All tunable numbers
│   ├── device.py               # torch device resolution for embed/rerank
│   ├── errors.py               # The project's exception types
│   ├── health.py               # Health and readiness reporting
│   ├── http_client.py          # JSON-over-HTTPS client for hosted model providers
│   ├── ignorer.py              # .aidbignore and path exclusion rules
│   ├── lazy.py                 # Lazy embedder/reranker (models load on first use)
│   ├── logger.py               # Standardized logging utilities
│   ├── mcp_config.py           # MCP client config snippets (`ai-db init-agent`)
│   ├── output_filter.py        # `ai-db run` / `filter`: command output under a budget
│   ├── output_store.py         # Stored raw outputs for re-filtering
│   ├── quiet.py                # Quiet stderr for third-party libraries
│   ├── tokens.py               # count_tokens (tiktoken), the one token counter
│   ├── utils.py                # Hashing, tokenization, project detection utilities
│   ├── watcher.py              # watchfiles: `ai-db watch` and the daemon's auto-watch
│   │
│   ├── transports/             # Protocol adapters over the dispatcher
│   │   └── mcp.py              # Stdio MCP server (JSON-RPC 2.0)
│   │
│   ├── daemon/                 # Opt-in HTTP daemon (FastAPI)
│   │   ├── server.py           # Project pool, auth, status page, auto-watch, routes
│   │   ├── client.py           # DaemonClient (the CLI/MCP side)
│   │   ├── manager.py          # start/stop/register (`ai-db daemon ...`)
│   │   ├── models.py           # Request/response models
│   │   └── _json.py            # JSON helpers
│   │
│   ├── storage/                # Pluggable storage layer
│   │   ├── backend.py          # StorageBackend ABC contract (all abstract)
│   │   ├── models.py           # Strongly typed slots=True domain DTOs
│   │   ├── factory.py          # StorageBackendFactory (URI & connection resolution)
│   │   ├── sqlite_backend.py   # SQLiteBackend: connection, schema, transactions
│   │   ├── sqlite_repos/       # SQLite repositories: chunks, vectors, graph, memory, cache, state
│   │   ├── conformance.py      # Backend-agnostic contract suite
│   │   └── database.py         # Backward-compatible Database wrapper
│   │
│   ├── parser/                 # Tree-sitter parsing, symbols and chunking
│   │   ├── languages.py        # Extension -> grammar map, per-language selection
│   │   ├── naming.py           # One naming rule for chunks and call-graph scopes
│   │   ├── ts_graph.py         # extract_graph: symbols, cross-refs, syntax errors
│   │   ├── receiver_types.py   # Python receiver -> class resolution
│   │   ├── queries/            # One .scm per language
│   │   ├── ast_visitor.py      # File outlines
│   │   ├── chunker.py          # Tree-sitter-backed code block chunker
│   │   ├── annotations.py      # Inline annotation scanner (TODO, FIXME, HACK)
│   │   └── linters.py          # parse_source and external linters for non-grammar files
│   │
│   ├── search/                 # Search orchestration & ranking
│   │   ├── indexer.py          # Incremental indexer, post-sync hooks, progress
│   │   ├── retriever.py        # Retrieval stages and snippet generation
│   │   ├── query.py            # QueryEngine: lexical / hybrid / vector
│   │   ├── query_builder.py    # FTS query construction
│   │   ├── ranking.py          # Score fusion, RRF, exact-symbol boost
│   │   ├── freshness.py        # Stale result files: re-sync or flag
│   │   ├── cache.py            # Query cache keys and generation invalidation
│   │   └── skills.py           # SKILL.md parsing and skill routing
│   │
│   ├── analysis/               # Investigation, tracing, per-file analysis
│   │   ├── investigate.py      # Investigator: locate/explain/impact/flow/diff packs
│   │   ├── pack.py             # InvestigationPack / EntryPoint / Evidence DTOs
│   │   ├── resolve.py          # Qualified-name resolver
│   │   ├── trace.py            # TraceEngine: call-flow execution order
│   │   ├── trace_format.py     # text / json / mermaid trace renderers
│   │   ├── analyze.py          # AnalyzerEngine: analyze / locate (progressive disclosure)
│   │   ├── refs.py             # ReferenceStore: `ref:<hash>` handles for `expand`
│   │   ├── diff.py             # The one diff engine (git hunks, index comparison)
│   │   └── formatters.py       # Pack/analysis serializers (stub, sexp, json)
│   │
│   ├── embed/                  # Embedding providers (base, registry, st_provider, hosted, indexing, defaults)
│   ├── rerank/                 # Reranking providers (base, registry, providers)
│   ├── ext/                    # Extensions: recall providers (recall.py, vector_recall.py), chunker base
│   ├── eval/                   # Offline evaluation harness (harness.py, metrics.py)
│   ├── memory/                 # Session context memory (context.py)
│   └── telemetry/              # Telemetry derived from the query log (tracker.py, stages.py)
│
├── packages/ai-db-scene/       # Plugin package: 3D scene and video tools (`ai_db.tools`)
├── tests/                      # Automated test suite (core; the plugin has its own)
├── eval/                       # Golden sets, baselines and recorded results
│   ├── golden/                 # Query sets per eval kind
│   ├── baseline*.json          # CI regression gates
│   └── results/                # Recorded measurements
│
└── .github/workflows/ci.yml    # pytest, ruff, mypy and the eval gates
```

### 7.1 Scene and video (plugin)
The 3D scene and video tools live in `packages/ai-db-scene/` (its own `pyproject.toml`,
tests and docs). Core knows nothing about them: the plugin registers its tools through
the `ai_db.tools` entry point (§4.2a) and reads its settings from `plugins.scene`. Core
runs, and its tests pass, with the plugin uninstalled.

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
