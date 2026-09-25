# ai-db: Code Intelligence & Retrieval Platform for AI Agents

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Dependencies: Zero](https://img.shields.io/badge/dependencies-0%20(stdlib%20only)-blueviolet.svg)](pyproject.toml)
[![Token Efficiency](https://img.shields.io/badge/token%20savings-80--95%25-success.svg)](#token-optimization-benchmarks)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy-informational.svg)](http://mypy-lang.org/)

A high-speed, self-contained local code intelligence engine, vector database, and AST analyzer designed specifically for AI coding assistants (such as Claude Desktop, Cursor, Google Antigravity, and autonomous coding agents) to **instantly recall codebase architecture, symbols, files, and session memories with 80%–95% token savings without repetitive codebase re-analysis on every chat**.

---

## Table of Contents

- [Overview & Value Proposition](#overview--value-proposition)
- [Key Features](#key-features)
- [Architecture at a Glance](#architecture-at-a-glance)
- [Installation & Quickstart](#installation--quickstart)
  - [GPU acceleration (optional)](#gpu-acceleration-optional)
  - [Configuration (required)](#configuration-required)
  - [One-call analysis for agents (`ai-db investigate`)](#one-call-analysis-for-agents-ai-db-investigate)
  - [Measuring retrieval quality](#measuring-retrieval-quality)
- [CLI Commands Reference](#cli-commands-reference)
  - [1. Code Health & Syntax Pre-Flight (`check`)](#1-code-health--syntax-pre-flight-ai-db-check)
  - [2. Exact Symbol Definition Resolution (`symbol`)](#2-exact-symbol-definition-resolution-ai-db-symbol)
  - [3. Instant File Outlines (`outline`)](#3-instant-file-outlines-ai-db-outline)
  - [4. Semantic & Full-Text Search (`query`)](#4-semantic--full-text-search-ai-db-query)
  - [5. Token-Optimized Code Analysis (`analyze`)](#5-token-optimized-code-analysis-ai-db-analyze)
  - [6. Progressive Disclosure Expansion (`expand`)](#6-progressive-disclosure-expansion-ai-db-expand)
  - [7. Natural Language Relevance Ranking (`locate`)](#7-natural-language-relevance-ranking-ai-db-locate)
  - [8. Session Context & Memory Recall (`context` / `remember`)](#8-session-context--memory-recall-ai-db-context--ai-db-remember)
  - [9. Performance & Token Telemetry (`telemetry`)](#9-performance--token-telemetry-ai-db-telemetry)
  - [10. HTTP REST API Server (`serve`)](#10-http-rest-api-server-ai-db-serve)
  - [11. Codebase Synchronization (`sync`, `sync-all`, `watch`)](#11-codebase-synchronization-ai-db-sync-sync-all-watch)
  - [12. Smart Skill Routing (`route-skill`)](#12-smart-skill-routing-ai-db-route-skill)
  - [13. Code Insights (`callers`, `diff`, `todos`)](#13-code-insights-ai-db-callers-diff-todos)
  - [14. Database Maintenance (`optimize`, `status`, `prune`)](#14-database-maintenance-ai-db-optimize-status-prune)
- [Model Context Protocol (MCP) Server](#model-context-protocol-mcp-server)
  - [Claude Desktop Configuration](#claude-desktop-configuration)
  - [Cursor IDE Configuration](#cursor-ide-configuration)
  - [Google Antigravity Configuration](#google-antigravity-configuration)
  - [MCP Tools Reference](#mcp-tools-reference)
    - [Progress notifications](#progress-notifications)
- [HTTP REST API Reference](#http-rest-api-reference)
- [Token Optimization Benchmarks](#token-optimization-benchmarks)
- [Project Scoping & Multi-Repo Access](#project-scoping--multi-repo-access)
- [Development & Testing](#development--testing)
- [Architecture & Documentation](#architecture--documentation)
- [License](#license)

---

## Overview & Value Proposition

### The Problem
Traditional AI coding workflows waste enormous context windows and API costs. When an AI coding assistant inspects a project, it repeatedly reads hundreds of raw files into its prompt context. This burns tens of thousands of tokens per turn, exceeds LLM context budgets, introduces high latency, and leads to context truncation or reasoning hallucinations.

### The Solution: `ai-db`
`ai-db` operates as a high-speed local code intelligence sidecar. Instead of ingesting raw code dumps, AI assistants query `ai-db` for:
- **Exact AST symbol definitions**: Instant jump-to-definition without full-text false positives.
- **Structural file skeletons**: Outline functions, class hierarchies, and type annotations consuming $\le 10\%$ of raw file tokens.
- **Progressive disclosure (`ref:hash`)**: Opaque node handles that allow agents to expand specific function bodies only when needed.
- **Token-compressed representations**: Native code skeletons (`--fmt stub`) and Lisp-like AST S-expressions (`--fmt sexp`) saving **80% to 95%** of token overhead.
- **Session context memory (`ai-db remember`)**: Persistent checkpointing of architectural decisions, open tasks, and active files across chats.

---

## Key Features

- **Zero External Dependencies**: The core platform (indexing, AST analysis, search, CLI, MCP stdio server, HTTP REST server) relies strictly on the Python 3 standard library (`sqlite3`, `ast`, `hashlib`, `json`, `http.server`, etc.).
- **Sub-Millisecond Search**: SQLite WAL mode with FTS5 BM25 ranking, identifier-aware tokenization, and zlib level-9 compression executes complex code queries in $< 2\text{ ms}$.
- **Pluggable Storage Layer (SOLID / Open-Closed)**: Decoupled `StorageBackend` abstraction with a built-in SQLite backend; other databases plug in as separate packages via the `ai_db.storage` entry-point group (see ARCHITECTURE.md §3.4).
- **Pluggable Transports**: Single unified `ServiceDispatcher` serving CLI commands (`ai-db`, `vectordb`), Model Context Protocol (stdio JSON-RPC 2.0), and threaded HTTP REST endpoints.
- **Performance & Token Telemetry**: Quantitative measurement of p50/p95/p99 query latencies, compression savings across 4 serialization formats, semantic cache hit rates, and codebase weak points (syntax error density, complexity hotspots).
- **Project Isolation & Scoping**: Auto-detects project boundaries via `.git`, `pyproject.toml`, or `package.json` to prevent cross-project context pollution while allowing explicit read-only sharing.

---

## Architecture at a Glance

```
                  ┌────────────────────────────────────────────────┐
                  │          AI Clients / Developer Tools          │
                  │   (Claude Desktop, Cursor, Antigravity, CLI)   │
                  └──────────────┬──────────────────┬──────────────┘
                                 │                  │
                         ┌───────▼────────┐  ┌──────▼───────┐
                         │   CLI / stdio  │  │   HTTP API   │
                         │ (ai-db, MCP)   │  │ (REST / JSON)│
                         └───────┬────────┘  └──────┬───────┘
                                 │                  │
                     ┌───────────▼──────────────────▼───────────┐
                     │         Unified ServiceDispatcher        │
                     │           (Tool Registry & Schema)       │
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
                            │ SQLiteBackend  │  │ Plugin (e.p.)│
                            │ (WAL,FTS5,vec) │  │  backends    │
                            └────────────────┘  └──────────────┘
```

For complete architectural details, design rationale, and extension guides, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Standard Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/example/ai-db.git
cd ai-db

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install editable package (core: tree-sitter, sqlite-vec, tiktoken, watchfiles)
pip install -e .
```

### Modular Dependency Extras

Install optional components as needed:

```bash
# Install local embedding / rerank model support (sentence-transformers + torch)
pip install -e ".[local-embed]"

# Install with development, linting, and testing tooling (pytest, ruff, mypy)
pip install -e ".[dev]"

# Install all optional dependencies
pip install -e ".[all]"
```

### GPU acceleration (optional)

`ai-db` never installs CUDA torch for you — pick the wheel that matches your driver:

```bash
# CUDA 12.8 (recommended; covers Blackwell / RTX 50xx and anything older back to Volta)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# CUDA 11.8 (older drivers, pre-Ampere)
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU only
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`ai-db init` writes `"device": "cuda"` when a usable GPU is detected and `"cpu"`
otherwise, so the usual flow needs no editing. To see what was resolved:

```bash
ai-db config check
# OK   embedding device=cuda (cuda available, torch.version.cuda=12.8)
```

A config that asks for `"device": "cuda"` on a machine without a usable GPU is a
**hard error**, not a silent downgrade to CPU — a large index that quietly runs
100× slower on CPU is worse than a clear failure. Override the auto-detected
value by editing `device` in the config.

Indexing a mid-sized repository typically takes a few minutes on CPU and
seconds on a modern GPU. This is a target, not a guarantee: it depends entirely
on your model size, hardware and sequence length.

### Verification
Both `ai-db` and `vectordb` console commands are installed as entry points:

```bash
ai-db --version
vectordb --version
```

### Configuration (required)

Every command except `init` needs a config file; there are no hidden runtime defaults.

```bash
ai-db init                                        # sqlite + lexical (BM25) retrieval
ai-db init --force --embedding sentence_transformers   # hybrid BM25 + local vectors
ai-db init --force --embedding voyage --rerank voyage  # hosted embeddings + rerank
ai-db init --migrate                              # convert an old unversioned config
ai-db config show                                 # resolved config (secrets hidden)
ai-db config check                                # builds every enabled provider, 1 test call each
```

The file lives at `--config PATH`, `$AI_DB_CONFIG`, `$XDG_CONFIG_HOME/ai-db/config.json`
or `~/.config/ai-db/config.json` (first match). Sections:

| Section | Values |
|---|---|
| `storage.provider` | `sqlite` (built in) or any installed `ai_db.storage` plugin; `storage.options` go to the plugin |
| `storage.options.path` | SQLite file path; `null` uses the platform default location |
| `storage.options.vector_index` | `exact` (default) or `vec0`. `vec0` runs the KNN in sqlite-vec's C loop instead of a Python loop — **not** an ANN index, so it is a constant-factor win, not an asymptotic one. See the caveat below before expecting a large speedup. |
| `retrieval.mode` | `lexical` or `hybrid` (hybrid requires an embedding provider and a `vector`-capable backend) |
| `retrieval.doc_weight` | `0.0`–`1.0`. Weight of file-level documentation chunks (module headers, markdown sections) against code chunks. `0.5` by default |
| `index.ignore` | Glob patterns excluded from indexing. Defaults to `[".agents/**"]` |
| `index.doc_weight` | Per-index counterpart of `retrieval.doc_weight` |
| `embedding.provider` | `none`, `sentence_transformers`, `openai_compatible`, `voyage`, or an `ai_db.embedding` plugin |
| `embedding.device` | `cpu` or `cuda`. `ai-db init` auto-detects; `cuda` on a machine without a usable GPU is a hard error, never a silent downgrade |
| `embedding.batch_size` | Documents per forward pass. Raise it on a GPU with VRAM to spare |
| `rerank.provider` | `none`, `sentence_transformers`, `voyage`, `cohere`, or an `ai_db.rerank` plugin |
| `rerank.top_n` | How many candidates the reranker re-scores. Default `10` |
| `access.cross_project` | `{"project": ["other-project", ...]}` read access grants |

Config files are versioned (`"version": 2`). `ai-db init --migrate` converts an
unversioned or v1 file in place; a v1 file loaded without migrating is rejected
with a message pointing at that command rather than being silently reinterpreted.

> **`vec0` is not an ANN index.** sqlite-vec's `vec0` KNN is a brute-force scan, so
> query time still grows linearly with corpus size. Measured on 100k x 1024
> normalized vectors, `vec0` is ~1.5x faster than the exact path with recall@10 of
> exactly 1.000. Choose it for the constant factor and for metadata pruning; do not
> expect order-of-magnitude gains. Reproduce with
> `uv run pytest tests/test_bench_vectors.py -m bench -q -s`.

Model names are only ever read from the config. `ai-db init` writes suggested models
(see `ai_db/embed/defaults.py`) with a `_note` to verify them on MTEB-Code/CoIR.
Changing the embedding model requires `ai-db reindex --embeddings`. Hosted providers
read their API key from the environment variable named in `api_key_env`.

### One-call analysis for agents (`ai-db investigate`)

| Mode | Question it answers | What it adds |
|---|---|---|
| `locate` | "where is X?" | the best matching chunks and nothing else |
| `explain` | "how does X work?" | callees, owning class, covering tests, and the direct callers that show how the code is entered |
| `impact` | "what breaks if X changes?" | transitive callers up to the configured depth, plus the tests that would catch it |
| `flow` | "what order does this run in?" | the call-flow trace from the most entry-like seed |
| `diff` | "what does this change touch?" | seeds from `git diff <since>`, plus the callers of every changed symbol — the review surface of a change |

```bash
ai-db investigate "how are search results ranked"            # mode: explain
ai-db investigate "search_chunks" --mode impact --budget 6000
ai-db investigate "where is the config validated" --mode locate
ai-db investigate --mode diff --since HEAD~1                 # review the working tree vs a ref
ai-db investigate --mode diff --since main...feature/x      # a range works too
```

`diff` mode reads `git diff --unified=0`, so the seeds are the symbols whose line
spans the diff actually touches. The pack adds a `changes` array with the changed
spans themselves. It raises a clear error rather than returning an empty pack if
`--since` is missing or the path is not a git repository — an empty pack would read
as "nothing to review" instead of "I could not look". Note that `git diff <ref>`
also spans *uncommitted* work; pass a range (`<a>..<b>`) to review committed history
only.

Returns a JSON evidence pack under the token budget: entry points with reasons, full
bodies of the best matches, stubs + `ref:` handles (`ai-db expand`) for classes, callees,
callers and covering tests, the call graph, recent commits and unresolved names. Also
available as the `investigate` MCP tool and `POST /investigate`.

Over MCP, a `tools/call` that carries `params._meta.progressToken` streams
`notifications/progress` frames while a long `sync` runs, so indexing a large
repository does not look like a hang. Clients that do not ask for progress get
nothing extra.

### Measuring retrieval quality

```bash
# Retrieval quality: recall@k, MRR, nDCG against a golden query set
ai-db eval --golden eval/golden/ai_db.jsonl --root . --baseline eval/baseline.json

# Investigation packs: does the pack actually contain the expected symbol?
ai-db eval --pack --golden eval/golden/ai_db_pack.jsonl --root . \
  --baseline eval/baseline_pack.json

# Polyglot coverage (TypeScript, Go, Rust) -- needs --all-files
ai-db eval --pack --golden eval/golden/polyglot_pack.jsonl \
  --root tests/fixtures/polyglot --all-files --baseline eval/baseline_pack_polyglot.json

# Diff-mode packs. Pins commit ranges, so it needs the full history and is
# deliberately not a CI gate; run it locally after a change to ranking.
ai-db eval --pack --golden eval/golden/ai_db_diff.jsonl --root . --all-files

# Skill routing: top-1 accuracy over a prompt -> skill golden set.
# NOTE: skills must already be indexed, or this reports skills_indexed: 0 and
# top1_accuracy: 0.0 without explaining why. Sync the skill dirs first.
AI_DB_SKILL_DIRS=tests/fixtures/skills ai-db sync tests/fixtures/skills
AI_DB_SKILL_DIRS=tests/fixtures/skills ai-db eval --skills --golden eval/golden/skills.jsonl
# -> {"queries": 20, "skills_indexed": 5, "top1_accuracy": 1.0}

ai-db log --slow 500        # slow queries with per-stage latency
```

Passing `--baseline` makes the command a regression gate: it exits non-zero if the
measured recall drops more than the tolerance in the baseline file, so CI fails on
a quality regression rather than only on a crash. Recorded results live in
`eval/results/`, and the numbers each baseline was measured at are in the
`_note` field alongside it.

---

## CLI Commands Reference

### 1. Code Health & Syntax Pre-Flight (`ai-db check`)
Fast AST validation to catch syntax errors with exact file, line, and column numbers before running builds or tests:

```bash
# Check current directory
ai-db check .

# Check specific file
ai-db check src/services/auth.py

# Continuous watch mode (re-checks instantly upon file saves)
ai-db check . --watch --interval 1.5
```

*Example Output:*
```
SYNTAX_ERROR: src/services/auth.py:42:18 Unterminated string literal
Total errors: 1
```

---

### 2. Exact Symbol Definition Resolution (`ai-db symbol`)
Resolves canonical `class`, `def`, and `interface` definitions directly via AST extraction without noisy full-text matches:

```bash
ai-db symbol AuthService
ai-db symbol verify_token
```

*Example Output:*
```
@src/services/auth.py:L18 (class AuthService)
@src/services/auth.py:L94 (def verify_token)
```

---

### 3. Instant File Outlines (`ai-db outline`)
Extracts class hierarchies, method signatures, and top-level definitions without blowing context tokens:

```bash
ai-db outline src/services/auth.py
```

*Example Output:*
```
File: src/services/auth.py (148 lines)
  L18: class AuthService
    L24: def __init__(self, db_client)
    L52: def create_session(self, user_id: str) -> str
    L94: def verify_token(self, token: str) -> dict
```

---

### 4. Semantic & Full-Text Search (`ai-db query`)
Executes sub-millisecond BM25 ranking across code chunks with identifier-aware tokenization:

```bash
ai-db query "sqlite fts5 bm25 ranking" --top 5
```

---

### 5. Token-Optimized Code Analysis (`ai-db analyze`)
Multi-depth AST code inspection designed for minimal token overhead:

```bash
# Outline signatures only (<= 10% of raw file tokens)
ai-db analyze src/services/auth.py --depth summary

# AST structure with class methods and types
ai-db analyze src/services/auth.py --depth structure

# Targeted: extract bodies only matching a question or concept filter
ai-db analyze src/services/auth.py -q "token expiration check" --depth targeted

# Range target: lines 50 to 90 with 5 lines surrounding context
ai-db analyze src/services/auth.py --span 50:90 --ctx 5

# Select serialization format: stub (code skeleton), sexp (S-expression), or json
ai-db analyze src/services/auth.py --depth summary --fmt stub
```

---

### 6. Progressive Disclosure Expansion (`ai-db expand`)
Expands opaque handles (`ref:hash`) returned by `analyze` on demand:

```bash
# Expand function body by reference
ai-db expand ref:a8f9c1

# Expand specific sub-span within a handle
ai-db expand ref:a8f9c1 --span 1:15
```

---

### 7. Natural Language Relevance Ranking (`ai-db locate`)
Finds top-k candidate files and snippets answering a natural question or concept:

```bash
ai-db locate "JWT token expiration and signature validation" -k 5
```

---

### 8. Session Context & Memory Recall (`ai-db context` / `ai-db remember`)
Persists conversation checkpoints, active files, architectural decisions, and open tasks across chats:

```bash
# Save session checkpoint
ai-db context save auth-refactor \
  --title "JWT Migration" \
  --summary "Migrating authentication service from cookie sessions to JWT" \
  --files "src/services/auth.py,src/models/user.py" \
  --tasks "Implement token refresh,Add unit tests for expired token"

# Instant context recall in a fresh chat session (/remember)
ai-db remember
```

*Example Output:*
```markdown
### [ai-db Context Memory: my-project / auth-refactor]
**Title**: JWT Migration
**Summary**: Migrating authentication service from cookie sessions to JWT

**Active Files**:
- src/services/auth.py
- src/models/user.py

**Pending Tasks**:
1. Implement token refresh
2. Add unit tests for expired token
```

---

### 9. Performance & Token Telemetry (`ai-db telemetry`)
Inspect query latencies (p50/p95/p99), token compression efficiency across 4 formats, cache hit rates, and codebase weak points:

```bash
# Human-readable summary
ai-db telemetry

# Machine-readable JSON output
ai-db telemetry --json
```

*Example Output:*
```
[ai-db Telemetry Summary]
Query Latency: p50=1.2ms | p95=3.4ms | p99=6.1ms (124 queries)
Token Efficiency: Raw=145,000 -> Stub=21,750 (85.0% savings) | S-Exp=13,050 (91.0% savings)
Cache Hit Rate: 92.4% (85 hits / 7 misses)
Codebase Weak Points: 0 syntax errors, 2 complexity hotspots
```

---

### 10. HTTP REST API Server (`ai-db serve`)
Launches the built-in HTTP server (standard library only, no web framework to install):

```bash
ai-db serve --port 8765 --host 127.0.0.1
```

See [HTTP REST API Reference](#http-rest-api-reference) for endpoint details.

---

### 11. Codebase Synchronization (`ai-db sync`, `sync-all`, `watch`)

```bash
# Sync current repository or path (incremental, SHA-256 change detection)
ai-db sync .

# Multi-repository synchronization from ~/.config/ai-db/config.json
ai-db sync-all

# Continuous file watcher (foreground)
ai-db watch .

# Continuous file watcher (background daemon)
ai-db watch . --daemon
```

---

### 12. Smart Skill Routing (`ai-db route-skill`)
Matches user prompts against installed agent skills in $< 2\text{ ms}$ without polluting LLM prompts:

```bash
ai-db route-skill "Design an ad banner for LinkedIn with glassmorphism style"
ai-db route-skill "Check if there are broken python files" --format json
ai-db route-skill "Create presentation slides" --format path
```

---

### 13. Code Insights (`ai-db callers`, `diff`, `todos`, `trace`)
 
```bash
# Find all call sites and references to a symbol
ai-db callers verify_token
 
# Show changed spans since last indexed snapshot or git ref
ai-db diff src/services/auth.py --since last
 
# Extract TODO, FIXME, and HACK annotations across project
ai-db todos --kind fixme
 
# Trace chronological call flow from an entry point (symbol, file:line, or script)
ai-db trace launch_stage_ov                                    # trace down from symbol
ai-db trace src/services/auth.py:94                            # trace from file:line
ai-db trace ./scripts/deploy.py                                # trace from script (__main__)
ai-db trace open_stage --direction up                          # trace callers up
ai-db trace launch_stage_ov --format mermaid                   # mermaid sequence diagram
ai-db trace launch_stage_ov --format json --with-code          # JSON with code context
ai-db trace launch_stage_ov --depth 10 --max-nodes 50          # limit depth/nodes
```
 
*Example Output (tree format):*
```
⏵ launch_stage_ov
    ⏸ open_stage L5
      ⇉ _load_async L11       # spawned (asyncio.create_task)
      │   ⏸ _fetch_assets L16
      ⏵ _init_ui L12
    ⏸ wait_for_stage_load L6
      ⏵ _wait_ready L28
    ⏸ preload_prefabs L7
```
 
**Readiness Summary** (printed after trace):
```
`_load_async` spawned by `open_stage` (not awaited)
`_fetch_assets` runs after `wait_for_stage_load` (sync, stage_manager.py:28)
```
 
**Markers:** `⏵` sync call · `⏸` awaited · `⇉` spawned/fire-and-forget · `↺` callback · `⌁` deferred (event handler)
 
---

### 14. Database Maintenance (`ai-db optimize`, `status`, `prune`)

```bash
# Defragment database, merge FTS5 indexes, and reclaim disk space
ai-db optimize

# Configure and persist default output serialization format
ai-db optimize --default-format stub

# Inspect database size and indexing statistics
ai-db status

# Prune index records for deleted files
ai-db prune
```

---

## Model Context Protocol (MCP) Server

`ai-db` includes a built-in stdio JSON-RPC 2.0 Model Context Protocol (MCP) server. AI assistants use this interface to invoke `ai-db` tools autonomously during reasoning sessions.

### Manual Stdio Testing

Test the MCP server directly from your terminal:

```bash
# Launch MCP server directly via CLI:
ai-db mcp

# Or launch directly with python:
python3 mcp_server.py
```

### Claude Desktop Configuration

Add `ai-db` to your Claude Desktop configuration file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Using the installed `ai-db` executable:
```json
{
  "mcpServers": {
    "ai-db": {
      "command": "ai-db",
      "args": ["mcp"]
    }
  }
}
```

*Alternative using Python virtualenv and `mcp_server.py`:*
```json
{
  "mcpServers": {
    "ai-db": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/ai-db/mcp_server.py"]
    }
  }
}
```

### Cursor IDE Configuration

Add `ai-db` in `.cursor/mcp.json` (workspace) or via **Settings > Features > MCP**:

```json
{
  "mcpServers": {
    "ai-db": {
      "command": "ai-db",
      "args": ["mcp"]
    }
  }
}
```

### Google Antigravity Configuration

Add `ai-db` to `~/.gemini/antigravity/mcp_config.json` or project MCP settings:

```json
{
  "mcpServers": {
    "ai-db": {
      "command": "ai-db",
      "args": ["mcp"]
    }
  }
}
```

### MCP Tools Reference

Every tool below is also reachable from the CLI and over HTTP, so an agent can use
whichever transport it already has.

| MCP Tool | Description | Key Arguments |
|---|---|---|
| `investigate` | **Start here.** One ranked evidence pack for a question, instead of many grep/read calls. `mode` selects `locate` / `explain` / `impact` / `flow` / `diff` | `query`, `mode`, `since`, `root`, `budget_tokens`, `project`, `allow_project`, `languages`, `chunk_types`, `modified_since` |
| `query` | Ranked chunk search (BM25, or hybrid with vectors) | `query`, `top`, `project`, `allow_project`, `languages`, `chunk_types`, `modified_since` |
| `locate` | Natural language concept and symbol search with snippet spans | `query`, `scope`, `k`, `format` |
| `symbol` | Exact symbol lookup (classes, functions, methods) with file and line | `name`, `project`, `allow_project` |
| `outline` | Top-level class and function signatures of one file | `path` |
| `analyze` | Token-optimized AST inspection with progressive depth control | `targets`, `depth`, `q`, `focus`, `span`, `ctx_lines`, `since`, `max_out`, `cursor`, `format`, `no_cache` |
| `expand` | Progressive disclosure of a `ref:hash` handle | `ref`, `depth`, `span` |
| `callers` | All call sites and references to a symbol | `name`, `project`, `allow_project` |
| `trace` | Chronological call flow from an entry point | `entry`, `direction`, `depth`, `max_nodes`, `include_tests`, `format`, `with_code`, `project`, `allow_project` |
| `check` | Syntax pre-flight with line/column on failure | `path`, `project`, `allow_project` |
| `diff` | Changed line spans in a file since the last snapshot or a git ref | `path`, `since` |
| `todos` | TODO, FIXME, HACK and NOTE annotations across indexed files | `kind`, `filepath`, `project` |
| `sync` | Scan a directory or file; update symbols, chunks and the FTS5 index | `path`, `project`, `verbose` |
| `sync_skills` | Scan and index `SKILL.md` definitions from directories | `skill_dirs`, `project` |
| `route_skill` | Route a user prompt to matching skills (lexical + vector) | `prompt`, `top`, `min_confidence`, `project` |
| `context_save` | Persist conversation memory, active files, tasks, architecture notes | `session_id`, `summary`, `title`, `active_files`, `open_tasks`, `notes`, `project` |
| `context_recall` | Recall session context and decisions across past conversations | `session_id`, `query`, `project` |
| `status` | Database health metrics and entity counts | — |
| `optimize` | Defragment the DB, merge FTS5 B-trees, update the query cache | `prune_missing`, `default_format` |
| `prune` | Drop records for deleted or missing files | — |
| `telemetry` | Latency percentiles and token-compression efficiency | `reset`, `detail` |

#### Progress notifications

A `tools/call` that carries `params._meta.progressToken` gets
`notifications/progress` frames streamed on stdout while the tool runs — most
usefully for `sync` on a large repository, which otherwise looks like a hang.
Updates are emitted at the first file, every 50th, and the last, each with
`progress`, `total` and a `message` naming the file. A client that does not send a
`progressToken` receives nothing extra, and progress is scoped to the call: it is
detached afterwards, so a later call cannot report through it.

---

## HTTP REST API Reference

Run `ai-db serve --port 8765` to start the lightweight HTTP server. All endpoints include automatic CORS preflight (`OPTIONS`) handling.

### Endpoints Overview

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check; returns `{"ok": true}` |
| `GET` | `/status` | Database statistics (files, chunks, symbols, size) |
| `GET` | `/tools` | List registered tools and JSON Schemas |
| `POST` | `/` | Execute tool via envelope `{"tool": "<name>", "args": {...}}` |
| `POST` | `/tools/{name}` | Execute tool with direct JSON arguments body |
| `GET` | `/telemetry` | Retrieve latency percentiles and token efficiency metrics |

### Example Requests

#### 1. Liveness Check
```bash
curl -s http://127.0.0.1:8765/health
```
```json
{
  "ok": true
}
```

#### 2. Locate Relevant Files & Symbols
```bash
curl -s -X POST http://127.0.0.1:8765/ \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "locate",
    "args": {
      "query": "authentication token verification",
      "scope": ".",
      "k": 3,
      "format": "json"
    }
  }'
```
```json
[
  {
    "file": "src/services/auth.py",
    "name": "verify_token",
    "span": [94, 120],
    "score": -3.12,
    "snippet": "def verify_token(self, token: str) -> dict:\n    payload = jwt.decode(token, self.secret)\n    return payload"
  }
]
```

#### 3. Save Session Context Memory
```bash
curl -s -X POST http://127.0.0.1:8765/ \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "context_save",
    "args": {
      "session_id": "auth-api",
      "title": "API Token Validation",
      "summary": "Implemented JWT bearer token authentication",
      "active_files": ["src/services/auth.py"],
      "open_tasks": ["Add token revocation blacklist"]
    }
  }'
```
```json
{
  "saved": true,
  "session_id": "auth-api",
  "project": "my-project",
  "timestamp": 1726639200.0
}
```

---

## Token Optimization Benchmarks

`ai-db` dramatically cuts token consumption by eliminating raw file ingestion in favor of structural and semantic representations:

| Representation / Format | CLI Flag | Token Footprint vs Raw | Token Savings | Optimal Use Case |
|---|---|---|---|---|
| **Raw File Dump** | *(traditional)* | 100% (Baseline) | 0% | Dumb context loading (wasteful) |
| **Standard JSON** | `--fmt json` | ~40%–55% | 45%–60% | Automated tooling & structured parsing |
| **Native Stub** | `--fmt stub` | ~12%–20% | **80%–88%** | LLM reasoning, types, signatures |
| **S-Expression** | `--fmt sexp` | ~5%–12% | **88%–95%** | Maximum context density & tree navigation |
| **Signature Summary** | `--depth summary`| ~4%–8% | **92%–96%** | High-level file scanning & discovery |

---

## Project Scoping & Multi-Repo Access

`ai-db` auto-detects project boundaries via `.git`, `pyproject.toml`, or `package.json`. Code chunks and symbol queries are isolated to the active project by default.

### Configuration (`~/.config/ai-db/config.json`)

Configure automatic project synchronization and cross-project read-only permissions:

```json
{
  "auto_sync_paths": [
    "~/projects/web-frontend",
    "~/projects/api-backend",
    "~/projects/shared-core"
  ],
  "cross_project_access": {
    "web-frontend": ["shared-core"],
    "api-backend": ["shared-core"]
  }
}
```

### CLI Cross-Project Access
Query an allowed external project on a read-only basis:

```bash
ai-db symbol SharedConfig --allow-project shared-core
ai-db query "database pool" --allow-project shared-core
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `AI_DB_PATH` | Path to primary SQLite index database | `~/.local/share/ai-db/codebase_knowledge.db` |
| `AI_DB_CONFIG_PATH` | Path to JSON configuration file | `~/.config/ai-db/config.json` |
| `AI_DB_SKILL_DIRS` | Colon-separated directories containing agent skills | Standard XDG skill paths |

---

## Development & Testing

### Running the Test Suite
The repository includes a comprehensive 5-tier test suite:

```bash
# Run all unit and integration tests
pytest

# Run repository sanitization and hygiene audit
pytest tests/test_sanitization.py -v

# Run storage layer tests
pytest tests/test_storage.py -v

# Run transport adapter tests (CLI, MCP, HTTP)
pytest tests/test_transports.py -v

# Run telemetry tests
pytest tests/test_telemetry.py -v
```

### Code Formatting & Type Checking
```bash
# Linting and formatting with Ruff
ruff check .
ruff format --check .

# Static type verification with Mypy
mypy ai_db
```

---

## Architecture & Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): Comprehensive architectural specification detailing SOLID design principles, pluggable `StorageBackend` implementation guide, custom transport adapters, and telemetry internals.
- [LICENSE](LICENSE): Full MIT License text.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).  
Copyright (c) 2026 ai-db Contributors.
