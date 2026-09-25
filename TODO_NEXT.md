# ai-db: implementation TODO (Phases 9–15)

This is a self-contained task list for an implementing model. Work top to bottom.
Tick each box (`[x]`) when its **Check** passes. Do not start a phase before the one it
depends on is complete.

---

## 0. Read first (rules for every task)

1. **No fallbacks.** If a dependency, grammar, model, device or git repo is required and
   missing, raise `AiDbConfigError` (from `ai_db/errors.py`) with a clear message. Never
   silently degrade, return `[]`, or switch to a weaker path.
2. **One code path per feature.** Do not keep a regex path next to a tree-sitter path.
3. **Exceptions:** only catch a specific exception type. The handler must re-raise it
   wrapped, report it at a transport boundary (HTTP/MCP, marked `# noqa: BLE001` with a
   reason), or carry a comment explaining why the case is expected.
4. **Models** are named only in the user config and `ai_db/embed/defaults.py`.
   Do not use outdated models (`all-MiniLM-*`, `bge-*-v1.5`, `text-embedding-ada-002`,
   `ms-marco-MiniLM` cross-encoders, `jina-embeddings-v2-*`).
5. **Tunable numbers** go in `ai_db/constants.py`, tuned only with `ai-db eval`.
6. **Every phase ends green** (run from the repo root with the venv):
   ```bash
   .venv/bin/python -m pytest -q
   .venv/bin/ruff check . --exclude token_benchmark.py
   .venv/bin/mypy ai_db mcp_server.py
   ```
7. **Git:** one commit per phase (more is fine), conventional-commit messages like the
   existing history (`feat(parser): ...`, `fix: ...`). Push after each phase.
   Never commit `token_benchmark.py`.
8. **Evals need a config.** Create one once: `.venv/bin/ai-db --config /tmp/lex.json init --force`.
   Use `AI_DB_PATH=/tmp/eval.db` when running eval commands so the user's DB is untouched.

### Decisions already made (do not ask the user)
- Config goes to **version 2** in Phase 11; `ai-db init --migrate` must upgrade 1 → 2.
- **Do not** install CUDA torch automatically; document it and support `device: cuda`.
- The `.agents/` folder is **ignored completely** (via `index.ignore`).
- The multi-language eval uses a **synthetic fixture** project (`tests/fixtures/polyglot/`).

### Orientation (where things are)
| Area | Files |
|---|---|
| Config / init | `ai_db/config.py`, `ai_db/config_template.py`, `ai_db/cli.py` (`_cmd_init`) |
| Chunking (tree-sitter) | `ai_db/parser/chunker.py` (`_parser`, `language_for`, `_CodeChunker._name`) |
| Symbols / refs (to replace) | `ai_db/parser/ast_visitor.py`, `ai_db/parser/cross_refs.py`, `ai_db/parser/syntax.py`, `ai_db/parser/linters.py` |
| Indexing | `ai_db/search/indexer.py` (`parse_file`, `sync`, `sync_paths`) |
| Storage | `ai_db/storage/sqlite_backend.py`, `ai_db/storage/backend.py`, `ai_db/storage/conformance.py` |
| Retrieval / ranking | `ai_db/search/retriever.py`, `ai_db/search/ranking.py`, `ai_db/search/query.py`, `ai_db/search/query_builder.py` |
| investigate | `ai_db/analysis/investigate.py`, `ai_db/analysis/pack.py` |
| Skills / memory | `ai_db/search/skills.py`, `ai_db/memory/context.py` |
| Analyzer | `ai_db/analyzer/engine.py`, `ai_db/analyzer/references.py` |
| Eval | `ai_db/eval/harness.py`, `eval/golden/*.jsonl`, `eval/results/*.json` |
| Transports | `ai_db/dispatcher.py`, `mcp_server.py`, `ai_db/server/http_server.py` |

---

## Phase 9 — Eval baseline + CI  (depends on: nothing)

- [ ] **9.1** Copy `eval/results/lexical.json` to `eval/baseline.json` (overwrite; recall@10 0.85).
- [ ] **9.2** Copy `eval/results/pack_lexical.json` to `eval/baseline_pack.json` (pack_recall 0.9).
- [ ] **9.3** In `ai_db/eval/harness.py` add
      `compare_pack_to_baseline(result, baseline, tolerance=0.02) -> str | None`
      comparing `pack_recall`; raise `ValueError` if the key is missing from `baseline`.
- [ ] **9.4** In `ai_db/cli.py` `_run_eval`: remove the error "`--baseline` ... cannot be
      combined with --pack"; when `--pack`, use `compare_pack_to_baseline`.
- [ ] **9.5** Add a unit test for `compare_pack_to_baseline` in `tests/test_eval_metrics.py`.
- [ ] **9.6** Create `.github/workflows/ci.yml`: on `push` and `pull_request`, `ubuntu-latest`,
      Python 3.12, `actions/checkout@v4` with `fetch-depth: 0`, install with `uv`:
      ```
      pip install uv
      uv pip install --system -e ".[dev]"
      ai-db --config ci.json init
      pytest -q
      ruff check . --exclude token_benchmark.py
      mypy ai_db mcp_server.py
      AI_DB_PATH=$RUNNER_TEMP/e.db ai-db --config ci.json eval --golden eval/golden/ai_db.jsonl --root . --baseline eval/baseline.json
      AI_DB_PATH=$RUNNER_TEMP/p.db ai-db --config ci.json eval --pack --golden eval/golden/ai_db_pack.jsonl --root . --baseline eval/baseline_pack.json
      ```
- [ ] **Check:** every command above passes locally. Commit `ci: ...`.

---

## Phase 10 — Tree-sitter symbols, cross-refs and syntax errors for all languages  (depends on: 9)

Goal: `investigate` gets callers/callees/tests for TS/JS/Go/Rust/C/C++/Java, not only Python.

- [ ] **10.1** Create `ai_db/parser/queries/<lang>.scm` for:
      python, javascript, typescript, tsx, go, rust, c, cpp, java.
      Captures (each def also captures `@name`):
      `@def.function`, `@def.method`, `@def.class`, `@def.interface`, `@def.type`,
      `@ref.call` (the called identifier: last identifier of the call target / selector /
      member expression), `@ref.import` (module path and imported names),
      `@ref.inherit` (base class, `extends`, `implements`, Rust `impl Trait for`).
      Add to `pyproject.toml`:
      ```toml
      [tool.setuptools.package-data]
      ai_db = ["parser/queries/*.scm"]
      ```
- [ ] **10.2** Create `ai_db/parser/ts_graph.py` with
      `extract_graph(filepath: str, content: str) -> tuple[list[dict], list[dict], list[tuple[int,int,str]]]`
      returning `(symbols, refs, syntax_errors)`:
      - symbols: `{"name", "symbol_type", "filepath", "line", "signature"}` (same shape as
        today's `extract_symbols`); `name` is the bare name, `signature` the first line.
      - refs: `{"caller_name", "caller_line", "callee_name", "ref_type"}` where
        `caller_name` is `module.<qualified>` (e.g. `module.Invoice.total_amount`), or
        `module` at top level. This must match `investigate.scope_of()`.
      - Qualified names must be built with the **same rules as `chunker._CodeChunker._name`**
        (Go receiver `S.M`, Rust impl type, C/C++ `A::b` → `A.b`). Factor the naming into a
        shared function used by both modules so they cannot diverge.
      - Imports: one ref with the dotted module path (`pkg.billing`) plus one per imported
        name (`pkg.billing.Invoice`). JS/TS relative imports (`./tax`, `../lib/x`) become the
        dotted path of the target relative to the importing file's directory with the
        extension dropped (`tax`, `lib.x`).
      - syntax_errors: first `ERROR` or `MISSING` node → `(line, col, "syntax error near '<text>'")`.
      - Unknown language → `ValueError` (callers only call it for `language_for(path)` files).
- [ ] **10.3** `ai_db/search/indexer.py` `parse_file`: for files with `language_for(path)`,
      use `extract_graph` for symbols, refs and syntax error. For markdown, symbols = one per
      `md` chunk (`symbol_type="section"`, name = chunk `name`). Other files: no symbols, no refs.
      Annotations stay as today.
- [ ] **10.4** `ai_db/parser/linters.py` is used only for extensions **without** a tree-sitter
      language (shell, yaml). On `subprocess.TimeoutExpired` return
      `(1, 1, "linter timeout")` instead of `None`.
- [ ] **10.5** Delete `ai_db/parser/cross_refs.py` and `ai_db/parser/syntax.py`. In
      `ai_db/parser/ast_visitor.py` delete `extract_symbols` and all regex code; re-implement
      `extract_file_outline(filepath, content)` from `extract_graph` symbols
      (`[(line, "<kind> <name>")]`, sorted). Update every import site (`grep -rn "cross_refs\|validate_python_syntax\|extract_symbols" ai_db tests vectordb.py`),
      including the re-exports in `ai_db/__init__.py`, `ai_db/parser/__init__.py` and `vectordb.py`
      (`extract_symbols` may be re-exported as a thin wrapper returning `extract_graph(...)[0]`).
- [ ] **10.6** Bump `SCHEMA_VERSION` in `sqlite_backend.py`.
- [ ] **10.7** `tests/test_ts_graph.py`: per language, a tiny fixture with a class/struct,
      a method, a call from one function to another, an import and an inheritance; assert the
      exact set of `(caller_name, callee_name, ref_type)` and symbol names. Update expectations in
      `tests/test_parser.py` that depended on regex behaviour (do not delete tests silently;
      replace them with the tree-sitter equivalent).
- [ ] **10.8** Create `tests/fixtures/polyglot/` (~15 files): a TS service calling a TS util,
      a Go package with a struct + methods used from another file, a Rust crate with a trait
      and impl, each with a test file. Add `eval/golden/polyglot_pack.jsonl` with 15
      explain/impact queries (paths relative to the fixture root). Because the fixture is
      inside the repo, run it with `--all-files --root tests/fixtures/polyglot`.
      Save the result to `eval/results/pack_polyglot.json` and `eval/baseline_pack_polyglot.json`;
      add the command to CI.
- [ ] **Check:**
      - Python pack eval not below `eval/baseline_pack.json` (−0.02 tolerance).
      - Polyglot pack_recall ≥ 0.8.
      - `ai-db investigate "search_chunks"` (on this repo) lists no `unresolved` names that are
        defined in this repo.
      - Commit `feat(parser): tree-sitter symbols, cross-refs and syntax errors for all languages`.

---

## Phase 10b — Optional SCIP graph provider  (depends on: 10; GATED)

**Gate — do this phase only if, after Phase 10, either:**
- polyglot pack_recall < 0.8, or
- `unresolved` in explain packs (Python golden set) averages > 5 names defined in the repo.
Otherwise mark every box "skipped (gate not met: <numbers>)" and move on.

SCIP (Sourcegraph Code Intelligence Protocol) indexers emit one `index.scip` protobuf with
compiler-resolved definitions and references. Use them instead of name matching, per language.

- [ ] **10b.1 Config** (version 2 schema from Phase 11 — if 11 is not done yet, add this key in
      11.1): `graph: {"providers": {"<lang>": "tree_sitter" | "scip"}}`. Required; template sets
      every language to `"tree_sitter"`. Unknown language or value → `AiDbConfigError`.
- [ ] **10b.2 Indexer registry** `ai_db/parser/scip.py`: fixed command per language
      (no auto-detection, no fallback):
      | lang | command (run in project root) |
      |---|---|
      | python | `scip-python index --output <tmp>/index.scip` |
      | typescript/tsx/javascript | `scip-typescript index --output <tmp>/index.scip` |
      | go | `scip-go --output <tmp>/index.scip` |
      | rust | `rust-analyzer scip . --output <tmp>/index.scip` |
      | java | `scip-java index --output <tmp>/index.scip` |
      | c/cpp | `scip-clang --compdb-path=compile_commands.json --index-output-path=<tmp>/index.scip` |
      Binary missing (`shutil.which`) or non-zero exit → `AiDbConfigError` with the command,
      exit code and last 20 stderr lines. Timeout from constant `SCIP_TIMEOUT_S = 900` → same error.
- [ ] **10b.3 Parsing:** add dependency `protobuf>=5` and vendor the generated `scip_pb2.py`
      (from `github.com/sourcegraph/scip/scip.proto`, record the upstream commit in a header
      comment) under `ai_db/parser/scip_pb2.py` (exclude from ruff/mypy via config).
      For each `Document` / `Occurrence`: `symbol_roles & Definition` → symbol; otherwise → ref.
      Map the enclosing definition range to the chunk `qualified_name` of that file/line
      (lookup via stored chunks) to build `caller_name = "module.<qualified>"`; `callee_name`
      = last descriptor of the SCIP symbol, and store the full SCIP symbol in a new nullable
      column `symbol_refs.callee_symbol` plus `chunks.scip_symbol` for definitions.
- [ ] **10b.4 Sync integration:** after chunking in `Indexer.sync`, for each language
      configured as `scip` that has changed files, run its indexer once for the project root and
      replace that language's refs/symbols for the project (SCIP is whole-project; do not use it
      in `sync_paths` — the watcher re-runs it debounced with constant `SCIP_WATCH_DEBOUNCE_S = 30`).
- [ ] **10b.5 Resolution:** in `investigate`, when both `callee_symbol` and `scip_symbol` exist,
      resolve by exact symbol equality and skip the name heuristics for that ref.
- [ ] **10b.6 Tests:** a recorded `index.scip` fixture for the polyglot TS project (commit the
      binary file) — parsing test does not require the indexer binaries. A test marked
      `@pytest.mark.scip` runs real `scip-typescript` if installed (excluded by default like `model`).
      Test that a missing binary raises `AiDbConfigError`.
- [ ] **Check:** with `typescript` set to `scip`, polyglot pack_recall improves over the Phase 10
      result; default config behaviour unchanged. Record `eval/results/pack_polyglot_scip.json`. Commit.

---

## Phase 11 — Unified retrieval, skills, doc weight, config v2  (depends on: 9)

- [ ] **11.1 Config v2** (`ai_db/config.py`, `ai_db/config_template.py`):
      - `CONFIG_VERSION = 2`. New required keys:
        `retrieval.doc_weight` (float in (0, 1]; template `0.5`),
        `rerank.top_n` (int ≥ 1, only when rerank provider ≠ none; template `10`),
        `index.ignore` (list of glob strings; template `[".agents/**"]`).
      - Add `index` to `TOP_LEVEL_KEYS` / `REQUIRED_TOP_LEVEL`; expose on `AppConfig`
        (`doc_weight`, `ignore_globs`, rerank `top_n` via options).
      - `migrate_legacy` handles unversioned → 2; add `migrate_v1(data) -> data` for 1 → 2
        (adds the three keys with template values). `ai-db init --migrate` detects the version
        and applies the right step. Version 1 files are **rejected** at load with
        "run: ai-db init --migrate".
      - Update `tests/test_config.py` and the conftest config fixture (it uses `build_template()`,
        and the session guard string in `tests/conftest.py` must become v2).
- [ ] **11.2 Ignore globs:** `Indexer.scan_directory` and `Indexer.sync_paths` skip paths
      matching `config.ignore_globs` (use `fnmatch` on the root-relative POSIX path) in
      addition to `.aidbignore`. Pass the globs into `Indexer` from `VectorDB.__init__`.
- [ ] **11.3 Doc weight:** in `Ranker.rank`, multiply the normalized fused score of candidates
      with `chunk_type in ("md", "txt")` by `doc_weight` before combining. Pass it from config.
- [ ] **11.4 `rerank.top_n`:** replace the constant `RERANK_TOP` use in `Ranker` with the
      configured value (keep the constant only as the template default).
- [ ] **11.5 locate:** `AnalyzerEngine.locate_targets` must call
      `query_engine.search(q, filters, k)` with `filters={"allowed_projects": None, "path_prefix": abs(scope)}`
      (add `path_prefix` support to `QueryEngine.search`/retrievers/`search_vectors` filters —
      `_append_filters` already supports it). Remove its direct `search_chunks` call.
      Inject `query_engine` into `AnalyzerEngine` in `VectorDB.__init__`.
- [ ] **11.6 Context + skill FTS:** `search_contexts` and `search_skills` in `sqlite_backend.py`
      build their MATCH expression with `query_builder.build_fts(expand_terms(q), base_terms(q))`
      (change their signatures to take the raw query text; update callers).
- [ ] **11.7 Hybrid for skills/contexts** (only when `retrieval.mode == "hybrid"`):
      tables `skill_vectors(name TEXT, project TEXT, embedding BLOB, PRIMARY KEY(name, project))`
      and `context_vectors(context_id INTEGER PRIMARY KEY REFERENCES contexts(id) ON DELETE CASCADE, embedding BLOB)`.
      Embed `description + triggers` at `sync_skills`, `title + summary` at `save_context`.
      Search = BM25 list + vector list fused with RRF (`RRF_K`). Add both tables to the schema
      rebuild hooks.
- [ ] **11.8 Skill router** (`ai_db/search/skills.py` `route_skills`): delete the fixed boosts
      (+4, +12, +14, +3.5, +1.5), the regex intent rules and stop-word special cases.
      New score = fused retrieval score + `SKILL_W_TRIGGER * (1 if an exact trigger phrase occurs in the prompt else 0)`.
      Confidence = min-max of scores over candidates. Keep `min_confidence` filtering and the
      `reasons` list (`"bm25 #n"`, `"vector #n"`, `"trigger match"`).
      Change the dispatcher tool description to "lexical (BM25) or hybrid, per retrieval.mode".
- [ ] **11.9 Skills eval:** `eval/golden/skills.jsonl` with 20 `{"prompt", "expected_skill"}`
      lines built from the user's installed skills (`ai-db` DB table `skills`, or `SKILL.md` files
      under `~/.gemini/config/skills` and `~/.local/share/ai-db/skills`). Add
      `ai-db eval --skills --golden ...` reporting top-1 accuracy (`harness.run_skills`). Tune
      `SKILL_W_TRIGGER` in `constants.py`. Do **not** add this eval to CI (it depends on local skills).
- [ ] **Check:** skills top-1 ≥ 0.8; code eval and pack eval not below baselines; all tests
      pass. Migrate the user's real config (`ai-db init --migrate`, back it up first to
      `~/.config/ai-db/config.v1.bak.json`) and run `ai-db sync-all`. Commit.

---

## Phase 12 — Remove remaining fallbacks  (depends on: 10)

- [ ] **12.1** `AnalyzerEngine.analyze_file`: build its symbol list from
      `ts_graph.extract_graph` symbols joined with the file's stored chunks (body = chunk
      content, `ref` = `_store_analysis_ref` of that chunk). Delete the Python-`ast` branch,
      the "Fallback to regex symbol extraction" branch and the `"AST unavailable"` note.
      Files without a language: return chunks only (md sections / text windows).
      Keep the public output shape (`symbols`, `meta`, depth handling) so existing tests pass.
- [ ] **12.2** `ai_db/analyzer/references.py` `_diff_spans`:
      `since in ("last", "db")` → compare with stored chunks only.
      Any other `since` → git required; if the file is not in a repo or `git show` fails,
      raise `AiDbConfigError(f"--since {since} needs the file to be in a git repository")`.
      Remove the "git unavailable → stored chunks" path.
- [ ] **12.3** Audit: `grep -rn "except" ai_db mcp_server.py`. Every handler must satisfy rule 3.
      Write the audit result (file:line → reason) in the commit message body.
- [ ] **Check:** tests green (update tests that relied on fallback behaviour to assert the new
      errors). Commit `fix: remove remaining fallbacks`.

---

## Phase 13 — Performance and scale  (depends on: 9; 13.5 after 11)

- [ ] **13.1 GPU:** README section: `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`
      (Blackwell / RTX 50xx needs cu128+). In `config_template.build_template`, when the
      embedding or rerank provider is `sentence_transformers`, write `"device": "cuda"` if
      `torch.cuda.is_available()` at init time, else `"cpu"` (import torch lazily; if torch is
      not installed, write `"cpu"`). In `st_provider.py` / `CrossEncoderReranker`: if device is
      `cuda` and `torch.cuda.is_available()` is false, raise `AiDbConfigError`.
      `ai-db config check` prints device and `torch.version.cuda`.
- [ ] **13.2 vec0 option:** `storage.options.vector_index` ∈ `{"exact", "vec0"}` (sqlite
      `OPTION_KEYS`; template `"exact"`; required key). `vec0` creates
      `vec_chunks USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[<dim>] distance_metric=cosine, project TEXT partition key, language TEXT, chunk_type TEXT)`
      in `ensure_vector_index`. `search_vectors` for vec0: `WHERE embedding MATCH ? AND k = ?`
      with project/language/chunk_type constraints; for `path_prefix` / `modified_since`
      fetch `k*4` and filter via a join, then truncate to `k`. Delete from `vec_chunks` in the
      chunk-delete hooks (`_before_delete_chunk_ids`, `_before_delete_file_chunks`).
      Parameterize `TestSQLiteConformance` over both modes.
- [ ] **13.3 Benchmark** `tests/test_bench_vectors.py` (`@pytest.mark.bench`): 100k random
      normalized 1024-dim vectors; report p50 query latency for both modes and recall@10 of
      vec0 vs exact.
- [ ] **13.4 Incremental centrality:** `rebuild_symbol_centrality(projects: list[str] | None)`
      recomputes only those projects (None = all). Indexer passes the set of projects written
      in the sync to the post-sync hooks (change hook signature to `hook(changed: bool, projects: set[str])`
      and update every registered hook).
- [ ] **13.5 Rerank tuning:** with the local cross-encoder configured, tune
      `RANK_W_*` rerank weights with `ai-db eval` (hybrid + rerank). Save to
      `eval/results/hybrid_rerank_tuned.json`. If it still does not beat
      `eval/results/hybrid_qwen3-0.6b.json` on recall@10 and MRR, add a comment next to the
      rerank suggestions in `ai_db/embed/defaults.py` stating that measured result.
      (These runs are slow on CPU; run in the background and poll.)
- [ ] **13.6 Write lock:** a class-level `threading.RLock` in `SQLiteBackend`, held for the
      whole outermost `transaction()` and around `_auto_commit` writes. Test: 8 threads calling
      dispatcher `sync` and `investigate` concurrently on one DB → no `database is locked`.
- [ ] **Check:** tests green; vec0 p50 ≥ 10× faster than exact at 100k with recall@10 ≥ 0.95;
      GPU indexing < 1 min is a documented target only if no GPU torch is installed.
      Commit.

---

## Phase 14 — Agent experience  (depends on: 10)

- [ ] **14.1 MCP progress:** in `mcp_server.py`, if a `tools/call` request has
      `params._meta.progressToken`, pass a `progress(done, total, message)` callback into
      `dispatcher.execute(..., progress=cb)`. The callback writes
      `{"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":..,"progress":done,"total":total,"message":..}}`
      to stdout immediately. `Indexer.sync` reports after parsing and per 50 written files;
      `Investigator.investigate` reports each stage (seeds, expand, tests, git, pack).
      Other transports pass no callback. Test with a fake stdout.
- [ ] **14.2 Diff mode:** `investigate(mode="diff", since=<git ref>)` (`since` required for this
      mode, error otherwise). Seeds = chunks overlapping changed spans from
      `git diff --unified=0 <since> -- <tracked files>` (parse `@@ -a,b +c,d @@`). Then run the
      impact expansion (callers depth `IMPACT_DEPTH`, tests). Pack gets
      `changes: [{"filepath", "spans": [[start, end]], "qualified_names": [...]}]`.
      Add to CLI (`--since`), dispatcher schema (`mode` enum + `since`), HTTP.
      Add 3 diff queries to a new `eval/golden/ai_db_diff.jsonl` pinned to existing commit
      hashes of this repo (e.g. `3875be9` → expected symbols from that commit's diff).
- [ ] **14.3 Cache isolation test:** two projects with identical files; `investigate` on each
      must not return the other's cached pack.
- [ ] **Check:** tests green. Commit.

---

## Phase 15 — Documentation  (depends on: all)

- [ ] **15.1** `ARCHITECTURE.md` §5–7: pipeline (chunker → ts_graph → FTS/vectors →
      retriever → ranker → investigate → cache → query log), one table per extension point
      (storage / embedding / rerank entry-point groups, contracts, conformance kit), current
      directory tree (generate with `git ls-files ai_db | sort`).
- [ ] **15.2** `README.md`: investigate modes incl. diff, config v2 keys
      (`retrieval.doc_weight`, `index.ignore`, `rerank.top_n`, `storage.options.vector_index`),
      GPU setup, skills eval.
- [ ] **15.3** Remove MySQL and "zero-dependency" claims from `PROJECT.md`,
      `ORIGINAL_REQUEST.md`, `TEST_INFRA.md` (keep the rest of their content).
- [ ] **Check:** `grep -rni "mysql\|zero.dependenc" *.md` returns nothing outdated. Commit `docs: ...`.
- Note: trace docs are covered by **16.7**.

---

## Phase 16 — Call-flow tracing (execution order), Python first  (depends on: 10, 11)

**Problem.** `locate`, `symbol` and `investigate` return isolated definitions. None trace
chronological execution order (e.g. `launch_stage_ov.py → open_stage → wait_for_stage_load →
preload_prefabs`) or show whether a call is awaited, spawned (fire-and-forget), a callback,
or guarded. Answering "is the stage fully loaded or still opening asynchronously?" today
requires reading raw line ranges by hand.

**Today.** `symbol_refs` (`ai_db/storage/sqlite_backend.py:353`) stores only
`caller_name, caller_line, callee_name, ref_type`. Callee/caller hops live in
`ai_db/analysis/investigate.py:243-303`.

**Scope.** Python only. Other languages follow in Phase 16b with the same schema. `trace` on a
non-Python entry point raises `AiDbConfigError("trace: language X not supported yet")` (no fallback).

- [ ] **16.1 Call-site facts (Python).** Add columns to `symbol_refs`: `call_col`, `seq`
      (call order within the caller body), `await_kind` (`sync|await|spawn|callback|deferred`),
      `guard` (short text of enclosing `if`/`while`/`try`/`with`, or NULL), `receiver`
      (`self.x`, module alias, …). Populate from the Phase 10 tree-sitter Python visitor:
  - `await`: `await f()`, `async with`, `async for`
  - `spawn`: `asyncio.create_task`, `ensure_future`, `loop.run_in_executor`,
    `threading.Thread(target=…)`, `executor.submit`
  - `callback`: function passed as an argument (`on_done=cb`, `add_done_callback(cb)`)
  - `deferred`: calls inside a function registered as a handler (decorator or
    `connect`/`subscribe` argument)
  - Bump the schema version; `ai-db reindex` rebuilds the table (no legacy rows kept).
  - **Check:** fixture `tests/fixtures/callflow/` (Python) asserts exact `seq`, `await_kind`, `guard`.
- [ ] **16.2 One resolver.** Move `_resolve`/`_linked` from `investigate.py` into
      `ai_db/analysis/resolve.py`. Order: same file → imported module → `self.`/class member →
      unique project-wide qualified name. Unresolvable → `unresolved` list, never guessed.
      `investigate` and `trace` share this single code path.
  - **Check:** existing investigate tests pass; one test per resolution step.
- [ ] **16.3 Trace engine** (`ai_db/analysis/trace.py`). Entry: symbol, `file:line`, or a script
      (module-level / `__main__` block is the root). Depth-first walk in `seq` order.
  - Options: `--depth` (default `TRACE_DEPTH`), `--max-nodes` (default `TRACE_MAX_NODES`),
    `--direction down|up`, `--include-tests`.
  - Node: qualified name, `file:start-end`, call line, `await_kind`, `guard`, recursion/revisit flag.
  - **Readiness summary:** for each `spawn`/`callback` edge, state that execution continues
    before it completes, and find the nearest later wait on the same path via wait patterns.
    Example: "`preload_prefabs` runs after `wait_for_stage_load` (sync, stage_manager.py:88);
    `open_stage` spawns `_load_async` (:71), which is not awaited."
  - **Wait patterns:** default `TRACE_WAIT_PATTERNS` in `ai_db/constants.py`
    (e.g. `("wait_*", "*_until*", "join", "result", "gather", "wait_for")`). Per-project override
    via `trace.wait_patterns` in the repo entry of the v2 config, plus `trace.wait_patterns_extend`
    to add to the defaults. Validated in `config.py`; invalid value → `AiDbConfigError`.
  - **Check:** fixture yields exact expected order and readiness notes; a second test shows a
    per-project pattern changing the readiness result.
- [ ] **16.4 Output.** Default: indented tree with `file:line` and markers `⏵` sync, `⏸` await,
      `⇉` spawn, `↺` callback, `⌁` deferred, `[if …]`. `--format json|mermaid` (mermaid =
      `sequenceDiagram`). `--with-code` adds the call line ± `TRACE_CONTEXT_LINES`. Output
      respects the token budget from `ai_db/analysis/pack.py`.
- [ ] **16.5 Transports.** `trace` subcommand in `ai_db/cli.py` (next to `callers`), dispatcher
      route, MCP tool `trace_flow`, HTTP `/trace`. `investigate --mode flow`: retriever picks
      seeds, then `trace` runs from the most entry-like seed.
- [ ] **16.6 Eval.** `eval/golden/flow.jsonl` (entry, ordered path, expected `await_kind`s) from the
      fixture plus a few ai-db self-traces. Metrics: order-accuracy (LCS / golden length) and
      edge-kind accuracy. Store `eval/baseline_flow.json`; add to the CI regression gate.
- [ ] **16.7 Docs.** `README.md` + `ARCHITECTURE.md`: trace command, markers, `trace.wait_patterns`.
      Document limits: dynamic dispatch, `getattr`/string calls and event buses appear only as
      `unresolved`/`deferred`; no runtime tracing.
- [ ] **Check:** pytest, ruff, mypy green. Commit `feat(analysis): call-flow trace (python)`.

## Phase 16b — Call-flow tracing for other languages  (depends on: 16)

- [ ] Same 16.1 facts for TS/JS, C#, Go, Rust (`spawn`: un-awaited Promise, `Task.Run`, `go`,
      `tokio::spawn`; `callback`: `.then(cb)`, …).
- [ ] Fixtures + flow-eval entries per language; then remove the non-Python guard.

**Out of scope:** runtime/instrumented tracing, data-flow analysis. SCIP (Phase 10b) may later
replace the 16.2 resolver.

---

## Final hand-back to the user
Report: commits (hash + subject), eval table (code recall@10/MRR, pack recall python and
polyglot, skills top-1, vec0 benchmark, flow order-accuracy / edge-kind accuracy), anything not completed and why, and any change
to the user's local config/DB.
