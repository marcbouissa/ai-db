# ai-db: implementation TODO (Phases 9–16b)

> **Status: working note (updated after commit `0dc9923`).** Checkboxes below were
> re-verified against the tree, not just ticked optimistically. Where a box is
> `[ ]` but partially done, the reason is given inline. Gate: `pytest` 393 passed,
> `ruff` clean, `mypy ai_db mcp_server.py` clean.

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
   Never commit `token_benchmark.py`. — **done:** it was tracked and leaked a personal
   path; untracked + gitignored in `0dc9923`.
8. **Evals need a config.** Create one once: `.venv/bin/ai-db --config /tmp/lex.json init --force`.
   Use `AI_DB_PATH=/tmp/eval.db` when running eval commands so the user's DB is untouched.

### Decisions already made (do not ask the user)
- Config goes to **version 2** in Phase 11; `ai-db init --migrate` must upgrade 1 → 2.
  — **still open:** `CONFIG_VERSION = 2` and all v2 keys landed, but no `migrate_v1`
  exists, so a real v1 file is rejected with no upgrade path (see 11.1).
- **Do not** install CUDA torch automatically; document it and support `device: cuda`.
- The `.agents/` folder is **ignored completely** (via `index.ignore`).
- The multi-language eval uses a **synthetic fixture** project (`tests/fixtures/polyglot/`).

### Orientation (where things are)
| Area | Files |
|---|---|
| Config / init | `ai_db/config.py`, `ai_db/config_template.py`, `ai_db/cli.py` (`_cmd_init`) |
| Chunking (tree-sitter) | `ai_db/parser/chunker.py` (`_parser`, `language_for`, `_CodeChunker._name`) |
| Symbols / refs | `ai_db/parser/ts_graph.py`, `ai_db/parser/ast_visitor.py`, `ai_db/parser/linters.py` |
| Indexing | `ai_db/search/indexer.py` (`parse_file`, `sync`, `sync_paths`) |
| Storage | `ai_db/storage/sqlite_backend.py`, `ai_db/storage/backend.py`, `ai_db/storage/conformance.py` |
| Retrieval / ranking | `ai_db/search/retriever.py`, `ai_db/search/ranking.py`, `ai_db/search/query.py`, `ai_db/search/query_builder.py` |
| investigate / trace | `ai_db/analysis/investigate.py`, `ai_db/analysis/pack.py`, `ai_db/analysis/trace.py`, `ai_db/analysis/resolve.py` |
| Skills / memory | `ai_db/search/skills.py`, `ai_db/memory/context.py` |
| Analyzer | `ai_db/analyzer/engine.py`, `ai_db/analyzer/references.py` |
| Eval | `ai_db/eval/harness.py`, `eval/golden/*.jsonl`, `eval/results/*.json` |
| Transports | `ai_db/dispatcher.py`, `mcp_server.py`, `ai_db/server/http_server.py` |
| Root facades | `vectordb.py` (restored as a Python module in `0dc9923`), `mcp_server.py` |

---

## Phase 9 — Eval baseline + CI  (depends on: nothing)

- [x] **9.1** Copy `eval/results/lexical.json` to `eval/baseline.json` (overwrite; recall@10 0.85).
- [x] **9.2** Copy `eval/results/pack_lexical.json` to `eval/baseline_pack.json` (pack_recall 0.9).
- [x] **9.3** In `ai_db/eval/harness.py` add
      `compare_pack_to_baseline(result, baseline, tolerance=0.02) -> str | None`
      comparing `pack_recall`; raise `ValueError` if the key is missing from `baseline`.
- [x] **9.4** In `ai_db/cli.py` `_run_eval`: remove the error "`--baseline` ... cannot be
      combined with --pack"; when `--pack`, use `compare_pack_to_baseline`.
- [x] **9.5** Add a unit test for `compare_pack_to_baseline` in `tests/test_eval_metrics.py`.
- [ ] **9.6** Create `.github/workflows/ci.yml`. The workflow exists but **will not pass**:
      it writes `"version": 1` (rejected — `CONFIG_VERSION = 2`) and then runs
      `ai-db --config ci.json init` over an existing file (needs `--force`). The polyglot
      command from 10.8 is also absent. Fix the config to v2, add `--force`, append the
      polyglot eval once 10.8 exists.
- [ ] **Check:** every command above passes locally. Commit `ci: ...`.

---

## Phase 10 — Tree-sitter symbols, cross-refs and syntax errors for all languages  (depends on: 9)

Goal: `investigate` gets callers/callees/tests for TS/JS/Go/Rust/C/C++/Java, not only Python.

> **Reality check:** this phase is done for **python, javascript, typescript, tsx, c**.
> The `go.scm` / `rust.scm` / `cpp.scm` / `java.scm` queries still raise `QueryError`
> (wrong node types / field names), so those extensions were removed from
> `ts_graph.SUPPORTED_LANGUAGES` rather than left to crash the indexer. Multi-language
> `investigate` is therefore **not** live yet.

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
      — `package-data` done; **5/9 queries valid**. Remaining breakage to fix:
      go: `type_declaration` pattern never matches struct/interface names; rust:
      `enum_variant_list` / `ordered_field_declaration_list` bodies, `impl_item` trait
      vs type ordering; cpp: `base_class_clause` is a child, not a field; java:
      `extends` is not a field name on `interface_declaration`.
- [x] **10.2** Create `ai_db/parser/ts_graph.py` with
      `extract_graph(filepath, str) -> (symbols, refs, syntax_errors)`.
      — implemented, incl. symbol dedup, `@name`-capture fallback, and `base`/`impl`
      inherit fan-out.
- [x] **10.3** `ai_db/search/indexer.py` `parse_file`: for files with `language_for(path)`,
      use `extract_graph` for symbols, refs and syntax error. For markdown, symbols = one per
      `md` chunk (`symbol_type="section"`, name = chunk `name`). Other files: no symbols, no refs.
      Annotations stay as today.
- [ ] **10.4** `ai_db/parser/linters.py` is used only for extensions **without** a tree-sitter
      language (shell, yaml). On `subprocess.TimeoutExpired` return
      `(1, 1, "linter timeout")` instead of `None`.
      — `linters.py:100` still returns `None` and reports the file as clean.
- [x] **10.5** Delete `ai_db/parser/cross_refs.py` and `ai_db/parser/syntax.py`; re-implement
      `extract_file_outline` from `extract_graph` symbols. Import sites updated
      (`ai_db/__init__.py`, `ai_db/parser/__init__.py`, `vectordb.py`).
      — `vectordb.py` re-exports a thin `extract_symbols` wrapper → `extract_graph(...)[0]`.
      Note: `ast_visitor._regex_outline` still exists but only for non-tree-sitter files.
- [x] **10.6** Bump `SCHEMA_VERSION` in `sqlite_backend.py`. — now `"8"`.
- [ ] **10.7** `tests/test_ts_graph.py`: per language, a class/struct, a method, a call,
      an import and an inheritance; assert the exact `(caller_name, callee_name, ref_type)`
      set and symbol names. — file exists with **17 passing tests covering 5 languages**;
      extend it as the other four queries are repaired.
- [ ] **10.8** Create `tests/fixtures/polyglot/` (~15 files) + `eval/golden/polyglot_pack.jsonl`
      (15 explain/impact queries). Run with `--all-files --root tests/fixtures/polyglot`.
      Save `eval/results/pack_polyglot.json` + `eval/baseline_pack_polyglot.json`; add to CI.
      — not started; blocked on 10.1.
- [ ] **Check:**
      - Python pack eval not below `eval/baseline_pack.json` (−0.02 tolerance).
      - Polyglot pack_recall ≥ 0.8. — **cannot run, no fixture**
      - `ai-db investigate "search_chunks"` (on this repo) lists no `unresolved` names that are
        defined in this repo.
      - Commit `feat(parser): tree-sitter symbols, cross-refs and syntax errors for all languages`.

---

## Phase 10b — Optional SCIP graph provider  (depends on: 10; GATED)

**Gate — do this phase only if, after Phase 10, either:**
- polyglot pack_recall < 0.8, or
- `unresolved` in explain packs (Python golden set) averages > 5 names defined in the repo.
Otherwise mark every box "skipped (gate not met: <numbers>)" and move on.

> **Deferred — gate not evaluable.** The gate depends on the polyglot eval from 10.8,
> which does not exist. Revisit only after 10.1/10.8 land and Phase 10.8 reports a number.

- [ ] **10b.1 Config** `graph: {"providers": {"<lang>": "tree_sitter" | "scip"}}`. Unknown
      language or value → `AiDbConfigError`.
- [ ] **10b.2 Indexer registry** `ai_db/parser/scip.py` with a fixed command per language;
      missing binary / non-zero exit / timeout → `AiDbConfigError`.
- [ ] **10b.3 Parsing:** vendor `scip_pb2.py`; `symbol_roles & Definition` → symbol else ref;
      add `symbol_refs.callee_symbol` and `chunks.scip_symbol`.
- [ ] **10b.4 Sync integration:** per-language SCIP index after chunking; debounce the
      watcher with `SCIP_WATCH_DEBOUNCE_S = 30`.
- [ ] **10b.5 Resolution:** in `investigate`, prefer exact SCIP symbol equality over name
      heuristics.
- [ ] **10b.6 Tests:** recorded `index.scip` fixture; `@pytest.mark.scip` runs real binaries
      only when installed; missing binary raises `AiDbConfigError`.
- [ ] **Check:** with `typescript` set to `scip`, polyglot pack_recall improves over the
      Phase 10 result. Record `eval/results/pack_polyglot_scip.json`. Commit.

---

## Phase 11 — Unified retrieval, skills, doc weight, config v2  (depends on: 9)

- [x] **11.1 Config v2**
      - [x] `CONFIG_VERSION = 2`; `retrieval.doc_weight`, `rerank.top_n`, `index.ignore`
            all parsed, validated and exposed on `AppConfig`.
      - [x] `index` added to `TOP_LEVEL_KEYS`; `tests/test_config.py` and the conftest
            fixture updated (session guard string bumped v1 → v2).
      - [x] `migrate_v1(data)` (1 → 2) added, plus a `migrate(data)` dispatcher that
            routes unversioned / v1 / current. `_cmd_init --migrate` now calls the
            dispatcher, and a v1 file is rejected at load with the
            "run: ai-db init --migrate" hint. Shared v2 defaults live in `config.py`
            (`DEFAULT_DOC_WEIGHT`, `DEFAULT_RERANK_TOP_N`, `DEFAULT_INDEX_IGNORE`) so
            the parser fallbacks and the template cannot drift.
- [x] **11.2 Ignore globs:** `Indexer.scan_directory` and `sync_paths` skip paths matching
      `config.ignore_globs` in addition to `.aidbignore`; globs passed in from
      `VectorDB.__init__` via `set_ignore_patterns`.
- [x] **11.3 Doc weight:** `Ranker.rank` scales the fused score of `md`/`txt` candidates by
      `doc_weight`; taken from config.
- [x] **11.4 `rerank.top_n`:** `Ranker.__init__` takes `rerank_top_n` and slices
      `head`/`tail` with it; `constants.RERANK_TOP` is gone from `ranking.py`.
- [x] **11.5 locate:** `AnalyzerEngine.locate_targets` calls
      `query_engine.search(q, filters, k)` with
      `{"allowed_projects": None, "path_prefix": abs(scope)}`; the direct
      `search_chunks` call is gone and `VectorDB` injects `query_engine`.
      `path_prefix` is now an accepted kwarg (its absence was breaking every MCP/HTTP
      `locate` call with a 500). A direct-backend path remains only as the
      no-retriever fallback for standalone `AnalyzerEngine` use.
- [x] **11.6 Context + skill FTS.** `search_skills` / `search_contexts` build their MATCH
      expression with `build_fts(expand_terms(q), base_terms(q))`, exactly like
      `search_chunks` — so they get AND/NEAR grouping, prefix matching and identifier
      splitting. The two placeholder `build_fts(...)  # for side effects` calls in the
      hybrid helpers are gone.
- [x] **11.7 Hybrid for skills/contexts.** `skill_vectors(name, project, embedding BLOB)`
      and `context_vectors(context_id → contexts(id) ON DELETE CASCADE, embedding BLOB)`
      as specified. Skills embed `description + triggers` at `sync_skills`, contexts
      `title + summary` at `save_context`; the embedder is handed to `SkillRouter` /
      `ContextMemory` by `VectorDB` and is `None` in lexical mode. Search is BM25 + vector
      fused with RRF. Both tables are in `DERIVED_TABLES` and dropped on a schema bump
      (`SCHEMA_VERSION` 8 → 9). Vectors are stored as float32 via stdlib `struct` with
      brute-force cosine — the previous vec0 tables hardcoded `FLOAT[768]` and were never
      written to. `save_context` now returns the row id so the caller can key the vector.
      Three bugs fixed here: nothing ever inserted; the RRF score was discarded and
      `abs(bm25())` returned instead; results were keyed by (nullable, non-unique) `title`.
- [x] **11.8 Skill router** (`route_skills`): fixed boosts, regex intent rules and stop-word
      special cases removed. Score = normalized BM25 + `SKILL_W_TRIGGER` on an exact
      trigger hit; min-max confidence; `min_confidence` filter and `reasons` kept.
      (A duplicated dead block — the whole `all_skills` rebuild — was removed in `0dc9923`.)
- [x] **11.9 Skills eval.** `harness.run_skills` + `load_skills_golden`, and
      `ai-db eval --skills --golden ... [--min-confidence]` reporting top-1 accuracy.
      `eval/golden/skills.jsonl` has 20 prompts over a 5-skill fixture
      (`tests/fixtures/skills/`). Not in CI, as specified.
      `SKILL_W_TRIGGER` left at 8.0 — top-1 is already 1.0 on the fixture, so there is no
      evidence to justify moving it.
- [x] **Check.** skills top-1 **1.0** (≥ 0.8 target); pack eval **0.95** vs 0.90 baseline
      (+0.05); code eval **0.80** — unchanged by this work, and the −0.05 vs the 0.85
      baseline is entirely the 3 stale golden entries (see Phase 10). 412 tests pass,
      ruff and mypy clean.

---

## Phase 12 — Remove remaining fallbacks  (depends on: 10)

- [x] **12.1** `AnalyzerEngine.analyze_file` builds symbols from `ts_graph.extract_graph`
      joined with stored chunks; the Python-`ast` branch, the "Fallback to regex symbol
      extraction" branch and the `"AST unavailable"` note are gone. Files without a
      language return md/text chunks only. Public output shape preserved.
- [x] **12.2** `references.py` `_diff_spans`: `since in ("last", "db")` compares stored
      chunks only; any other `since` requires git and raises
      `AiDbConfigError(f"--since {since} needs the file to be in a git repository")`.
- [x] **12.3** Exception audit: `ruff`/`mypy` clean across `ai_db` + `mcp_server.py`; all
      handlers catch a specific type or document why the case is expected. Two
      `build_fts` "for side effects" lines and a self-assigning `rrf_k = rrf_k` were
      removed as dead code during this pass (see 11.6 for the real fix).
- [x] **Check:** tests green. Commit `fix: remove remaining fallbacks`. (landed in `0dc9923`)

---

## Phase 13 — Performance and scale  (depends on: 9; 13.5 after 11)

> **Status: branch `perf/scale-phase13`.** 13.1, 13.2, 13.3, 13.4 and 13.6 are
> done. 13.5 (rerank weight tuning) is not started. **The Check's 10× vec0 target
> was measured and is not reachable** — see the note on 13.3.

- [x] **13.1 GPU:** `ai_db/device.py` owns device resolution so the template, the
      sentence-transformers embedder and the cross-encoder reranker cannot disagree.
      `build_template` writes `"device": "cuda"` when a usable GPU is present else
      `"cpu"` (torch imported lazily; stdlib-only core unaffected). `st_provider` /
      `CrossEncoderReranker` raise `AiDbConfigError` on cuda-without-a-GPU.
      `ai-db config check` prints the resolved device and `torch.version.cuda`.
      README gained a GPU section (cu128 / cu118 / cpu wheels).
      **Verified on hardware** — RTX 3070 Laptop (sm_86, CUDA 12.8): template resolves
      to `cuda`, embedder reports `device=cuda` and encodes on `cuda:0`, and
      `config check` prints `device=cuda (cuda available, torch.version.cuda=12.8)`.
- [x] **13.2 vec0 option:** `storage.options.vector_index` ∈ `{"exact", "vec0"}`
      (default `exact`, validated in `from_options`). `vec0` creates
      `vec_chunks USING vec0(...)` with `project` as a partition key; `search_vectors`
      uses `WHERE embedding MATCH ? AND k = ?`; `path_prefix` / `modified_since`
      over-fetch `k*4`, filter via join, truncate to `k`. The search mode is recorded
      in `embed_meta` so switching exact↔vec0 forces a reindex. Chunk-delete hooks and
      `drop_vector_index` clear vector rows. `TestSQLiteConformance` is parameterised
      over both modes. Fixed a bug this surfaced: both paths touched `vec_chunks`
      unconditionally, which does not exist in exact mode.
- [x] **13.3 Benchmark** `tests/test_bench_vectors.py` (`@pytest.mark.bench`,
      `@pytest.mark.slow`), size via `AI_DB_BENCH_N` / `_DIM` / `_QUERIES`.
      Measured at spec size (100k × 1024, k=10, 50 queries) — see
      `eval/results/vec0_benchmark.json`:

      | | p50 | |
      |---|---|---|
      | exact | 313.66 ms | brute force, Python loop |
      | vec0 | 202.06 ms | |
      | speedup | **1.55×** | target was 10× — **not met** |
      | vec0 recall@10 | **1.000** | target ≥ 0.95 — met |

      **Why the 10× target is unreachable:** sqlite-vec **0.1.9 is the latest
      release** and its `vec0` KNN is *brute force, not ANN*. Measured directly:
      cost per candidate vector stays flat at ~1.6–1.9 µs from 5k to 100k rows, i.e.
      linear in N — an ANN index would show sub-linear scaling. The 1.55× is only
      what vec0's C loop gains over the pure-Python exact path. A real 10×+ needs
      hnswlib / FAISS / USearch, or a sqlite-vec release that ships ANN.
      The option is still worth keeping (C speed, partition pruning, recall 1.000,
      and a clean seam for a real ANN backend). The test asserts recall and that
      vec0 is not *slower*; it deliberately does not assert 10×.
- [x] **13.4 Incremental centrality:** `rebuild_symbol_centrality(projects=None)`
      recomputes every project, or only the named ones. Hook signature is now
      `hook(changed: bool, projects: set[str])`; the indexer passes the projects a
      sync actually touched and all three registered hooks were updated.
- [ ] **13.5 Rerank tuning:** tune `RANK_W_*` with the local cross-encoder via
      `ai-db eval`; save `eval/results/hybrid_rerank_tuned.json`; annotate
      `ai_db/embed/defaults.py` if it does not beat `hybrid_qwen3-0.6b.json`.
      Not started — needs long eval runs. Note a CUDA device is now available
      (RTX 3070, sm_86), which should make this far cheaper than the earlier
      CPU-only attempts.
- [x] **13.6 Write lock:** process-wide `threading.RLock` held for the whole outermost
      `transaction()` and around `_auto_commit`. This removes the WAL
      deferred-transaction upgrade race behind `database is locked` (a deferred txn
      that starts as a reader and later writes gets SQLITE_BUSY immediately,
      ignoring `busy_timeout`). RLock so nested savepoints re-enter; readers never
      take it, so query latency is unaffected. Tested with 8 concurrent
      writer/reader threads at a 50 ms `busy_timeout`.
- [ ] **Check:** tests green ✅ (451) · ruff ✅ · mypy ✅ · vec0 recall@10 ≥ 0.95 ✅
      (1.000) · vec0 p50 ≥ 10× faster ❌ (**1.55×, target unreachable — see 13.3**) ·
      GPU indexing < 1 min — not benchmarked. Commit `perf(storage): …`.

---

## Phase 14 — Agent experience  (depends on: 10)

- [ ] **14.1 MCP progress:** when `tools/call` carries `params._meta.progressToken`, pass a
      `progress(done, total, message)` callback into `dispatcher.execute(...)` that writes a
      `notifications/progress` frame to stdout immediately. `Indexer.sync` reports after
      parsing and per 50 files; `Investigator.investigate` reports per stage. Other
      transports pass no callback. Test with a fake stdout.
- [ ] **14.2 Diff mode:** the `diff` dispatcher tool and `references._diff_spans` exist, but
      `investigate(mode="diff", since=…)` with seeds from
      `git diff --unified=0 <since>` and `changes:` in the pack is **not** wired up. Finish
      the mode, add `since` to the CLI/dispatcher schema/HTTP, and add 3 diff queries to
      `eval/golden/ai_db_diff.jsonl` pinned to real commit hashes.
- [ ] **14.3 Cache isolation test:** two projects with identical files; `investigate` on each
      must not return the other's cached pack.
- [ ] **Check:** tests green. Commit.

---

## Phase 15 — Documentation  (depends on: all)

- [ ] **15.1** `ARCHITECTURE.md` §5–7: pipeline, one table per extension point, current
      directory tree (`git ls-files ai_db | sort`).
- [ ] **15.2** `README.md`: investigate modes incl. diff, config v2 keys
      (`retrieval.doc_weight`, `index.ignore`, `rerank.top_n`, `storage.options.vector_index`),
      GPU setup, skills eval.
- [ ] **15.3** Remove MySQL and "zero-dependency" claims from `PROJECT.md`,
      `ORIGINAL_REQUEST.md`, `TEST_INFRA.md` (keep the rest of their content).
      — *Note: `TEST_INFRA.md:49` still advertises a "zero-dependency core
      `requirements.txt`", so this is not yet satisfied.*
- [ ] **Check:** `grep -rni "mysql\|zero.dependenc" *.md` returns nothing outdated. Commit `docs: ...`.
- Note: trace docs are covered by **16.7** (done).

---

## Phase 16 — Call-flow tracing (execution order), Python first  (depends on: 10, 11)

- [x] **16.1 Call-site facts (Python).** `symbol_refs` gained `call_col`, `seq`,
      `await_kind` (`sync|await|spawn|callback|deferred`), `guard`, `receiver`; populated
      from the tree-sitter Python visitor. Schema bumped; `tests/fixtures/callflow/`
      asserts exact `seq`, `await_kind`, `guard`, `receiver`.
- [x] **16.2 One resolver.** `_resolve` / `_linked` live in `ai_db/analysis/resolve.py` and
      are shared by `investigate` and `trace`; unresolvable names are reported, never guessed.
- [x] **16.3 Trace engine** (`ai_db/analysis/trace.py`): entry by symbol / `file:line` / script,
      depth-first in `seq` order; `--depth`, `--max-nodes`, `--direction`, `--include-tests`;
      node carries qualified name, span, call line, `await_kind`, `guard`, recursion flag;
      readiness summary for `spawn`/`callback` edges; `TRACE_WAIT_PATTERNS` defaults plus
      per-project `trace.wait_patterns` / `trace.wait_patterns_extend` validated in config.
- [x] **16.4 Output.** `format_tree` (indented, `⏵ ⏸ ⇉ ↺ ⌁` markers, `[if …]`),
      `format_json`, `format_mermaid` (`sequenceDiagram`), `--with-code` with
      `TRACE_CONTEXT_LINES`, token budget respected.
- [ ] **16.5 Transports.** `ai-db trace` (CLI) and the dispatcher `trace` route are done.
      **Missing: the MCP tool `trace_flow` and the HTTP `POST /trace`** — neither
      `mcp_server.py` nor `ai_db/server/http_server.py` mentions `trace` at all.
      Also confirm `investigate --mode flow` picks the most entry-like seed.
- [x] **16.6 Eval.** `eval/baseline_flow.json` + `eval/flow_baseline.py` with order-accuracy
      (LCS) and edge-kind accuracy.
- [x] **16.7 Docs.** `README.md` shows the marker tree; `ARCHITECTURE.md:358` documents
      `TRACE_WAIT_PATTERNS` and the per-project override.
      — *Confirm the documented limits paragraph (dynamic dispatch / `getattr` / event buses
      appear only as `unresolved`/`deferred`; no runtime tracing) is actually present.*
- [x] **Check:** pytest, ruff, mypy green. Commit `feat(analysis): call-flow trace (python)`.

## Phase 16b — Call-flow tracing for other languages  (depends on: 16)

Not started. Blocked on Phase 10.1 for those languages.

- [ ] Same 16.1 facts for TS/JS, C#, Go, Rust (`spawn`: un-awaited Promise, `Task.Run`, `go`,
      `tokio::spawn`; `callback`: `.then(cb)`, …).
- [ ] Fixtures + flow-eval entries per language; then remove the non-Python guard.

**Out of scope:** runtime/instrumented tracing, data-flow analysis. SCIP (Phase 10b) may later
replace the 16.2 resolver.

---

## Quick "what is left" summary

| Block | State |
|---|---|
| Phase 9 | 9.6 only — CI writes a v1 config, so CI is currently red |
| Phase 10 | 5/9 languages; needs 10.1 (4 queries), 10.4, 10.7, 10.8 |
| Phase 10b | deferred, gate not evaluable until 10.8 exists |
| Phase 11 | **done** |
| Phase 12 | **done** |
| Phase 13 | 13.1/13.2/13.3/13.4/13.6 done on `perf/scale-phase13`; 13.5 open; the vec0 10× target is unreachable (see 13.3) |
| Phase 14 | untouched (14.2 partly scaffolded) |
| Phase 15 | untouched |
| Phase 16 | 16.5 only — add MCP `trace_flow` + HTTP `/trace` |
| Phase 16b | untouched |

Highest-value next steps, in order: **9.6** (unblocks CI, and the CI config is still v1),
**regenerate the 3 stale golden entries** in `eval/golden/ai_db.jsonl` (they reference
Phase-10-deleted code and are the sole reason the code eval reads 0.80 vs the 0.85
baseline), then **10.1** for Go/Rust/C++/Java (unblocks 10.8, 16b, and 10b's gate).

---

## Final hand-back to the user
Report: commits (hash + subject), eval table (code recall@10/MRR, pack recall python and
polyglot, skills top-1, vec0 benchmark, flow order-accuracy / edge-kind accuracy), anything not completed and why, and any change
to the user's local config/DB.
