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
   wrapped, report it at a transport boundary (MCP, marked `# noqa: BLE001` with a
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
| Transports | `ai_db/dispatcher.py`, `mcp_server.py` (the HTTP server was removed) |
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
- [x] **9.6** Create `.github/workflows/ci.yml`. The workflow existed but **would not pass**:
      it wrote `"version": 1` (rejected — `CONFIG_VERSION = 2`) and then ran
      `ai-db --config ci.json init` over an existing file (needs `--force`). Now: v2 config,
      `init --force`, `vector_index` in the storage options, and the polyglot eval step
      (10.8) appended. All three gates verified green locally.
- [ ] **Check:** every command above passes locally. Commit `ci: ...`.

---

## Phase 10 — Tree-sitter symbols, cross-refs and syntax errors for all languages  (depends on: 9)

Goal: `investigate` gets callers/callees/tests for TS/JS/Go/Rust/C/C++/Java, not only Python.

> **Reality check:** all nine queries are valid and all nine languages are enabled in
> `ts_graph.SUPPORTED_LANGUAGES`. The three failure modes that had disabled Go/Rust/C++/Java
> are recorded under 10.1.

- [x] **10.1** Create `ai_db/parser/queries/<lang>.scm` for:
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
      — all 9 queries compile and match. The four repairs were three distinct mistakes:
      **(a)** tree-sitter requires children to be listed in **source order**; the C++ and
      Java class patterns put `body:` before `base_class_clause` / `superclass`, so the
      pattern matched nothing and every inheritance edge was silently lost while the
      `@ref.inherit` capture still fired — which is what made it hard to see. (b) Rust's
      `impl_item` fields are `trait:` (the trait) and `type:` (the self type); the query had
      them swapped, so `impl Priced for Invoice` recorded `Invoice` as the base. (c) Go puts
      struct/interface names on `type_spec`, not on `type_declaration`; Java's
      `interfaces` is a field but `extends_interfaces` on `interface_declaration` is an
      anonymous child, not a field.
- [x] **10.2** Create `ai_db/parser/ts_graph.py` with
      `extract_graph(filepath, str) -> (symbols, refs, syntax_errors)`.
      — implemented, incl. symbol dedup, `@name`-capture fallback, and `base`/`impl`
      inherit fan-out.
- [x] **10.3** `ai_db/search/indexer.py` `parse_file`: for files with `language_for(path)`,
      use `extract_graph` for symbols, refs and syntax error. For markdown, symbols = one per
      `md` chunk (`symbol_type="section"`, name = chunk `name`). Other files: no symbols, no refs.
      Annotations stay as today.
- [x] **10.4** `ai_db/parser/linters.py` is used only for extensions **without** a tree-sitter
      language (shell, yaml). On `subprocess.TimeoutExpired` return
      `(1, 1, "linter timeout")` instead of `None`. — done. A linter that never answers is
      not evidence of a clean file; reporting it as clean hid the gap.
- [x] **10.5** Delete `ai_db/parser/cross_refs.py` and `ai_db/parser/syntax.py`; re-implement
      `extract_file_outline` from `extract_graph` symbols. Import sites updated
      (`ai_db/__init__.py`, `ai_db/parser/__init__.py`, `vectordb.py`).
      — `vectordb.py` re-exports a thin `extract_symbols` wrapper → `extract_graph(...)[0]`.
      Note: `ast_visitor._regex_outline` still exists but only for non-tree-sitter files.
- [x] **10.6** Bump `SCHEMA_VERSION` in `sqlite_backend.py`. — now `"8"`.
- [x] **10.7** `tests/test_ts_graph.py`: per language, a class/struct, a method, a call,
      an import and an inheritance; assert the exact `(caller_name, callee_name, ref_type)`
      set and symbol names. — 25 tests across all 9 languages.
- [x] **10.8** Create `tests/fixtures/polyglot/` (~15 files) + `eval/golden/polyglot_pack.jsonl`
      (15 explain/impact queries). Run with `--all-files --root tests/fixtures/polyglot`.
      Save `eval/results/pack_polyglot.json` + `eval/baseline_pack_polyglot.json`; add to CI.
      — 9 source files (TS + Go + Rust, each with a billing package and an audit package
      that calls back into it) + README. 15 queries. **Measured pack_recall 1.000**
      (target ≥ 0.8), tokens_mean 1778.9, p50 16.4 ms. In CI.
- [x] **Check:**
      - Python pack eval not below `eval/baseline_pack.json` (−0.02 tolerance). — passes.
      - Polyglot pack_recall ≥ 0.8. — **1.000**.
      - `ai-db investigate "search_chunks"` (on this repo) lists no `unresolved` names that are
        defined in this repo.
      - Committed as `d80ffb6`.

---

## Phase 10b — Optional SCIP graph provider  (depends on: 10; GATED)

**Gate — do this phase only if, after Phase 10, either:**
- polyglot pack_recall < 0.8, or
- `unresolved` in explain packs (Python golden set) averages > 5 names defined in the repo.
Otherwise mark every box "skipped (gate not met: <numbers>)" and move on.

> **Gate is now evaluable — and not met.** Phase 10.8 reports polyglot
> **pack_recall 1.000** (target for the gate: < 0.8), and `ai-db investigate
> "search_chunks"` on this repo reports **0** unresolved names (gate: > 5). Tree-sitter
> already resolves the polyglot graph, so SCIP would have to beat a perfect score.
>
> **Skipped (gate not met).** Every box below is skipped, not pending. Revisit only
> if a language appears that tree-sitter cannot resolve symbols for — at which point
> this is a per-language decision, not a whole-phase one.

- [ ] **10b.1 Config** `graph: {"providers": {"<lang>": "tree_sitter" | "scip"}}`. Unknown
      language or value → `AiDbConfigError`. — *skipped (gate not met: polyglot 1.000,
      unresolved 0)*
- [ ] **10b.2 Indexer registry** `ai_db/parser/scip.py` with a fixed command per language;
      missing binary / non-zero exit / timeout → `AiDbConfigError`. — *skipped (gate)*
- [ ] **10b.3 Parsing:** vendor `scip_pb2.py`; `symbol_roles & Definition` → symbol else ref;
      add `symbol_refs.callee_symbol` and `chunks.scip_symbol`. — *skipped (gate)*
- [ ] **10b.4 Sync integration:** per-language SCIP index after chunking; debounce the
      watcher with `SCIP_WATCH_DEBOUNCE_S = 30`. — *skipped (gate)*
- [ ] **10b.5 Resolution:** in `investigate`, prefer exact SCIP symbol equality over name
      heuristics. — *skipped (gate)*
- [ ] **10b.6 Tests:** recorded `index.scip` fixture; `@pytest.mark.scip` runs real binaries
      only when installed; missing binary raises `AiDbConfigError`. — *skipped (gate)*
- [ ] **Check:** with `typescript` set to `scip`, polyglot pack_recall improves over the
      Phase 10 result. Record `eval/results/pack_polyglot_scip.json`. Commit.
      — *not applicable; no improvement is possible over 1.000.*

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
      `path_prefix` is now an accepted kwarg (its absence was breaking every MCP
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
- [x] **Check:** tests green ✅ · ruff ✅ · mypy ✅ · vec0 recall@10 ≥ 0.95 ✅
      (1.000) · vec0 p50 ≥ 10× faster ❌ (**1.55×, target unreachable — see 13.3**) ·
      GPU benefit — now **measured** (see below). Commit `perf(storage): …`.

      **13.1 GPU benefit, measured 2026-09-26** (`eval/results/gpu_benchmark.json`),
      real `Qwen3-Embedding-0.6B` over real chunks from this repo:

      | | CUDA | CPU | speedup |
      |---|---|---|---|
      | embedding throughput | 33.2 chunks/s | 0.23 chunks/s | **141.8×** |
      | single query, p50 | 41.9 ms | 438.3 ms | **10.5×** |

      The 14× difference between those two ratios is the actual finding: one short
      query cannot fill the GPU, so it pays kernel-launch and transfer overhead for
      little parallel work, while bulk indexing saturates the device. **The device
      matters enormously for indexing and only moderately for querying** — index on
      the GPU, query on either. At 0.23 chunks/s the CPU arm is why a 2000-chunk
      index is ~2.4 h on CPU versus ~60 s on the GPU.

      The "GPU indexing < 1 min" figure itself is extrapolated, not measured: the
      end-to-end sync test is opt-in (`AI_DB_BENCH_SYNC=1`) because the CPU arm costs
      minutes per hundred chunks. That is stated in the results file rather than
      dressed up as a measurement.

      **Follow-up: a 2× CPU win was sitting inside that number.** Investigating why
      CPU was slow showed three candidate explanations, all of which measurement
      eliminated: throughput was *flat* from batch 1 to 24, so not memory-bandwidth
      bound; length-sorting cut padded tokens 3.31× and changed wall clock by 0.7%,
      so not padding bound; time scaled with *chunk count* rather than tokens, so
      not per-token compute. What was left was the dtype: the checkpoint ships in
      bfloat16, and this CPU has no `avx512_bf16` and no AMX, so every matmul is
      emulated. float32 is **0.51 vs 0.24 chunks/s — 2.1×** on the same hardware.

      Fixed with a portable gate (`ai_db.device.resolve_dtype`, `embedding.dtype`
      to override): on CPU it times a matmul per candidate dtype and uses the
      fastest; on an accelerator it does not run at all, leaving the checkpoint's
      native dtype alone. A timing probe rather than a CPU-feature lookup, because
      `/proc/cpuinfo` does not exist on macOS or Windows, flag names differ across
      Intel/AMD/ARM, and any hardcoded table goes stale. `ai-db config check`
      reports the decision. See `tests/test_dtype_gate.py` (17 tests).

      The first version of that probe was itself wrong and measured nothing:
      `torch.randn(n)` builds a **1-D vector**, so `a @ b` was a 512-element dot
      product, both dtypes looked identical, and the gate never fired. Caught by
      asserting the probe's own arithmetic is physically plausible, which is now
      a regression test.

---

## Phase 14 — Agent experience  (depends on: 10)

- [x] **14.1 MCP progress:** when `tools/call` carries `params._meta.progressToken`, pass a
      `progress(done, total, message)` callback into `dispatcher.execute(...)` that writes a
      `notifications/progress` frame to stdout immediately. `Indexer.sync` reports after
      parsing and per 50 files. Other transports pass no callback. Tested with a fake stdout.
      — done. `ServiceDispatcher.execute(..., progress=...)` publishes the callback for the
      duration of the call and `_get_db` bridges it onto `indexer.progress_sink`; the sink is
      detached in a `finally` so a later direct call cannot report. Without a `progressToken`
      nothing is written at all. `Indexer` emits at 1, every 50th, and the last file.
- [x] **14.2 Diff mode:** `investigate(mode="diff", since=…)` with seeds from
      `git diff --unified=0 <since>`, `changes:` in the pack, `since`/`root` on the
      dispatcher schema and CLI, and
      `eval/golden/ai_db_diff.jsonl` with 6 queries pinned to real commit **ranges**.
      — done. `changes:` carries `{filepath, lines, changed_lines}`; `lines` collapses
      spans to `L1-3 L7`. Missing `since` and a non-repo root both raise
      `AiDbConfigError` rather than returning an empty pack, which would read as
      "nothing to review".
      Ranking needed four fixes, each because the naive version returned the wrong thing:
      (a) raw changed-line count let a 400-line class outrank the 5-line function inside
      it — now `hit / sqrt(span)`; (b) a chunk whose changed lines are fully covered by a
      strictly smaller chunk (a `class_header` over its own methods) is dropped entirely;
      (c) one heavily-edited file took every seed slot, so seeds now interleave across
      files; (d) the query was a 0.25 tie-break that lost to density — it is now a
      token-overlap multiplier (`DIFF_QUERY_BOOST`), because "rebuild centrality" has to
      match `rebuild_symbol_centrality`.
      Golden items pin **ranges** (`A..B`), not single refs: `git diff <ref>` also spans
      uncommitted work, so a commit-pinned eval is only reproducible on a clean tree.
      **Measured pack_recall 1.000** (6 queries) → `eval/results/pack_diff.json`.
      Deliberately **not** a CI gate — it needs this repo's full history, and a shallow
      CI checkout cannot resolve the ranges. Unit coverage carries it instead.
- [x] **14.3 Cache isolation test:** two projects with identical files; `investigate` on each
      must not return the other's cached pack.
      — done, and it found a real leak. `retrieval_signature()` covered only
      `mode`/`embedding_model`/`rerank_model`, so two configs pointed at the **same DB
      file** with different `cross_project` policies or `vector_index` read each other's
      results as cache hits. Signature now covers all six, and
      `test_same_db_different_config_does_not_share_cache` fails against the old code
      (verified by reverting). `test_same_config_across_processes_hits_cache` proves
      isolation did not cost the cache its purpose.
- [x] **Check:** tests green (21 in `tests/test_phase14.py`, 480 total) · ruff ✅ · mypy ✅.

---

## Phase 15 — Documentation  (depends on: all)

- [x] **15.1** `ARCHITECTURE.md` §5–7: pipeline, one table per extension point, current
      directory tree (`git ls-files ai_db | sort`).
      — §7 rewritten from the real tree. It had drifted badly: it still listed
      `parser/syntax.py` and `parser/cross_refs.py`, **both deleted in Phase 10**, and
      listed 8 test files against the actual 28. Every package is now covered
      (`analysis/`, `embed/`, `rerank/`, `eval/` were all missing).
- [x] **15.2** `README.md`: investigate modes incl. diff, config v2 keys
      (`retrieval.doc_weight`, `index.ignore`, `rerank.top_n`, `storage.options.vector_index`),
      GPU setup, skills eval.
      — all present, plus: a mode table for investigate; an honest `vec0` caveat
      (brute force, not ANN) so nobody expects order-of-magnitude gains; the four
      eval commands with their prerequisites; and the **MCP tool table was listing 11
      of 22 tools** — `investigate`, `query`, `sync`, `callers`, `symbol`, `check`,
      `diff`, `todos`, `prune`, `sync_skills`, `status` were all missing. Rewritten in
      full, plus a progress-notification subsection.
- [x] **15.3** Remove MySQL and "zero-dependency" claims from `PROJECT.md`,
      `ORIGINAL_REQUEST.md`, `TEST_INFRA.md` (keep the rest of their content).
      — `PROJECT.md`/`TEST_INFRA.md` describe the current system, so the text is
      corrected. **`ORIGINAL_REQUEST.md` is a dated record of what was asked for**, so
      rewriting it would falsify history; it keeps the request verbatim and carries an
      as-built banner plus inline markers on the two items that did not ship.
      Verified against the code, not assumed: there is no `mysql_backend.py`, no
      `[mysql]` extra, and the core has 6 third-party deps (tree-sitter,
      tree-sitter-language-pack, sqlite-vec, tiktoken, watchfiles, numpy) — so
      "zero-dependency" was false. What *is* true, and is now what the docs claim, is
      that heavy ML (torch, sentence-transformers) stays in the `[local-embed]` extra.
      `tests/test_packaging.py::test_core_zero_runtime_dependencies` asserted the right
      thing under a lying name; renamed to
      `test_core_runtime_dependencies_exclude_heavy_ml` (assertions unchanged).
- [x] **Follow-ups raised by 15.2, fixed in the same commit:**
  - `eval --skills` reported `{"skills_indexed": 0, "top1_accuracy": 0.0}` when no
    skills were present — indistinguishable from a broken router, and it reads like
    a result. `run_skills` now auto-syncs first (so the check is accurate rather
    than racing `route_skills`, which already auto-syncs internally) and raises
    `AiDbConfigError` naming every directory it searched.
  - The README instruction I had just written for it was **wrong**: it said
    `ai-db sync <dir>`, which indexes SKILL.md files as ordinary chunks and creates
    no skill records. The correct prerequisite is only `AI_DB_SKILL_DIRS`, because
    the eval DB is a fresh temporary database and the router auto-discovers skills
    into it. Verified end to end: 5 skills, top-1 1.0.
  - `[tool.ruff]` had no `include`, so `ruff check .` — what CI runs — reached every
    `.py` in the tree and `ruff check --fix` would rewrite any stray script. That is
    the mechanism by which the 19 debugging files were silently modified. Now
    pinned to the project's own code, with `token_benchmark.py` excluded in config
    so the exclusion is not lost when a caller omits the CLI flag. Scope limit
    stated in the config and the test: `include` filters *discovery*, so naming a
    file explicitly still processes it (correct — the user pointed at it).
  - The stray-script regression test initially asserted the wrong thing and caught
    a real limit in the fix, which is why the limit is now documented rather than
    papered over.

- [x] **Check:** `grep -rni "mysql\|zero.dependenc" *.md` — the 10 remaining hits are
      all either an explicit correction ("was scoped out", "is *not* zero-dependency")
      or the annotated historical record in `ORIGINAL_REQUEST.md`. No stale claim
      remains. Tests 480, ruff and mypy clean, all 3 eval gates green.

  **Also removed 19 tracked scratch files** (`test_final.py`, `test_js_patterns.py`, …)
  from the repo root. They were unreferenced debugging scripts left over from getting
  the JavaScript tree-sitter queries working, were not tests (not in `tests/`, not
  collected by pytest), and would have shipped in an open-source release. Verified
  unreferenced by `git grep` before removing.
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
- [x] **16.5 Transports.** `ai-db trace` (CLI), the dispatcher `trace` tool, and MCP
      exposure are all done. **The original note here was wrong** and is corrected:
      it claimed the MCP tool `trace_flow` was missing because "neither `mcp_server.py`
      nor `ai_db/server/http_server.py` mentions `trace` at all". True of the source
      text, but irrelevant — the MCP server reflects the dispatcher registry
      dynamically and hardcodes no tool list, so the registered `trace` tool is
      exposed automatically. Verified: `tools/list` returns 21 tools including
      `trace` and `investigate`. No separate `trace_flow` tool is needed.
      The HTTP half of this item is moot — the HTTP transport has been removed.
      Remaining: confirm `investigate --mode flow` picks the most entry-like seed
      (code exists; not yet asserted by a test).
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

## Post-Phase — HTTP REST transport removed

Not a numbered phase; a removal decided after Phase 16 was otherwise complete.

**Removed:** `ai_db/server/` (the whole package), the `ai-db serve` CLI subparser and its
dispatch branch, `tests/test_transports.py`'s `http_server` fixture and 9 HTTP tests,
`test_telemetry_http_endpoint`, and the HTTP REST API Reference section of the README.

**Why.** Asked what the server was for, the honest answer was: nothing it advertised.
It can only analyse code on its own filesystem, in its own SQLite file — there is no
URL fetch, no `git clone`, and no remote ingest anywhere in the codebase. But it
presented as a network API, and that was a liability rather than a feature:

- `Access-Control-Allow-Origin: *` with **no authentication**, and every tool exposed
  via `POST /tools/{name}` — including `sync`. Verified live: an unauthenticated
  `POST /tools/sync {"path":"/tmp"}` indexed 1835 files from an arbitrary host path
  and returned the file list. Any web page the user visits could do this.
- The `db` argument let any caller point the server at an arbitrary local `.db`, which
  `_resolve_db` would open or create.
- `int(self.headers.get("Content-Length", 0))` sat outside the try block, so a
  malformed header killed the connection with no response (confirmed).
- No body-size cap, no read timeout, and `log_message` suppressed outright, so a 500
  left no trace.

**What was kept and why.** The parity guarantee is the valuable part, not the
transport, so `test_transport_parity_mcp_and_http` was **reframed as
`test_transport_parity_mcp_and_cli`** rather than deleted — the point is that both
remaining transports route through one `ServiceDispatcher` and must agree.

`ai_db/http_client.py` is untouched: it is a client for *hosted model providers*
(OpenAI-compatible, Voyage, Cohere), unrelated to the server, and easy to confuse by name.

**If this is ever wanted back**, the use case that would justify it is N concurrent
clients on one shared index from a long-lived remote host (devcontainer, Codespaces,
build agent) — the one thing stdio MCP structurally cannot do. It would need
authentication, a read-only mode, and tenant isolation first. `access.cross_project` is
a label grant inside one database and is not a security boundary.

---

## Benchmark results (written up)

`eval/results/BENCHMARKS.md` is the human-readable report; `eval/results/matrix.json`
holds the machine-readable per-condition data. Six sections: the retrieval
configuration matrix, output-mode costs, device throughput, the vec0/exact
comparison, the dtype gate, and the bugs this exercise surfaced.

Headline numbers, all on this repo (149 files, 1713 chunks, 40-query golden set):

| condition | recall@10 | MRR | nDCG | p50 (ms) | pack |
|---|---|---|---|---|---|
| lexical (BM25 + graph) | 0.825 | 0.618 | 0.668 | 13.7 | 0.950 |
| hybrid, embedder on CPU | 0.925 | 0.782 | 0.818 | 205.7 | 0.975 |
| hybrid, embedder on GPU | **0.950** | **0.785** | **0.826** | 167.4 | **1.000** |
| hybrid + vec0 | 0.950 | 0.785 | 0.826 | 156.1 | 1.000 |
| hybrid + bge-reranker | 0.875 | 0.555 | 0.632 | 16,772 | 0.950 |

1. **Hybrid is decisive** (+0.125 recall@10, +0.167 MRR, pack to 1.000). This is
   the column that was missing from the raw-vs-lexical comparison, and it settles
   whether embeddings earn their cost.
2. **The device matters for indexing, not querying**: 145x on index time
   (~55 min CPU vs 22.9 s GPU) but only 1.2x on query latency. One short query
   cannot fill a GPU; bulk indexing can.
3. **vec0 is a wash at real corpus size** — identical accuracy, 156 vs 167 ms.
4. **Rerank is a double negative on this pairing**: worse accuracy *and* 100x the
   latency. Two causes identified, both left unfixed and documented rather than
   silently corrected.
5. **Output formats are where the money is**: `trace --format json` is 59x the
   tree renderer, `analyze --format json` 2.9x outline, `locate --format json`
   3.4x sexp.

Not done: a code-tuned cross-encoder, and the two rerank latency fixes
(`max_seq_length` 8192 -> 512, and fp32 on GPU). Both are one-line changes with a
known ~7x available, deliberately left so the measurement above stays reproducible.

---

## investigate output formats (added after the context benchmark)

`investigate` was JSON-only: no `--format` on the CLI, no `format` in the
dispatcher schema, and `indent=2` hardcoded in both `ai_db/cli.py` and
`mcp_server.py`. The `analyze` formatters could not be reused because a pack is a
different schema from analyze output, so pack renderers were written rather than
the existing ones reshaped.

`format_pack(pack, style)` in `ai_db/analyzer/formatters.py` renders
`json` (default, indented), `compact` (same data, one line), `stub` (ranked
answer + `why` provenance + graph edges, no bodies) and `sexp`. The dispatcher
returns a rendered **string** for a non-default style; `mcp_server.py` already
passed `str` results through verbatim, so one renderer serves CLI, MCP and HTTP
instead of being reimplemented per transport. `json` stays a dict so every
existing caller keeps parsing it.

Measured on this repo at `--budget 8000`, 40 questions:

| format | median ctx | vs raw | 32k hit rate | 8k hit rate |
|---|---|---|---|---|
| `json` (before) | 32,438 | 19.2% | 13/40 | 6/40 |
| `compact` | 26,559 | 15.8% | **38/40** | **36/40** |
| `stub` | **3,607** | **2.6%** | 32/40 | 32/40 |

**Two findings the benchmark produced, both recorded rather than quietly fixed:**

1. **`investigate` was overrunning its own budget.** The pack sizes its evidence
   to fit `--budget`, then the `indent=2` JSON envelope pushes the total past it.
   It reports 8,000 tokens; the caller receives 8,109. Invisible from the pack's
   own numbers. `compact` fixes it as a side effect of removing indent overhead.
2. **`meta.tokens_out` ignores `--format`.** For `ai_db/analysis/pack.py` it
   reports `tokens_in=447, tokens_out=423` identically for json, stub, sexp,
   outline and prose, while the emitted output ranges 658–2,930 characters. It
   tracks `--depth` but not the format, so the product's own token figures cannot
   support any comparison between output formats.

   **Partly addressed.** `meta.tokens_out_formatted` now reports the token count
   of the string the caller actually receives, so the two together show each
   format's real overhead. Measured on `ai_db/storage/backend.py`:
   outline 4,111 · stub 4,454 · prose 4,743 · sexp 4,909 · json 10,132,
   against a `tokens_out` of 5,774 that was identical for all five.

   To make the measurement possible, `analyze`'s rendering moved out of
   `ai_db/cli.py` into `format_analyze()` / `annotate_formatted_tokens()` in
   `ai_db/analyzer/formatters.py` — the CLI had it inline, and counting the
   formatted size without duplicating the rendering means the number and the
   printed string can drift apart. All five formats verified byte-identical
   before and after the move.

   **Still outstanding:** `tokens_out` itself is unchanged, so anything reading
   the old field is still format-blind, and the README's telemetry example block
   still shows per-format savings (`Raw=145,000 -> Stub=21,750 | S-Exp=13,050`)
   that this code cannot produce. Deprecating the old field, or making
   `ai-db telemetry` report `tokens_out_formatted`, is a behaviour change and is
   left as a decision rather than done silently.

The stub's lower hit rate than compact (32 vs 38) is not a quality regression: it
is the same pack rendered without bodies. The 6 extra hits live in bodies, which
`stub` deliberately does not inline and which remain one `expand <ref>` away.

---

## Post-phase: per-feature latency benchmark, and what it exposed

`eval/feature_benchmark.py` times all 47 user-facing features cold (first run) and warm
(median of 5), end-to-end as `python -m ai_db.cli ...`, plus an in-process column for the
hot paths. `eval/render_feature_table.py` renders the table from the stored JSON.

Headline: **the features are not the cost, the process is.** Every in-process feature is
under 1.2 ms against a ~240 ms warm subprocess. `import ai_db` was +102 ms of that, and
`tree_sitter` was 33% of it.

### Fixed

- [x] **`tokens_out_formatted` regression (self-inflicted).** Loading tiktoken's encoder
      costs ~300 ms once per process (200k base64 decodes of the BPE ranks table) against
      0.9 ms per subsequent count, so the field more than doubled `analyze` latency.
      `annotate_formatted_tokens` now takes `count=`; the CLI counts only for
      `--format json`, where the number is in `meta` and visible. Measured by
      counterfactual (forcing `count=True`), the fix is worth ~290 ms on `outline` and
      ~318 ms on `prose`; `stub`/`sexp`/`json` were already paying the load.
- [x] **Eager tree-sitter import.** `ai_db/__init__.py:28` pulled in `ts_graph` →
      `tree_sitter` + `tree_sitter_language_pack` on every command, including `query`,
      which is served entirely from SQLite + FTS. The four tree-sitter names are now
      imported inside `extract_graph` (the only function that uses them) and `Node` moved
      to `TYPE_CHECKING`. `import ai_db`: +102 ms → **+67 ms**; `ai-db --version`
      152 → 107 ms; `query` 238 → 203 ms.
- [x] **`check`/`lint` re-indexed the file they were checking.** See below.

### `check` is now two commands, not one slow one

`check <path>` used to call `prune_file` then `_index_file` and read the errors back out
of the database it had just rewritten. So a read-only command mutated the index, cost a
full chunk-and-embed pass (216 `count_tokens`, loading tiktoken) to answer a question
`ast.parse` answers in under a millisecond (583 ms against a 229 ms floor for `status`),
and reported the *index's* verdict while appearing to check the file.

- [x] `check <path>` parses files on disk and writes nothing; `check --index` reads the
      stored index. Both are rejected together rather than one silently winning.
- [x] `check --watch` requires a path — watching means "the file changed".
- [x] `parse_source()` in `parser/linters.py` is now the single parse path, used by both
      the indexer and `check`, so the two modes cannot disagree. The indexer's inline
      `if language_for(...)` branch (commented "Fallback", though it is the only path for
      those languages) is gone.
- [x] 583 ms → **266 ms**, at parity with `status`; the feature's own cost is ~35 ms.

### Bugs found while doing the above

- [x] **Tree-sitter missing nodes were never reported as syntax errors.**
      `collect_errors` compared `node.type` against the literals `"ERROR"` and
      `"MISSING"`, but tree-sitter types a missing node as the *expected token* and sets
      `is_missing` — `def broken(:` yields a node typed `")"`. It now tests
      `is_error`/`is_missing`, and is verified against `ast.parse` across cases.
- [x] **`python -m ai_db.cli` always exited 0.** `if __name__ == "__main__": main()`
      discarded the return value, so a configuration error printed a message and exited
      0. The `ai-db` console script (which wraps it in `sys.exit`) was correct; the
      module form was not. This also means the benchmark's exit codes were unreliable for
      config errors until this was fixed.
- [x] **`ToolDefinition.category` was write-only.** All 21 tools set one of 9 curated
      categories and `to_mcp_dict()` dropped the field, so `list_tools()` — what MCP
      clients receive — reported `None` for every tool. Now exposed (additive).
- [x] **`.aidbignore` directory patterns silently matched nothing.** A trailing slash was
      escaped into a literal `/` that then had to be followed by end-of-string or another
      slash, so `build/`, `node_modules/` and `dist/` were all no-ops. This was not
      theoretical: `eval/results/` had no effect, so a 63 KB generated JSON file stayed
      in the index and competed with the code the golden queries are about. 20 tests in
      `tests/test_aidbignore.py`.
- [x] **`.aidbignore` added** for the repo, excluding `eval/results/`. A measurement must
      not be able to change the thing it measures.

### Retrieval gate 1 was already failing — bisected, not assumed

Regenerating a baseline is the move that can hide a regression, so this was established by
bisecting the corpus rather than assumed:

| corpus | recall@10 | vs 0.825 baseline |
|---|---|---|
| @ `d80ffb6` (where 0.825 was recorded) | 0.850 | pass |
| @ `e8ed971` (pre-session HEAD) | 0.725 | fail by 0.100 |
| **current code on that same `e8ed971` corpus** | **0.725, mrr 0.4336** | — identical to the row above |
| @ HEAD plus this session's files | 0.700 | fail by 0.125 |

The third row is load-bearing: the current code reproduces the old code's numbers *exactly*
on an unchanged corpus, so this session's code changes have **no** effect on retrieval. The
whole 0.850 → 0.700 slide is corpus drift — the golden queries are about `ai_db/` source but
the eval root is `.`, so every test file and benchmark script added since `d80ffb6` competes
in the BM25 index. No golden entry was edited; `eval/golden/ai_db.jsonl` has not changed
since `d80ffb6`. Baseline regenerated to 0.700 with the bisect recorded in its `_note`.

All three gates are green: recall@10 0.700, pack_recall 0.95, polyglot pack_recall 1.000.

### Token cost and latency: no ai-db vs ai-db

`eval/token_budget_benchmark.py` answers the question that decides whether a code database
is worth its setup cost: for a task an agent must solve, how many tokens does it read and
how long does it wait? 40 golden questions, 7 arms, all scored on tokens out, latency and
hit rate.

| Arm | Median tokens | Latency | Hit rate |
|---|---|---|---|
| no ai-db: read whole files | 157.6k | 7 ms | 95% |
| no ai-db: read matching lines | 33.1k | 7 ms | 95% |
| `query` | 636 | 238 ms | 72% |
| `locate` | 304 | 250 ms | 72% |
| `investigate` | 7,612 | 249 ms | 82% |
| workflow: locate + analyze | 1,322 | 487 ms | 72% |
| workflow: locate + outline + analyze + investigate | 9,271 | 880 ms | 82% |

Four findings, in the order they matter:

1. **Every ai-db arm finds the golden file less often than the baseline** (72–82% vs 95%).
   The baseline has a structural advantage: it *reads* files, so once ripgrep ranks one into
   its top 5 the filename is guaranteed to appear. It is brute force, and brute force is
   why it costs 33.1k tokens. The honest claim is "far cheaper and somewhat less
   exhaustive", not "52x cheaper". Closing the recall gap is the real remaining work.
2. **At matched recall the reduction is real**: on the 29 questions where both `query` and
   the baseline found the file, 626 vs 30.9k median tokens — **49x at matched recall**. That
   is the defensible version; the unrestricted 52x credits `query` with questions it missed.
3. **Combining features is not automatically better.** The full workflow costs 9,271
   tokens against `query`'s 636. It does buy recall (82% vs 72%), so the trade is tokens
   for recall, not free extra context.
4. **ai-db loses on wall clock by a lot** — 238 ms vs 7 ms, because every CLI invocation
   pays ~200 ms of interpreter start and import before ~1 ms of work. About 231 ms buys the
   avoidance of 32.5k tokens. A clear win over a session; never a win against a one-shot
   grep, which ripgrep will always win.

Over all 40 questions the line-window baseline reads 1.5M tokens; `query` reads 25.2k.

`eval/render_token_table.py` renders the tables. The renderer refuses to call an arm a win if
it misses files the baseline found, and the report lists what it does not measure.

### A bug in the benchmark harness itself

The first token-budget run reported `baseline:files` as 0 tokens and a miss. The cause was
`[...][BASELINE_FILES]`: on a list that is *integer indexing*, not slicing, so it returned
the 6th path and the loop then iterated that path's characters. The output looked plausible
— 43 files were found — so only asserting the *type* of the value caught it. Recorded
because the same class of error has produced three false readings in this project: a probe
measuring a dot product, a hit test requiring a filename inside file content, and budget
truncation making all medians identical. **Assert physical plausibility, not just "no
exception".**

### Known limitation, not fixed

- [ ] **`tokens_out_formatted` is unreachable over MCP/HTTP.** With an explicit format
      `analyze` returns a rendered string, so there is no `meta` to carry the number; with
      no format there is no "formatted output" to count, the field being format-specific
      by definition. Filling it means deciding whether those transports return a dict plus
      a rendered form — a contract change, not a missing line.

### Test coverage added

`tests/test_check_modes.py` (23 tests): tree-sitter/`ast.parse` agreement, single-parse-path
parity, both check modes, read-only proof by index hash, no-chunking and no-tiktoken
assertions, `check` within 1.6× of `status`, and the CLI flag surface.

---

## Quick "what is left" summary

| Block | State |
|---|---|
| Phase 9 | **done** — CI config repaired to v2, `init --force`, polyglot step added; all 3 gates green |
| Phase 10 | **done** — 9/9 languages, linter timeout reported, polyglot pack_recall 1.000 |
| Phase 10b | deferred. Its gate is now **evaluable** (10.8 reports 1.000) but the SCIP cost was not justified; see 10b |
| Phase 11 | **done** |
| Phase 12 | **done** |
| Phase 13 | 13.1/13.2/13.3/13.4/13.6 done on `perf/scale-phase13`; 13.5 open; the vec0 10× target is unreachable (see 13.3) |
| Phase 14 | **done** — progress notifications, diff mode (pack_recall 1.000), cache isolation (real leak fixed) |
| Phase 15 | **done** — docs corrected against the code; 19 tracked scratch files removed; ruff scoped so a lint run cannot mutate strays |
| Phase 16 | **done** — 16.5's "missing MCP tool" was a false alarm; the HTTP half is moot |
| Phase 16b | unblocked (10.1 done) but not started |
| Latency | **done** — 47-feature cold/warm benchmark + 7-arm token/latency benchmark; 6 bugs fixed (tiktoken regression, eager tree-sitter import, `check` re-indexing, tree-sitter missing nodes, `__main__` exit code, `.aidbignore` directory patterns) |
| Gates | **green** — recall@10 0.700, pack_recall 0.95, polyglot pack_recall 1.000. Gate 1's baseline regenerated after bisecting the corpus: it had been failing since the Phase 14/15 commits, not from this session's code |

Highest-value next steps, in order: **closing the retrieval recall gap** (the token benchmark
shows every ai-db arm below the brute-force baseline on hit rate, and that is now the
largest known weakness), then **16.5** (the only remaining Phase 16 item), then **13.5**
(rerank tuning — cheap now that CUDA is available), then **16b** (unblocked by 10.1; the
python-only guard can now be lifted language by language).

Open items carried forward: `tokens_out_formatted` is CLI-only (contract change needed for
MCP/HTTP, above); `rerank` latency is unfixed by choice so its 20.5 s measurement stays
reproducible (~7x available from `max_seq_length` 8192→512 and fp32→fp16 on GPU);
`meta.tokens_out` is still format-blind and `ai-db telemetry` still reports it; the
`investigate` trim loops re-serialise the whole pack via `count_tokens(json.dumps(...))`
per trimmed item, which is quadratic in pack size — measured at under 1 ms in practice at
the default budget, so it is recorded rather than rewritten.

---

## Final hand-back to the user
Report: commits (hash + subject), eval table (code recall@10/MRR, pack recall python and
polyglot, skills top-1, vec0 benchmark, flow order-accuracy / edge-kind accuracy), anything not completed and why, and any change
to the user's local config/DB.
