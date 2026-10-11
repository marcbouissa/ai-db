# ai-db: agent-usability and architecture fixes

> **Source:** two audits.
> 1. 2026-10-03: hands-on agent audit (items A–C, all landed; see "Done").
> 2. 2026-10-07: full code review of every subsystem (transports, storage, parser/analysis,
>    scene/video) plus a hands-on CLI run on an isolated DB (items D–K).
>
> Each item lists what was observed, the verified root cause with `file:line`, the fix, and a
> **Check** that must pass before ticking the box. Line numbers are as of commit `faf67ff`
> and drift as you edit: **always `grep -n` the named function first**; never trust a raw
> line number.
>
> **Written for an implementing agent (Sonnet).** Each item is self-contained. Do them in
> order. If an item says **DECISION**, stop and ask the user before doing it.

---

## 0. Rules

Follow `TODO_NEXT.md` §0 (no fallbacks, one code path, specific exceptions, tunables in
`ai_db/constants.py`, every phase ends green). Gates, from the repo root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/mypy ai_db mcp_server.py
```

- One conventional commit per item (`fix(storage): ...`, `refactor(daemon): ...`). Only stage
  the files you changed; the working tree has unrelated showcase changes.
- Every bug fix gets a regression test that **fails before the fix**. Run it before fixing.
- Never run tests or experiments against the user's DB
  (`~/.local/share/ai-db/codebase_knowledge.db`) or `~/.ai-db/`. Use
  `AI_DB_PATH=$SCRATCH/fix.db AI_DB_CONFIG=$SCRATCH/cfg.json` (`ai-db init --embedding none`).
- Schema changes bump `SCHEMA_VERSION` in `ai_db/storage/sqlite_backend.py`.
- New tunables go in `ai_db/constants.py` with a one-line comment. No magic numbers.
- Do not swallow exceptions. `except Exception` is only allowed at a transport boundary
  (MCP/daemon request handler, CLI `main`) and must carry `# noqa: BLE001 - <reason>`.

Priority: **P0 → P1 → P2**. Inside a priority, top to bottom.

---

## Done (audit 1, verified 2026-10-07)

| Item | What | Landed in |
|---|---|---|
| A1 | Test isolation guard, `context forget`, recall shows age + missing files | `dbeaa39`, `bb6d78b`, `c2d77c6`, `tests/conftest.py` |
| A2 | `callers`: call-site line/receiver, `Class.method`, truncation reported | `3f5a4de`, `dd62b45` |
| A3 | One symbol per Python def, exact/prefix label | `12c9542` (2513 symbols, 2513 distinct) |
| A4 | Stale files re-synced or flagged; diff defaults to HEAD, includes untracked | `df250cf`, `1efaea1` |
| B1 | Exact lookups do not load models (`symbol` 0.3 s) | lazy providers |
| B2 | Quiet stderr (`ai_db/quiet.py`) | `6f9724d` |
| B3 | Session `default_format`, MCP investigate defaults to stub | `6f9724d`, `4e623dc` |
| B4 | Outline: scope-only, indented, deduped | `e86ae7d` |
| C1 | `ai-db init-agent` | `6f9724d` |
| C2 | Eval suites for symbols and callers | `08d4be9` |
| C3 | Scene tools split into `_handle_*` functions | `ai_db/mcp_tools/scene_tools.py` |

---

## P0: Security, data loss, red CI

### D1. Live API keys in `benchmark.sh` (USER ACTION, still open)
- [ ] **Observed:** `benchmark.sh:3-4` hardcodes two OpenRouter keys. The file is kept out of git
  only by the local `.git/info/exclude`. It was never committed.
- **Fix:** the user rotates both keys. Then replace the literals with
  `: "${OPENROUTER_API_KEY_NAIVE:?set it}"` and add `benchmark.sh` to `.gitignore` (not only
  `.git/info/exclude`). **The agent must not print, copy or move the keys.**
- **Check:** `grep -c 'sk-or-' benchmark.sh` prints `0`.

### D2. Syncing one project can prune another project's files
- [x] **Observed:** `SQLiteBackend.get_files_by_prefix` runs
  `filepath LIKE '<prefix>/%'` with `_`/`%` unescaped; SQLite `LIKE` is also ASCII
  case-insensitive. `get_files_by_prefix('/src/a_b')` returns `/src/a-b/x.py` and
  `/src/A_B/x.py`. `Indexer` (`ai_db/search/indexer.py`, the `to_prune` computation) then
  prunes every returned file not found on disk under the root, deleting the sibling's index.
  `_append_filters` (`path_prefix` filter) has the same bug, so search filters leak too.
- **Fix:** add one helper `_path_prefix_clause(column, prefix) -> (sql, params)` that emits
  `(col = ? OR substr(col, 1, ?) = ?)` with `prefix.rstrip('/') + '/'` and its length. Use it
  in `get_files_by_prefix` and `_append_filters`. Grep the file for any other
  `LIKE ? ` built from a path and route it through the helper.
- **Check:** test in `tests/test_storage.py`: index `/src/a_b/x.py`, `/src/a-b/x.py`,
  `/src/A_B/x.py`; `get_files_by_prefix('/src/a_b')` returns only the first; the
  `path_prefix` search filter likewise.

### D3. CI is red: ruff reports 125 errors
- [x] **Observed:** `ruff check .` → 125 errors (44 BLE001/S110/S112 blind excepts, 23 I001,
  17 RUF012, 16 PIE790, ...), mostly `ai_db/daemon/*`, `ai_db/scene/storage/*`,
  `ai_db/ext/segmentation.py`, `ai_db/video/*`, `eval/showcase/*`.
- **Fix:**
  1. `ruff check . --fix` for the auto-fixable ones; review the diff.
  2. Blind excepts: narrow each to the real exception(s) (`OSError`, `sqlite3.Error`,
     `httpx.HTTPError`, `json.JSONDecodeError`, `ValueError`...). Keep a broad catch only at a
     transport boundary, with `# noqa: BLE001 - <reason>` and the error logged
     (`logger.exception`), never `pass`.
  3. RUF012: annotate mutable class attributes with `ClassVar[...]` (or make them tuples).
  4. If `eval/showcase/*` is a throwaway harness, add it to `[tool.ruff] extend-exclude` in
     `pyproject.toml` rather than fixing it (see K9).
- **Check:** `ruff check .` prints `All checks passed!`; pytest and mypy green.
- **Done:** `eval/showcase` excluded in `pyproject.toml`; new `AiDbDaemonError` for daemon
  management; scene JSON store now raises `AiDbStorageError` on a corrupt file instead of
  skipping it (a corrupt index used to be silently overwritten by `save_scene`).

---

## P0: Storage correctness (`ai_db/storage/sqlite_backend.py`)

### E1. A failed write outside `transaction()` is committed by the next unrelated write
- [x] **Observed:** Python 3.12 sqlite3 opens an implicit `BEGIN` on the first DML.
  `_insert_chunk_rows` and `_resolve_parents` can raise after rows are inserted; nothing rolls
  back, and the next `_auto_commit()` (e.g. from `set_state`) commits the partial rows.
  Reproduced with a bad `parent_index`. Also, the comment at `_WRITE_LOCK` claims writers are
  serialised, but `_auto_commit` holds the lock only for `COMMIT`; the statements run outside it.
- **Fix:** one code path: every public mutating method runs inside `with self.transaction():`
  (which takes `_WRITE_LOCK`, commits on success, rolls back on exception when it is the
  outermost level). Remove `_auto_commit` calls those methods made. Grep for
  `_auto_commit(` and convert each caller. Update the `_WRITE_LOCK` comment to say what is true.
- **Check:** test: monkeypatch `_resolve_parents` to raise during `replace_file_chunks`, assert
  the exception propagates, then call `set_state("k","v")` and assert a **fresh connection**
  sees zero chunks for that file and `conn.in_transaction` is False.

### E2. vec0: kept chunks keep their old project in `vec_chunks`
- [x] **Observed:** `replace_file_chunks` (diff path for unchanged chunk ids) updates
  `project`, `language`, `chunk_type` on `chunks` but not on the metadata copies in
  `vec_chunks` written once by `upsert_embeddings`. After moving a file's project A→B,
  `search_vectors(allowed_projects=['B'])` returns `[]` and `['A']` returns it (cross-project
  leak). `ai_db/search/retriever.py` does not re-check the project.
- **Fix:** in the same transaction, `UPDATE vec_chunks SET project=?, language=?, chunk_type=?
  WHERE chunk_id IN (...)` for the kept ids (only when the vec0 table exists). If vec0 metadata
  columns cannot be updated in place, delete + reinsert those rows with their stored vectors.
- **Check:** test with `vector_index="vec0"` (skip if sqlite-vec missing): reassign, assert
  results under B and none under A.

### E3. Kept chunks keep stale FTS text and stale `symbol_name`
- [x] **Observed:** same diff path updates `qualified_name` but not the `fts_index` row
  (`_index_chunks_fts`) nor `symbol_name` (derived in `_insert_chunk_rows`). After renaming class
  `Old`→`New`, unchanged methods are found only by "Old"; `find_chunks_by_symbol` reads
  `symbol_name`.
- **Fix:** for kept chunks whose `qualified_name` changed, recompute `symbol_name` with the same
  helper `_insert_chunk_rows` uses (extract it if inline) and delete + reinsert their FTS rows.
- **Check:** test: index `class Old: def foo`, re-index as `class New: def foo` with identical
  body; BM25 "New" finds `foo`, "Old" does not; `find_chunks_by_symbol(["New.foo"])` hits.

### E4. `reindex --embeddings` with a different dimension fails in vec0 mode
- [x] **Observed:** `drop_vector_index` deletes rows only; `_create_vec0_table` is
  `CREATE ... IF NOT EXISTS`, so the table keeps `float[old_dim]` → "Expected 4 dimensions but
  received 8".
- **Fix:** `drop_vector_index` runs `DROP TABLE IF EXISTS vec_chunks` (vec0 mode) and clears
  the stored dimension state.
- **Check:** test: embed with dim 4, drop, embed with dim 8, search works.

### E5. Small storage bugs (one commit)
- [x] `delete_skill(filepath, project, name=None)` deletes from `skills` but not `fts_skills` or
  `skill_vectors` → `search_skills` returns deleted names. Delete from all three.
- [ ] **(deferred to K7)** `cosine_distance` returns "worst" on a dimension mismatch instead of raising (§0.1).
  Raise `AiDbQueryError` with both dimensions. Deferred: `tests/test_phase11.py` pins the
  current behaviour and its only callers are the skill/context vector paths K7 may delete.
- [x] `get_weak_points` joins `files.id = symbols.file_id`; neither column exists, so it always
  crashes. Nothing calls it. Delete it and the `backend.py` stub that returns empty results;
  the working implementation was `ai_db/storage/state.py:get_telemetry_weak_points` (now `SqliteBackend.get_weak_points`, K3).
- [ ] **(partly done; rest deferred to K7)** `search_skills_hybrid` keys RRF by `name` only, but `skills` PK is `(name, project)`.
  Key by `(name, project)` (needs `search_skills`/`search_skills_vector` to return the project;
  only worth it if K7 keeps the vector path). `rrf_k` now defaults to `RRF_K` (done).
- **Check:** one test per bullet.

---

## P0: Dispatcher / CLI correctness (`ai_db/dispatcher.py`)

### F1. `prune` bypasses cache invalidation
- [x] **Observed:** `_handle_prune` calls `backend.delete_file` through a 4-way `hasattr` chain,
  skipping `bump_index_generation` and the centrality rebuild that `VectorDB.prune_file`
  (`ai_db/__init__.py`) performs. Cached `query`/`investigate` results keep returning deleted
  files.
- **Fix:** call `db.prune_file(path)` only. Delete the `hasattr` chain.
- **Check:** test: query (cached), prune the file, same query no longer returns it.

### F2. Single-file `sync` ignores `.aidbignore` and extension filters
- [x] **Observed:** `_handle_sync` for a file path does `prune_file` + `_index_file` and always
  reports `added: 1`; an ignored file gets indexed.
- **Fix:** route through `db.sync_paths(root, [file])` (the same path `Indexer.sync_paths`
  uses), and report its real counts. Remove the `db.conn.commit()` in that branch.
- **Check:** test: ignored file → `added: 0`, not in DB.

### F3. Skill `min_confidence` default always overrides the real threshold
- [x] **Observed:** dispatcher schema and `_handle_route_skills` default `min_confidence` to
  `0.0`; `cli.py` does too. That always overrides the `0.15` hardcoded in
  `ai_db/search/skills.py` (`route_skills`).
- **Fix:** add `SKILL_MIN_CONFIDENCE = 0.15` to constants; default `None` everywhere and let
  `route_skills` apply the constant when `None`. Move the other skills tunables (5.0, 2.5, 20,
  4000) to constants in the same commit.
- **Check:** test: a 0.05-confidence match is not returned by default.

### F4. "Unknown tool" hides real `KeyError`s
- [x] **Observed:** `mcp_server.py` and `ai_db/daemon/server.py` catch `KeyError` around
  `dispatcher.execute` and report "Unknown tool", so `args["x"]` bugs inside handlers are
  misreported.
- **Fix:** add `UnknownToolError(AiDbError)` in `ai_db/errors.py`; `ServiceDispatcher.execute`
  raises it for an unknown name; transports catch only that.
- **Check:** test: a handler that raises `KeyError` surfaces as an internal error, not
  "Unknown tool".

### F5. Query cache key ignores weights
- [x] **Observed:** `VectorDB.retrieval_signature` omits `index.doc_weight` and
  `rerank.top_n`, though its docstring promises isolation by weights.
- **Fix:** include them.
- **Check:** test: change `doc_weight`, same query is a cache miss.

### F6. Any MCP caller can open an arbitrary SQLite file
- [x] **Observed:** `ServiceDispatcher._get_db` honours an undocumented `args["db"]`.
- **Fix:** remove the `db` argument from tool calls (CLI `--db` sets the path at dispatcher
  construction instead). Update the CLI to construct the dispatcher with the override.
- **Check:** test: MCP call with `"db": "/tmp/x.db"` is rejected (unknown argument) or ignored
  by construction; CLI `--db` still works.
- **Done:** `_resolve_db` ignores tool arguments; the CLI already passed `--db` at construction.

### F7. `locate_targets` bypasses cross-project policy
- [x] **Observed:** `AnalyzerEngine.locate_targets` falls back to backend search with
  `allowed_projects=None` when no query engine is given.
- **Fix:** remove the fallback; require the query engine; pass the resolved allowed projects.
- **Check:** test: a second project's file is never returned from `locate`.
- **Done, narrower than planned:** the fallback is gone and `locate` accepts `project` /
  `allow_project` (applied with `access.cross_project`). Without `project`, results stay limited
  by the `scope` path only: deriving the project from the directory name broke every caller
  that synced under an explicit `--project` name different from the directory.

---

## P1: Call graph, trace, analysis

### G1. Call graph is empty outside Python
- [x] **Observed:** in `ai_db/parser/ts_graph.py` the caller walk recognises only
  `function_definition`, `function_declaration`, `function_item`, `method_declaration`,
  `async_item`. JS/TS methods, arrow functions and Java constructors get caller `module`;
  C/C++ `function_definition` has no `name` field → `module.module`; Go methods become
  `module.m` but the chunk is `S.m`; Rust impl methods become `module.fmt` vs chunk
  `Display for Foo.fmt`. `investigate` (`_add_callees`) and `trace` look up refs by
  `scope_of(chunk.qualified_name)`, so they find no callees.
- **Fix:** one function computes a scope's qualified name, used by **both** the chunker
  (`ai_db/parser/chunker.py`) and the caller walk, so names agree by construction. Cover:
  JS/TS `method_definition`, arrow functions bound to a variable, Java
  `constructor_declaration`, C/C++ declarator-chain name extraction, Go receiver type,
  Rust `impl` type.
- **Check:** `tests/test_ts_graph.py`: per language, a method calling a function yields a ref
  whose caller equals the method chunk's `qualified_name`; `trace` on the polyglot fixture
  returns ≥1 callee for each language.
- **Done 2026-10-11:** `ai_db/parser/naming.py` (`SCOPE_CONTAINERS`, `definition_name`,
  `qualified_name`) names both chunks and caller scopes; JS/TS methods and bound arrow
  functions are callers, C/C++ names come from the declarator chain, Go methods are
  `Recv.m`, Rust impl methods are `Type.m` (trait/generics stripped; was `Trait for Type`).
  Python refs on the repo are byte-identical. Tests in `tests/test_ts_graph.py`.

### G12. Run-through 2026-10-08: fixes and performance (done)
- [x] `trace`: dotted entries (`Class.method`) resolve; `file:line` takes the innermost
  chunk (was the class header) and snaps from a blank/decorator line; default scope is the
  caller's project (was `global` only, so nothing in a repo was found); a call on an
  unknown receiver no longer links to an unrelated class's same-named method.
- [x] `symbol_refs` now stores `seq`/`await_kind`/`guard`/`receiver` (schema 12); trace
  shows real call lines in source order with await/guard markers. `seq` is per file
  (was a never-reset global: non-deterministic parses).
- [x] `diff --since last` compares hashes, then chunk content hashes (it rebuilt the old
  text from compacted chunks and reported changes in unchanged files).
- [x] `todos` hides docstrings by default (196 docstrings vs 3 TODOs) and prints one line
  per hit; tag regex no longer matches `# todos` / `# note that`.
- [x] `investigate --mode diff` header names `changes since <ref>` instead of `''`.
- [x] Output filter: a single line larger than the budget is truncated, not dropped.
- [x] Performance: receiver resolution memoised per scope (cli.py 4.1 s -> 0.07 s to
  parse; identical refs on all 213 Python files); full sync 9.1 s -> 1.3 s; CLI start
  0.56 s -> 0.15 s import (daemon/fastapi, httpx and the scene plugin's numpy no longer
  imported by every command); every read command ~0.17 s (was 0.6-1.1 s).
- [x] Next perf step: K3 (lazy `ai_db/__init__`): done, CLI import 40 ms.

### G1b. C# and Apple Pkl grammars (done 2026-10-07)
- [x] `.cs` -> `csharp`, `.pkl` -> `pkl` (`ai_db/parser/languages.py`), queries in
  `ai_db/parser/queries/{csharp,pkl}.scm`, chunker `LangSpec`s. Callers of calls inside
  methods match chunk names (namespaces/classes qualify both). `ts_graph`'s four copies of
  the caller node list became `_CALLER_NODES`/`_QUALIFYING_NODES` (constructors now count
  as callers for Java too); `declared_name()` handles grammars without a `name` field
  (Pkl). `.cs`/`.pkl`/`.mjs`/`.cjs`/`.cc` were missing from `INDEXABLE_EXTENSIONS` (the
  last three silently never indexed); a test now keeps it a superset of `EXT_TO_LANG`.
  Tests: `tests/test_csharp_pkl.py`. Open: `callers Class.method` needs receiver->class
  resolution, which only Python has (`self.`); same gap as the other non-Python languages.

### G2. Trace ordering and async markers never apply
- [x] **Observed:** `SQLiteBackend.get_refs_from` selects only
  `id, caller_filepath, caller_name, caller_line, callee_name, ref_type, project`, never
  `seq, await_kind, guard, receiver, call_line, call_col`. So the sort in `trace.py` is a no-op,
  readiness is always empty, and `child.call_line` is the def line. `seq` also comes from a
  module-global counter in `ts_graph.py` that is never reset (re-parse → 4, 7, 8).
- **Fix:** select the columns and map them onto the ref model; sort by `(call_line, call_col)`;
  delete the global counter (derive `seq` per caller from position).
- **Check:** test: `trace` of a function calling `a()`, `await b()`, `c()` prints them in source
  order with the await marker and real call lines; parsing the same file twice gives identical
  rows.

### G3. Readiness message inverted
- [x] **Observed:** `trace.py` `_generate_readiness`: an ancestor with `await_kind == "await"`
  prints "(not awaited)".
- **Fix:** invert the condition. **Check:** test asserting both branches.
- **Done 2026-10-11:** an awaited ancestor now reads "(awaited, file:line)"; both branches
  tested in `tests/test_trace.py`.

### G4. Trace entry by qualified name fails
- [x] **Observed:** `_find_entry` passes `"A.m"` to `find_chunks_by_symbol`, which matches
  `symbol_name` (last component) only → "Entry point not found". Flow mode's
  `_pick_entry_seed` prefers module chunks (`imports/globals`) that never resolve.
- **Fix:** match `qualified_name` suffix for dotted names (same rule `callers` uses for
  `Class.method`); `_pick_entry_seed` skips module chunks.
- **Check:** `ai-db trace ServiceDispatcher.execute` on this repo returns a tree.

### G5. Per-language tree-sitter query bugs (`ai_db/parser/queries/*.scm`)
- [x] Python: add `aliased_import` / `dotted_name as` patterns; restrict `@def.type` to
  annotated or `TypeAlias` assignments at module/class scope (not `app = Flask()`, not
  `x = field()`); capture `attribute` bases (`abc.ABC`).
- [x] C: typedef pattern must match `type_definition` declarators directly; handle
  `pointer_declarator` for function names.
- [x] C++: in-class `field_identifier` methods, destructors, `template_declaration` (and the
  chunker must not put templates into `imports/globals`).
- [x] JS: `lexical_declaration` → `variable_declarator` with `arrow_function` value.
- [x] TS: `abstract_class_declaration`.
- [x] Go: struct emitted as both `struct` and `type` (dedupe); `selector_expression` alone is
  a field read, only capture it as a call under `call_expression.function`.
- [x] Rust: `impl Trait for Type` → caller scope is `Type`; record `Trait` as an implemented
  interface, not a base; generic impls.
- [x] Java: enum `implements` child order; `extends Base<T>` (`generic_type`);
  `package_declaration` is not an import.
- [x] Chunker comment detection: also `line_comment`, `block_comment` (Rust/Java).
- **Check:** extend `tests/test_ts_graph.py` with one case per bullet. Polyglot eval
  `pack_recall` must not drop (`ai-db eval --suite polyglot` vs baseline).
- **Done 2026-10-11:** `test_g5_*` in `tests/test_ts_graph.py`; polyglot `pack_recall` 1.0
  (= baseline). Also: Python `from x import a` now emits one import ref per imported name
  (the old pattern skipped single-name imports and duplicated others: 657 refs gained on
  this repo, 0 lost), C macro calls are no longer counted twice, a Java enum's methods are
  chunked (`Color.n`), Java supertypes match one per list entry (a quantified list captured
  only the first). Lexical (0.8 vs 0.825) and pack (0.875 vs 0.9) evals were already below
  baseline at `386cc81`, before this item.

### G6. `analyze_file` bodies are one line
- [x] **Observed:** `ai_db/analyzer/engine.py` end-line lookup compares `chunk.name == sym_name`,
  but chunk names are `"def A.m"` / `"class A"` → every span `[l, l]`; `depth=full` returns the
  signature only. Text files return no symbols (checks `"text"/"section"`, chunker emits
  `"txt"`).
- **Fix:** match on `qualified_name` (or `symbol_name`) and use the chunker's language ids.
- **Check:** test: `depth=full` on a 5-line function returns all 5 lines.
- **Done 2026-10-11:** `analyze_file` takes its symbols from `file_symbols()` (the chunker,
  the index's own code path, run on the content on disk so spans are never stale):
  qualified names (`Store.save`), whole-definition spans (split parts merged), text/markdown
  sections included; the `extract_graph` + name-matching path is gone. `tests/test_analyze.py`.

### G7. Diff spans are wrong
- [ ] `ai_db/analyzer/references.py` `_diff_spans(since="db")` rebuilds old text from
  bloat-stripped chunk contents → unchanged files report added spans. Store/compare the file
  `sha256` first (unchanged → no spans); for changed files diff against git or the stored raw
  text, not chunks.
- [ ] A file new since `since` raises `AiDbConfigError` from `git show`; treat it as fully added.
- [ ] `investigate._git_changed_spans` ignores pure-deletion hunks (`+N,0`); seed the
  enclosing chunk at line N.
- **Fix also:** there must be **one** diff module (see K1); do G7 inside it if K1 lands first.
- **Check:** tests for unchanged file (no spans), new file (all added), deletion-only hunk
  (seed present).

### G8. Untracked new files are listed but not indexed
- [ ] **Observed (hands-on):** create `new.py` in a git repo; `ai-db symbol gamma` → not found;
  `inv --mode diff` lists `new.py L1` under changes but has no evidence for it. Also prints
  `investigate diff: ''`.
- **Fix:** before answering in diff mode, `sync_paths` the untracked, non-ignored files
  (bounded by `STALE_RESYNC_MAX_FILES`). Print `investigate diff: since HEAD` instead of `''`.
- **Check:** test in a temp git repo.

### G9. Ref ids collide
- [x] **Observed:** analysis ref ids are `sha1[:8]` (32 bits) with `INSERT OR REPLACE`; ~10k
  refs in the 7-day window → ~1% collision; `expand` then returns another body.
- **Fix:** 12 hex chars (48 bits) and `INSERT` that raises on conflict with a different body.
  Constant `REF_ID_HEX_LEN`.
- **Check:** test: forced collision raises instead of overwriting.
- **Done 2026-10-11:** `REF_ID_HEX_LEN = 12`; `store_analysis_ref` re-storing the same
  ref only refreshes its timestamp, a different file/span/body under the same id raises
  `AiDbStorageError`. Test in `tests/test_analyze.py`.

### G10. Eval inflates recall
- [ ] **Observed:** `ai_db/eval/metrics.py:_matches` uses `exp_symbol in name` (`run` matches
  `prune`). Diff golden items use `A..B` ranges against a working-tree index. Without
  `--all-files`, `materialize_tracked` copies to a non-git dir so diff mode raises.
- **Fix:** match the final dotted component exactly (or the full qualified name when the
  expectation is dotted); materialize as a git worktree (`git worktree add --detach`).
  Re-record baselines in the same commit and state old → new numbers in the message.
- **Check:** metrics unit test (`run` ≠ `prune`); `ai-db eval` runs green.

### G11. Receiver resolution is O(calls × scope)
- [x] `ai_db/parser/receiver_types.py:resolve_receiver_class` re-walks the class/module per
  call: 94% of `extract_graph` time on `sqlite_backend.py`. Build one assignment table per
  scope, then look up. **Check:** same refs as before on the fixtures; `extract_graph` on
  `sqlite_backend.py` < 0.1 s.

---

## P1: Daemon (`ai_db/daemon/`)

> **Decided: fix it (2026-10-07). All of H1–H7 are done.** `ai_db/daemon/server.py` was
> rewritten; tests in `tests/test_daemon_server.py` (14) plus a live run (real daemon,
> CLI from a subdirectory, auth on). `ai_db/daemon/config.py` deleted (one `DaemonConfig`
> in `ai_db/config.py`); `/templates` removed; daemon errors reach the CLI as
> `[ai-db] daemon error: ...` instead of a traceback.

### H1. Daemon resolves paths against `/`
- [x] `manager.py` daemonize does `os.chdir("/")`; the dispatcher uses `os.getcwd()`
  (`relative_to=`, `sync` default `"."`, `check` without path); `cli.py` sends `check`'s path
  unresolved. **Fix:** the client makes every path absolute and sends `cwd` explicitly; the
  dispatcher never calls `os.getcwd()` (grep it). **Check:** daemon test: `check foo.py` from a
  subdir works.

### H2. Progress cannot work; blocking calls stall the loop
- [x] `_meta` is a pydantic private attribute (`models.py`) → dropped. Use
  `meta: dict | None = Field(None, alias="_meta")` with `populate_by_name`.
- [x] `get_progress_callback` returns an `async def`, called synchronously by
  `report_progress`. Make it sync: `loop.call_soon_threadsafe(queue.put_nowait, event)`.
- [x] `call_tool` runs `dispatcher.execute` on the event loop. Use
  `await asyncio.to_thread(...)`.
- **Check:** test: a `sync` with a progress token yields ≥1 progress event, and `/health`
  answers while the sync runs.

### H3. Auth and rate limit broken
- [x] `HTTPException` raised in `@app.middleware` → 500. Return `JSONResponse(status_code=...)`.
- [x] `request.path_params` is empty in middleware → one global bucket. Parse the project from
  `request.url.path`.
- [x] `/daemon/shutdown`, `/daemon/reload`, `POST /projects` (takes arbitrary
  `db_path`/`config_path`), `/query/cross-project` bypass auth. Protect every route except
  `/health`.
- [x] Tunables (rate 100/60, TTL 300, intervals 60/300/2.0, client timeout 30) → constants.
- **Check:** tests: missing key → 401, bad key → 403, flood → 429 per project; admin routes 401.

### H4. `daemon --config` ignored
- [x] `cli.py` returns for `daemon` before setting `AI_DB_CONFIG`; `server.py` lifespan calls
  `load_config()` with no path. Set the env (or pass the path) before dispatch. **Check:** test.

### H5. Tool sets diverge
- [x] Scene/video tools are registered only in `mcp_server.py`; `ProjectPool.get_or_create`
  does not register them. One factory `build_dispatcher(...)` used by CLI, MCP and daemon.
  Dedupe the copied tool-list normalization and the single vs batch tool-call bodies in
  `server.py`. **Check:** `tools/list` is identical across MCP and daemon.

### H6. Locks and reload
- [x] `warm_project` takes the non-reentrant lock then calls `get_or_create`, which takes it
  again → deadlock for an unloaded project.
- [x] `remove()` deletes the lock it holds.
- [x] Two reload implementations (`_reload_config` vs `/daemon/reload`) differ; keep one.
- [x] Manager `register` writes `source_path`, `unregister` writes the default path.
- [x] `DaemonConfig`/`DaemonProjectConfig` defined in both `ai_db/config.py` and
  `ai_db/daemon/config.py`; keep one.
- **Check:** tests for warm of an unloaded project and register/unregister round-trip.

### H7. `DaemonClient` leaks
- [x] `_ensure_client` creates a new `AsyncClient` per call, never closed; `_client` never
  assigned; `arguments.pop("project")` mutates the caller's dict; listeners swallow errors.
  One long-lived client, copy args, narrow excepts. Daemon logs go to `/dev/null` after
  daemonizing: log to a file under the XDG state dir.

---

## P1: Scene / video

> **Moved:** the scene/video code now lives in `packages/ai-db-scene/ai_db_scene/` (K10).
> Read `ai_db/scene/X` below as `ai_db_scene/X`, `ai_db/video/` as `ai_db_scene/video/`,
> `ai_db/mcp_tools/` as `ai_db_scene/tools/`, `ai_db/ext/blender_client.py` as
> `ai_db_scene/blender_client.py`. Run its tests with `pytest packages/ai-db-scene/tests`.

### I1. `delete_scene` orphans rows
- [ ] `ai_db/scene/storage/sqlite_vec.py` never sets `PRAGMA foreign_keys=ON`, so
  `ON DELETE CASCADE` never fires (also on `INSERT OR REPLACE`). Enable it on connect.
  **Check:** test counts `scene_objects` and `prompt_contexts` rows after delete → 0.

### I2. Scene prompt search crashes on normal input
- [ ] `search_scenes_by_prompt` passes raw text to FTS5 `MATCH`; `desk-chair`, `desk:chair`,
  `AND`, `chair OR` raise `OperationalError`. Reuse the core FTS sanitizer (find it with
  `grep -n "def .*fts\|MATCH" ai_db/storage/sqlite_backend.py`). **Check:** parametrized test.

### I3. MCP startup imports torch (~1.1 s)
- [x] `ai_db/video/processors/base.py` loads entry points at import time;
  `check_dependencies` does `__import__("torch")`. `mcp_server.py` calls
  `load_video_processors()` again. **Fix:** no import-time loading; check deps with
  `importlib.util.find_spec`; load processors on first video tool call. **Check:**
  `python -X importtime -c "import mcp_server"` shows no `torch`.

### I4. Video dependency gate wrong both ways
- [x] `opencv-python-headless` → `opencv_python_headless` (should be `cv2`); the `cv2` branch is
  a no-op. Each processor module self-registers at import regardless. **Fix:** explicit
  `{dist: module}` map; registration only via the entry-point loader. **Check:** test with a
  missing module → processor not listed; present → listed.

### I5. Blender capture wired only once
- [ ] `scene_tools.get_scene_memory` caches the memory and attaches the Blender capture only if
  Blender answered on the first call. Resolve the client inside `snapshot_scene`. **Check:**
  test: memory created while Blender down, later snapshot works when a fake client is up.

### I6. Scene storage drift
- [ ] Default path `~/.ai-db/scene_memory.db` ignores XDG/`AI_DB_PATH`; derive from core's
  data dir (constants). Do **not** migrate the user's existing file automatically; print a
  one-line notice if the old file exists.
- [ ] `prompt_embeddings` vec0 table (384-dim, banned model name) is never written; the
  `has_vectors ... pass` branch is dead; `fts_query` unused. Delete all three.
- [ ] `BlenderClient._roundtrip`: monotonic overall deadline + max buffer size (constants).

---

## P2: Architecture decay

### K1. Merge `ai_db/analyzer/` into `ai_db/analysis/`
- [ ] Two diff engines (`references._diff_spans`, `investigate._git_changed_spans`), three
  "symbols of a file" paths (indexed `symbols`, `analyze_file` re-parse, `ast_visitor`), two
  token counters (`len//4` vs tiktoken). `investigate` calls the private
  `analyzer_engine._store_analysis_ref`; `analyzer/formatters.py` renders investigate packs.
- **Plan:** (1) `analysis/diff.py`: the single git-hunk diff engine; (2) `analysis/refs.py`:
  the ref store (public API); (3) `analyze_file` reads indexed chunks/symbols instead of
  re-parsing; (4) one `count_tokens` in `ai_db/tokens.py`; (5) move formatters to
  `analysis/formatters.py`; (6) delete `analyzer/`. One commit per step, tests green each time.
- Also remove: `Investigator._import_cache`, the "populate resolver cache" call in
  `_add_callees`, the `"methods"` branches, the unused `_BUILTINS` in `resolve.py`,
  `ast_visitor._regex_outline`, `ts_graph`'s `fallback_name_node` path, `getattr` defaults in
  `QueryEngine`, `hasattr` session-state paths.

### K2. Split `SQLiteBackend` (2306-line god class)
- [ ] Repositories sharing one connection + `transaction()`: `chunks` (+FTS), `vectors`,
  `graph` (symbols/refs/centrality), `memory` (contexts/skills), `cache` (query cache/log),
  `state`. `SQLiteBackend` composes them. Do it **after** E1 (single transaction path).
- [ ] Make `backend.py` a real contract: abstract methods only; delete the
  `NotImplementedError` "defaults" and aliases (`search_chunks_bm25`, `upsert_symbols`,
  `search_symbols`) after updating callers.
- [ ] Table counts exist three times (`backend.py`, `sqlite_backend.py`, `state.py`): keep one.
- [ ] Delete unused `module_summaries` table and the empty "Analysis References" header.

### K3. Slim `ai_db/__init__.py`
- [x] Move `VectorDB` to `ai_db/facade.py`; `__init__` exports lazily (`__getattr__`), so
  `import ai_db.constants` loads nothing else. Remove private passthroughs (`_index_file`,
  `_store_analysis_ref`, `_diff_spans`) and the `hasattr` session-state fallback.
  **Done 2026-10-08:** `VectorDB`/`run_watch` live in `ai_db/facade.py`; `ai_db`,
  `ai_db.parser`, `.search`, `.storage`, `.analyzer`, `.analysis` export lazily through
  `ai_db/_lazy.py` (with `TYPE_CHECKING` imports for mypy); the CLI imports the facade only
  in `eval`/`reindex`/`watch`. `import ai_db.constants` loads nothing else; `import
  ai_db.cli` 150 ms -> 40 ms (560 ms at the start of the 10-08 run); `ai-db --version`
  0.07 s. Tests: `tests/test_lazy_imports.py`.
  **Done 2026-10-09:** passthroughs removed; `ReferenceStore.store_ref`/`diff_spans` are
  public and called directly (engine, investigate, `diff_file`); session state goes
  straight to `backend.get_state`/`set_state` (facade, engine, dispatcher), no `hasattr`.
- [x] Stop handing the raw `backend.conn` to `TelemetryTracker`, `state.py` and the dispatcher
  (`db.conn.commit()`); they go through backend methods.
  **Done 2026-10-09:** `TelemetryTracker(backend)` uses `get_state`/`set_state`/
  `get_table_counts`/new `get_weak_points`; `ai_db/storage/state.py` deleted (and its
  exports from `ai_db`, `ai_db.storage`, `vectordb.py`); the dispatcher no longer builds a
  tracker from `db.conn`. No production code outside `sqlite_backend.py` touches `.conn`.

### K4. Transport layout
- [ ] Move root `mcp_server.py` to `ai_db/transports/mcp.py` (keep a 3-line root shim only if
  an external config references it; `ai-db mcp` is the documented entry). `cli.py` must not
  import a root module.
- [ ] Deduplicate span parsing in `dispatcher.py` (`_handle_expand` and the next handler).

### K5. Telemetry
- [ ] `TelemetryTracker.record_query` rewrites a JSON blob of ≤1000 samples and commits on every
  query; `avg_ms` is over the window but `count` is lifetime; multi-process last-writer-wins.
  Derive telemetry from the existing `query_log` table and delete the blob.
- [ ] `_handle_telemetry` returns a fabricated all-zero summary as fallback; raise instead.
- [ ] `video_extract_keyframes` returns `{"error": ...}` as success; raise.

### K6. Constants sweep
- [x] Language maps unified in `ai_db/parser/languages.py`; grammars are now selectable per
  language (`index.languages`, `ai-db grammars`, tests in `tests/test_languages.py`).
- [ ] Move to `ai_db/constants.py`: BM25 weights, `QUERY_LOG_KEEP`, vec0 over-fetch factor
  (`k*4`), 7-day eviction (duplicated in storage and `analyzer/engine.py`), FTS 4000-char
  truncation, `ZLIB_LEVEL`, investigate decay factors 0.6/0.5/0.4, `limit=100`, `per_file=5`,
  `[:8]`, analyzer 32 targets / ctx 10 lines, linter timeout 15, `META_MAX_CHARS`,
  `SNIPPET_CHARS`, `COMPOUND_MAX_DEFINITIONS`. Use the existing `GIT_TIMEOUT_S` instead of
  literal 5/10. Unify `ts_graph.SUPPORTED_LANGUAGES`/`language_for` with
  `constants.EXT_TO_LANG`/`chunker.language_for`.

### K7. Dead or half-wired features (DECISION per bullet; default = delete)
- [x] **Skill/context vectors: done (2026-10-07), user decision "an extension, but built in".**
  `ai_db/ext/recall.py` (`RecallProvider`, `ai_db.recall` entry points, `recall.providers`)
  with the built-in `vectors` provider in `ai_db/ext/vector_recall.py`. Now actually used:
  `route_skills` adds a similarity bonus, `query_contexts` RRF-fuses it with BM25. Model id
  tracked (re-embed on change), backfill of unseen skills/contexts, errors propagate,
  `search_*_hybrid` deleted, `cosine_distance` raises on mixed dimensions (E5 bullet).
  Tests: `tests/test_recall.py`. Still to do: tune `SKILL_W_RECALL` / `SKILL_RECALL_MIN_SIM`
  with the skills eval in hybrid mode (real model).
- [ ] `retrieval.doc_weight` is parsed and validated (`config.py`) but never read; the Ranker
  uses `index.doc_weight`. Remove the `retrieval` copy (and migrate configs that set it).
- [ ] `QueryEngine.__init__` re-derives `doc_weight`/`rerank_top_n` with `getattr` defaults
  0.5/30 that `VectorDB` then overrides; `Ranker.__init__` repeats the same literal defaults.
  One source: `DEFAULT_DOC_WEIGHT`/`DEFAULT_RERANK_TOP_N`.
- [ ] Skill/context vectors: embedded on every save (errors swallowed), never read outside
  tests. Wire into `route_skills`/`query_contexts` with an eval showing gain, or delete
  `search_*_hybrid`, `search_*_vector`, `skill_vectors`, the embed calls.
- [ ] `ai_db.chunkers` entry points (markdown/usd/segmentation in `pyproject.toml`) are never
  loaded: `ai_db/ext/markdown.py`, `usd.py`, `segmentation.py` are dead and untested.
- [ ] Custom linter validators from config never load (`get_linter()` without config);
  `.js/.ts` validator entries are dead.
- [ ] `/templates` daemon endpoint hardcodes banned model names: delete.
- [ ] Root `vectordb.py` (nothing imports it): delete.

### K8. Docs drift
- [ ] `ARCHITECTURE.md`: HTTP "removed" claim (§4) vs the daemon (per H decision);
  nonexistent `ai_db/transports/` (until K4); `vectordb` console script not in
  `pyproject.toml`; daemon, scene and video subsystems undocumented. Same "HTTP removed" claim
  in `TODO_NEXT.md` orientation.
- **Check:** every path mentioned in `ARCHITECTURE.md` exists
  (`grep -o 'ai_db/[a-z_/]*\.py' ARCHITECTURE.md | xargs ls`).

### K9. Repo hygiene (binaries done)
- [x] ~12.5 MB of renders/`.blend` tracked under `eval/showcase/oval_office_v2_out/`; ~19 MB
  more untracked (`*.blend1`, `oval_office_out/`, `material_sheet_v*.png`,
  `recipe_sheet_v*.png`). Add to `.gitignore`: `*.blend1`, `eval/showcase/*_out/`,
  `eval/showcase/*_sheet_v*.png`; `git rm --cached` the tracked outputs (**DECISION**: the user
  may want them in Git LFS instead). Do not rewrite history.
  **Done:** user decision "never commit binaries". `.gitignore` covers `*.blend*`, `*.exr`,
  `*.glb`, `*.fbx`, `*.usdc`, `*.usdz`, `packages/*/showcase/*_out/` and the sheet PNGs; the 8
  tracked binaries are untracked (still on disk). ~12 MB stays in history (not rewritten).
- [ ] Delete the stray `C:\data\project_db.sqlite` in the repo root and find the code that
  built a Windows path on Linux (`grep -rn 'C:' ai_db tests`); root `__pycache__/`.

### K10. Extract scene/video into a plugin package (done 2026-10-07)
- [x] Coupling is one-way: scene/video uses only `logger`, `errors`, a few constants,
  `config.SceneConfig`; core touches it only via `mcp_server.py` registration. Create
  `ai-db-scene` (its own `pyproject.toml`), register tools through an
  `ai_db.tools` entry point loaded by the K4 dispatcher factory, remove the `scene` key from
  the core config schema and `scene/priors.json` from core package-data.
  **Done:** `packages/ai-db-scene/` (code, tests, showcase, docs, own pyproject with the
  video/judge extras). Core loads plugins in `ServiceDispatcher.__init__`
  (`ai_db/plugins.py`), which also closes H5. Config: `scene` → generic `plugins.scene`.
  Core runs and its tests pass with the plugin uninstalled (24 tools; 65 with it). CI
  installs and type-checks both. Remaining K10 debt: K7 decides what happens to the
  `ai_db.chunkers` USD/segmentation chunkers that moved with it.

### K11. Showcase scripts bypass the pipeline
- [ ] `eval/showcase/oval_office_v2.py` is hand-authored by the lead model (hardcoded
  coordinates/yaws/asset specs) and imports nothing from `ai_db`; it uses a room-centre origin
  while the pipeline uses `[0,W]x[0,D]`. Project rule: the small model generates 3D content,
  the lead only builds tools and judges. **Fix:** turn the script into a harness that feeds a
  small-model-generated spec through `get_scene_spec_schema → solve_layout → apply_layout`;
  keep the hand-authored file only as a labelled reference baseline. Fix the stale v1
  docstring ("solver handles rectangular rooms only").

---

## Done when
CI is green. No sync can touch another project's rows. A failed write never becomes visible.
`trace`/`investigate --mode flow` return callees for every supported language in source order.
Eval recall is measured with exact matching. The daemon either works with auth or is gone.
`import mcp_server` does not import torch. `ARCHITECTURE.md` describes the code that exists.
