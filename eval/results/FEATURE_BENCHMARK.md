# Per-feature latency: cold vs prewarmed

Every user-facing feature, timed as `python -m ai_db.cli ...` — what a user
or an agent loop actually pays.

## What the columns mean

| Column | Definition |
|---|---|
| **Cold (first run)** | First subprocess invocation: fresh interpreter, cold page cache, no loaded model or CUDA context. |
| **Warm (steady state)** | Median of further subprocess runs. OS page cache and query cache are populated; the interpreter still starts fresh. |
| **In-process** | Median of repeated calls inside one process. Excludes interpreter startup and model loading, so it is the feature's own cost. |
| **Init overhead** | Cold − warm: the one-time cost. |
| **Cold ÷ warm** | How much of the first call is setup rather than work. |

Both cold and warm are end-to-end CLI latency on purpose: an agent driving
ai-db through the CLI pays interpreter startup on every single call, so a
library-only measurement would be flattering and wrong. The in-process column
is there to show what that startup costs.

## What the numbers say

1. **The features are not the cost; the process is.** Every measured feature runs in under 1.2 ms in-process, while the same call as a subprocess takes ~240 ms — 269–3028× longer. A median warm call spends ~28 ms starting the interpreter and ~102 ms importing ai_db, leaving single-digit milliseconds of actual work. The in-process column is the reason this table is worth reading twice: it shows the library is fast and the CLI is what costs.

2. **The genuinely expensive features are the ones that load a model.** Anything hybrid pays ~10.5 s on *every* invocation, because the embedding model is reloaded per process:

| Feature | Warm | Why |
|---|---|---|
| config check | 11.3s | validates the embedding backend |
| reindex --embeddings | 10.5s | re-embeds the whole corpus |
| query (hybrid) | 10.5s | reloads the embedding model |
| investigate (hybrid) | 10.3s | reloads the embedding model |
| eval --pack | 3.55s | builds packs for 40 queries |
| eval (retrieval, 10 queries) | 2.44s | builds packs for 10 queries |
| check | 583ms | re-indexes the file it is checking |
| lint | 564ms | re-indexes the file it is checking |

3. **Cold and warm are close for most features.** A lexical query has no model to warm, so the one-time cost is small — several rows show a *negative* init overhead, which is just the first run happening to beat the median of the next five. The genuine one-time costs are `sync` (2.9×, one-time parse and index build) and the four `investigate` modes (2.1–2.6×, one-time pack assembly).

4. **`check` and `lint` re-index the file they are checking.** At ~580 ms they are the slowest non-model commands, and the cause is not the syntax check — `ast.parse` on the same file is 0 ms. `_handle_check` calls `prune_file` then `_index_file` on a file path, so it re-parses and re-chunks the file (216 `count_tokens` calls, which loads tiktoken) and only then reads the syntax errors back out of the database it just rewrote. On a hybrid config this would also re-embed. The check itself needs none of that.

5. **`analyze --format json` is ~2.4× the other formats for one telemetry field.** Loading tiktoken's encoder costs ~300 ms once per process (200k base64 decodes of the BPE ranks table) against 0.9 ms per subsequent count. The load is now skipped for the four formats whose output is read directly and is only paid for json, where the number appears in `meta`.

## Indexing

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| sync | 756ms | 260ms | — | 496ms | 2.91× |
| sync-all | 134ms | 121ms | — | 12.8ms | 1.11× |

## Retrieval

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| query | 256ms | 238ms | 0.63ms | 17.5ms | 1.07× |
| query (hybrid) | 10.7s | 10.5s | — | 231ms | 1.02× |
| locate | 239ms | 229ms | 0.63ms | 9.2ms | 1.04× |
| symbol | 235ms | 231ms | 0.86ms | 3.7ms | 1.02× |

## Analysis

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| check | 586ms | 583ms | — | 3.0ms | 1.01× |
| lint | 547ms | 564ms | — | noise | 0.97× |
| outline | 122ms | 148ms | — | noise | 0.82× |
| analyze | 202ms | 221ms | 0.20ms | noise | 0.91× |
| analyze --fmt json | 530ms | 528ms | 0.46ms | 1.5ms | 1.00× |
| analyze --fmt sexp | 213ms | 233ms | — | noise | 0.91× |
| analyze --fmt outline | 252ms | 239ms | — | 12.7ms | 1.05× |
| analyze --fmt prose | 232ms | 221ms | — | 11.3ms | 1.05× |
| analyze --depth summary | 222ms | 230ms | — | noise | 0.96× |
| analyze --depth full | 211ms | 224ms | — | noise | 0.94× |
| callers | 234ms | 223ms | — | 10.6ms | 1.05× |
| todos | 240ms | 236ms | — | 4.0ms | 1.02× |
| diff | 260ms | 236ms | — | 23.7ms | 1.10× |
| trace | 292ms | 268ms | — | 24.2ms | 1.09× |
| trace --format mermaid | 267ms | 266ms | — | 0.30ms | 1.00× |
| investigate --mode locate | 578ms | 235ms | 0.22ms | 343ms | 2.46× |
| investigate --mode explain | 619ms | 236ms | 0.26ms | 383ms | 2.62× |
| investigate --mode impact | 654ms | 310ms | 0.21ms | 344ms | 2.11× |
| investigate --mode flow | 672ms | 276ms | 0.14ms | 396ms | 2.44× |
| investigate --format compact | 233ms | 247ms | — | noise | 0.94× |
| investigate --format stub | 253ms | 242ms | — | 11.5ms | 1.05× |
| investigate --format sexp | 234ms | 263ms | — | noise | 0.89× |
| investigate (hybrid) | 10.5s | 10.3s | — | 195ms | 1.02× |
| expand | — | — | 0.09ms | — | — |

## Context memory

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| remember | 224ms | 229ms | — | noise | 0.98× |
| context save | 269ms | 257ms | — | 12.6ms | 1.05× |
| context recall | 246ms | 240ms | — | 5.9ms | 1.02× |
| context list | 247ms | 258ms | — | noise | 0.96× |

## Skills

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| sync-skills | 221ms | 232ms | — | noise | 0.96× |
| route-skill | 245ms | 246ms | — | noise | 1.00× |

## Maintenance

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| status | 230ms | 242ms | 0.08ms | noise | 0.95× |
| telemetry | 247ms | 252ms | — | noise | 0.98× |
| log --slow | 237ms | 241ms | — | noise | 0.98× |
| prune | 248ms | 252ms | — | noise | 0.99× |
| optimize | 321ms | 278ms | — | 43.6ms | 1.16× |
| vacuum | 288ms | 274ms | — | 14.7ms | 1.05× |
| reindex --embeddings | 10.4s | 10.5s | — | noise | 0.99× |

## Evaluation

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| eval (retrieval, 10 queries) | 2.48s | 2.44s | — | 36.5ms | 1.01× |
| eval --pack | 3.36s | 3.55s | — | noise | 0.95× |

## Config & diagnostics

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| config show | 128ms | 137ms | — | noise | 0.93× |
| config check | 11.5s | 11.3s | — | 199ms | 1.02× |

## Reproducing

```bash
uv run python eval/matrix_configs.py /tmp/feature_bench /tmp/feature_bench/db
uv run python eval/feature_benchmark.py --repeats 5
uv run python eval/render_feature_table.py   # regenerate this file
```

Destructive features (`prune`, `optimize`, `vacuum`, `reindex`) run against a
scratch copy of the index, never the real one.
