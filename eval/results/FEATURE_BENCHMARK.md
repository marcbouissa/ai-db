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

1. **The features are not the cost; the process is.** Every measured feature runs in under 1.2 ms in-process, while the same call as a subprocess takes ~240 ms — 233–3572× longer. A median warm call spends ~28 ms starting the interpreter and ~102 ms importing ai_db, leaving single-digit milliseconds of actual work. The in-process column is the reason this table is worth reading twice: it shows the library is fast and the CLI is what costs.

2. **The genuinely expensive features are the ones that load a model.** Anything hybrid pays ~10.5 s on *every* invocation, because the embedding model is reloaded per process:

| Feature | Warm | Why |
|---|---|---|
| config check | 11.2s | validates the embedding backend |
| reindex --embeddings | 10.6s | re-embeds the whole corpus |
| query (hybrid) | 10.6s | reloads the embedding model |
| investigate (hybrid) | 10.4s | reloads the embedding model |
| eval --pack | 3.52s | builds packs for 40 queries |
| eval (retrieval, 10 queries) | 2.51s | builds packs for 10 queries |
| analyze --fmt json | 550ms | loads tiktoken to fill one meta field |
| trace --format mermaid | 274ms | — |

3. **Cold and warm are close for most features.** A lexical query has no model to warm, so the one-time cost is small — several rows show a *negative* init overhead, which is just the first run happening to beat the median of the next five. The genuine one-time costs are `sync` (2.9×, one-time parse and index build) and the four `investigate` modes (2.1–2.6×, one-time pack assembly).

4. **`analyze --format json` is ~2.4× the other formats for one telemetry field.** Loading tiktoken's encoder costs ~300 ms once per process (200k base64 decodes of the BPE ranks table) against 0.9 ms per subsequent count. Measured by counterfactual, forcing the count on for every format costs ~290 ms on `outline` and ~318 ms on `prose`; `stub` and `sexp` already paid the load via the analyze path. The count is now taken only for `json`, where the number appears in `meta` and is actually read.

5. **Three costs found by this benchmark have been fixed** since the first run: the eager tree-sitter import (`import ai_db` +102 ms → +67 ms), `check`/`lint` re-indexing the file they were checking (583 ms → 265 ms, now at parity with `status`), and the tiktoken regression above. `check` also now has two modes — `check <path>` validates files on disk and writes nothing, `check --index` reads the stored index.

## Indexing

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| sync | 727ms | 240ms | — | 487ms | 3.03× |
| sync-all | 118ms | 110ms | — | 8.4ms | 1.08× |

## Retrieval

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| query | 252ms | 235ms | 0.77ms | 16.2ms | 1.07× |
| query (hybrid) | 10.5s | 10.6s | — | noise | 0.99× |
| locate | 232ms | 238ms | 0.81ms | noise | 0.98× |
| symbol | 229ms | 236ms | 1.0ms | noise | 0.97× |

## Analysis

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| check <file> | 299ms | 266ms | — | 33.1ms | 1.12× |
| check --index | 227ms | 220ms | — | 7.3ms | 1.03× |
| lint <file> | 273ms | 264ms | — | 9.1ms | 1.03× |
| outline | 129ms | 147ms | — | noise | 0.88× |
| analyze | 221ms | 219ms | 0.17ms | 2.0ms | 1.01× |
| analyze --fmt json | 552ms | 550ms | 0.40ms | 2.0ms | 1.00× |
| analyze --fmt sexp | 240ms | 229ms | — | 11.5ms | 1.05× |
| analyze --fmt outline | 230ms | 236ms | — | noise | 0.98× |
| analyze --fmt prose | 273ms | 231ms | — | 42.0ms | 1.18× |
| analyze --depth summary | 228ms | 221ms | — | 7.3ms | 1.03× |
| analyze --depth full | 220ms | 240ms | — | noise | 0.92× |
| callers | 233ms | 226ms | — | 7.8ms | 1.03× |
| todos | 235ms | 234ms | — | 0.60ms | 1.00× |
| diff | 257ms | 235ms | — | 22.5ms | 1.10× |
| trace | 273ms | 266ms | — | 6.4ms | 1.02× |
| trace --format mermaid | 258ms | 274ms | — | noise | 0.94× |
| investigate --mode locate | 648ms | 232ms | 0.20ms | 416ms | 2.79× |
| investigate --mode explain | 607ms | 246ms | 0.25ms | 360ms | 2.46× |
| investigate --mode impact | 622ms | 241ms | 0.19ms | 381ms | 2.58× |
| investigate --mode flow | 624ms | 246ms | 0.12ms | 378ms | 2.53× |
| investigate --format compact | 238ms | 250ms | — | noise | 0.95× |
| investigate --format stub | 254ms | 240ms | — | 14.0ms | 1.06× |
| investigate --format sexp | 235ms | 239ms | — | noise | 0.98× |
| investigate (hybrid) | 10.7s | 10.4s | — | 330ms | 1.03× |
| expand | — | — | 0.07ms | — | — |

## Context memory

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| remember | 218ms | 230ms | — | noise | 0.95× |
| context save | 237ms | 230ms | — | 6.8ms | 1.03× |
| context recall | 222ms | 238ms | — | noise | 0.93× |
| context list | 216ms | 222ms | — | noise | 0.98× |

## Skills

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| sync-skills | 248ms | 236ms | — | 12.0ms | 1.05× |
| route-skill | 248ms | 213ms | — | 34.9ms | 1.16× |

## Maintenance

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| status | 218ms | 214ms | 0.06ms | 4.2ms | 1.02× |
| telemetry | 251ms | 258ms | — | noise | 0.97× |
| log --slow | 222ms | 217ms | — | 5.5ms | 1.03× |
| prune | 208ms | 227ms | — | noise | 0.92× |
| optimize | 276ms | 264ms | — | 11.9ms | 1.05× |
| vacuum | 287ms | 268ms | — | 19.0ms | 1.07× |
| reindex --embeddings | 10.9s | 10.6s | — | 374ms | 1.04× |

## Evaluation

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| eval (retrieval, 10 queries) | 2.50s | 2.51s | — | noise | 1.00× |
| eval --pack | 3.34s | 3.52s | — | noise | 0.95× |

## Config & diagnostics

| Feature | Cold (first run) | Warm (steady state) | In-process | Init overhead | Cold ÷ warm |
|---|---|---|---|---|---|
| config show | 131ms | 104ms | — | 27.5ms | 1.26× |
| config check | 10.9s | 11.2s | — | noise | 0.98× |

## Reproducing

```bash
uv run python eval/matrix_configs.py /tmp/feature_bench /tmp/feature_bench/db
uv run python eval/feature_benchmark.py --repeats 5
uv run python eval/render_feature_table.py   # regenerate this file
```

Destructive features (`prune`, `optimize`, `vacuum`, `reindex`) run against a
scratch copy of the index, never the real one.
