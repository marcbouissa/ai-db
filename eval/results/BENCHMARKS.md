# Benchmark results

Everything here was measured on one machine, on this repository, with the commands
shown. Where a number is extrapolated rather than measured, it says so.

**Hardware.** NVIDIA GeForce RTX 3070 Laptop GPU (sm_86, 8 GB), 16 CPU threads
(Intel i7-11800H, 8 physical cores, AVX-512, no `avx512_bf16`, no AMX), torch
2.11.0+cu128.

**Corpus.** This repository: 149 files, ~1713 chunks, 3204 symbols, 9 languages.
**Golden set.** `eval/golden/ai_db.jsonl` (40 queries) and
`eval/golden/ai_db_pack.jsonl` (20 queries).
**Models.** `Qwen/Qwen3-Embedding-0.6B` (1024 dims), `BAAI/bge-reranker-v2-m3`.

---

## 1. Retrieval configuration matrix

The axis that accuracy actually depends on.

| condition | recall@10 | MRR | nDCG@10 | p50 (ms) | pack_recall |
|---|---|---|---|---|---|
| lexical (BM25 + graph) | 0.825 | 0.618 | 0.668 | **13.7** | 0.950 |
| hybrid, local embedder on CPU | 0.925 | 0.782 | 0.818 | 205.7 | 0.975 |
| **hybrid, local embedder on GPU** | **0.950** | **0.785** | **0.826** | 167.4 | **1.000** |
| hybrid + vec0 index | 0.950 | 0.785 | 0.826 | 156.1 | 1.000 |
| hybrid + bge-reranker-v2-m3 | 0.875 | 0.555 | 0.632 | 16,771.7 | 0.950 |

### Hybrid is decisive, not a wash

+0.125 recall@10, +0.167 MRR, +0.158 nDCG over lexical, and pack recall to 1.000.
Embeddings earn their cost on this corpus. This is the column that was missing
from the raw-vs-lexical comparison, and it settles the question.

**CPU and GPU embeddings are not interchangeable.** The quality gap (0.925 vs
0.950 recall@10) is one query out of 40, and the cause is numeric: the CPU loads
`float32` while the GPU loads the checkpoint's `bfloat16`, and the resulting
vectors differ enough to flip one borderline ranking. MRR is effectively tied
(0.782 vs 0.785).

### The device matters for indexing, not for querying

| | index this repo (1713 chunks) | query p50 |
|---|---|---|
| CPU | ~55 min | 205.7 ms |
| GPU | 22.9 s | 167.4 ms |

**145× on indexing, 1.2× on querying.** A single short query cannot fill a GPU, so
it pays launch and transfer overhead for little parallel work. This is the single
most actionable number in the document: if you index once and query many times,
put the embedder on a GPU. If you are CPU-only, everything still works — it is
just slow to build the index.

### vec0 is not a speedup at this size

Identical accuracy to three decimals and 156 ms vs 167 ms — noise. Consistent
with §4: `vec0` is a constant-factor C-loop win, not ANN, so it does not grow
with the corpus. Keep it for the C loop and metadata pruning; do not expect it to
transform query time.

### Rerank makes retrieval worse, and costs 100× the latency

MRR 0.785 → 0.555, recall@10 0.950 → 0.875, p50 167 ms → 16.8 s. A double
negative. `bge-reranker-v2-m3` is a general MS-Marco-style cross-encoder, not tuned
for code, so re-scoring an already-good fused ranking discards signal. A
code-tuned cross-encoder may behave differently; this measurement says *this
model hurts this corpus*.

The latency has two identified causes, both left unfixed and documented rather
than silently corrected:

1. The `CrossEncoder` loads at `max_seq_length=8192` while the model was trained
   at 512. Measured on the same 10 real candidates: **27,575 ms → 14,965 ms** at
   512 → **4,088 ms** at 256.
2. It loads `float32` on GPU. The dtype gate in §5 covers the embedder only.

---

## 2. Output modes

Formats render identical evidence, so only the cost an agent pays differs. Chars
for a fixed target, `lexical` condition.

| command | mode | chars | vs cheapest |
|---|---|---|---|
| `analyze` | outline | 5,256 | 1.00× |
| | stub | 5,658 | 1.08× |
| | prose | 6,399 | 1.22× |
| | sexp | 6,727 | 1.28× |
| | **json** | 15,221 | **2.90×** |
| `locate` | sexp | 583 | 1.00× |
| | stub | 942 | 1.62× |
| | **json** | 1,994 | **3.42×** |
| `trace` | tree | 2,268 | 1.00× |
| | mermaid | 5,006 | 2.21× |
| | **json** | 134,245 | **59.2×** |
| `investigate` | flow | 13,602 | 1.00× |
| | locate | 18,166 | 1.34× |
| | explain | 31,287 | 2.30× |
| | **impact** | 32,694 | **2.40×** |

**JSON is the expensive default and should rarely be the choice.** For `analyze`
and `locate`, `sexp` or `stub` cost a third of what JSON does. For `trace`, JSON
is 59× the tree renderer — a formatting decision, not a data difference.

`investigate` mode choice is a real accuracy/cost trade: `flow` is 2.4× cheaper
than `impact` and, on this corpus, retrieved a smaller pack for the questions
asked. `impact` earns its size only when you specifically need transitive callers.

---

## 3. Device: embedding throughput and query latency

`Qwen/Qwen3-Embedding-0.6B`, 24 real chunks, batch 32.

| | CUDA | CPU | speedup |
|---|---|---|---|
| embedding throughput | 33.2 chunks/s | 0.23 chunks/s | **141.8×** |
| single query (p50) | 41.9 ms | 438.3 ms | **10.5×** |

The 14× difference between those two ratios is the finding. One short query
cannot fill a GPU; bulk indexing can. Same conclusion as §1 from a different
measurement.

With the dtype gate applied, the CPU arm becomes 0.51 chunks/s — **2.1×** better
than the 0.23 the checkpoint's `bfloat16` gives on this hardware.

---

## 4. Vector index: exact vs sqlite-vec vec0

100,000 × 1024 normalised vectors, k=10, 50 queries.

| | p50 | recall@10 |
|---|---|---|
| exact | 541.82 ms | — (ground truth) |
| vec0 | 367.02 ms | 1.000 |
| speedup | **1.5×** | |

**The 10× target is unreachable with sqlite-vec 0.1.9**, whose `vec0` KNN is a
brute-force scan, not ANN. Measured directly: per-candidate cost stays flat at
~1.6–1.9 µs from 5k to 100k rows, which is linear in N. The observed 1.5× is
only what the C loop gains over a pure-Python exact path, and §1 confirms it
disappears at realistic corpus sizes.

Keep vec0 because recall is exactly 1.000 (a correct drop-in) and it provides a
clean seam for a real ANN backend later. Do not adopt it expecting order-of-magnitude
gains.

---

## 5. The dtype gate

Most modern embedding checkpoints ship in `bfloat16`. Correct on a GPU. On a CPU
without `avx512_bf16` or AMX it is emulated in software and runs *slower* than
`float32`:

| dtype | chunks/s (i7-11800H) |
|---|---|
| `bfloat16` (checkpoint default) | 0.24 |
| `float16` | 0.11 |
| **`float32`** | **0.51** |

Three explanations were measured and eliminated before finding this one:

- **not** memory-bandwidth bound — throughput is flat from batch 1 to 24
- **not** padding bound — length-sorting cut padded tokens 3.31×, wall clock moved 0.7%
- **not** per-token compute — time scales with *chunk count*, not tokens

`ai_db.device.resolve_dtype` now times a matmul per candidate dtype on CPU and
uses the fastest. On an accelerator it does not run at all — the native dtype is
left alone, because forcing `float32` on tensor cores would waste VRAM. It is a
timing probe rather than a CPU-feature lookup on purpose: `/proc/cpuinfo` does
not exist on macOS or Windows, flag names differ across Intel/AMD/ARM, and any
hardcoded table goes stale. `ai-db config check` reports the decision; set
`embedding.dtype` to override.

---

## 6. Bugs this exercise found

- **Rerank was impossible to configure from a file.** The `rerank` section was
  key-checked against `{provider, options, top_n}` while the option extraction
  assumed flat siblings. Every shape of a non-`none` provider was rejected —
  verified across all five plausible shapes. This is why the previously recorded
  rerank result had no reproducible config path. Fixed; tests added.
- **`trace` resolves bare symbol names only.** `Class.method` and
  `path/file.py:symbol` both return `No trace found`, and an entry with no
  outgoing edges yields a one-node tree. Pre-existing.
- **`evaluate --skills` reported `top1_accuracy: 0.0`** when no skills were
  indexed, indistinguishable from a broken router. Now an error naming the
  directories searched.
- **`ruff check .` reached stray root scripts** and `--fix` rewrote them. Now
  pinned to the project's own code.

---

## Caveats

- One corpus, one machine, no repeated runs. p50 varies run to run — the rerank
  condition ranged 12.7–20.5 s across three runs.
- Rerank quality is model-dependent. "This model hurts this corpus" is a finding
  about this pairing, not a general claim about reranking.
- `float16` is never auto-selected, but is still selectable by hand.
- §3's "index this repo ≈ 55 min on CPU" is wall-clock from a single run and
  includes parsing and SQLite writes, not just embedding.

## Reproduce

```bash
# configuration matrix (§1, §2)
uv run python eval/matrix_runner.py --out eval/results/matrix.json --work /tmp/matrix

# evaluate an already-built index without re-indexing it
uv run python eval/measure_index.py <condition> <config.json> <db> <out.json>

# device throughput and query latency (§3)
uv run pytest tests/test_bench_gpu.py -m bench -q -s

# vector index (§4)
AI_DB_BENCH_N=100000 AI_DB_BENCH_DIM=1024 uv run pytest tests/test_bench_vectors.py -m bench -q -s
```

The GPU benchmark skips rather than fails without CUDA or a warm model cache, and
refuses to benchmark a cold one (you would be timing the download). Its
end-to-end sync comparison is opt-in via `AI_DB_BENCH_SYNC=1`, because the CPU
arm costs minutes per hundred chunks.
