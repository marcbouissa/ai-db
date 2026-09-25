"""TODO 13.1 -- GPU vs CPU embedding throughput, and end-to-end index time.

Phase 13 wired device resolution (see ``ai_db/device.py``) and verified the GPU
was usable, but the benefit was never measured: the "GPU indexing < 1 min"
check in TODO_NEXT.md has no number behind it. This fills that in.

**Sizing matters more than anything else here.** A 0.6B model embeds at roughly
0.24 chunks/s on 16 CPU threads versus ~33 chunks/s on an RTX 3070 -- a ~135x
gap. A corpus sized to be comfortable on the GPU therefore takes *hours* on the
CPU arm. The default below is deliberately small: throughput per chunk is
stable well before the sample is large, and the ratio is what this measures, not
the absolute corpus size.

Measurements:

1. **Embedding throughput** -- chunks/second per device over real code chunks.
2. **Query latency** -- a single ``embed_query`` per device: the interactive path.
3. **End-to-end index time** -- a real ``sync()``, opt-in, because the CPU arm
   costs minutes per hundred chunks.

The corpus is real code from this repo, not synthetic padding: embed latency is
sensitive to sequence length and padding would flatter the GPU.

Run with::

    uv run pytest tests/test_bench_gpu.py -m bench -q -s

Skips (not fails) when the embedding model is not cached locally or torch has no
CUDA device: this benchmark needs assets CI does not have.
"""
from __future__ import annotations

import os
import statistics
import time

import pytest

pytestmark = pytest.mark.bench

# 24 chunks is ~100 s on CPU and ~1 s on CUDA: enough for a stable per-chunk
# rate, small enough that the CPU arm does not dominate the run.
N_CHUNKS = int(os.environ.get("AI_DB_BENCH_CHUNKS", "24"))
MODEL = os.environ.get("AI_DB_BENCH_MODEL", "Qwen/Qwen3-Embedding-0.6B")
BATCH = int(os.environ.get("AI_DB_BENCH_BATCH", "32"))
N_QUERIES = int(os.environ.get("AI_DB_BENCH_QUERIES", "10"))
# The full-sync comparison costs minutes on the CPU arm, so it is opt-in.
RUN_SYNC = os.environ.get("AI_DB_BENCH_SYNC") == "1"
SYNC_ROOT = os.environ.get("AI_DB_BENCH_SYNC_ROOT", "ai_db/parser")
QUERY_PROMPT = "Instruct: Given a code search query, retrieve relevant code\nQuery: "


def _model_is_cached() -> bool:
    """True when the embedding model is already in the HF cache.

    Downloading a model inside a benchmark makes the numbers depend on network
    speed, so a cold cache skips rather than quietly benchmarking the download.
    """
    from ai_db.device import cuda_available

    if not cuda_available():
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    hit = try_to_load_from_cache(MODEL, "config.json")
    return isinstance(hit, str)


requires_gpu = pytest.mark.skipif(
    not _model_is_cached(),
    reason=f"needs a CUDA device and a cached {MODEL}",
)


def _real_chunks(limit: int) -> list[str]:
    """Real code chunks from this repo, not synthetic padding."""
    from ai_db.parser.chunker import chunk_file

    texts: list[str] = []
    for root, dirs, files in os.walk("ai_db"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
            for chunk in chunk_file(path, source):
                content = chunk.get("content", "")
                if content.strip():
                    texts.append(content)
            if len(texts) >= limit:
                return texts[:limit]
    return texts


def _write_config(path, db_path, device: str):
    """A real config file, so the benchmark exercises the user path."""
    import json

    path.write_text(json.dumps({
        "version": 2,
        "storage": {"provider": "sqlite",
                    "options": {"path": str(db_path), "vector_index": "exact"}},
        "retrieval": {"mode": "hybrid"},
        "embedding": {"provider": "sentence_transformers", "model": MODEL,
                      "device": device, "batch_size": BATCH,
                      "query_prompt": QUERY_PROMPT},
        "rerank": {"provider": "none"},
        "access": {"cross_project": {}},
        "auto_sync_paths": [],
        "trace": {"wait_patterns": None, "wait_patterns_extend": None},
    }, indent=2), encoding="utf-8")
    return str(path)


def _embedder(device: str):
    from ai_db.embed.st_provider import SentenceTransformersEmbedder

    return SentenceTransformersEmbedder({
        "model": MODEL, "device": device, "batch_size": BATCH,
        "query_prompt": QUERY_PROMPT,
    })


def _sync(device: str) -> None:
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def _report(title: str, report: dict[str, object]) -> None:
    print(f"\n=== {title} ===", flush=True)
    for key, value in report.items():
        print(f"  {key}: {value}", flush=True)
    print("=== end ===", flush=True)


@requires_gpu
def test_embedding_throughput_cpu_vs_cuda():
    """Chunks/second per device, and the ratio between them."""
    texts = _real_chunks(N_CHUNKS)
    assert len(texts) >= 8, f"only found {len(texts)} chunks to embed"

    report: dict[str, object] = {"model": MODEL, "chunks": len(texts),
                                 "batch_size": BATCH}
    rates: dict[str, float] = {}
    for device in ("cuda", "cpu"):
        print(f"  [throughput] loading {device}...", flush=True)
        embedder = _embedder(device)
        # Warm up: the first call pays lazy model load and CUDA context setup,
        # and charging that to the throughput number would be meaningless.
        embedder.embed_documents(texts[:BATCH])
        _sync(device)
        t0 = time.perf_counter()
        for start in range(0, len(texts), BATCH):
            embedder.embed_documents(texts[start:start + BATCH])
        _sync(device)
        elapsed = time.perf_counter() - t0
        rate = len(texts) / elapsed
        rates[device] = rate
        report[f"{device}_chunks_per_s"] = round(rate, 2)
        report[f"{device}_s_per_1k_chunks"] = round(elapsed / len(texts) * 1000, 1)
        del embedder

    speedup = rates["cuda"] / rates["cpu"] if rates["cpu"] else 0.0
    report["cuda_speedup"] = round(speedup, 1)
    _report("embedding throughput", report)
    assert speedup > 1.0, f"CUDA was not faster: {report}"


@requires_gpu
def test_query_latency_cpu_vs_cuda():
    """A single embed_query() per device: the interactive path."""
    report: dict[str, object] = {"queries": N_QUERIES}
    p50: dict[str, float] = {}
    for device in ("cuda", "cpu"):
        print(f"  [query latency] loading {device}...", flush=True)
        embedder = _embedder(device)
        embedder.embed_query("warm up the model")
        _sync(device)
        samples = []
        for _ in range(N_QUERIES):
            t0 = time.perf_counter()
            embedder.embed_query("where is the sqlite backend opened")
            _sync(device)
            samples.append((time.perf_counter() - t0) * 1000)
        ordered = sorted(samples)
        p50[device] = statistics.median(samples)
        report[f"{device}_p50_ms"] = round(p50[device], 1)
        report[f"{device}_p95_ms"] = round(ordered[int(len(ordered) * 0.95) - 1], 1)
        del embedder

    report["cuda_speedup"] = (round(p50["cpu"] / p50["cuda"], 1)
                              if p50["cuda"] else 0.0)
    _report("query latency", report)


@pytest.mark.skipif(not RUN_SYNC, reason="set AI_DB_BENCH_SYNC=1 (minutes on the CPU arm)")
@requires_gpu
def test_end_to_end_sync_time_cpu_vs_cuda(tmp_path, monkeypatch):
    """A real sync() on each device -- what a user actually waits for."""
    from ai_db import VectorDB
    from ai_db.config import load_config

    report: dict[str, object] = {"model": MODEL, "root": SYNC_ROOT}
    totals: dict[str, float] = {}
    for device in ("cuda", "cpu"):
        print(f"  [sync] {device} over {SYNC_ROOT} ...", flush=True)
        db_path = tmp_path / f"{device}.db"
        cfg_path = _write_config(tmp_path / f"{device}.json", db_path, device)
        monkeypatch.setenv("AI_DB_CONFIG", cfg_path)

        db = VectorDB(str(db_path), config=load_config(cfg_path))
        t0 = time.perf_counter()
        stats = db.sync(SYNC_ROOT, project="bench", verbose=False)
        totals[device] = time.perf_counter() - t0
        report[f"{device}_sync_s"] = round(totals[device], 1)
        report[f"{device}_chunks"] = stats.get("inserted", 0) + stats.get("kept", 0)
        db.close()

    if totals["cuda"] > 0:
        report["cuda_speedup"] = round(totals["cpu"] / totals["cuda"], 1)
        report["gpu_under_60s"] = bool(totals["cuda"] < 60)
    _report("end-to-end sync", report)
