"""TODO 13.1 -- GPU vs CPU embedding throughput, and end-to-end index time.

Phase 13 wired device resolution (see ``ai_db/device.py``) and verified the GPU
was usable, but the benefit was never measured: the "GPU indexing < 1 min"
check in TODO_NEXT.md has no number behind it. This fills that in.

Two measurements, because they answer different questions:

1. **Embedding throughput** -- chunks/second for the real embedding model on
   each device. Isolates the device, with no database in the way.
2. **End-to-end index time** -- a real ``sync()`` of a real tree, CPU then
   CUDA. This is what a user waits for, and it includes parsing, chunking,
   FTS and the vector writes, so it is always slower than (1) alone.

The corpus is the ai-db source tree itself, truncated to ``AI_DB_BENCH_FILES``
files, so the chunks are real code chunks rather than lorem ipsum -- embed
latency is sensitive to sequence length and synthetic padding would flatter
the GPU.

Run with::

    AI_DB_BENCH_FILES=400 pytest tests/test_bench_gpu.py -m bench -q -s

Skipped (not failed) when the embedding model is not cached locally or torch
has no CUDA device: this benchmark needs assets CI does not have.
"""
from __future__ import annotations

import os
import statistics
import time

import pytest

pytestmark = pytest.mark.bench

N_FILES = int(os.environ.get("AI_DB_BENCH_FILES", "400"))
MODEL = os.environ.get("AI_DB_BENCH_MODEL", "Qwen/Qwen3-Embedding-0.6B")
BATCH = int(os.environ.get("AI_DB_BENCH_BATCH", "32"))
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


def _real_chunks(n_files: int) -> list[str]:
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
            if len(texts) >= n_files * 4:
                return texts
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


@requires_gpu
def test_embedding_throughput_cpu_vs_cuda():
    """Chunks/second per device, and the ratio between them."""
    texts = _real_chunks(N_FILES)
    assert len(texts) >= 50, f"only found {len(texts)} chunks to embed"

    report: dict[str, object] = {"model": MODEL, "chunks": len(texts),
                                 "batch_size": BATCH}
    for device in ("cpu", "cuda"):
        embedder = _embedder(device)
        # Warm up: the first call pays lazy model load and CUDA context setup,
        # and charging that to the throughput number would be meaningless.
        embedder.embed_documents(texts[:BATCH])
        if device == "cuda":
            import torch
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for start in range(0, len(texts), BATCH):
            embedder.embed_documents(texts[start:start + BATCH])
        if device == "cuda":
            import torch
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        report[f"{device}_s_per_1k_chunks"] = round(elapsed / len(texts) * 1000, 1)
        report[f"{device}_chunks_per_s"] = round(len(texts) / elapsed, 1)
        del embedder

    speedup = (report["cpu_s_per_1k_chunks"] / report["cuda_s_per_1k_chunks"]
               if isinstance(report["cpu_s_per_1k_chunks"], float)
               and isinstance(report["cuda_s_per_1k_chunks"], float) else 0.0)
    report["speedup"] = round(speedup, 2)
    print("\n=== embedding throughput ===")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print("=== end ===")
    assert speedup > 1.0, f"CUDA was not faster: {report}"


@requires_gpu
def test_end_to_end_sync_time_cpu_vs_cuda(tmp_path, monkeypatch):
    """A real sync() on each device -- what a user actually waits for."""
    from ai_db import VectorDB
    from ai_db.config import load_config

    report: dict[str, object] = {"model": MODEL, "root": "ai_db"}
    for device in ("cpu", "cuda"):
        db_path = tmp_path / f"{device}.db"
        cfg_path = _write_config(tmp_path / f"{device}.json", db_path, device)
        monkeypatch.setenv("AI_DB_CONFIG", cfg_path)

        db = VectorDB(str(db_path), config=load_config(cfg_path))
        t0 = time.perf_counter()
        stats = db.sync("ai_db", project="bench", verbose=False)
        elapsed = time.perf_counter() - t0
        report[f"{device}_sync_s"] = round(elapsed, 1)
        report[f"{device}_chunks_indexed"] = stats.get("inserted", 0) + stats.get("kept", 0)
        db.close()

    cpu = report["cpu_sync_s"]
    cuda = report["cuda_sync_s"]
    if isinstance(cpu, float) and isinstance(cuda, float) and cuda > 0:
        report["sync_speedup"] = round(cpu / cuda, 2)
        report["gpu_under_60s"] = bool(cuda < 60)
    print("\n=== end-to-end sync (ai_db tree) ===")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print("=== end ===")


@requires_gpu
def test_query_latency_cpu_vs_cuda(tmp_path):
    """A single embed_query() per device: the interactive path."""

    report: dict[str, object] = {}
    for device in ("cpu", "cuda"):
        embedder = _embedder(device)
        embedder.embed_query("warm up the model")
        if device == "cuda":
            import torch
            torch.cuda.synchronize()
        samples = []
        for _ in range(20):
            t0 = time.perf_counter()
            embedder.embed_query("where is the sqlite backend opened")
            if device == "cuda":
                import torch
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)
        report[f"{device}_query_p50_ms"] = round(statistics.median(samples), 1)
        report[f"{device}_query_p95_ms"] = round(
            sorted(samples)[int(len(samples) * 0.95) - 1], 1)
        del embedder

    cpu = report["cpu_query_p50_ms"]
    cuda = report["cuda_query_p50_ms"]
    if isinstance(cpu, float) and isinstance(cuda, float) and cuda > 0:
        report["query_speedup"] = round(cpu / cuda, 2)
    print("\n=== query latency ===")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print("=== end ===")
