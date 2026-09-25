"""TODO 13.3 -- vector search benchmark: exact vs sqlite-vec (vec0).

Builds a synthetic corpus of normalised vectors, then reports p50 query latency
for both ``storage.options.vector_index`` modes plus recall@10 of vec0 against
the exact baseline (the ground truth).

Run with::

    pytest tests/test_bench_vectors.py -m bench -q -s

Size knobs (defaults follow the spec: 100k vectors, 1024 dims)::

    AI_DB_BENCH_N=100000 AI_DB_BENCH_DIM=1024 AI_DB_BENCH_QUERIES=50
"""
from __future__ import annotations

import os
import random
import statistics
import time

import pytest

from ai_db.storage.models import ChunkRecord, FileRecord
from ai_db.storage.sqlite_backend import SQLiteBackend

pytestmark = pytest.mark.bench

N_VECTORS = int(os.environ.get("AI_DB_BENCH_N", "100000"))
DIM = int(os.environ.get("AI_DB_BENCH_DIM", "1024"))
QUERIES = int(os.environ.get("AI_DB_BENCH_QUERIES", "50"))
K = 10

# Keep the default run bounded: a full 100k x 1024 float32 corpus is ~400 MB.
# CI-friendly default is smaller; the spec number is one env var away.
if os.environ.get("CI") and N_VECTORS > 20000:
    N_VECTORS = 20000


def _unit(rng: random.Random) -> list[float]:
    v = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def _build(mode: str, path: str, rng_seed: int) -> tuple[SQLiteBackend, list[int], list[list[float]]]:
    """Seed ``N_VECTORS`` chunks + vectors; returns the backend, ids and probe vectors.

    Vector i always comes from ``random.Random(i)`` (not a shared stream) so the
    exact and vec0 corpora are byte-identical and recall is a fair comparison.
    """
    db = SQLiteBackend(path, vector_index=mode)
    db.initialize()
    db.ensure_vector_index(DIM, "bench:synthetic", vector_index=mode)

    t0 = time.perf_counter()
    with db.transaction():
        for i in range(N_VECTORS):
            fp = f"/bench/dir{i % 64}/f{i}.py"
            db.upsert_file(FileRecord(filepath=fp, sha256=f"h{i}",
                                      last_modified=time.time(), chunk_count=1, project="p"))
            db.insert_chunks([ChunkRecord(
                filepath=fp, name=f"n{i}", qualified_name=f"module.n{i}", chunk_type="code",
                content="x", start_line=1, end_line=1, language="python", project="p")])
    ids = [r["id"] for r in db.conn.execute(
        "SELECT id FROM chunks ORDER BY id").fetchall()]

    probes = [_unit(random.Random(9999 + i)) for i in range(QUERIES)]
    db.upsert_embeddings(list(zip(ids, (_unit(random.Random(i)) for i in range(N_VECTORS)))))
    print(f"[bench:{mode}] built {N_VECTORS} vectors in {time.perf_counter() - t0:.1f}s")
    return db, ids, probes


def _p50_search(db: SQLiteBackend, probes: list[list[float]], filters: dict) -> float:
    lat = []
    for v in probes:
        t0 = time.perf_counter()
        db.search_vectors(v, K, filters)
        lat.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(lat)


@pytest.mark.slow
def test_vec0_beats_exact_and_keeps_recall(tmp_path):
    """p50 latency for both modes + recall@10 of vec0 against exact ground truth."""
    t_start = time.perf_counter()
    db_exact, _ids, probes = _build("exact", str(tmp_path / "exact.db"), 42)
    try:
        exact_ms = _p50_search(db_exact, probes, {"allowed_projects": ["p"]})
        truth: list[list[int]] = []
        for v in probes:
            truth.append([cid for cid, _ in
                          db_exact.search_vectors(v, K, {"allowed_projects": ["p"]})])
    finally:
        db_exact.close()

    db_vec, _, _ = _build("vec0", str(tmp_path / "vec0.db"), 42)
    try:
        vec_ms = _p50_search(db_vec, probes, {"allowed_projects": ["p"]})
        hits: list[set[int]] = []
        for v in probes:
            hits.append({cid for cid, _ in
                         db_vec.search_vectors(v, K, {"allowed_projects": ["p"]})})
    finally:
        db_vec.close()

    recalls = [
        len(hits[i] & set(truth[i])) / max(1, len(truth[i]))
        for i in range(len(truth))
    ]
    recall10 = statistics.fmean(recalls) if recalls else 0.0
    speedup = (exact_ms / vec_ms) if vec_ms > 0 else float("inf")

    print("\n" + "=" * 64)
    print(f"corpus        : {N_VECTORS:,} vectors x {DIM} dims (normalised)")
    print(f"queries       : {QUERIES}, k={K}, filter=project")
    print(f"exact   p50   : {exact_ms:9.2f} ms")
    print(f"vec0    p50   : {vec_ms:9.2f} ms")
    print(f"speedup       : {speedup:9.1f}x")
    print(f"vec0 recall@10: {recall10:9.3f}  (vs exact)")
    print(f"wall clock    : {time.perf_counter() - t_start:.1f}s")
    print("=" * 64)

    # Correctness first: an ANN index that loses the neighbourhood is not usable.
    assert recall10 >= 0.95, f"vec0 recall@10 {recall10:.3f} < 0.95"
    # The spec's performance target. Reported either way, but only enforced at
    # the full spec size -- smaller corpora are dominated by setup, not search.
    if N_VECTORS >= 100_000:
        assert speedup >= 10.0, f"vec0 only {speedup:.1f}x faster (target 10x)"
