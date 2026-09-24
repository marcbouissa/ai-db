"""Run a golden query set against an index and report averaged retrieval metrics."""

import json
import os
import statistics
import time
from typing import Any

from ai_db.eval.metrics import mrr, ndcg_at_k, recall_at_k

VALID_KINDS = ("locate", "explain", "impact")
EVAL_PROJECT = "ai-db-eval"


def load_golden(golden_path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item.get("query"), str) or not item["query"].strip():
                raise ValueError(f"{golden_path}:{lineno}: 'query' must be a non-empty string")
            if item.get("kind") not in VALID_KINDS:
                raise ValueError(f"{golden_path}:{lineno}: 'kind' must be one of {VALID_KINDS}")
            expected = item.get("expected")
            if not isinstance(expected, list) or not expected:
                raise ValueError(f"{golden_path}:{lineno}: 'expected' must be a non-empty list")
            for exp in expected:
                if not isinstance(exp.get("filepath"), str):
                    raise TypeError(f"{golden_path}:{lineno}: expected.filepath must be a string")
            items.append(item)
    return items


def _expected_tuples(item: dict[str, Any]) -> list[tuple[str, str | None]]:
    return [(e["filepath"], e.get("symbol")) for e in item["expected"]]


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(pct / 100.0 * (len(ordered) - 1)))
    return ordered[idx]


def run(golden_path: str, root: str, db: Any, k: int = 10, sync: bool = True) -> dict[str, Any]:
    """Index ``root`` into ``db`` (a VectorDB) and evaluate every golden query.

    Golden filepaths are relative to ``root``.
    """
    root = os.path.abspath(root)
    golden = load_golden(golden_path)
    if sync:
        db.sync(root, project=EVAL_PROJECT, verbose=False)

    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    latencies: list[float] = []
    per_query: list[dict[str, Any]] = []

    for item in golden:
        t0 = time.perf_counter()
        hits = db.query(item["query"], top_k=k, project=EVAL_PROJECT)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ranked = [(os.path.relpath(h["abs_path"], root), h["name"]) for h in hits]
        expected = _expected_tuples(item)
        r = recall_at_k(ranked, expected, k)
        m = mrr(ranked, expected)
        n = ndcg_at_k(ranked, expected, k)
        recalls.append(r)
        mrrs.append(m)
        ndcgs.append(n)
        per_query.append({"query": item["query"], "recall": r, "mrr": m, "ndcg": n})

    return {
        "k": k,
        "queries": len(golden),
        f"recall@{k}": round(statistics.fmean(recalls), 4),
        "mrr": round(statistics.fmean(mrrs), 4),
        f"ndcg@{k}": round(statistics.fmean(ndcgs), 4),
        "latency_ms_p50": round(_percentile(latencies, 50), 2),
        "latency_ms_p95": round(_percentile(latencies, 95), 2),
        "per_query": per_query,
    }


def compare_to_baseline(result: dict[str, Any], baseline: dict[str, Any],
                        tolerance: float = 0.02) -> str | None:
    """Return an error message when recall@k regressed more than ``tolerance``."""
    key = f"recall@{result['k']}"
    if key not in baseline:
        raise ValueError(f"baseline has no '{key}' entry")
    drop = baseline[key] - result[key]
    if drop > tolerance:
        return f"{key} regressed by {drop:.4f} (baseline {baseline[key]}, now {result[key]})"
    return None
