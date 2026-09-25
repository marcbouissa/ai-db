"""Run a golden query set against an index and report averaged retrieval metrics."""

import json
import os
import shutil
import statistics
import subprocess
import time
from typing import Any

from ai_db.eval.metrics import mrr, ndcg_at_k, recall_at_k

VALID_KINDS = ("locate", "explain", "impact", "diff")
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


def materialize_tracked(root: str, dest: str) -> str:
    """Copy the git-tracked files of ``root`` (working-tree versions) into ``dest``.

    Makes eval results independent of untracked/ignored local files.
    """
    out = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True, check=True)
    for rel in out.stdout.decode("utf-8").split("\0"):
        if not rel:
            continue
        src = os.path.join(root, rel)
        if not os.path.isfile(src):
            continue  # deleted in the working tree
        target = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(src, target)
    return dest


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


def compare_pack_to_baseline(result: dict[str, Any], baseline: dict[str, Any],
                             tolerance: float = 0.02) -> str | None:
    """Return an error message when pack_recall regressed more than ``tolerance``."""
    key = "pack_recall"
    if key not in baseline:
        raise ValueError(f"baseline has no '{key}' entry")
    drop = baseline[key] - result[key]
    if drop > tolerance:
        return f"{key} regressed by {drop:.4f} (baseline {baseline[key]}, now {result[key]})"
    return None


def run_pack(golden_path: str, root: str, db: Any, budget_tokens: int = 8000,
             sync: bool = True) -> dict[str, Any]:
    """Pack recall: fraction of expected symbols present in ``investigate`` evidence.

    Each golden item's ``kind`` is used as the investigate mode.
    """
    root = os.path.abspath(root)
    golden = load_golden(golden_path)
    if sync:
        db.sync(root, project=EVAL_PROJECT, verbose=False)
    recalls: list[float] = []
    tokens: list[int] = []
    latencies: list[float] = []
    per_query: list[dict[str, Any]] = []
    for item in golden:
        t0 = time.perf_counter()
        # Diff-mode items name the git ref to diff against; the other modes
        # have no such field and must not be handed a made-up one.
        if item["kind"] == "diff" and not item.get("since"):
            raise ValueError(
                f"diff golden item {item['query']!r} needs a 'since' git ref")
        pack = db.investigate(item["query"], budget_tokens=budget_tokens, mode=item["kind"],
                              project=EVAL_PROJECT, since=item.get("since"),
                              root=item.get("root", root))
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ranked = [(os.path.relpath(e["filepath"], root), e["qualified_name"]) for e in pack["evidence"]]
        r = recall_at_k(ranked, _expected_tuples(item), len(ranked) or 1)
        recalls.append(r)
        tokens.append(pack["token_count"])
        per_query.append({"query": item["query"], "mode": item["kind"], "pack_recall": r,
                          "tokens": pack["token_count"]})
    return {
        "queries": len(golden),
        "budget_tokens": budget_tokens,
        "pack_recall": round(statistics.fmean(recalls), 4),
        "tokens_mean": round(statistics.fmean(tokens), 1),
        "latency_ms_p50": round(_percentile(latencies, 50), 2),
        "latency_ms_p95": round(_percentile(latencies, 95), 2),
        "per_query": per_query,
    }


def load_skills_golden(golden_path: str) -> list[dict[str, Any]]:
    """Load a skill-routing golden set: one ``{"prompt", "expected_skill"}`` per line."""
    items: list[dict[str, Any]] = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
                raise ValueError(f"{golden_path}:{lineno}: 'prompt' must be a non-empty string")
            if not isinstance(item.get("expected_skill"), str) or not item["expected_skill"].strip():
                raise ValueError(f"{golden_path}:{lineno}: 'expected_skill' must be a non-empty string")
            items.append(item)
    return items


def run_skills(golden_path: str, db: Any, top_k: int = 3,
               min_confidence: float | None = None) -> dict[str, Any]:
    """Skill routing: top-1 accuracy of ``route_skills`` over a golden prompt set.

    Depends on the *locally installed* skills, so this is deliberately not part
    of the CI regression gate (see TODO 11.9).
    """
    golden = load_skills_golden(golden_path)
    router = db.skill_router
    hits: list[float] = []
    per_query: list[dict[str, Any]] = []
    for item in golden:
        routed = router.route_skills(item["prompt"], top_k=top_k,
                                     min_confidence=min_confidence)
        top = routed[0]["name"] if routed else None
        ok = 1.0 if top == item["expected_skill"] else 0.0
        hits.append(ok)
        per_query.append({
            "prompt": item["prompt"],
            "expected_skill": item["expected_skill"],
            "routed": top,
            "top1": ok,
        })
    return {
        "queries": len(golden),
        "skills_indexed": len(router.db.get_skills()),
        "top1_accuracy": round(statistics.fmean(hits), 4) if hits else 0.0,
        "per_query": per_query,
    }
