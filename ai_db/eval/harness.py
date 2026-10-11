"""Run a golden query set against an index and report averaged retrieval metrics."""

import contextlib
import json
import os
import statistics
import subprocess
import tempfile
import time
from collections.abc import Iterator
from typing import Any

from ai_db.constants import DEFAULT_SKILL_DIRS, EVAL_GIT_TIMEOUT_S
from ai_db.errors import AiDbConfigError
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


@contextlib.contextmanager
def git_worktree(repo: str, dest: str, rev: str = "HEAD") -> Iterator[str]:
    """A detached git worktree of ``repo`` at ``rev`` in ``dest``, removed on exit.

    The eval corpus is the committed tree (no untracked or ignored local files), and it
    is a real git checkout, so diff-mode items can run ``git diff`` in it.
    """
    _git(repo, "worktree", "add", "--detach", "--quiet", dest, rev)
    try:
        yield dest
    finally:
        _git(repo, "worktree", "remove", "--force", dest)


def _git(repo: str, *args: str) -> None:
    try:
        subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True,
                       timeout=EVAL_GIT_TIMEOUT_S)
    except subprocess.CalledProcessError as exc:
        raise AiDbConfigError(
            f"git {' '.join(args)} failed in {repo}: {exc.stderr.decode(errors='replace').strip()}"
        ) from exc


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



def load_refs_golden(golden_path: str) -> list[dict[str, Any]]:
    """Golden set for ``--symbols`` / ``--callers``: ``{"name", "expected": [{"filepath", "line", "kind"?}]}``."""
    items: list[dict[str, Any]] = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item.get("name"), str) or not item["name"].strip():
                raise ValueError(f"{golden_path}:{lineno}: 'name' must be a non-empty string")
            expected = item.get("expected")
            if not isinstance(expected, list) or not expected:
                raise ValueError(f"{golden_path}:{lineno}: 'expected' must be a non-empty list")
            for exp in expected:
                if not isinstance(exp.get("filepath"), str) or not isinstance(exp.get("line"), int):
                    raise TypeError(f"{golden_path}:{lineno}: expected entries need filepath (str) and line (int)")
            items.append(item)
    return items


def _score_sets(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-averaged precision/recall over (filepath, line[, kind]) sets."""
    tp = sum(len(q["hits"]) for q in per_query)
    fp = sum(len(q["wrong"]) for q in per_query)
    fn = sum(len(q["missed"]) for q in per_query)
    return {
        "queries": len(per_query),
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "exact_queries": sum(1 for q in per_query if not q["wrong"] and not q["missed"]),
    }


def _diff(expected: set[tuple[Any, ...]], got: set[tuple[Any, ...]], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "hits": sorted(expected & got),
        "missed": sorted(expected - got),
        "wrong": sorted(got - expected),
    }


def run_symbols(golden_path: str, root: str, db: Any, sync: bool = True) -> dict[str, Any]:
    """Exact symbol lookup: each name must return exactly its definitions, with the right kind."""
    root = os.path.abspath(root)
    golden = load_refs_golden(golden_path)
    if sync:
        db.sync(root, project=EVAL_PROJECT, verbose=False)
    per_query = []
    for item in golden:
        hits = db.query_symbol(item["name"], project=EVAL_PROJECT)
        got = {(os.path.relpath(h["abs_path"], root), h["line"], h["symbol_type"]) for h in hits}
        expected = {(e["filepath"], e["line"], e.get("kind")) for e in item["expected"]}
        per_query.append(_diff(expected, got, item["name"]))
    return {"suite": "symbols", **_score_sets(per_query), "per_query": per_query}


def run_callers(golden_path: str, root: str, db: Any, sync: bool = True) -> dict[str, Any]:
    """Qualified caller lookup: ``Class.method`` must return exactly its call sites."""
    root = os.path.abspath(root)
    golden = load_refs_golden(golden_path)
    if sync:
        db.sync(root, project=EVAL_PROJECT, verbose=False)
    per_query = []
    for item in golden:
        res = db.query_callers(item["name"], relative_to=root, project=EVAL_PROJECT, top_k=1000)
        if res["truncated"]:
            raise AssertionError(f"callers {item['name']}: truncated at top_k=1000")
        got = {(r["file"], r["line"]) for r in res["results"]}
        expected = {(e["filepath"], e["line"]) for e in item["expected"]}
        per_query.append(_diff(expected, got, item["name"]))
    return {"suite": "callers", **_score_sets(per_query), "per_query": per_query}


def compare_refs_to_baseline(result: dict[str, Any], baseline: dict[str, Any],
                             tolerance: float = 0.0) -> str | None:
    """Precision and recall may not drop at all by default: these are exact lookups."""
    for key in ("precision", "recall"):
        if key not in baseline:
            raise ValueError(f"baseline has no '{key}' entry")
        drop = baseline[key] - result[key]
        if drop > tolerance:
            return f"{result['suite']} {key} regressed by {drop:.4f} (baseline {baseline[key]}, now {result[key]})"
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
    with contextlib.ExitStack() as stack:
        # `A..B` diff items: the index must be the tree at B, not the working tree, or
        # the diff's line spans land on unrelated chunks. One worktree + project per B.
        snapshots: dict[str, tuple[str, str]] = {}
        for item in golden:
            # Diff-mode items name the git ref to diff against; the other modes
            # have no such field and must not be handed a made-up one.
            if item["kind"] == "diff" and not item.get("since"):
                raise ValueError(
                    f"diff golden item {item['query']!r} needs a 'since' git ref")
            item_root, project, since = item.get("root", root), EVAL_PROJECT, item.get("since")
            if item["kind"] == "diff" and since and ".." in since:
                base, _, head = since.partition("..")
                if head not in snapshots:
                    tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="ai_db_eval_rev_"))
                    tree = stack.enter_context(git_worktree(root, os.path.join(tmp, "tree"), head))
                    snapshots[head] = (tree, f"{EVAL_PROJECT}@{head[:12]}")
                    db.sync(tree, project=snapshots[head][1], verbose=False)
                item_root, project = snapshots[head]
                since = base
            t0 = time.perf_counter()
            pack = db.investigate(item["query"], budget_tokens=budget_tokens, mode=item["kind"],
                                  project=project, since=since, root=item_root)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            ranked = [(os.path.relpath(e["filepath"], item_root), e["qualified_name"])
                      for e in pack["evidence"]]
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

    # route_skills auto-syncs from the default skill dirs on its first call, so
    # the eval DB (which is a fresh temp database) would only fill in as a side
    # effect of scoring the first prompt. Do it up front so the emptiness check
    # below is accurate rather than racing the router.
    if not router.db.get_skills():
        router.sync_skills(verbose=False)

    # Still nothing means the prompt set cannot be scored: every prompt would
    # miss, reporting as top1_accuracy 0.0 -- indistinguishable from a
    # genuinely broken router. A silent 0.0 looks like a result; fail with the
    # fix instead (TODO rule 1: no silent degradation).
    indexed = len(router.db.get_skills())
    if indexed == 0:
        raise AiDbConfigError(
            f"no skills found, so {golden_path} cannot be scored "
            f"({len(golden)} prompts would all miss). Point AI_DB_SKILL_DIRS at "
            f"the directories holding your SKILL.md files (colon-separated for "
            f"several), or install them under a default skill dir. Note that "
            f"AI_DB_SKILL_DIRS is read at import time, so it must be set in the "
            f"environment of the process. Searched: "
            f"{', '.join(DEFAULT_SKILL_DIRS)}"
        )

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
        "skills_indexed": indexed,
        "top1_accuracy": round(statistics.fmean(hits), 4) if hits else 0.0,
        "per_query": per_query,
    }
