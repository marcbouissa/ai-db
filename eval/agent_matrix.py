"""Matrix 5: agent-realistic configurations — config x transport, on held-out data.

The finding that motivates this matrix
--------------------------------------
Measured on this repo, same index, same query:

| transport | first call | subsequent calls |
|---|---|---|
| CLI subprocess, lexical | ~240 ms | ~240 ms, every call |
| CLI subprocess, hybrid | ~10.5 s | ~10.5 s, every call |
| persistent MCP | 8-286 ms | **0.2-1.4 ms** |

Transport dominates configuration. The hybrid index is not slow; *reloading the
embedding model once per `subprocess`* is. An agent holds an MCP session open, so
it pays that once. A CLI-only harness therefore does not add noise to the hybrid
comparison — it inverts it, making the highest-recall configuration look like the
slowest by three orders of magnitude.

Two guards, both of which exist because the alternative is a flattering lie:

- **Held-out scoring.** Config selection reads the `tune` half of the golden set;
  reporting reads `holdout`. Choosing a config to maximise recall@10 on the same
  40 queries that gate CI would manufacture the number being reported. See
  `eval/heldout.py`.
- **Empty-pack detection.** A pack with no `project` argument matches nothing and
  returns ~500 chars of empty JSON in ~8 ms. Scored naively that is a *fast, cheap,
  zero-recall success* — it improves latency and tokens while destroying recall.
  Every result is checked for emptiness and empty results are counted separately
  rather than averaged in.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))

from benchmark_runner import (
    REPO as _REPO,
)
from benchmark_runner import (
    count_tokens,
    cuda_available,
    ensure_index,
    index_counts,
    project_name,
    score,
    timed,
)
from heldout import HeldOutSplit
from mcp_client import McpError, PersistentMcpClient

SCRATCH = "/tmp/ai_db_bench"


def is_empty_pack(text: str) -> bool:
    """True if the output is a well-formed pack that found nothing."""
    try:
        pack = json.loads(text)
    except (ValueError, TypeError):
        return False
    if not isinstance(pack, dict) or "evidence" not in pack:
        return False
    return not pack.get("evidence") and not pack.get("entry_points")


def with_doc_weight(config_path: str, weight: float) -> str:
    """A sibling config differing only in `retrieval.doc_weight`.

    Written to a new file rather than mutating the shared one, so conditions
    cannot contaminate each other or the Matrix 3 configs.
    """
    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.setdefault("retrieval", {})["doc_weight"] = weight
    tag = f"w{str(weight).replace('.', '_')}"
    out = f"{os.path.splitext(config_path)[0]}_{tag}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    return out


def _truth(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "recall": [e["filepath"] for e in row.get("expected", [])],
        "symbol_recall": [e["symbol"] for e in row["expected"] if e.get("symbol")],
    }


def _record(query: str, text: str, first_ms: float | None,
            warm_ms: float | None, exit_code: int,
            truth: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    s = score(text, truth, _REPO)
    s["empty"] = is_empty_pack(text)
    return {
        "query": query,
        "first_ms": round(first_ms, 2) if first_ms is not None else None,
        "warm_ms": round(warm_ms, 2) if warm_ms is not None else None,
        "tokens": count_tokens(text) if text else 0,
        "chars": len(text),
        "exit": exit_code,
        "error": error,
        "score": s,
    }


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    tok = [r["tokens"] for r in records]
    lat = [r["warm_ms"] for r in records if r["warm_ms"] is not None]
    rec = [r["score"]["recall"] for r in records if r["score"]["recall"] is not None]
    sym = [r["score"]["symbol_recall"] for r in records
           if r["score"]["symbol_recall"] is not None]
    return {
        "n": len(records),
        "tokens_median": round(statistics.median(tok), 1) if tok else None,
        "tokens_total": sum(tok),
        "warm_ms": round(statistics.median(lat), 2) if lat else None,
        "first_ms": records[0]["first_ms"] if records else None,
        "recall": round(sum(rec) / len(rec), 4) if rec else None,
        "symbol_recall": round(sum(sym) / len(sym), 4) if sym else None,
        "empty_results": sum(1 for r in records if r["score"].get("empty")),
        "errors": sum(1 for r in records if r.get("error")),
    }


def measure_cli(config: str, split: HeldOutSplit, am: dict[str, Any],
                project: str, repeats: int) -> dict[str, Any]:
    """One process per call: what a shell-scripting agent pays."""
    out: dict[str, Any] = {"transport": "cli-subprocess"}
    for half, rows in (("tune", split.tune), ("holdout", split.holdout),
                       ("all", split.all)):
        records = []
        for row in rows:
            argv = ["investigate", row["query"], "--mode", am["mode"],
                    "--budget", str(am["budget_tokens"]), "--project", project]
            r = timed(config, argv, repeats)
            records.append(_record(row["query"], r["stdout"], r["ms_cold"],
                                   r["ms"], r["exit"], _truth(row),
                                   r["stderr"] or None))
        out[half] = records
        out.setdefault("summary", {})[half] = summarise(records)
    return out


def measure_mcp(config: str, split: HeldOutSplit, am: dict[str, Any],
                project: str, repeats: int) -> dict[str, Any]:
    """One process for every call: what an MCP-hosted agent pays."""
    out: dict[str, Any] = {"transport": "persistent-mcp"}
    try:
        client = PersistentMcpClient(config).start()
    except (McpError, OSError) as exc:
        return {"transport": "persistent-mcp", "error": str(exc)[:200],
                "summary": {h: {"n": 0, "error": "server did not start"}
                            for h in ("tune", "holdout", "all")}}
    try:
        out["handshake_ms"] = round(client.handshake_ms or 0.0, 1)
        for half, rows in (("tune", split.tune), ("holdout", split.holdout),
                           ("all", split.all)):
            records = []
            for row in rows:
                args = {"query": row["query"], "mode": am["mode"],
                        "budget_tokens": am["budget_tokens"], "project": project}
                first_ms: float | None = None
                samples: list[float] = []
                text = ""
                err: str | None = None
                for _ in range(max(1, repeats)):
                    try:
                        text, ms = client.call_tool("investigate", args)
                    except McpError as exc:
                        err = str(exc)[:200]
                        break
                    if first_ms is None:
                        first_ms = ms
                    samples.append(ms)
                records.append(_record(
                    row["query"], text, first_ms,
                    statistics.median(samples) if samples else None,
                    1 if err else 0, _truth(row), err))
            out[half] = records
            out.setdefault("summary", {})[half] = summarise(records)
    finally:
        # Always terminate: an orphaned server holds a database lock.
        client.close()
    return out


def measure_baseline(split: HeldOutSplit, am: dict[str, Any],
                     repeats: int, baseline_cfg: dict[str, Any]) -> dict[str, Any]:
    """The no-ai-db reference, scored on the same halves under the same conditions.

    Included so the config and transport comparison has something to be better
    than. Both shell baselines are measured, not just the naive one: reading whole
    files is what an agent does with `cat`, and reading matching lines is the
    strongest thing it can do without ai-db. Quoting only the first would make
    ai-db look better than the alternative it actually has to beat.
    """
    from benchmark_runner import baseline as raw_baseline

    out: dict[str, Any] = {"transport": "none (ripgrep)"}
    for kind in ("windows", "files"):
        for half, rows in (("tune", split.tune), ("holdout", split.holdout),
                           ("all", split.all)):
            records = []
            for row in rows:
                best_ms: float | None = None
                text = ""
                for _ in range(max(1, repeats)):
                    b = raw_baseline(row["query"], kind, baseline_cfg)
                    if best_ms is None:
                        best_ms = b["ms"]
                    text = b["stdout"]
                records.append(_record(row["query"], text, best_ms, best_ms,
                                       0, _truth(row)))
            out.setdefault(kind, {})[half] = records
            out.setdefault("summary", {}).setdefault(kind, {})[half] = summarise(records)
    return out


def run(agent_matrix: dict[str, Any], repeats: int,
        baseline_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    am = agent_matrix
    split = HeldOutSplit.from_golden(os.path.join(REPO, am["golden"]))
    project = project_name(REPO)

    cfg_dir = os.path.join(SCRATCH, "agent_cfg")
    db_dir = os.path.join(SCRATCH, "agent_db")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(REPO, "eval/matrix_configs.py"),
                    cfg_dir, db_dir], capture_output=True, text=True, cwd=REPO,
                   check=True)
    with open(os.path.join(cfg_dir, "index.json"), encoding="utf-8") as fh:
        index = json.load(fh)

    has_cuda = cuda_available()
    out: dict[str, Any] = {
        "golden": am["golden"],
        "mode": am["mode"],
        "budget_tokens": am["budget_tokens"],
        "split": split.describe(),
        "cuda_available": has_cuda,
        "rows": [],
    }
    if baseline_cfg:
        # Measured once: the shell baselines do not depend on the ai-db config,
        # so repeating them per condition would only add noise to the totals.
        print("  [baseline] ripgrep arms on the same halves", flush=True)
        out["baseline"] = measure_baseline(split, am, repeats, baseline_cfg)
        for kind in ("windows", "files"):
            h = out["baseline"]["summary"][kind]["holdout"]
            print(f"    baseline {kind:8s} holdout recall={h['recall']} "
                  f"tokens={h['tokens_median']} {h['warm_ms']}ms", flush=True)

    for cond in am["conditions"]:
        meta = index.get(cond)
        if meta is None:
            out["rows"].append({"condition": cond, "status": "skipped",
                                "reason": "not produced by matrix_configs.py"})
            continue
        if meta.get("device") == "cuda" and not has_cuda:
            out["rows"].append({"condition": cond, "status": "skipped",
                                "reason": "requires CUDA; none present on this host"})
            continue
        # A lexical config has no fusion weight; sweeping it would be a no-op
        # reported as if it were a datapoint.
        if meta.get("retrieval_mode") != "hybrid":
            weights = [am["doc_weights"][0]]
        else:
            weights = am["doc_weights"]

        for weight in weights:
            config = with_doc_weight(meta["config"], weight)
            build = ensure_index(config, f"{cond}@{weight}", am["sync_root"])
            if build.get("exit") not in (0, None):
                out["rows"].append({
                    "condition": cond, "doc_weight": weight, "status": "skipped",
                    "reason": f"index build failed: {str(build.get('stderr'))[:120]}"})
                continue
            counts = index_counts(config)
            cli = measure_cli(config, split, am, project, repeats)
            mcp = measure_mcp(config, split, am, project, repeats)
            out["rows"].append({
                "condition": cond, "doc_weight": weight, "status": "ok",
                "retrieval_mode": meta.get("retrieval_mode"),
                "vector_index": meta.get("vector_index"),
                "device": meta.get("device"), "dtype": meta.get("dtype"),
                "sync_root": am["sync_root"],
                "index_build_ms": build["ms"],
                "files": counts.get("files"), "chunks": counts.get("chunks"),
                "cli": cli, "mcp": mcp,
            })
            c, m = cli["summary"]["holdout"], mcp["summary"]["holdout"]
            print(f"  [{cond} w={weight}] holdout recall cli={c['recall']} "
                  f"mcp={m['recall']} | per-call cli {c['warm_ms']}ms "
                  f"mcp {m['warm_ms']}ms | empty {c['empty_results']}/"
                  f"{m['empty_results']}", flush=True)
    return out
