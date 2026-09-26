"""Run the evaluation across every configuration the local machine can run.

Two orthogonal axes:

1. **Retrieval configuration** -- the thing the accuracy numbers depend on.
   lexical (no embedder) vs hybrid with the local embedder on GPU vs on CPU,
   vs the vec0 vector index, vs hybrid + a cross-encoder rerank. Rerank is
   query-time only, so it reuses the same index as the hybrid condition rather
   than re-embedding anything.

2. **Output mode** -- what an agent actually pays. `analyze` serialises a file
   five ways, `locate` three, `trace` three, and `investigate` has five
   retrieval modes. These are measured for tokens and latency, not accuracy:
   the same underlying evidence, rendered differently.

Run:  uv run python eval/matrix_runner.py --out eval/results/matrix.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_CONFIGS = os.path.join(REPO, "eval", "matrix_configs.py")

# Conditions that need an index built before they can be queried.
NEEDS_INDEX = ("lexical", "hybrid_cuda", "hybrid_cuda_vec0", "hybrid_cpu")
# hybrid_cuda_rerank reuses hybrid_cuda's index: rerank never re-embeds.
INDEX_SOURCE = {"hybrid_cuda_rerank": "hybrid_cuda"}


def _run(cmd: list[str], timeout: int = 7200) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          cwd=REPO, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def _ai_db(config: str, *args: str, timeout: int = 7200) -> tuple[int, str, str]:
    return _run([sys.executable, "-m", "ai_db.cli", "--config", config, *args],
                timeout=timeout)


def build_configs(out_dir: str) -> dict[str, dict[str, str]]:
    os.makedirs(out_dir, exist_ok=True)
    _run([sys.executable, MATRIX_CONFIGS, out_dir,
          os.path.join(out_dir, "db")], timeout=300)
    with open(os.path.join(out_dir, "index.json"), encoding="utf-8") as fh:
        return json.load(fh)


def sync_condition(name: str, cfg: str) -> dict[str, object]:
    """Index the repo under one config. Returns timing, not correctness."""
    source = INDEX_SOURCE.get(name, name)
    # Copy the source index when the condition only differs at query time.
    if source != name:
        import shutil
        shutil.copyfile(_CONFIGS[source]["db"], _CONFIGS[name]["db"])
        return {"indexed": False, "note": f"copied index from {source}",
                "seconds": 0.0}

    # Sync under the auto-detected project name so that `analyze` and `locate`
    # -- which take no --project flag and detect from cwd -- see the same rows.
    t0 = time.perf_counter()
    code, out, err = _ai_db(cfg, "sync", ".")
    elapsed = time.perf_counter() - t0
    if code != 0:
        return {"indexed": False, "error": (err or out)[-400:], "seconds": elapsed}
    return {"indexed": True, "seconds": round(elapsed, 1),
            "summary": (out or "").strip().splitlines()[-1] if out.strip() else ""}


def eval_retrieval(name: str, cfg: str) -> dict[str, object]:
    """recall@10 / MRR / nDCG over the code golden set."""
    t0 = time.perf_counter()
    code, out, err = _ai_db(cfg, "eval", "--golden", "eval/golden/ai_db.jsonl",
                            "--root", ".", "-k", "10")
    if code != 0:
        return {"error": (err or out)[-400:]}
    data = json.loads(out[out.index("{"):])
    data.pop("per_query", None)
    data["seconds"] = round(time.perf_counter() - t0, 1)
    return data


def eval_pack(name: str, cfg: str) -> dict[str, object]:
    """pack_recall over the pack golden set."""
    t0 = time.perf_counter()
    code, out, err = _ai_db(cfg, "eval", "--pack", "--golden",
                            "eval/golden/ai_db_pack.jsonl", "--root", ".",
                            "--budget", "8000")
    if code != 0:
        return {"error": (err or out)[-400:]}
    data = json.loads(out[out.index("{"):])
    data.pop("per_query", None)
    data["seconds"] = round(time.perf_counter() - t0, 1)
    return data


def eval_output_modes(name: str, cfg: str) -> dict[str, object]:
    """Token cost and latency of every output mode, per command.

    Accuracy is identical across formats by construction -- they render the same
    parsed evidence -- so only cost is measured here.
    """
    results: dict[str, object] = {}

    def timed(*args: str, timeout: int = 1800) -> dict[str, object]:
        t0 = time.perf_counter()
        code, out, err = _ai_db(cfg, *args, timeout=timeout)
        return {"exit": code, "ms": round((time.perf_counter() - t0) * 1000, 1),
                "chars": len(out), "error": (err or "")[-200:] if code else ""}

    results["analyze"] = {
        fmt: timed("analyze", "ai_db/analysis/investigate.py", "--format", fmt)
        for fmt in ("stub", "sexp", "outline", "prose", "json")
    }
    results["locate"] = {
        fmt: timed("locate", "how are search results ranked", "--format", fmt)
        for fmt in ("json", "stub", "sexp")
    }
    # NB: trace resolves bare symbol names only. "Class.method" and
    # "path/file.py:symbol" both return "No trace found" (see notes), so use a
    # bare name that actually has internal callees -- an entry with no outgoing
    # edges yields a one-node tree and measures nothing.
    results["trace"] = {
        fmt: timed("trace", "sync_skills", "--format", fmt)
        for fmt in ("tree", "json", "mermaid")
    }
    results["investigate"] = {
        mode: timed("investigate", "how are search results ranked", "--mode", mode)
        for mode in ("locate", "explain", "impact", "flow")
    }
    return results


_CONFIGS: dict[str, dict[str, str]] = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results", "matrix.json"))
    ap.add_argument("--work", default="/tmp/matrix")
    ap.add_argument("--only", default="", help="comma-separated condition names")
    args = ap.parse_args()

    global _CONFIGS
    _CONFIGS = build_configs(args.work)
    wanted = [c.strip() for c in args.only.split(",") if c.strip()] or list(_CONFIGS)

    report: dict[str, object] = {
        "generated": time.strftime("%Y-%m-%d"),
        "repo": REPO,
        "conditions": {},
    }
    for name in wanted:
        entry = _CONFIGS.get(name)
        if entry is None:
            print(f"skip unknown condition {name}", file=sys.stderr)
            continue
        cfg = entry["config"]
        print(f"[{name}] indexing...", flush=True)
        cond: dict[str, object] = {"config": cfg, "db": entry["db"]}
        cond["index"] = sync_condition(name, cfg)
        if name in NEEDS_INDEX and not cond["index"].get("indexed") \
                and "error" in cond["index"]:
            report["conditions"][name] = cond
            continue
        print(f"[{name}] retrieval eval...", flush=True)
        cond["retrieval"] = eval_retrieval(name, cfg)
        print(f"[{name}] pack eval...", flush=True)
        cond["pack"] = eval_pack(name, cfg)
        print(f"[{name}] output modes...", flush=True)
        cond["output_modes"] = eval_output_modes(name, cfg)
        report["conditions"][name] = cond

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
