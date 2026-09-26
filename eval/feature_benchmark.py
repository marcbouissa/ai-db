"""Exhaustive per-feature latency benchmark: cold vs prewarmed.

Measures every user-facing feature of ai-db and answers one question for each:
**what does it cost the first time, and what does it cost once warm?**

Definitions, because "cold" and "warm" are ambiguous if you do not pin them down:

- **cold** -- the first subprocess invocation. Fresh interpreter, cold OS page
  cache for the code, and (for the hybrid configs) no loaded embedding model or
  CUDA context yet. This is what a user waits the very first time.
- **warm** -- median of N further subprocess invocations. The OS page cache and
  the query-result cache are populated; the interpreter still starts fresh. This
  is steady-state CLI latency, which is what an agent loop pays per call.
- **in-process** -- the same call made repeatedly inside one process, so
  interpreter startup and model loading are excluded. Reported only for the
  hot paths, where the gap between `warm` and `in-process` is the interesting
  part: it is what the feature itself costs versus what the process costs.

Everything is timed end to end as ``python -m ai_db.cli ...`` because that is
what a user or an agent actually runs. Destructive features run against a scratch
copy of the index, never the real one.

Run:
    uv run python eval/feature_benchmark.py --out eval/results/feature_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))

LEXICAL_CONFIG = "/tmp/feature_bench/lexical.json"
HYBRID_CONFIG = "/tmp/feature_bench/hybrid_cuda.json"
SKILL_DIRS = os.path.join(REPO, "tests/fixtures/skills")

# name -> (category, config, argv, extra env, safety)
#   safety: "ro" read-only, "rw" writes, "destructive" run on a scratch copy only
PROBES: list[tuple[str, str, str, list[str], dict[str, str], str]] = []


def probe(name: str, category: str, argv: list[str], config: str = LEXICAL_CONFIG,
          env: dict[str, str] | None = None, safety: str = "ro") -> None:
    PROBES.append((name, category, config, argv, env or {}, safety))


def make_ref() -> str:
    """Mint a ref handle for the in-process `expand` measurement.

    Ref handles live in an in-memory ``ReferenceStore`` (see
    ``ai_db/analyzer/references.py``), so they cannot cross a process boundary:
    a handle minted by one subprocess is unknown to the next. This is why
    `expand` is measured in-process and is absent from the CLI probes.
    """
    import re

    from ai_db import VectorDB

    cfg = load_config_for(LEXICAL_CONFIG)
    db = VectorDB(cfg.storage.options["path"], config=cfg)
    out = db.analyze_file("ai_db/device.py", no_cache=True)
    text = VectorDB.format_as_stub(out)
    match = re.search(r"ref:[0-9a-f]+", text)
    db.close()
    return match.group(0) if match else "ref:unknown"


def load_config_for(path: str):
    from ai_db.config import load_config
    return load_config(path)


# ---------------------------------------------------------------- indexing
probe("sync", "indexing", ["sync", "."], safety="rw")
probe("sync-all", "indexing", ["sync-all"], safety="rw")
probe("query", "retrieval", ["query", "where is bm25 ranking of chunks done"])
probe("query (hybrid)", "retrieval", ["query", "where is bm25 ranking of chunks done"],
      config=HYBRID_CONFIG)
probe("locate", "retrieval", ["locate", "how are search results ranked"])
probe("symbol", "retrieval", ["symbol", "resolve_dtype"])
probe("check <file>", "analysis", ["check", "ai_db/device.py"])
probe("check --index", "analysis", ["check", "--index"])
probe("lint <file>", "analysis", ["lint", "ai_db/device.py"])
probe("outline", "analysis", ["outline", "ai_db/analysis/pack.py"])
probe("analyze", "analysis", ["analyze", "ai_db/device.py", "--format", "stub"])
probe("analyze --fmt json", "analysis", ["analyze", "ai_db/device.py", "--format", "json"])
probe("analyze --fmt sexp", "analysis", ["analyze", "ai_db/device.py", "--format", "sexp"])
probe("analyze --fmt outline", "analysis", ["analyze", "ai_db/device.py", "--format", "outline"])
probe("analyze --fmt prose", "analysis", ["analyze", "ai_db/device.py", "--format", "prose"])
probe("analyze --depth summary", "analysis",
      ["analyze", "ai_db/device.py", "--format", "stub", "--depth", "summary"])
probe("analyze --depth full", "analysis",
      ["analyze", "ai_db/device.py", "--format", "stub", "--depth", "full"])
# `expand` is deliberately absent from the subprocess probes: ref handles live
# in an in-memory ReferenceStore, so a handle minted by one process does not
# exist in the next. It is measured in-process instead, below.
probe("callers", "analysis", ["callers", "resolve_dtype"])
probe("todos", "analysis", ["todos"])
probe("diff", "analysis", ["diff", "ai_db/device.py", "--since", "last"])
probe("trace", "analysis", ["trace", "sync_skills"])
probe("trace --format mermaid", "analysis", ["trace", "sync_skills", "--format", "mermaid"])
for _mode in ("locate", "explain", "impact", "flow"):
    probe(f"investigate --mode {_mode}", "analysis",
          ["investigate", "how are search results ranked", "--mode", _mode])
probe("investigate --format compact", "analysis",
      ["investigate", "how are search results ranked", "--format", "compact"])
probe("investigate --format stub", "analysis",
      ["investigate", "how are search results ranked", "--format", "stub"])
probe("investigate --format sexp", "analysis",
      ["investigate", "how are search results ranked", "--format", "sexp"])
probe("investigate (hybrid)", "analysis",
      ["investigate", "how are search results ranked"],
      config=HYBRID_CONFIG)

# ------------------------------------------------------------------ memory
probe("remember", "memory", ["remember", "bench-session"], safety="rw")
probe("context save", "memory", ["context", "save", "bench-session",
                                 "--summary", "a note from the benchmark"],
      safety="rw")
probe("context recall", "memory", ["context", "query", "benchmark"])

# ------------------------------------------------------------------ skills
probe("sync-skills", "skills", ["sync-skills", "--dir", SKILL_DIRS], safety="rw")
probe("route-skill", "skills", ["route-skill", "check for broken python files"])
probe("context list", "memory", ["context", "list"])

# ------------------------------------------------------------- maintenance
probe("status", "maintenance", ["status"])
probe("telemetry", "maintenance", ["telemetry"])
probe("log --slow", "maintenance", ["log", "--slow", "0"])
probe("prune", "maintenance", ["prune"], safety="destructive")
probe("optimize", "maintenance", ["optimize", "--default-format", "stub"],
      safety="destructive")
probe("vacuum", "maintenance", ["vacuum"], safety="destructive")
probe("reindex --embeddings", "maintenance", ["reindex", "--embeddings"],
      config=HYBRID_CONFIG, safety="destructive")

# -------------------------------------------------------------------- eval
probe("eval (retrieval, 10 queries)", "eval",
      ["eval", "--golden", "eval/golden/ai_db.jsonl", "--root", "."])
probe("eval --pack", "eval",
      ["eval", "--pack", "--golden", "eval/golden/ai_db_pack.jsonl", "--root", "."])

# ------------------------------------------------------------------ config
probe("config show", "config", ["config", "show"])
probe("config check", "config", ["config", "check"], config=HYBRID_CONFIG)


def run_once(config: str, argv: list[str], env_extra: dict[str, str],
             cwd: str = REPO) -> tuple[int, float, str]:
    env = dict(os.environ)
    env["AI_DB_CONFIG"] = config
    env.update(env_extra)
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "ai_db.cli", "--config", config, *argv],
                          capture_output=True, text=True, cwd=cwd, env=env,
                          timeout=900, check=False)
    return proc.returncode, (time.perf_counter() - t0) * 1000, proc.stderr


def scratch_config(src: str, db_name: str) -> str:
    """A config pointing at a copy of the index, for destructive features."""
    out_dir = os.path.dirname(src)
    with open(src, encoding="utf-8") as fh:
        src_cfg = json.load(fh)
    src_db = src_cfg["storage"]["options"]["path"]
    dst_db = os.path.join(out_dir, f"scratch_{db_name}.db")
    shutil.copyfile(src_db, dst_db)
    cfg = json.loads(json.dumps(src_cfg))
    cfg["storage"]["options"]["path"] = dst_db
    path = os.path.join(out_dir, f"scratch_{db_name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "eval/results/feature_benchmark.json"))
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    for cfg in (LEXICAL_CONFIG, HYBRID_CONFIG):
        if not os.path.exists(cfg):
            print(f"missing {cfg}; run eval/matrix_configs.py first", file=sys.stderr)
            return 1

    results = []
    for name, category, config, argv, env_extra, safety in PROBES:
        use_config = config
        if safety == "destructive":
            use_config = scratch_config(config, name.replace(" ", "_").replace("/", "_"))
        cold_code, cold_ms, cold_err = run_once(use_config, argv, env_extra)
        warm = []
        for _ in range(args.repeats):
            code, ms, _ = run_once(use_config, argv, env_extra)
            if code == 0:
                warm.append(ms)
        row = {
            "feature": name, "category": category, "argv": " ".join(argv),
            "config": os.path.basename(use_config), "safety": safety,
            "cold_ms": round(cold_ms, 1), "exit_cold": cold_code,
            "warm_ms": round(statistics.median(warm), 1) if warm else None,
            "warm_min_ms": round(min(warm), 1) if warm else None,
            "warm_samples": len(warm),
            "error": (cold_err or "").strip()[-200:] if cold_code != 0 else "",
        }
        if warm:
            row["init_overhead_ms"] = round(row["cold_ms"] - row["warm_ms"], 1)
            row["cold_over_warm"] = (round(row["cold_ms"] / row["warm_ms"], 2)
                                      if row["warm_ms"] else None)
        results.append(row)
        flag = "" if cold_code == 0 else f"  !! exit={cold_code} {cold_err.strip()[-90:]}"
        warm_txt = f"{row['warm_ms']:.1f}" if row["warm_ms"] is not None else "n/a"
        print(f"  {name:36s} cold {row['cold_ms']:9.1f}ms  "
              f"warm {warm_txt:>10s}ms{flag}", flush=True)

    # In-process comparison for the hot paths: separates interpreter startup
    # and model loading from the feature's own cost.
    inproc = inprocess_probe()
    for row in results:
        if row["feature"] in inproc:
            row["in_process_ms"] = inproc[row["feature"]]

    report = {
        "generated": time.strftime("%Y-%m-%d"),
        "repo": REPO,
        "definitions": {
            "cold_ms": "first subprocess run: fresh interpreter, cold page cache, "
                       "no loaded model or CUDA context",
            "warm_ms": f"median of {args.repeats} further subprocess runs: OS page "
                       "cache and query cache warm, interpreter still fresh",
            "in_process_ms": "median of repeated calls inside one process; excludes "
                             "interpreter startup and model loading",
            "init_overhead_ms": "cold_ms - warm_ms: the one-time cost",
        },
        "features": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


def inprocess_probe() -> dict[str, float]:
    """Repeated in-process calls for the hot paths."""
    import copy

    os.environ["AI_DB_CONFIG"] = LEXICAL_CONFIG
    from ai_db import VectorDB
    from ai_db.utils import detect_project_name

    cfg = load_config_for(LEXICAL_CONFIG)
    db = VectorDB(cfg.storage.options["path"], config=cfg)
    project = detect_project_name(REPO)
    out: dict[str, float] = {}

    def timed(label: str, fn, n: int = 20) -> None:
        fn()
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000)
        out[label] = round(statistics.median(samples), 2)

    timed("query", lambda: db.query("where is bm25 ranking of chunks done",
                                    top_k=10, project=project))
    timed("locate", lambda: db.query("how are search results ranked", top_k=10,
                                     project=project))
    timed("symbol", lambda: db.query_symbol("resolve_dtype", project=project))
    for mode in ("locate", "explain", "impact", "flow"):
        timed(f"investigate --mode {mode}",
              lambda m=mode: db.investigate("how are search results ranked",
                                            budget_tokens=8000, mode=m, project=project))
    data = copy.deepcopy(db.analyze_file("ai_db/device.py", no_cache=True))
    from ai_db.analyzer.formatters import annotate_formatted_tokens
    for fmt in ("stub", "json"):
        timed(f"analyze (fmt {fmt})",
              lambda f=fmt: annotate_formatted_tokens(copy.deepcopy(data), f))
    # expand needs a handle minted in this same process; reuse the data already
    # parsed above and re-mint per call so each sample does real work.
    import re as _re

    def _expand() -> None:
        local = copy.deepcopy(data)
        text = VectorDB.format_as_stub(local)
        handle = _re.search(r"ref:[0-9a-f]+", text)
        if handle:
            db.expand_ref(handle.group(0), depth="targeted")

    timed("expand", _expand, n=10)

    timed("status", lambda: db.backend.get_all_filepaths())
    db.close()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
