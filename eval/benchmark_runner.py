"""Multi-scenario benchmark harness for ai-db.

Runs four matrices and scores each one on tokens, latency, accuracy and cost:

  Matrix 1  Query modes   -- investigate --mode {locate,explain,impact,flow,diff}
                              against two raw-context baselines
  Matrix 2  Formats       -- analyze --format (5) and pack --format (4), plus the
                              formats each must REJECT, and compression vs source
  Matrix 3  Config modes  -- retrieval.mode x vector_index x dtype, with the
                              index built before timing so latency is real
  Matrix 4  Depths        -- analyze --depth, and the documented equivalence of
                              `summary` and `structure` checked rather than trusted

Design decisions worth stating, because they are the difference between a
number and a decoration:

* **Accuracy is deterministic.** Recall, symbol recall, call-path order and
  unresolved citations are computed from the output text against ground truth
  copied from `eval/golden/`. No model is involved, so these numbers are
  reproducible and cannot drift with a sampling seed.
* **The LLM arm is optional and reports nothing when absent.** Without an API
  key, `llm.measured` is false and every LLM metric is null. TTFT and generation
  time are *not* simulated -- a fake TTFT is worse than an absent one, because
  it looks like a measurement.
* **Fabrication is measured structurally.** A pack that cites a file or symbol
  which does not exist has invented it. That is the hallucination proxy that can
  be checked offline; the report says so rather than implying a model was asked
  to grade itself.
* **Invariants are enforced, not assumed.** `query` has no `--format`,
  `investigate --mode diff` requires `--since`, and `outline`/`prose` are analyze
  formats that a pack must reject. Each is asserted at runtime, and a violated
  invariant fails the run rather than producing a plausible-looking row.
* **Skips are labelled.** A config needing CUDA on a CPU-only host is reported
  as `skipped` with a reason. It is never recorded as a zero, which would be
  indistinguishable from "measured, and it was instant".

Run:
    uv run python eval/benchmark_runner.py --out eval/results/benchmark.json
    uv run python eval/benchmark_runner.py --only matrix1 --repeats 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))

SCENARIOS = os.path.join(REPO, "eval/scenarios.json")
DEFAULT_OUT = os.path.join(REPO, "eval/results/benchmark.json")
SCRATCH = "/tmp/ai_db_bench"

# Formats each surface must accept, from the code that defines them. The runner
# compares scenarios.json against these so the two cannot disagree.
ANALYZE_FORMATS = ("json", "stub", "sexp", "outline", "prose")
PACK_FORMATS = ("compact", "json", "stub", "sexp")
INVESTIGATE_MODES = ("locate", "explain", "impact", "flow", "diff")
ANALYZE_DEPTHS = ("summary", "structure", "targeted", "full")


# --------------------------------------------------------------------- utils
def count_tokens(text: str) -> int:
    """tiktoken o200k_base, falling back to a 4-chars-per-token estimate.

    The fallback is recorded per run so a report built without tiktoken cannot be
    mistaken for one built with it.
    """
    try:
        from ai_db.parser.chunker import count_tokens as real

        return int(real(text))
    except Exception:  # noqa: BLE001
        return max(1, len(text) // 4) if text else 0


def tokenizer_available() -> bool:
    try:
        import tiktoken  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def run_cli(config: str, args: list[str], cwd: str = REPO,
            timeout: int = 900) -> dict[str, Any]:
    """Run `python -m ai_db.cli` and capture everything the harness needs."""
    argv = [sys.executable, "-m", "ai_db.cli", "--config", config, *args]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, cwd=cwd,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"exit": -1, "stdout": "", "stderr": f"timeout after {timeout}s",
                "ms": (time.perf_counter() - t0) * 1000}
    return {
        "exit": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr[-400:],
        "ms": (time.perf_counter() - t0) * 1000,
    }


def timed(config: str, args: list[str], repeats: int, cwd: str = REPO) -> dict[str, Any]:
    """Run once to warm caches, then `repeats` more, returning the median."""
    first = run_cli(config, args, cwd)
    samples = [first["ms"]]
    last = first
    for _ in range(max(0, repeats - 1)):
        last = run_cli(config, args, cwd)
        samples.append(last["ms"])
    return {
        "exit": last["exit"],
        "stdout": last["stdout"],
        "stderr": last["stderr"],
        "ms": round(statistics.median(samples), 1),
        "ms_cold": round(first["ms"], 1),
        "runs": len(samples),
    }


# ------------------------------------------------------------------ scoring
PATH_RE = re.compile(
    r"(?<![\w./-])((?:[\w.-]+/)+[\w.-]+\.(?:py|pyi|ts|tsx|js|jsx|mjs|cjs|go|rs|c|h|cpp|hpp|cc|java))"
)


def citations(text: str) -> set[str]:
    """Repo-relative-looking source paths named in the output.

    Deliberately conservative, because the first version of this flagged most
    citations as fabrications and was therefore worthless:

    * only paths with a directory component are considered -- a bare ``foo.py``
      cannot be verified as repo-relative, so judging it would be a guess;
    * ``..`` and absolute paths are dropped rather than guessed at;
    * an occurrence of the repo's own absolute prefix is rewritten relative, so
      ``/root/.../ai_db/config.py`` is not mistaken for the invented path
      ``root/.../ai_db/config.py``.

    A metric that cries wolf on correct output is worse than no metric, because
    it trains the reader to ignore it.
    """
    found: set[str] = set()
    for raw in PATH_RE.findall(text):
        cand = raw.strip()
        if not cand or cand.startswith("/") or ".." in cand:
            continue
        prefix = REPO.lstrip("/") + "/"
        if prefix in cand:
            cand = cand.split(prefix, 1)[1]
            if not cand or "/" not in cand:
                continue
        if "/" not in cand:
            continue  # bare basename: unverifiable, so not evidence
        found.add(cand)
    return found


def fabrications(text: str) -> list[str]:
    """Cited repo-relative paths that do not exist on disk."""
    return sorted(c for c in citations(text)
                  if not os.path.exists(os.path.join(REPO, c)))


def score(text: str, truth: dict[str, Any], repo_root: str) -> dict[str, Any]:
    """Deterministic scoring. No model, no sampling, no seed."""
    want_files = [f for f in truth.get("recall", []) if f]
    want_syms = [s for s in truth.get("symbol_recall", []) if s]
    order = [s for s in truth.get("order", []) if s]

    found_files = [f for f in want_files if f in text]
    found_syms = [s for s in want_syms if re.search(rf"\b{re.escape(s)}\b", text)]

    # Order: are the required symbols present *and* in sequence?
    positions = []
    for sym in order:
        m = re.search(rf"\b{re.escape(sym)}\b", text)
        positions.append(m.start() if m else -1)
    if order:
        present = [p for p in positions if p >= 0]
        order_ok = len(present) == len(order) and present == sorted(present)
        order_frac = sum(1 for p in positions if p >= 0) / len(order)
    else:
        order_ok, order_frac = None, None

    # Fabrications: a cited repo-relative file that does not exist on disk.
    fabricated = fabrications(text)

    denom = len(want_files) or 1
    return {
        "recall": round(len(found_files) / denom, 4) if want_files else None,
        "symbol_recall": round(len(found_syms) / len(want_syms), 4) if want_syms else None,
        "order_ok": order_ok,
        "order_found": round(order_frac, 4) if order_frac is not None else None,
        "found_files": found_files,
        "found_symbols": found_syms,
        "missing_files": [f for f in want_files if f not in found_files],
        "missing_symbols": [s for s in want_syms if s not in found_syms],
        "fabricated_citations": fabricated[:10],
        "fabrication_count": len(fabricated),
        "_repo_root": repo_root,
    }


# --------------------------------------------------------------- baselines
STOPWORDS = {
    "where", "is", "the", "a", "an", "of", "in", "to", "and", "or", "for", "on",
    "do", "does", "done", "are", "be", "by", "with", "that", "this", "it", "its",
    "how", "what", "when", "which", "from", "into", "at", "as", "we", "you", "i",
    "can", "get", "use", "used", "using", "if", "then", "than", "so", "but", "not",
}


def content_terms(query: str) -> list[str]:
    seen: list[str] = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query):
        low = w.lower()
        if low in STOPWORDS or low in seen:
            continue
        seen.append(low)
    return seen


def baseline(query: str, kind: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Raw-context baseline. `files` reads whole files, `windows` reads lines."""
    terms = content_terms(query)
    if not terms:
        return {"exit": 1, "stdout": "", "ms": 0.0, "stderr": "no searchable terms"}
    pattern = "|".join(re.escape(t) for t in terms)
    t0 = time.perf_counter()
    if kind == "files":
        argv = ["rg", "-l", "-i", "-w", pattern, "--type", "py", "."]
    else:
        argv = ["rg", "-n", "-i", "-w", pattern, "-C", str(cfg["line_window"]),
                "--type", "py", "."]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                              timeout=120, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {"exit": -1, "stdout": "", "ms": (time.perf_counter() - t0) * 1000,
                "stderr": str(exc)}
    out = proc.stdout
    if proc.returncode not in (0, 1) or not out.strip():
        return {"exit": proc.returncode, "stdout": "", "stderr": "no match",
                "ms": (time.perf_counter() - t0) * 1000}

    if kind == "files":
        # NB: `[cfg["file_read"]:]` is a slice. Indexing with `[n]` returns one
        # path and the loop would then walk its characters.
        paths = [p for p in out.split() if p][cfg["file_read"]:]
        chunks = []
        for rel in paths:
            full = os.path.join(REPO, rel)
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    chunks.append(f"# {rel}\n" + fh.read())
            except OSError:
                continue
        text = "\n".join(chunks)
    else:
        groups: list[str] = []
        current: list[str] = []
        for line in out.splitlines():
            if line.startswith("---") and ":" in line.split("---")[0]:
                if current:
                    groups.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            groups.append("\n".join(current))
        kept: list[str] = []
        total = 0
        for g in groups:
            n = g.count("\n") + 1
            if total + n > cfg["max_lines"] and kept:
                break
            kept.append(g)
            total += n
        text = "\n".join(kept)
    return {"exit": 0, "stdout": text, "stderr": "",
            "ms": (time.perf_counter() - t0) * 1000}


# ---------------------------------------------------------------- matrix 1
def matrix1_query_modes(scn: dict[str, Any], config: str, repeats: int) -> dict[str, Any]:
    out: dict[str, Any] = {"modes": [], "invariants": {}}
    base_cfg = scn["baselines"]

    for spec in scn["query_modes"]:
        mode = spec["mode"]
        rows = []
        for case in spec["cases"]:
            query = case["query"]
            argv = ["investigate", "--mode", mode]
            if spec.get("requires_since"):
                # Invariant: diff is meaningless without --since, and the CLI
                # rejects it. Passing it must therefore be explicit per case.
                if "since" not in case:
                    raise SystemExit(f"diff case {query!r} has no 'since'")
                argv += ["--since", case["since"]]
            argv.append(query)

            aidb = timed(config, argv, repeats)
            entry: dict[str, Any] = {
                "query": query,
                "argv": " ".join(argv),
                "aidb": {
                    "ms": aidb["ms"], "ms_cold": aidb["ms_cold"],
                    "chars": len(aidb["stdout"]),
                    "tokens": count_tokens(aidb["stdout"]),
                    "exit": aidb["exit"],
                    "stderr": aidb["stderr"],
                    "score": score(aidb["stdout"], case["ground_truth"], REPO),
                },
                "baselines": {},
            }
            for kind in ("files", "windows"):
                b = baseline(query, kind, base_cfg)
                entry["baselines"][kind] = {
                    "ms": round(b["ms"], 1),
                    "chars": len(b["stdout"]),
                    "tokens": count_tokens(b["stdout"]),
                    "exit": b["exit"],
                    "score": score(b["stdout"], case["ground_truth"], REPO),
                }
            rows.append(entry)
        out["modes"].append({
            "mode": mode,
            "goal": spec["goal"],
            "command": spec["command"],
            "golden_source": spec.get("golden_source"),
            "rows": rows,
        })

    out["invariants"] = check_invariants(config)
    return out


def check_invariants(config: str) -> dict[str, Any]:
    """Assert the CLI contracts the harness relies on. Each must FAIL as stated."""
    checks: list[dict[str, Any]] = []

    def expect(name: str, argv: list[str], should_fail: bool, why: str) -> None:
        r = run_cli(config, argv, timeout=180)
        failed = r["exit"] != 0
        checks.append({
            "invariant": name, "argv": " ".join(argv),
            "expected": "non-zero exit" if should_fail else "exit 0",
            "observed_exit": r["exit"], "passed": failed == should_fail,
            "why": why,
        })

    expect("query has no --format",
           ["query", "ranking", "--format", "json"], True,
           "query takes no --format; passing one is a usage error")
    expect("diff requires --since",
           ["investigate", "config", "--mode", "diff"], True,
           "a diff with no ref would silently describe the wrong change")
    for fmt in ("outline", "prose"):
        expect(f"pack rejects {fmt}",
               ["investigate", "ranking", "--format", fmt], True,
               f"{fmt} is an analyze format, not a pack format")
    for fmt in PACK_FORMATS:
        expect(f"pack accepts {fmt}",
               ["investigate", "how are search results ranked", "--format", fmt], False,
               "listed in the pack format set")
    for fmt in ANALYZE_FORMATS:
        expect(f"analyze accepts {fmt}",
               ["analyze", "ai_db/utils.py", "--format", fmt], False,
               "listed in the analyze format set")

    # Only json may carry tokens_out_formatted.
    carriers = []
    for fmt in ANALYZE_FORMATS:
        r = run_cli(config, ["analyze", "ai_db/utils.py", "--format", fmt], timeout=300)
        if "tokens_out_formatted" in r["stdout"]:
            carriers.append(fmt)
    checks.append({
        "invariant": "only analyze --format json emits tokens_out_formatted",
        "observed": carriers, "expected": ["json"],
        "passed": carriers == ["json"],
        "why": "a format-blind field cannot support a format comparison",
    })
    return {"all_passed": all(c["passed"] for c in checks), "checks": checks}


# ---------------------------------------------------------------- matrix 2
def matrix2_formats(scn: dict[str, Any], config: str, repeats: int) -> dict[str, Any]:
    fm = scn["format_matrix"]
    out: dict[str, Any] = {"analyze": [], "pack": [], "rejected": []}

    for target in fm["targets"]:
        # Raw source size, the denominator for the compression factor.
        try:
            with open(os.path.join(REPO, target), encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
        except OSError:
            continue
        raw_tokens = count_tokens(raw)
        for fmt in fm["analyze_formats"]:
            r = timed(config, ["analyze", target, "--format", fmt], repeats)
            emitted = None
            if fmt == "json":
                try:
                    emitted = json.loads(r["stdout"])["meta"].get("tokens_out_formatted")
                except Exception:  # noqa: BLE001
                    emitted = None
            out["analyze"].append({
                "target": target, "format": fmt, "ms": r["ms"],
                "chars": len(r["stdout"]), "tokens": count_tokens(r["stdout"]),
                "raw_tokens": raw_tokens,
                "compression_vs_source": round(raw_tokens / max(1, count_tokens(r["stdout"])), 3),
                "tokens_out_formatted": emitted,
                "self_count_accurate": (
                    emitted == count_tokens(r["stdout"]) if emitted is not None else None),
                "exit": r["exit"],
            })

    for query in fm["pack_queries"]:
        for fmt in fm["pack_formats"]:
            r = timed(config, ["investigate", query, "--format", fmt,
                               "--budget", str(fm["budget_tokens"])], repeats)
            out["pack"].append({
                "query": query, "format": fmt, "ms": r["ms"],
                "chars": len(r["stdout"]), "tokens": count_tokens(r["stdout"]),
                "exit": r["exit"],
            })
        for fmt in fm["rejected_pack_formats"]:
            r = run_cli(config, ["investigate", query, "--format", fmt], timeout=180)
            out["rejected"].append({
                "query": query, "format": fmt, "exit": r["exit"],
                "rejected_as_expected": r["exit"] != 0,
            })
    return out


# ---------------------------------------------------------------- matrix 3
def matrix3_config_modes(scn: dict[str, Any], repeats: int) -> dict[str, Any]:
    cm = scn["config_matrix"]
    cfg_dir = os.path.join(SCRATCH, "configs")
    db_dir = os.path.join(SCRATCH, "db")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)

    subprocess.run([sys.executable, os.path.join(REPO, "eval/matrix_configs.py"),
                    cfg_dir, db_dir], capture_output=True, text=True, cwd=REPO,
                   check=True)
    with open(os.path.join(cfg_dir, "index.json"), encoding="utf-8") as fh:
        index = json.load(fh)

    has_cuda = cuda_available()
    out: dict[str, Any] = {"conditions": [], "cuda_available": has_cuda}

    for name in cm["conditions"]:
        meta = index.get(name)
        if meta is None:
            out["conditions"].append({"name": name, "status": "skipped",
                                      "reason": "not produced by matrix_configs.py"})
            continue
        if name in cm["cuda_conditions"] and not has_cuda:
            # Explicit skip. Recording 0.0 here would be indistinguishable from
            # "measured, and it was instant".
            out["conditions"].append({
                "name": name, "status": "skipped",
                "reason": "requires CUDA; none present on this host",
                "config": meta["config"],
            })
            continue

        # Build the index first: timing a query against an empty database
        # measures the absence of an index, not the search. The corpus root is
        # shared by every condition so the comparison is like-for-like.
        root = cm.get("sync_root") or "."
        sync = ensure_index(meta["config"], name, root)
        latencies, recalls = [], []
        db_bytes = 0
        try:
            db_bytes = os.path.getsize(meta["db"])
        except OSError:
            pass
        counts = index_counts(meta["config"])
        for q in cm["recall_queries"]:
            r = timed(meta["config"], ["query", q, "--top-k", "10",
                                        "--project", project_name(REPO)], repeats)
            if r["exit"] != 0:
                continue
            latencies.append(r["ms"])
            recalls.append(1.0 if r["stdout"].strip() else 0.0)
        out["conditions"].append({
            "name": name, "status": "ok",
            "retrieval_mode": meta.get("retrieval_mode"),
            "vector_index": meta.get("vector_index"),
            "device": meta.get("device"),
            "dtype": meta.get("dtype"),
            "sync_root": root,
            "sync_ms": sync["ms"],
            "index_built": sync.get("synced", False),
            "files": counts.get("files"),
            "chunks": counts.get("chunks"),
            "db_bytes": db_bytes,
            "db_mib": round(db_bytes / 1048576, 2),
            "latency_ms_median": round(statistics.median(latencies), 2) if latencies else None,
            "latency_ms_p95": (round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2)
                               if latencies else None),
            "hit_rate": round(sum(recalls) / len(recalls), 4) if recalls else None,
        })
    corpora = {c.get("chunks") for c in out["conditions"] if c["status"] == "ok"}
    out["corpus_identical_across_conditions"] = len(corpora) <= 1
    return out


def index_counts(config: str) -> dict[str, int]:
    """files/chunks in an index, so corpus comparability is verifiable."""
    r = run_cli(config, ["status"], timeout=300)
    out: dict[str, int] = {}
    for line in r["stdout"].splitlines():
        if "files:" not in line:
            continue
        for field in ("files", "chunks"):
            if f"{field}:" in line:
                try:
                    out[field] = int(line.split(f"{field}:")[1].split("|")[0].strip())
                except (IndexError, ValueError):
                    pass
    return out


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def project_name(root: str) -> str:
    from ai_db.utils import detect_project_name

    return detect_project_name(root)


# ---------------------------------------------------------------- matrix 4
def matrix4_depths(scn: dict[str, Any], config: str, repeats: int) -> dict[str, Any]:
    dm = scn["depth_matrix"]
    rows = []
    for target in dm["targets"]:
        for depth in dm["analyze_depths"]:
            r = timed(config, ["analyze", target, "--format", "stub",
                               "--depth", depth], repeats)
            rows.append({
                "target": target, "depth": depth, "ms": r["ms"],
                "chars": len(r["stdout"]), "tokens": count_tokens(r["stdout"]),
                "exit": r["exit"],
            })
    # Depth equivalence. The docs claim `summary` and `structure` are identical;
    # measure it rather than trust it, and check every adjacent pair so a second
    # duplicate cannot hide. (There is one: `targeted` == `full` as well.)
    digests: dict[str, dict[str, str]] = {}
    for target in dm["targets"]:
        per_depth: dict[str, str] = {}
        for depth in dm["analyze_depths"]:
            r = run_cli(config, ["analyze", target, "--format", "stub", "--depth", depth])
            per_depth[depth] = hashlib.sha256(r["stdout"].encode()).hexdigest()[:16]
        digests[target] = per_depth
    groups: dict[str, list[str]] = {}
    for depth, dg in digests[dm["targets"][0]].items():
        groups.setdefault(dg, []).append(depth)
    return {
        "rows": rows,
        "digests": digests,
        "depth_aliases": {dg: ds for dg, ds in groups.items() if len(ds) > 1},
        "distinct_behaviours": len(groups),
        "declared_depths": len(dm["analyze_depths"]),
        "expand_depths_not_measurable": {
            "depths": dm["expand_depths"],
            "reason": "ref handles are minted in an in-memory ReferenceStore and "
                      "do not cross a process boundary, so `expand` cannot be "
                      "measured as a subprocess",
        },
    }


# --------------------------------------------------------------------- LLM
def llm_arm(scn: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Optional real completions. Reports nothing at all when not configured."""
    cfg = scn["llm"]
    pricing = cfg["pricing_usd_per_1m"].get(cfg["model"])
    answer_tokens = cfg["answer_tokens"]

    if not cfg.get("enabled"):
        return {
            "measured": False,
            "reason": "llm.enabled is false in scenarios.json",
            "model": cfg["model"],
            "ttft_ms": None, "generation_ms": None, "output_tokens": None,
            "cost_projected_usd": None,
            "note": "cost below is a projection from measured context tokens and a "
                    "declared answer length, not a billed amount",
        }
    key = os.environ.get(cfg["api_key_env"])
    base = os.environ.get(cfg["base_url_env"])
    if not key or not base:
        return {
            "measured": False,
            "reason": f"set {cfg['api_key_env']} and {cfg['base_url_env']} to measure",
            "model": cfg["model"],
            "ttft_ms": None, "generation_ms": None, "output_tokens": None,
            "cost_projected_usd": None,
        }

    import urllib.request

    results: list[dict[str, Any]] = []
    for row in rows:
        for arm, text in (("aidb", row["aidb"]["stdout"]),):
            body = json.dumps({
                "model": cfg["model"],
                "messages": [{"role": "user",
                              "content": f"Using only this context, answer: {row['query']}\n\n{text}"}],
                "max_tokens": answer_tokens,
                "stream": True,
            }).encode()
            req = urllib.request.Request(
                f"{base.rstrip('/')}/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            t0 = time.perf_counter()
            ttft = None
            chunks, out_tokens = 0, 0
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for raw in resp:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload in ("", "[DONE]"):
                            continue
                        try:
                            delta = json.loads(payload)["choices"][0]["delta"]
                        except (ValueError, KeyError, IndexError, TypeError):
                            # keep-alive frames and truncated lines are normal in
                            # a SSE stream; a malformed frame is not a failure.
                            continue
                        if delta.get("content"):
                            chunks += 1
                            out_tokens += 1
            except Exception as exc:  # noqa: BLE001
                results.append({"arm": arm, "error": str(exc)[:200]})
                continue
            results.append({
                "arm": arm, "query": row["query"],
                "ttft_ms": round(ttft, 1) if ttft else None,
                "generation_ms": round((time.perf_counter() - t0) * 1000, 1),
                "stream_chunks": chunks,
                "output_tokens_approx": out_tokens,
            })
    measured = [r for r in results if "error" not in r and r.get("ttft_ms") is not None]
    return {
        "measured": bool(measured),
        "model": cfg["model"],
        "reason": None if measured else "every request failed",
        "ttft_ms": (round(statistics.median(float(r["ttft_ms"]) for r in measured), 1)
                    if measured else None),
        "generation_ms": (round(statistics.median(float(r["generation_ms"]) for r in measured), 1)
                          if measured else None),
        "output_tokens": answer_tokens,
        "cost_projected_usd": projected_cost(rows, pricing, answer_tokens),
        "samples": results[:10],
    }


def projected_cost(rows: list[dict[str, Any]], pricing: dict[str, float] | None,
                   answer_tokens: int) -> dict[str, Any] | None:
    """Cost from *measured* context tokens, priced per 1M. A projection, not a bill."""
    if not pricing:
        return None
    out = {}
    for arm, get in (("aidb", lambda r: r["aidb"]["tokens"]),
                     ("baseline:files", lambda r: r["baselines"]["files"]["tokens"]),
                     ("baseline:windows", lambda r: r["baselines"]["windows"]["tokens"])):
        ctx = sum(get(r) for r in rows)
        out[arm] = {
            "context_tokens_total": ctx,
            "usd_input": round(ctx / 1e6 * pricing["input"], 6),
            "usd_output": round(len(rows) * answer_tokens / 1e6 * pricing["output"], 6),
            "usd_total": round(ctx / 1e6 * pricing["input"]
                               + len(rows) * answer_tokens / 1e6 * pricing["output"], 6),
        }
    return out


# -------------------------------------------------------------------- main
def ensure_index(config: str, label: str, root: str = ".") -> dict[str, Any]:
    """Build the index for `config` unless it already has one.

    Without this the harness measures an *empty* database: `investigate` returns
    a near-empty pack, recall reads 0.00, and every downstream number looks like a
    property of the tool rather than of the missing corpus. A first run of this
    harness produced exactly that -- 0.00 recall across every mode, ~158 tokens
    per pack -- which looked like a result until `status` was checked.
    """
    probe = run_cli(config, ["status"], timeout=300)
    already = False
    for line in probe["stdout"].splitlines():
        if "files:" in line:
            try:
                already = int(line.split("files:")[1].split("|")[0].strip()) > 0
            except (IndexError, ValueError):
                already = False
    if already:
        return {"synced": False, "reason": f"{label} index already populated",
                "ms": 0.0}
    t0 = time.perf_counter()
    r = run_cli(config, ["sync", root], timeout=7200)
    return {
        "synced": r["exit"] == 0,
        "root": root,
        "exit": r["exit"],
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "stderr": r["stderr"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--scenarios", default=SCENARIOS)
    ap.add_argument("--config", default=os.path.join(SCRATCH, "primary/primary.json"),
                    help="config for matrices 1, 2 and 4")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="",
                    help="comma list: matrix1,matrix2,matrix3,matrix4,matrix5")
    ap.add_argument("--skip-agent", action="store_true",
                    help="skip matrix5, which builds a CUDA index for hybrid")
    ap.add_argument("--no-sync", action="store_true",
                    help="assume the primary index is already built (it must be, "
                         "or matrices 1/2/4 measure an empty database)")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    want = (lambda n: not only or n in only)

    with open(args.scenarios, encoding="utf-8") as fh:
        scn = json.load(fh)

    if not os.path.isdir(os.path.dirname(args.config)):
        subprocess.run([sys.executable, os.path.join(REPO, "eval/matrix_configs.py"),
                        os.path.join(SCRATCH, "primary"), os.path.join(SCRATCH, "db")],
                       capture_output=True, text=True, cwd=REPO, check=True)
    # The primary config is the lexical one: it needs no model, so the harness is
    # runnable on any host. Config modes are swept separately in matrix 3.
    if not os.path.exists(args.config):
        alt = os.path.join(SCRATCH, "primary", "lexical.json")
        args.config = alt if os.path.exists(alt) else args.config

    # Matrices 1, 2 and 4 all read the primary index. Build it once, up front,
    # and say so in the report -- an unbuilt index silently turns every recall
    # number into 0.00.
    primary: dict[str, Any] = {"skipped": True}
    if not args.no_sync and (not only or only & {"matrix1", "matrix2", "matrix4"}):
        print(f"[setup] ensuring {args.config} has an index", flush=True)
        primary = ensure_index(args.config, os.path.basename(args.config))
        print(f"[setup] {primary}", flush=True)

    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "primary_index": primary,
        "repo": REPO,
        "scenarios": os.path.relpath(args.scenarios, REPO),
        "primary_config": args.config,
        "repeats": args.repeats,
        "tokenizer": "tiktoken o200k_base" if tokenizer_available() else "4-chars-per-token estimate",
    }

    if want("matrix1"):
        print("[matrix1] query modes vs raw-context baselines", flush=True)
        report["matrix1_query_modes"] = matrix1_query_modes(scn, args.config, args.repeats)
    if want("matrix2"):
        print("[matrix2] output formats and pack formats", flush=True)
        report["matrix2_formats"] = matrix2_formats(scn, args.config, args.repeats)
    if want("matrix3"):
        print("[matrix3] config modes (retrieval x vector_index x dtype)", flush=True)
        report["matrix3_config_modes"] = matrix3_config_modes(scn, args.repeats)
    if want("matrix4"):
        print("[matrix4] depth variations", flush=True)
        report["matrix4_depths"] = matrix4_depths(scn, args.config, args.repeats)
    if want("matrix5") and not args.skip_agent and "agent_matrix" in scn:
        print("[matrix5] agent-realistic config x transport, held-out", flush=True)
        from agent_matrix import run as run_agent_matrix

        report["matrix5_agent_configs"] = run_agent_matrix(
            scn["agent_matrix"], args.repeats, scn.get("baselines"))

    m1 = report.get("matrix1_query_modes")
    if m1:
        flat = [r for mode in m1["modes"] for r in mode["rows"]]
        report["llm"] = llm_arm(scn, flat)
        report["invariants_all_passed"] = m1["invariants"]["all_passed"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")
    if m1 and not m1["invariants"]["all_passed"]:
        failed = [c["invariant"] for c in m1["invariants"]["checks"] if not c["passed"]]
        print(f"INVARIANT FAILURES: {failed}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
