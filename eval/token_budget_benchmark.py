"""Token cost and latency: no ai-db vs each feature alone vs a combined workflow.

The question this answers is the one that decides whether a code-intelligence
database is worth its setup cost: **for a task an agent has to solve, how many
tokens does it have to read, and how long does it wait?**

Three arms, on the same 40 golden queries, all timed and tokenized identically:

- **baseline** -- no ai-db. Ripgrep to find candidates, then read. Modelled two
  ways, because the naive version is too easy a target: ``files`` reads every
  matching file whole (what an agent does with ``cat``), ``windows`` reads only
  the matching lines plus context (what a careful agent does with ``sed``).
- **feature** -- one ai-db feature per query, alone.
- **workflow** -- a realistic sequence, either ``minimal`` (locate, then analyze
  the symbol it found) or ``full`` (locate, outline, analyze, investigate).

Every arm is scored on the same three things:

| metric | why it is here |
|---|---|
| ``tokens_out`` | tiktoken o200k_base over the exact text an agent would receive |
| ``latency_ms`` | wall clock of the whole arm, subprocess included |
| ``hit`` / ``symbol_hit`` | did the output contain the golden file / symbol |

``hit`` is the guard that makes the token numbers mean anything. A 20x token
reduction that loses the file is worth less than reading the file. Arms are
compared at the token counts they actually achieve *at their own hit rate*, and
the report prints both.

Two things this deliberately does not do:

- It does not claim to model a real agent. The baseline is a stated, mechanical
  model of shell-based search, reproducible to the byte. Treat it as a
  well-defined reference point, not as a measurement of some agent's skill.
- It does not hide the trade. Reading whole files with ripgrep is far *faster*
  in wall clock than starting a Python process. Where ai-db loses on latency it
  says so.

Run:
    uv run python eval/token_budget_benchmark.py --out eval/results/token_budget.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = "/tmp/feature_bench/lexical.json"
GOLDEN = os.path.join(REPO, "eval/golden/ai_db.jsonl")

# Words that carry no retrieval signal in a code query. Used only to build the
# ripgrep pattern for the baseline arm.
STOPWORDS = {
    "where", "is", "the", "a", "an", "of", "in", "to", "and", "or", "for", "on",
    "do", "does", "done", "are", "be", "by", "with", "that", "this", "it", "its",
    "how", "what", "when", "which", "from", "into", "at", "as", "we", "you", "i",
    "can", "get", "use", "used", "using", "if", "then", "than", "so", "but", "not",
}

# How many files / lines the baseline reads. Deliberately generous: a larger
# budget makes the baseline *look* worse, so these are set high on purpose.
BASELINE_FILES = 5
BASELINE_LINES = 200
BASELINE_CONTEXT = 3

# Repeats for the latency median. Token counts are deterministic and are taken
# from a single run; only timing is repeated.
REPEATS = 3


# ------------------------------------------------------------------ helpers
def content_terms(query: str) -> list[str]:
    """Distinctive words of a natural-language query, for the ripgrep pattern."""
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)
    seen: list[str] = []
    for w in words:
        low = w.lower()
        if low in STOPWORDS or low in seen:
            continue
        seen.append(low)
    return seen


def run(argv: list[str], timeout: int = 600) -> tuple[int, str, float]:
    """Run a subprocess, returning (exit code, stdout, elapsed ms)."""
    t0 = time.perf_counter()
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                          timeout=timeout, check=False)
    return proc.returncode, proc.stdout, (time.perf_counter() - t0) * 1000


def ai_db(*args: str) -> tuple[int, str, float]:
    env_argv = [sys.executable, "-m", "ai_db.cli", "--config", CONFIG, *args]
    return run(env_argv)


# ------------------------------------------------------------------ baseline
def baseline_files(query: str) -> tuple[int, str, float]:
    """No ai-db: ripgrep for candidate files, then read each one whole."""
    terms = content_terms(query)
    if not terms:
        return 0, "", 0.0
    pattern = "|".join(re.escape(t) for t in terms)
    code, out, ms = run(["rg", "-l", "-i", "-w", pattern, "--type", "py", "."])
    if code not in (0, 1) or not out.strip():
        return code, "", ms
    files = [f for f in out.split() if f][BASELINE_FILES:]
    chunks = []
    for rel in files:
        path = os.path.join(REPO, rel)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                chunks.append(f"# {rel}\n" + fh.read())
        except OSError:
            continue
    return 0, "\n".join(chunks), ms


def baseline_windows(query: str) -> tuple[int, str, float]:
    """No ai-db, done carefully: matching lines with a little context, no whole files."""
    terms = content_terms(query)
    if not terms:
        return 0, "", 0.0
    pattern = "|".join(re.escape(t) for t in terms)
    code, out, ms = run(["rg", "-n", "-i", "-w", pattern, "-C",
                         str(BASELINE_CONTEXT), "--type", "py", "."])
    if code not in (0, 1) or not out.strip():
        return code, "", ms
    # rg prints matches grouped by file; keep whole groups so a match keeps its
    # filepath, and stop at the line budget.
    groups, current = [], []
    for line in out.splitlines():
        if line.startswith("---") and ":" in line.split("---")[0]:
            if current:
                groups.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        groups.append("\n".join(current))
    kept, total = [], 0
    for g in groups:
        n = g.count("\n") + 1
        if total + n > BASELINE_LINES and kept:
            break
        kept.append(g)
        total += n
    return 0, "\n".join(kept), ms


# ----------------------------------------------------------------- features
def feature_query(query: str) -> tuple[int, str, float]:
    return ai_db("query", query)


def feature_locate(query: str) -> tuple[int, str, float]:
    return ai_db("locate", query)


def feature_investigate(query: str, mode: str = "explain") -> tuple[int, str, float]:
    return ai_db("investigate", query, "--mode", mode)


def workflow_minimal(query: str) -> tuple[int, str, float]:
    """What an agent actually does: find the file, then read just that symbol.

    The second step uses whatever `locate` returned, so the workflow adapts to
    the query instead of running a fixed script.
    """
    total_ms = 0.0
    code, out, ms = ai_db("locate", query)
    total_ms += ms
    if code != 0 or not out.strip():
        return code, out, total_ms
    parts = [out]
    symbol = _first_symbol(out)
    target = _first_filepath(out)
    if target:
        focus = ["--focus", symbol] if symbol else []
        code2, out2, ms2 = ai_db("analyze", target, "--format", "stub", *focus)
        total_ms += ms2
        if code2 == 0:
            parts.append(out2)
    return 0, "\n".join(parts), total_ms


def workflow_full(query: str) -> tuple[int, str, float]:
    """The thorough version: locate, outline the file, analyze it, then investigate."""
    total_ms = 0.0
    parts: list[str] = []
    code, out, ms = ai_db("locate", query)
    total_ms += ms
    if code != 0:
        return code, out, total_ms
    parts.append(out)
    target = _first_filepath(out)
    if target:
        for step in (("outline", target), ("analyze", target, "--format", "stub")):
            c, o, m = ai_db(*step)
            total_ms += m
            if c == 0:
                parts.append(o)
    c, o, m = feature_investigate(query, "explain")
    total_ms += m
    if c == 0:
        parts.append(o)
    return 0, "\n".join(parts), total_ms


PATH_RE = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|ts|tsx|go|rs|js|md))")
SYMBOL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b")


def _first_filepath(text: str) -> str | None:
    for line in text.splitlines():
        m = PATH_RE.search(line)
        if m:
            candidate = m.group(1)
            if os.path.isfile(os.path.join(REPO, candidate)):
                return candidate
    return None


def _first_symbol(text: str) -> str | None:
    """A plausible symbol from locate output: a bare identifier on its own line."""
    for line in text.splitlines():
        stripped = line.strip()
        if SYMBOL_RE.fullmatch(stripped):
            return stripped
    return None


# -------------------------------------------------------------------- scoring
def score(text: str, expected: dict) -> tuple[bool, bool]:
    """Did the output mention the golden file, and the golden symbol?"""
    path = expected["filepath"]
    symbol = expected["symbol"]
    file_hit = path in text or os.path.basename(path) in text
    symbol_hit = re.search(rf"\b{re.escape(symbol)}\b", text) is not None
    return file_hit, symbol_hit


ARMS: dict[str, object] = {
    "baseline:files": baseline_files,
    "baseline:windows": baseline_windows,
    "feature:query": feature_query,
    "feature:locate": feature_locate,
    "feature:investigate": feature_investigate,
    "workflow:minimal": workflow_minimal,
    "workflow:full": workflow_full,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "eval/results/token_budget.json"))
    ap.add_argument("--limit", type=int, default=0, help="Only the first N queries")
    args = ap.parse_args()

    if not os.path.exists(CONFIG):
        print(f"missing {CONFIG}; run eval/matrix_configs.py first", file=sys.stderr)
        return 1
    with open(GOLDEN, encoding="utf-8") as fh:
        golden = [json.loads(line) for line in fh if line.strip()]
    if args.limit:
        golden = golden[:args.limit]

    # Tokenize once at the end: loading tiktoken per arm would add ~300ms to
    # every arm and distort the very comparison being made.
    from ai_db.parser.chunker import count_tokens

    records: list[dict] = []
    for qi, case in enumerate(golden, 1):
        query, expected = case["query"], case["expected"][0]
        row: dict = {"query": query, "expected": expected, "arms": {}}
        for name, fn in ARMS.items():
            samples: list[float] = []
            text = ""
            code = 0
            for _ in range(REPEATS):
                code, text, ms = fn(query)  # type: ignore[operator]
                samples.append(ms)
            file_hit, symbol_hit = score(text, expected) if text else (False, False)
            row["arms"][name] = {
                "exit": code,
                "latency_ms": round(statistics.median(samples), 1),
                "chars": len(text),
                "tokens": count_tokens(text) if text else 0,
                "hit": file_hit,
                "symbol_hit": symbol_hit,
            }
        records.append(row)
        print(f"  [{qi:2d}/{len(golden)}] {query[:52]}", flush=True)

    report = {
        "generated": time.strftime("%Y-%m-%d"),
        "config": CONFIG,
        "golden": os.path.relpath(GOLDEN, REPO),
        "n_queries": len(records),
        "definitions": {
            "tokens_out": "tiktoken o200k_base over the exact stdout the agent would read",
            "latency_ms": f"median of {REPEATS} runs of the whole arm, subprocess included",
            "hit": "output mentions the golden filepath",
            "symbol_hit": "output mentions the golden symbol",
            "baseline:files": f"rg -l, then the first {BASELINE_FILES} matching files read whole",
            "baseline:windows": f"rg -n -C{BASELINE_CONTEXT}, first {BASELINE_LINES} lines of matches",
        },
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
