"""Context-cost benchmark: no ai-db vs ai-db, across representations and modes.

Answers one question with a controlled comparison: **for the same questions and
the same context budget, how much context does it cost to get the right answer
with ai-db, and which representation costs least?**

The design holds three things constant so the representation is the only variable:

  * the **question set** -- ``eval/golden/ai_db.jsonl`` (40 queries)
  * the **budget** -- ``BUDGET_CHARS``, the same figure the original raw-vs-ai-db
    comparison used, so the numbers are comparable with it
  * the **hit test** -- a query hits when the expected (file, symbol) pair is
    both visible in the context actually emitted

What varies is the row: the discovery mechanism (ripgrep, or ai-db lexical or
hybrid) and the representation (raw file text, sexp, stub, outline, prose, json,
or an investigate mode).

Two axes, because they are independent questions:

  * **representation** -- discovery fixed at ai-db lexical, varying the format.
    Accuracy should barely move; cost moves a lot.
  * **discovery** -- representation fixed at sexp, varying lexical vs hybrid.
    Accuracy moves; cost barely moves.

Character counts are the unit because that is what the raw baseline can be
measured in. ``ai-db``'s own internal counters are token-based; tokens here are
the repo's usual chars/4 approximation and are labelled as such.

Run:
    uv run python eval/context_benchmark.py --out eval/results/context_benchmark.json
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
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same figure as the original raw-vs-ai-db comparison, so results line up.
BUDGET_CHARS = 8000
CHARS_PER_TOKEN = 4
# Stop emitting once we are this far past budget: no representation that cannot
# answer inside 8,000 chars is going to answer inside 80,000.
MAX_BUDGET_OVERSHOOT = 10

STOP = frozenset(
    ["a", "an", "the", "of", "to", "in", "is", "are", "be", "for", "on", "by", "with", "from", "this", "that", "it", "its", "where", "what", "how", "which", "does", "do", "done", "when", "why", "who", "can", "should", "would", "there", "into", "and", "or", "as", "at", "if", "then", "else", "use", "used", "using"]
)


RG_EXCLUDES = ["-g", "!.venv", "-g", "!node_modules", "-g", "!.git",
               "-t", "py"]

#: The baseline every other row is compared against, per query.
RAW_CONDITION = "raw ripgrep + read (no ai-db)"


# --------------------------------------------------------------------------- util

def content_terms(q: str) -> list[str]:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", q)
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        tl = t.lower()
        if tl in STOP or len(tl) < 3 or tl in seen:
            continue
        seen.add(tl)
        out.append(t)
    return out


def load_golden() -> list[dict]:
    items = []
    with open(os.path.join(REPO, "eval/golden/ai_db.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(json.loads(line))
    return items


def expected_pairs(item: dict) -> list[tuple[str, str]]:
    return [(e["filepath"], e.get("symbol", e.get("name", ""))) for e in item["expected"]]


def symbol_visible(context: str, symbol: str) -> bool:
    """Is the symbol name literally present in the emitted text?

    Used when a single file is being read or emitted, so the file identity is
    already known and only the symbol has to be found. Requiring the filename
    too would score every raw-file read as a miss, because a file's content
    never contains its own name.
    """
    return not symbol or symbol in context


def pair_visible(context: str, filepath: str, symbol: str) -> bool:
    """Is this (file, symbol) pair present in a *multi-file* document?

    Used for investigate packs, where one payload covers many files and so has
    to say which file a symbol came from. The file test matches on the path
    tail so an absolute header and a repo-relative golden path both work.
    """
    base = os.path.basename(filepath)
    if base and base not in context:
        return False
    return symbol_visible(context, symbol)


# ------------------------------------------------------------------- conditions

def cond_raw(question: str, item: dict) -> dict:
    """No ai-db: ripgrep the query terms, rank files, read them whole.

    This is what an agent does with a bare shell -- the honest baseline.
    """
    terms = content_terms(question)
    if not terms:
        return {"hit": False, "chars": 0}
    pattern = "|".join(terms)
    try:
        out = subprocess.run(
            ["rg", "--no-messages", "--no-heading", "-n", "-i", "-e", pattern,
             *RG_EXCLUDES, REPO],
            capture_output=True, text=True, timeout=60, check=False).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"hit": False, "chars": 0, "error": str(exc)}

    weights: Counter[str] = Counter()
    for line in out.splitlines():
        fpath, _, text = line.partition(":")
        low = text.lower()
        for t in terms:
            if t.lower() in low:
                weights[fpath] += 1

    exp = expected_pairs(item)
    exp_files = {os.path.abspath(p) for p, _ in exp}
    ranked = [f for f, _ in weights.most_common()]
    ranked += [f for f in sorted(exp_files) if f not in ranked]
    ranked.sort(key=lambda f: (-weights.get(f, 0), f))

    used = 0
    tokens_read = 0
    ranked_seen: set[str] = set()
    seen_text: dict[str, str] = {}
    for fpath in ranked:
        try:
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        ranked_seen.add(fpath)
        seen_text[fpath] = text
        # Cost to answer: emit until the expected pair is actually visible,
        # unbounded. The budget is applied afterwards as a gate, not as a
        # truncation point -- truncating every condition at 8,000 made every
        # representation report the same median and hid exactly the differences
        # this benchmark exists to measure.
        used += len(text)
        tokens_read += len(text) // CHARS_PER_TOKEN
        for want_file, want_symbol in exp:
            if os.path.abspath(want_file) == fpath and \
                    symbol_visible(text, want_symbol):
                # A raw read compresses nothing: tokens_out == tokens_in.
                return {"chars_to_hit": used, "within_budget": used <= BUDGET_CHARS,
                        "hit_flag": True, "tokens_in": tokens_read,
                        "tokens_out": tokens_read}
    found = any(os.path.abspath(f) in ranked_seen
                and symbol_visible(seen_text.get(f, ""), sym)
                for f, sym in exp)
    return {"chars_to_hit": used, "within_budget": False, "hit_flag": found,
            "tokens_in": tokens_read, "tokens_out": tokens_read}


def _ranked_files(db, question: str, project: str, limit: int = 12) -> list[str]:
    """Files ai-db would surface for this question, best first.

    ``project`` must be passed explicitly: the index is stored under the
    auto-detected project name, and a query without it silently matches nothing.
    """
    try:
        hits = db.query(question, top_k=limit * 3, project=project)
    except Exception:  # noqa: BLE001 - a retrieval failure is a miss, not a crash
        return []
    ordered: list[str] = []
    for h in hits:
        p = h.get("abs_path") or h.get("file")
        if p and p not in ordered and os.path.exists(p):
            ordered.append(p)
        if len(ordered) >= limit:
            break
    return ordered


def cond_aidb_format(db, question: str, item: dict, fmt: str, project: str,
                     depth: str = "structure") -> dict:
    """ai-db discovery, then emit each candidate file in one representation.

    Files are emitted one at a time and the budget is enforced per file, so this
    is directly comparable with the raw baseline, which also stops at the budget.
    """
    exp = expected_pairs(item)
    used = 0
    emitted = 0
    seen_any = False
    tokens_in = 0        # tokens of the raw files the representation was built from
    tokens_out = 0       # tokens actually emitted
    for fpath in _ranked_files(db, question, project):
        try:
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                tokens_in += len(fh.read()) // CHARS_PER_TOKEN
        except OSError:
            pass
        cmd = [sys.executable, "-m", "ai_db.cli", "--config", db._cfg_path,
               "analyze", fpath, "--format", fmt, "--depth", depth]
        try:
            text = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=120, check=False).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        if not text.strip():
            continue
        seen_any = True
        used += len(text)
        tokens_out += len(text) // CHARS_PER_TOKEN
        for want_file, want_symbol in exp:
            if os.path.abspath(want_file) == os.path.abspath(fpath) and \
                    symbol_visible(text, want_symbol):
                return {"chars_to_hit": used, "within_budget": used <= BUDGET_CHARS,
                        "files": emitted + 1, "hit_flag": True,
                        "tokens_in": tokens_in, "tokens_out": tokens_out}
        emitted += 1
        if used > BUDGET_CHARS * MAX_BUDGET_OVERSHOOT:
            break   # already far outside any plausible budget; stop paying for more
    return {"chars_to_hit": used if seen_any else 0, "within_budget": False,
            "hit_flag": False, "tokens_in": tokens_in, "tokens_out": tokens_out}


def cond_investigate(db, question: str, item: dict, mode: str, project: str,
                     fmt: str = "json") -> dict:
    """An investigate pack as an agent receives it, in one of its output formats.

    The pack is rendered through the same formatter a caller would use, so a
    format's cost here is its real cost, not a re-render.
    """
    try:
        from ai_db.analyzer.formatters import format_pack
        pack = db.investigate(question, budget_tokens=BUDGET_CHARS // CHARS_PER_TOKEN,
                              mode=mode, project=project)
        text = format_pack(pack, fmt)
    except Exception:  # noqa: BLE001
        return {"chars_to_hit": 0, "within_budget": False, "hit_flag": False,
                "tokens_in": 0, "tokens_out": 0}
    # Measure the RENDERED output, not the pack's own token_count. The pack
    # sizes its evidence selection to the budget, and that count is identical
    # for every format -- so using it would report the same number for json and
    # stub, which is the opposite of what this benchmark is for. The pack's own
    # claim is kept alongside so the two can be compared.
    chars = len(text)
    hit = any(pair_visible(text, f, s) for f, s in expected_pairs(item))
    return {"chars_to_hit": chars if hit else 0,
            "within_budget": hit and chars <= BUDGET_CHARS,
            "hit_flag": hit,
            "tokens_in": None,
            "tokens_out": chars // CHARS_PER_TOKEN,
            "pack_claim_tokens": int(pack.get("token_count", 0))}


# ------------------------------------------------------------------------- driver

def summarise(rows: list[dict], raw_rows: list[dict] | None = None) -> dict:
    """Hit rate within budget, cost of the answers reached, and the token story.

    ``median_chars`` is over hits only -- including misses would drag every
    representation toward the corpus size and say nothing.

    ``vs_raw_paired`` is the median over queries where BOTH this condition and
    the raw baseline found the answer, of ``this / raw`` for that same query.
    A plain ratio of medians is not robust: the two medians can be taken over
    different question sets (raw found 39/40, outline 38/40), so a ratio of
    medians compares two different populations. Pairing removes that.
    """
    n = len(rows) or 1
    hits = [r for r in rows if r.get("within_budget")]
    found = [r for r in rows if r["chars_to_hit"] and (r.get("within_budget")
                                                      or r.get("hit_flag"))]
    out: dict[str, object] = {
        "queries": len(rows),
        "hits_in_budget": len(hits),
        "hit_rate": round(len(hits) / n, 4),
        "found": len(found),
        "found_rate": round(len(found) / n, 4),
        "median_chars": int(statistics.median([r["chars_to_hit"] for r in found])) if found else 0,
        "total_chars": sum(r["chars_to_hit"] for r in found),
    }
    out["median_tokens"] = out["median_chars"] // CHARS_PER_TOKEN
    out["total_tokens"] = out["total_chars"] // CHARS_PER_TOKEN

    if found:
        t_in = sum(r.get("tokens_in") or 0 for r in found)
        t_out = sum(r.get("tokens_out") or 0 for r in found)
        out["tokens_in_total"] = t_in
        out["tokens_out_total"] = t_out
        if t_in:
            out["token_saving"] = round(1 - t_out / t_in, 4)
        out["median_tokens_in"] = int(statistics.median(
            [r.get("tokens_in") or 0 for r in found]))
        out["median_tokens_out"] = int(statistics.median(
            [r.get("tokens_out") or 0 for r in found]))

    claims = [r["pack_claim_tokens"] for r in rows if r.get("pack_claim_tokens")]
    if claims:
        out["pack_claim_median_tokens"] = int(statistics.median(claims))
        if out["median_tokens_out"]:
            out["claim_vs_actual"] = round(
                statistics.median(claims) / max(1, out["median_tokens_out"]), 2)

    if raw_rows:
        raw_by_q = {r["query"]: r for r in raw_rows if r.get("hit_flag")}
        pairs = []
        for r in rows:
            if not (r.get("hit_flag") or r.get("within_budget")):
                continue
            raw = raw_by_q.get(r["query"])
            if raw and raw["chars_to_hit"]:
                pairs.append(r["chars_to_hit"] / raw["chars_to_hit"])
        if pairs:
            out["paired_n"] = len(pairs)
            out["vs_raw_paired"] = round(statistics.median(pairs), 4)
            out["vs_raw_paired_pct"] = round(statistics.median(pairs) * 100, 1)
    return out


def main() -> int:
    global BUDGET_CHARS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "eval/results/context_benchmark.json"))
    ap.add_argument("--config", default="/tmp/matrix/lexical.json")
    ap.add_argument("--limit", type=int, default=0, help="only the first N queries")
    ap.add_argument("--budget", type=int, default=BUDGET_CHARS, help="context budget in chars")
    args = ap.parse_args()

    BUDGET_CHARS = args.budget
    items = load_golden()
    if args.limit:
        items = items[:args.limit]

    os.environ["AI_DB_CONFIG"] = args.config
    from ai_db import VectorDB
    from ai_db.config import load_config

    cfg = load_config(args.config)
    results: dict[str, list[dict]] = {}

    def run(name: str, fn) -> None:
        t0 = time.perf_counter()
        rows = []
        for item in items:
            row = fn(item)
            row["query"] = item["query"]
            rows.append(row)
        results[name] = rows
        s = summarise(rows, results.get(RAW_CONDITION))
        pair = (f"{s['vs_raw_paired_pct']:6.1f}%" if "vs_raw_paired_pct" in s
                else "     -")
        save = (f"{s['token_saving'] * 100:5.1f}%" if "token_saving" in s
                else "    -")
        print(f"  {name:32s} in-budget {s['hits_in_budget']:2d}/{s['queries']} "
              f"({s['hit_rate'] * 100:4.1f}%)  median {s['median_chars']:7,d}ch  "
              f"tok {s.get('median_tokens_out', 0):6d}  paired {pair}  "
              f"saving {save}  [{time.perf_counter() - t0:5.1f}s]", flush=True)

    print(f"budget {BUDGET_CHARS} chars "
          f"(~{BUDGET_CHARS // CHARS_PER_TOKEN} tokens); "
          f"baseline: no ai-db (ripgrep + read)")
    run("raw ripgrep + read (no ai-db)", lambda it: cond_raw(it["query"], it))

    db = VectorDB(cfg.storage.options["path"], config=cfg)
    db._cfg_path = args.config  # subprocesses need the same config
    from ai_db.utils import detect_project_name
    project = detect_project_name(REPO)
    print(f"ai-db conditions: config={args.config} project={project}")
    try:
        for fmt in ("sexp", "stub", "outline", "prose", "json"):
            run(f"ai-db lexical | {fmt}",
                lambda it, f=fmt: cond_aidb_format(db, it["query"], it, f, project))
        run("ai-db lexical | summary depth",
            lambda it: cond_aidb_format(db, it["query"], it, "stub", project, "summary"))
        run("ai-db lexical | structure depth",
            lambda it: cond_aidb_format(db, it["query"], it, "stub", project, "structure"))
        for mode in ("locate", "explain", "impact", "flow"):
            run(f"ai-db investigate --mode {mode}",
                lambda it, m=mode: cond_investigate(db, it["query"], it, m, project))
        for fmt in ("compact", "stub"):
            run(f"ai-db investigate --mode explain --format {fmt}",
                lambda it, f=fmt: cond_investigate(db, it["query"], it, "explain",
                                                    project, f))
    finally:
        db.close()

    report = {
        "generated": time.strftime("%Y-%m-%d"),
        "repo": REPO,
        "budget_chars": BUDGET_CHARS,
        "chars_per_token": CHARS_PER_TOKEN,
        "golden": "eval/golden/ai_db.jsonl",
        "config": args.config,
        "summary": {name: summarise(
            rows, results.get(RAW_CONDITION)) for name, rows in results.items()},
        "per_query": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
