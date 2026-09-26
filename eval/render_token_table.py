"""Render eval/results/token_budget.json as the token/latency comparison tables.

Kept separate from the benchmark so the measurement and its presentation can
change independently, and so the tables can be regenerated from a stored run.

The reporting rule that matters here: **a token reduction is only worth
something at an equal or better hit rate.** Every table therefore prints hit
rate next to token count, and the summary refuses to call an arm a win if it
misses files the baseline found.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARM_ORDER = [
    "baseline:files",
    "baseline:windows",
    "feature:query",
    "feature:locate",
    "feature:investigate",
    "workflow:minimal",
    "workflow:full",
]
ARM_LABELS = {
    "baseline:files": "no ai-db: read whole files",
    "baseline:windows": "no ai-db: read matching lines",
    "feature:query": "ai-db `query`",
    "feature:locate": "ai-db `locate`",
    "feature:investigate": "ai-db `investigate`",
    "workflow:minimal": "workflow: locate + analyze",
    "workflow:full": "workflow: locate + outline + analyze + investigate",
}
BASELINE = "baseline:windows"


def agg(records: list[dict], arm: str) -> dict:
    rows = [r["arms"][arm] for r in records if arm in r["arms"]]
    if not rows:
        return {}
    toks = sorted(r["tokens"] for r in rows)
    lat = sorted(r["latency_ms"] for r in rows)
    return {
        "tokens_median": statistics.median(toks),
        "tokens_mean": statistics.fmean(toks),
        "tokens_p90": toks[int(len(toks) * 0.9) - 1] if len(toks) > 1 else toks[0],
        "latency_median": statistics.median(lat),
        "hit_rate": sum(r["hit"] for r in rows) / len(rows),
        "symbol_rate": sum(r["symbol_hit"] for r in rows) / len(rows),
        "n": len(rows),
        "errors": sum(1 for r in rows if r.get("exit")),
    }


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def num(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1000:.1f}k"
    return f"{value:,.0f}"


def matched(records: list[dict], arm: str, other: str) -> dict | None:
    """Arm vs `other`, restricted to the queries where *both* found the file.

    This is the only comparison that is like-for-like. Unrestricted, the
    baseline wins on recall simply because reading whole files is exhaustive, so
    a token-per-query average quietly charges ai-db for the queries it missed.
    """
    pairs = [(r["arms"][arm], r["arms"][other]) for r in records
             if arm in r["arms"] and other in r["arms"]
             and r["arms"][arm]["hit"] and r["arms"][other]["hit"]]
    if not pairs:
        return None
    a = statistics.median(p[0]["tokens"] for p in pairs)
    b = statistics.median(p[1]["tokens"] for p in pairs)
    return {"n": len(pairs), "arm_tokens": a, "other_tokens": b, "ratio": b / a if a else 0}


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        REPO, "eval/results/token_budget.json")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        REPO, "eval/results/TOKEN_BUDGET.md")
    with open(src, encoding="utf-8") as fh:
        report = json.load(fh)
    records = report["records"]
    stats = {arm: agg(records, arm) for arm in ARM_ORDER if agg(records, arm)}
    base = stats[BASELINE]

    doc: list[str] = [
        "# Token cost and latency: no ai-db vs ai-db",
        "",
        f"{report['n_queries']} questions from `{report['golden']}`, each asking where",
        "something in this repo is implemented. Every arm answers the same question with",
        "the same scorer.",
        "",
        "## What is being compared",
        "",
        "| Arm | What it does |",
        "|---|---|",
        "| no ai-db: read whole files | `rg -l` for candidates, then reads the first 5 matching files end to end |",
        "| no ai-db: read matching lines | `rg -n -C3`, first 200 lines of matches. A careful agent without ai-db |",
        "| ai-db `query` | one lexical query |",
        "| ai-db `locate` | one locate call |",
        "| ai-db `investigate` | one investigate, explain mode |",
        "| workflow: locate + analyze | locate, then analyze the symbol it found |",
        "| workflow: full | locate, outline, analyze, then investigate |",
        "",
        "The two baselines are both included on purpose. Reporting only the naive one",
        "would flatter ai-db; the line-window baseline is the strongest thing an agent",
        "can do with a shell, and it is the comparison that matters.",
        "",
        "## Results",
        "",
        "| Arm | Median tokens | vs baseline | p90 tokens | Median latency | Hit rate | Symbol rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ARM_ORDER:
        s = stats.get(arm)
        if not s:
            continue
        ratio = base["tokens_median"] / s["tokens_median"] if s["tokens_median"] else 0
        better = "baseline" if arm.startswith("baseline") else f"**{ratio:.0f}x less**"
        doc.append(
            f"| {ARM_LABELS[arm]} | {num(s['tokens_median'])} | {better} | "
            f"{num(s['tokens_p90'])} | {s['latency_median']:.0f} ms | "
            f"{pct(s['hit_rate'])} | {pct(s['symbol_rate'])} |"
        )

    doc.extend(summary(stats, base, records))
    doc.append("")
    doc.append("## Reproducing")
    doc.append("")
    doc.append("```bash")
    doc.append("uv run python eval/matrix_configs.py /tmp/feature_bench /tmp/feature_bench/db")
    doc.append("uv run python eval/token_budget_benchmark.py")
    doc.append("uv run python eval/render_token_table.py")
    doc.append("```")
    doc.append("")

    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(doc))
    print(f"wrote {dst} ({len(records)} queries, {len(stats)} arms)")
    return 0


def summary(stats: dict, base: dict, records: list[dict]) -> list[str]:
    out = ["", "## What the numbers say", ""]

    q = stats.get("feature:query", {})
    base_hit = base["hit_rate"]
    worse = {a: s for a, s in stats.items()
             if not a.startswith("baseline") and s["hit_rate"] < base_hit}

    # The recall caveat leads, because it bounds every other claim.
    if worse:
        best = max(s["hit_rate"] for s in worse.values())
        weakest = min(worse.items(), key=lambda kv: kv[1]["hit_rate"])
        out.append(
            f"1. **Every ai-db arm finds the golden file less often than the "
            f"baseline.** Best ai-db recall is {best:.0%}, against "
            f"{base_hit:.0%} for the line-window baseline and "
            f"{stats['baseline:files']['hit_rate']:.0%} for whole-file reads. The "
            f"baseline has a structural advantage worth naming: it *reads* files, "
            f"so once ripgrep ranks a file into its top 5 the filename is "
            f"guaranteed to appear in the output. It is brute force, and brute "
            f"force is exactly why it costs {num(base['tokens_median'])} tokens. "
            f"So the honest reading is not \"ai-db is "
            f"{base['tokens_median'] / q['tokens_median']:.0f}x cheaper\", it is "
            f"\"ai-db is far cheaper and somewhat less exhaustive\". Closing the "
            f"recall gap is the real work: {ARM_LABELS[weakest[0]]} is the weakest "
            f"arm here at {weakest[1]['hit_rate']:.0%}.")
    else:
        out.append(
            "1. **No ai-db arm loses files the baseline found**, so the token "
            "figures are like-for-like rather than a trade of recall for size.")
    out.append("")

    # Like-for-like: only the queries both arms actually found.
    m = matched(records, "feature:query", BASELINE)
    if m:
        out.append(
            f"2. **On the {m['n']} questions where both arms found the file**, "
            f"`query` reads {num(m['arm_tokens'])} median tokens against the "
            f"baseline's {num(m['other_tokens'])} -- a **{m['ratio']:.0f}x "
            f"reduction at matched recall**. This is the defensible version of the "
            f"headline; the unrestricted "
            f"{base['tokens_median'] / q['tokens_median']:.0f}x overstates it by "
            f"quietly crediting ai-db with questions it never answered.")
        out.append("")

    full = stats.get("workflow:full")
    mini = stats.get("workflow:minimal")
    if full and q:
        out.append(
            f"3. **Combining features is not automatically better.** The full "
            f"workflow costs {num(full['tokens_median'])} tokens against "
            f"`query`'s {num(q['tokens_median'])}; the minimal workflow "
            f"{num(mini['tokens_median']) if mini else 'n/a'}. Chaining re-reads "
            f"code `query` already summarised, and the agent pays for the "
            f"concatenation. It does buy recall "
            f"({full['hit_rate']:.0%} vs {q['hit_rate']:.0%}), so the trade is "
            f"tokens for recall, not free extra context. The lesson is \"use "
            f"`query` unless you need a body only `analyze` can give you\", not "
            f"\"use more features\".")
        out.append("")

    if q and base["latency_median"]:
        out.append(
            f"4. **ai-db loses on wall clock, by a lot.** `query` takes "
            f"{q['latency_median']:.0f} ms against the baseline's "
            f"{base['latency_median']:.0f} ms: every CLI invocation pays ~200 ms "
            f"of interpreter start and import before doing ~1 ms of work. The "
            f"exchange rate is explicit -- about "
            f"{q['latency_median'] - base['latency_median']:.0f} ms of wall clock "
            f"to avoid re-reading "
            f"{num(base['tokens_median'] - q['tokens_median'])} tokens. Over a "
            f"session that is a clear win; for a single one-shot grep it is not, "
            f"and ripgrep will always win that case.")
        out.append("")

    total = sum(r["arms"][BASELINE]["tokens"] for r in records if BASELINE in r["arms"])
    qt = sum(r["arms"]["feature:query"]["tokens"] for r in records
             if "feature:query" in r["arms"])
    out.append(
        f"Across all {len(records)} questions the line-window baseline reads "
        f"{num(total)} tokens in total; `query` reads {num(qt)}, {num(total - qt)} "
        f"fewer ({100 - qt / total * 100:.0f}%). That total is the figure to "
        f"quote for context-window pressure, and unlike the per-query medians it "
        f"does not depend on the recall caveat -- it is what both arms cost when "
        f"asked all {len(records)} questions.")
    out.append("")
    out.append("### What this does not measure")
    out.append("")
    out.append("- **It is not a measurement of agent skill.** The baselines are a "
               "stated, mechanical")
    out.append("  model of shell search, reproducible to the byte. A real agent "
               "that reads a file,")
    out.append("  understands it and answers without re-reading would beat both arms.")
    out.append("- **Tokens out, not tokens in.** Query text, tool schemas and "
               "retries are uncounted.")
    out.append("  That favours neither side strongly, but it does favour repeated "
               "baseline calls,")
    out.append("  which re-send the query every time.")
    out.append("- **The hit test is a filename mention**, matched on either the full "
               "relative path")
    out.append("  or the basename. It is deliberately generous to the baseline, "
               "which names every")
    out.append("  file it read, and it does not check that the *right part* of the "
               "file was surfaced.")
    out.append("- **Lexical config only.** The hybrid config was not measured because "
               "it costs")
    out.append("  ~10.5 s per invocation, which would dominate the latency column "
               "for reasons")
    out.append("  unrelated to token cost.")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
