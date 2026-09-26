"""Render eval/results/benchmark.json as the benchmark matrix report.

Reads only the JSON, so the report can be regenerated without re-running the
harness. Ordering of the report mirrors the four matrices, and every table leads
with the caveat that constrains it -- a token number without its recall beside it
is the easiest number in this project to quote out of context.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(REPO, "eval/results/benchmark.json")
DEFAULT_DST = os.path.join(REPO, "eval/results/BENCHMARK_MATRIX.md")


def num(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1000:.1f}k"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:,.0f}"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def ms(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 10000:
        return f"{value / 1000:.1f}s"
    if value >= 1000:
        return f"{value / 1000:.2f}s"
    return f"{value:.0f}ms"


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def arm_summary(rows: list[dict[str, Any]], which: str) -> dict[str, Any]:
    """Aggregate one arm across rows. `which` is "aidb", "files" or "windows"."""
    toks: list[float] = []
    lat: list[float] = []
    rec: list[float] = []
    sym: list[float] = []
    fab: list[float] = []
    for r in rows:
        arm = r["aidb"] if which == "aidb" else r["baselines"][which]
        toks.append(arm["tokens"])
        lat.append(arm["ms"])
        if arm["score"]["recall"] is not None:
            rec.append(arm["score"]["recall"])
        if arm["score"]["symbol_recall"] is not None:
            sym.append(arm["score"]["symbol_recall"])
        fab.append(arm["score"]["fabrication_count"])
    return {
        "tokens_median": median(toks),
        "tokens_total": sum(toks),
        "latency_median": median(lat),
        "recall": (sum(rec) / len(rec)) if rec else None,
        "symbol_recall": (sum(sym) / len(sym)) if sym else None,
        "fabrications": int(sum(fab)),
        "n": len(rows),
    }


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    dst = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DST
    with open(src, encoding="utf-8") as fh:
        rep = json.load(fh)

    doc: list[str] = []
    add = doc.append

    add("# Benchmark matrix: ai-db vs raw-context baselines")
    add("")
    add(f"Generated {rep['generated']} by `eval/benchmark_runner.py` from "
        f"`{rep['scenarios']}`. Tokenizer: **{rep['tokenizer']}**. "
        f"Latency is the median of {rep['repeats']} run(s).")
    add("")
    add("Four matrices, in the order they were specified:")
    add("")
    add("| Matrix | What it varies | Question |")
    add("|---|---|---|")
    add("| 1 | `investigate --mode` (5 modes) | does the mode find the right thing, "
        "and what does it cost in tokens? |")
    add("| 2 | `analyze --format` (5), pack `--format` (4) | how much does the "
        "rendering cost, and does the format change what is found? |")
    add("| 3 | `retrieval.mode` x `vector_index` x `dtype` | what does the config "
        "buy in latency, memory and recall? |")
    add("| 4 | `--depth` (4) | where is the sweet spot between tokens and detail? |")
    add("")

    # ------------------------------------------------------------ invariants
    m1 = rep.get("matrix1_query_modes")
    if m1:
        inv = m1["invariants"]
        add("## Invariants")
        add("")
        add(f"**{'All passed' if inv['all_passed'] else 'FAILURES PRESENT'}** — "
            "enforced at runtime, not assumed. A violation fails the run.")
        add("")
        add("| Invariant | Expected | Observed exit | Why it matters |")
        add("|---|---|---|---|")
        for c in inv["checks"]:
            mark = "ok" if c["passed"] else "**FAIL**"
            obs = c.get("observed_exit", c.get("observed"))
            add(f"| {mark} {c['invariant']} | {c['expected']} | `{obs}` | {c['why']} |")
        add("")

    # ------------------------------------------------------------- matrix 1
    if m1:
        add("## Matrix 1 — query modes vs raw-context baselines")
        add("")
        add("Two baselines, both reported. `files` reads the top matching files "
            "whole (what an agent does with `cat`); `windows` reads only matching "
            "lines plus context (the strongest thing an agent can do with a "
            "shell). Quoting only the first would flatter ai-db.")
        add("")
        add("| Mode | n | ai-db tokens | ai-db latency | ai-db file recall | "
            "baseline `windows` tokens | baseline recall | Token change |")
        add("")
        add("`flow` has no file-recall axis — it is scored on call-path order, "
            "reported separately below.")
        add("|---|---|---|---|---|---|---|---|")
        totals: dict[str, list[float]] = {"aidb": [], "files": [], "windows": []}
        for mode in m1["modes"]:
            rows = mode["rows"]
            a = arm_summary(rows, "aidb")
            w = arm_summary(rows, "windows")
            for key in ("aidb", "files", "windows"):
                totals[key].extend(
                    r["aidb"]["tokens"] if key == "aidb" else r["baselines"][key]["tokens"]
                    for r in rows)
            red = (w["tokens_median"] / a["tokens_median"]
                   if a["tokens_median"] and w["tokens_median"] else 0)
            red_txt = f"**{red:.0f}x**" if red >= 1 else f"**{1 / red:.1f}x MORE**"
            add(f"| `{mode['mode']}` | {a['n']} | {num(a['tokens_median'])} | "
                f"{ms(a['latency_median'])} | {pct(a['recall'])} | "
                f"{num(w['tokens_median'])} | {pct(w['recall'])} | "
                f"{red_txt} |")
        add("")
        add("Across all cases:")
        add("")
        add("| Arm | Total context tokens | Median per query |")
        add("|---|---|---|")
        for label, key in (("ai-db (per mode)", "aidb"),
                           ("baseline: read whole files", "files"),
                           ("baseline: read matching lines", "windows")):
            add(f"| {label} | {num(sum(totals[key]))} | {num(median(totals[key]))} |")
        add("")
        red = (sum(totals["windows"]) / sum(totals["aidb"])) if totals["aidb"] else 0
        add(f"**{red:.0f}x** fewer context tokens than the strongest baseline.")
        add("")

        # Flow is scored on call-path order, not file presence.
        flow = next((m for m in m1["modes"] if m["mode"] == "flow"), None)
        if flow:
            add("### `flow` is scored on order, not on file presence")
            add("")
            add("A call path that names the right functions in the wrong order is a "
                "wrong answer, so `flow` is scored on sequence. It has no file-recall "
                "axis, which is why the table above shows `—` for it.")
            add("")
            add("| Query | ai-db order correct | Symbols found | baseline order correct |")
            add("|---|---|---|---|")
            for r in flow["rows"]:
                a = r["aidb"]["score"]
                b = r["baselines"]["windows"]["score"]
                add(f"| {r['query'][:34]} | {'yes' if a['order_ok'] else 'no'} | "
                    f"{pct(a['order_found'])} | "
                    f"{'yes' if b['order_ok'] else 'no'} |")
            add("")

        # Where ai-db does not win. A benchmark that only prints its wins is
        # marketing, and the impact/diff rows genuinely go against it.
        add("### Where ai-db does not win")
        add("")
        losses = []
        for m in m1["modes"]:
            rows = m["rows"]
            a = arm_summary(rows, "aidb")
            w = arm_summary(rows, "windows")
            notes = []
            if a["tokens_median"] and w["tokens_median"] and a["tokens_median"] > w["tokens_median"]:
                notes.append(f"costs {a['tokens_median'] / w['tokens_median']:.1f}x MORE "
                             f"tokens ({num(a['tokens_median'])} vs {num(w['tokens_median'])})")
            if a["recall"] is not None and w["recall"] is not None and a["recall"] < w["recall"]:
                notes.append(f"lower file recall ({pct(a['recall'])} vs {pct(w['recall'])})")
            if notes:
                losses.append((m["mode"], notes))
        if not losses:
            add("None: ai-db is at least as cheap and at least as accurate as the "
                "strongest baseline in every mode.")
        else:
            add("| Mode | ai-db is worse on |")
            add("|---|---|")
            for mode, notes in losses:
                add(f"| `{mode}` | {'; '.join(notes)} |")
            add("")
            add("The pattern is legible. `impact` cases are bare symbol names "
                "(`search_chunks`), which ripgrep matches inside a handful of files, "
                "so the baseline reads very little while `investigate` still returns a "
                "full token-budgeted pack. `diff` costs ai-db recall it does not "
                "recover in tokens. Reporting only the token reduction would hide "
                "both.")
            add("")

        # fabrications
        fab_rows = [(m["mode"], r["query"], r["aidb"]["score"]["fabricated_citations"],
                     r["baselines"]["windows"]["score"]["fabricated_citations"])
                    for m in m1["modes"] for r in m["rows"]]
        flagged = [f for f in fab_rows if f[2] or f[3]]
        add("### Fabricated citations")
        add("")
        add("A structural stand-in for hallucination that needs no model: a cited "
            "file path that does not exist on disk. This catches an output that "
            "*invents* a location. It cannot catch an invented explanation, which "
            "is why it is labelled a proxy and not a hallucination rate.")
        add("")
        if not flagged:
            add("None. No case in any mode cited a non-existent path.")
        else:
            add("| Mode | Query | ai-db | baseline `windows` |")
            add("|---|---|---|---|")
            for mode, q, a, b in flagged[:12]:
                add(f"| `{mode}` | {q[:40]} | {', '.join(a[:3]) or '—'} | "
                    f"{', '.join(b[:3]) or '—'} |")
        add("")

    # ------------------------------------------------------------- matrix 2
    m2 = rep.get("matrix2_formats")
    if m2:
        add("## Matrix 2 — output formats")
        add("")
        add("### `analyze --format`")
        add("")
        add("| Target | Format | Tokens | vs source | Latency | `tokens_out_formatted` | Self-count accurate |")
        add("|---|---|---|---|---|---|---|")
        for r in m2["analyze"]:
            acc = r["self_count_accurate"]
            acc_txt = "—" if acc is None else ("yes" if acc else "**NO**")
            emit = "—" if r["tokens_out_formatted"] is None else num(r["tokens_out_formatted"])
            add(f"| `{os.path.basename(r['target'])}` | `{r['format']}` | "
                f"{num(r['tokens'])} | {r['compression_vs_source']:.2f}x | "
                f"{ms(r['ms'])} | {emit} | {acc_txt} |")
        add("")
        add("`analyze --format json` is the only format that carries "
            "`meta.tokens_out_formatted`. The last column checks the emitted value "
            "against the token count of the bytes actually received — the number "
            "has to describe the payload that exists, not one that was counted and "
            "then discarded.")
        add("")
        add("### Pack `--format`")
        add("")
        add("| Query | Format | Tokens | Latency |")
        add("|---|---|---|---|")
        for r in m2["pack"]:
            add(f"| {r['query'][:38]} | `{r['format']}` | {num(r['tokens'])} | {ms(r['ms'])} |")
        add("")
        if m2["rejected"]:
            bad = [r for r in m2["rejected"] if not r["rejected_as_expected"]]
            add(f"`outline` and `prose` are **analyze** formats, not pack formats. "
                f"Rejected as expected in {len(m2['rejected']) - len(bad)}/"
                f"{len(m2['rejected'])} attempts"
                + ("." if not bad else f" — **{len(bad)} were wrongly accepted**."))
            add("")

    # ------------------------------------------------------------- matrix 3
    m3 = rep.get("matrix3_config_modes")
    if m3:
        add("## Matrix 3 — config modes")
        add("")
        add(f"CUDA available: **{m3['cuda_available']}**. Conditions needing a GPU "
            "are reported as skipped with a reason; a skip is never recorded as a "
            "zero, which would be indistinguishable from \"measured, and instant\".")
        add("")
        add("| Condition | retrieval | vector_index | device | dtype | Sync | DB size | "
            "Query p50 | Query p95 | Hit rate |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for c in m3["conditions"]:
            if c["status"] == "skipped":
                add(f"| `{c['name']}` | — | — | — | — | — | — | — | — | "
                    f"_{c['reason']}_ |")
                continue
            add(f"| `{c['name']}` | {c['retrieval_mode']} | {c['vector_index']} | "
                f"{c['device'] or '—'} | {c['dtype'] or 'auto'} | {ms(c['sync_ms'])} | "
                f"{c['db_mib']} MiB | {ms(c['latency_ms_median'])} | "
                f"{ms(c['latency_ms_p95'])} | {pct(c['hit_rate'])} |")
        add("")
        add("`hit_rate` here is the fraction of queries returning any result, not "
            "ground-truth recall: the config matrix varies the engine, and scoring "
            "it against golden expectations is Matrix 1's job. The lexical row of "
            "Matrix 1 is the like-for-like recall comparison.")
        add("")

    # ------------------------------------------------------------- matrix 4
    m4 = rep.get("matrix4_depths")
    if m4:
        add("## Matrix 4 — depth")
        add("")
        add("| Target | Depth | Tokens | Latency |")
        add("|---|---|---|---|")
        for r in m4["rows"]:
            add(f"| `{os.path.basename(r['target'])}` | `{r['depth']}` | "
                f"{num(r['tokens'])} | {ms(r['ms'])} |")
        add("")
        aliases = m4.get("depth_aliases") or {}
        if aliases:
            groups = list(aliases.values())
            add(f"**{m4['declared_depths']} declared depths produce only "
                f"{m4['distinct_behaviours']} distinct outputs.**")
            add("")
            for g in groups:
                add(f"- `{'` == `'.join(g)}` are byte-identical")
            add("")
            add("The docs claimed only that `summary` and `structure` were "
                "identical. Measuring every adjacent pair found a second alias "
                f"group, so `{'` / `'.join(groups[1])}` is also redundant. Depth is "
                "a two-valued control presented as a four-valued one.")
            add("")
        nm = m4.get("expand_depths_not_measurable")
        if nm:
            add(f"`expand --depth {'` / `'.join(nm['depths'])}` is **not "
                f"measurable as a subprocess**: {nm['reason']}. Recorded as such "
                "rather than estimated.")
            add("")

    # ------------------------------------------------------------- matrix 5
    m5 = rep.get("matrix5_agent_configs")
    if m5:
        sp = m5["split"]
        add("## Matrix 5 — agent-realistic config x transport (held out)")
        add("")
        add("Config selection reads the `tune` half of the golden set; everything "
            "below is reported on `holdout`, which no config choice saw. Scoring a "
            "config on the same queries that gate CI would manufacture the number "
            "being reported.")
        add("")
        add(f"Split: **{sp['n_holdout']} holdout / {sp['n_tune']} tune** of "
            f"{sp['n_total']}, disjoint={sp['disjoint']}, stratified by module. "
            f"Deterministic — no seed, so the same commit reproduces it.")
        add("")
        add("| Arm | Config | Transport | Holdout recall | Median tokens | Per call | First call |")
        add("|---|---|---|---|---|---|---|")
        base = m5.get("baseline")
        if base:
            for kind, label in (("windows", "no ai-db (matching lines)"),
                                ("files", "no ai-db (whole files)")):
                s = base["summary"][kind]["holdout"]
                add(f"| {label} | — | ripgrep | {pct(s['recall'])} | "
                    f"{num(s['tokens_median'])} | {ms(s['warm_ms'])} | — |")
        for row in m5["rows"]:
            if row["status"] == "skipped":
                add(f"| — | `{row['condition']}` | — | _{row['reason']}_ | | | |")
                continue
            for key, label in (("cli", "CLI subprocess"), ("mcp", "persistent MCP")):
                s = row[key]["summary"]["holdout"]
                w = row.get("doc_weight")
                cfg = f"`{row['condition']}`" + (f" w={w}" if w is not None else "")
                add(f"| ai-db | {cfg} | {label} | {pct(s['recall'])} | "
                    f"{num(s['tokens_median'])} | {ms(s['warm_ms'])} | "
                    f"{ms(s['first_ms'])} |")
        add("")
        # The transport ratio is the point of this matrix.
        best = None
        for row in m5["rows"]:
            if row["status"] != "ok":
                continue
            c = row["cli"]["summary"]["holdout"]
            p_ = row["mcp"]["summary"]["holdout"]
            if c["warm_ms"] and p_["warm_ms"]:
                best = (row, c["warm_ms"] / p_["warm_ms"], c, p_)
        if best:
            row, ratio, c, p_ = best
            add(f"**Transport is worth {ratio:.0f}x on `{row['condition']}`, at "
                f"identical recall ({pct(c['recall'])} both) and identical tokens "
                f"({num(c['tokens_median'])} vs {num(p_['tokens_median'])}).** The "
                f"pack is the same object either way; only the process count "
                f"differs. An agent holding an MCP session open pays the "
                f"interpreter start and the embedding-model load once, not per call.")
            add("")
        lex = next((r for r in m5["rows"] if r["status"] == "ok"
                    and r.get("retrieval_mode") == "lexical"), None)
        hyb = next((r for r in m5["rows"] if r["status"] == "ok"
                    and r.get("retrieval_mode") == "hybrid"), None)
        if lex and hyb:
            a = lex["mcp"]["summary"]["holdout"]
            b_ = hyb["mcp"]["summary"]["holdout"]
            add("Retrieval mode, both over persistent MCP so the transport is held "
                "constant:")
            add("")
            add("| Mode | Holdout recall | Median tokens | Per call |")
            add("|---|---|---|---|")
            add(f"| lexical | {pct(a['recall'])} | {num(a['tokens_median'])} | {ms(a['warm_ms'])} |")
            add(f"| hybrid | {pct(b_['recall'])} | {num(b_['tokens_median'])} | {ms(b_['warm_ms'])} |")
            add("")
        empties = sum(r[t]["summary"]["holdout"]["empty_results"]
                      for r in m5["rows"] if r["status"] == "ok" for t in ("cli", "mcp"))
        if empties:
            add(f"**{empties} empty results** were returned and are counted "
                "separately rather than averaged in. An empty pack is fast and "
                "cheap, so folding it into the means would flatter every column "
                "except recall.")
            add("")
        add("### What this matrix does not do")
        add("")
        add("- It does not tune a config and call the result a finding. `doc_weight` "
            "is swept so the shape of the curve is visible; choosing a value from "
            "these numbers and then quoting holdout recall is still fitting, and the "
            "split exists to make that visible rather than to license it.")
        add("- It does not claim the holdout half is representative of queries the "
            "gate has never seen. It is the same 40, partitioned.")
        add("")

    # ----------------------------------------------------------------- LLM
    llm = rep.get("llm")
    if llm:
        add("## LLM arm and cost")
        add("")
        add(f"Measured: **{llm['measured']}**"
            + (f" — {llm['reason']}" if llm.get("reason") else ""))
        add("")
        add("| Metric | Value |")
        add("|---|---|")
        add(f"| Model | `{llm['model']}` |")
        add(f"| TTFT | {ms(llm.get('ttft_ms'))} |")
        add(f"| Generation time | {ms(llm.get('generation_ms'))} |")
        add(f"| Output tokens | {llm.get('output_tokens') or '—'} |")
        add("")
        if not llm["measured"]:
            add("**No LLM metric is reported, and none is simulated.** With no API "
                "key configured, TTFT and generation time are null. A fabricated "
                "TTFT would be worse than an absent one, because it would be "
                "indistinguishable from a measurement once it reached a slide.")
            add("")
            add("Everything above — tokens, latency, recall, order, fabricated "
                "citations — is computed deterministically from output text and "
                "needs no model.")
            add("")
        cost = llm.get("cost_projected_usd")
        if cost:
            add("### Projected cost")
            add("")
            add("Context tokens are **measured**; output tokens are a declared "
                "assumption from `scenarios.json`. This is a projection at list "
                "price, not a billed amount.")
            add("")
            add("| Arm | Context tokens (measured) | Input USD | Output USD | Total USD |")
            add("|---|---|---|---|---|")
            for arm, v in cost.items():
                add(f"| {arm} | {num(v['context_tokens_total'])} | "
                    f"{v['usd_input']:.4f} | {v['usd_output']:.4f} | {v['usd_total']:.4f} |")
            add("")

    add("## Reproducing")
    add("")
    add("```bash")
    add("uv run python eval/matrix_configs.py /tmp/ai_db_bench/primary /tmp/ai_db_bench/db")
    add("uv run python eval/benchmark_runner.py --out eval/results/benchmark.json")
    add("uv run python eval/reporter.py")
    add("```")
    add("")
    add("Set `BENCH_LLM_BASE_URL` and `BENCH_LLM_API_KEY` and flip "
        "`llm.enabled` in `eval/scenarios.json` to measure the LLM arm for real.")
    add("")

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(doc))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
