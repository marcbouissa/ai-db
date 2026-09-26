# Context-cost benchmark: no ai-db vs ai-db

Does using ai-db actually cost less context to answer the same questions, and
which representation costs least?

Run:
```bash
uv run python eval/context_benchmark.py --budget 32000 --out eval/results/context_benchmark_32k.json
uv run python eval/context_benchmark.py --budget 8000  --out eval/results/context_benchmark_8k.json
```

## Method

Three things are held constant so the representation is the only variable:

- **Questions** — `eval/golden/ai_db.jsonl`, 40 queries. (The older raw-vs-ai-db
  comparison used 37; the 3 stale entries were repointed at current code in
  Phase 15, so all 40 are usable now.)
- **Hit test** — a query hits when the expected symbol is literally visible in the
  context emitted, in the file the golden entry names. Not "the right chunk id
  came back" — *can an agent act on this*.
- **Budget** — the context window the agent is allowed. Two are reported.

What varies per row: discovery (ripgrep, or ai-db lexical) and representation
(raw file text, sexp, stub, outline, prose, json, or an investigate mode).

`Median ctx` is the **cost to answer**: characters emitted before the expected
symbol became visible, over the queries where it was found. It is not capped at
the budget, because capping made every representation report the same number and
hid exactly the differences this benchmark exists to measure. The budget instead
acts as a gate on hit rate.

## Results

### Budget 32,000 chars (~8k tokens)

| Condition | Hit rate | Median ctx (ch) | Total ctx (ch) | Median vs raw |
|---|---|---|---|---|
| raw ripgrep + read  (no ai-db) | 4/40 (10%) | 182,258 | 8,817,371 | 100.0% |
| ai-db lexical · outline | 35/40 (88%) | 2,743 | 384,683 | 1.5% |
| ai-db lexical · stub | 35/40 (88%) | 2,995 | 416,208 | 1.6% |
| ai-db lexical · prose | 35/40 (88%) | 3,528 | 501,503 | 1.9% |
| ai-db lexical · sexp | 35/40 (88%) | 3,770 | 528,066 | 2.1% |
| ai-db lexical · json | 25/40 (62%) | 9,514 | 1,142,323 | 5.2% |
| ai-db lexical · stub `--depth summary` | 35/40 (88%) | 2,995 | 416,208 | 1.6% |
| ai-db lexical · stub `--depth structure` | 35/40 (88%) | 2,995 | 416,208 | 1.6% |
| ai-db investigate `--mode locate` | 37/40 (92%) | 20,668 | 763,416 | 11.3% |
| ai-db investigate `--mode explain` | 38/40 (95%) | 31,586 | 1,173,164 | 17.3% |
| ai-db investigate `--mode impact` | 36/40 (90%) | 29,440 | 969,152 | 16.2% |
| ai-db investigate `--mode flow` | 31/40 (78%) | 17,996 | 560,816 | 9.9% |

### Budget 8,000 chars (~2k tokens)

| Condition | Hit rate | Median ctx (ch) | Total ctx (ch) | Median vs raw |
|---|---|---|---|---|
| raw ripgrep + read  (no ai-db) | 1/40 (2%) | 182,258 | 8,817,371 | 100.0% |
| ai-db lexical · outline | 22/40 (55%) | 2,743 | 384,683 | 1.5% |
| ai-db lexical · stub | 22/40 (55%) | 2,995 | 416,208 | 1.6% |
| ai-db lexical · prose | 21/40 (52%) | 3,528 | 501,503 | 1.9% |
| ai-db lexical · sexp | 21/40 (52%) | 3,770 | 528,066 | 2.1% |
| ai-db lexical · json | 15/40 (38%) | 8,135 | 689,023 | 4.5% |
| ai-db lexical · stub `--depth summary` | 22/40 (55%) | 2,995 | 415,938 | 1.6% |
| ai-db lexical · stub `--depth structure` | 22/40 (55%) | 2,995 | 416,208 | 1.6% |
| ai-db investigate `--mode locate` | 35/40 (88%) | 7,900 | 273,924 | 4.3% |
| ai-db investigate `--mode explain` | 36/40 (90%) | 7,932 | 283,252 | 4.4% |
| ai-db investigate `--mode impact` | 32/40 (80%) | 7,904 | 251,508 | 4.3% |
| ai-db investigate `--mode flow` | 30/40 (75%) | 7,924 | 236,040 | 4.3% |

## What the numbers say

**The raw baseline cannot answer these questions in any realistic context
window.** It needs a median of **182,258 characters — about 45,000 tokens** — to
find the answer, and lands 4/40 even with 32,000 characters available. A single
Python file in this repo is larger than a typical agent's whole budget. This is
the honest shape of the problem: not "ai-db saves tokens" but *without it you
cannot answer at all*.

**Every ai-db representation costs 1.5–5% of that.** The cheapest,
`--format outline`, is **1.5% of raw** and hits 88% at a 32k budget. The
expensive one, `--format json`, is 5.2% and still hits 62%.

**Cheapest is not `sexp`, and the extra structure buys nothing here.** The
ranking by cost is `outline` (2,743) < `stub` (2,995) < `prose` (3,528) <
`sexp` (3,770) << `json` (9,514). All four of the first four score an identical
35/40, because the symbol name appears in a signature in all of them. **If you
are looking for a symbol, `outline` is 27% cheaper than `sexp` for the same
result.** The README's token table ranks `sexp` as the most context-dense
format; on this task it is the fourth most expensive of five, and buys nothing.

**JSON is a trap.** It is 3.3x the cost of `outline` *and* scores worse — 62%
versus 88% at a 32k budget, 38% versus 55% at 8k — because being larger is how
it overflows the budget and loses the answer. It is the right choice for a
program consuming the output, and the wrong choice for an agent reading it.

**`--depth summary` is the same as `--depth structure`.** Byte-identical medians
(2,995 both), and per-file differences of 0–9 characters on a header line. The
README's token table lists "Signature Summary" as a distinct format at ~4% of raw
versus stub's ~12%; in fact the two produce the same output. That row should be
removed rather than corrected — there is nothing to distinguish.

**`investigate --mode explain` is the most accurate and the most expensive.**
95% at 31,586 chars, versus `locate` at 92% for 20,668 — 35% cheaper for 3
points of hit rate. `flow` is the cheapest at 17,996 but drops to 78%: it
answers a different question, and when you need the symbol's surroundings it
does not find them. `impact` sits at 90% for 29,440 and is worth its cost only
when you specifically need transitive callers.

**The budget dominates everything.** At 8,000 characters the raw baseline
collapses to 1/40 and the analyze-based representations to ~55%, while every
`investigate` mode stays above 75% — because `investigate` is the only path that
takes the budget as an input and sizes its own output to it. Representations
that ignore the budget do not degrade gracefully as it shrinks; they overflow.

## Correcting the published token table

`README.md` claims these footprints "vs raw". Measured here, against a baseline
of 182,258 characters:

| Format | README claim | Measured | Verdict |
|---|---|---|---|
| json | 45–60% saving | 94.8% saving | conservative |
| stub | 80–88% saving | 98.4% saving | conservative |
| sexp | 88–95% saving | 97.9% saving | conservative, but see below |
| summary | 92–96% saving | *identical to stub* | **wrong — not a distinct format** |

The savings percentages are *understated*, because the README never defines its
raw baseline and this one is expensive (whole Python files, top-down). The
absolute numbers are the trustworthy part. The `sexp` row is the one worth
changing regardless of baseline: it is not the most dense format for this task,
and the density advantage it advertises does not show up in hit rate.

## Caveats

- The analyze-based rows emit a **whole file** in the chosen representation, so
  they measure the cost of the containing file, not of the symbol alone. A
  span- or symbol-targeted extraction would be cheaper for all of them, and the
  ranking between formats would likely hold.
- One corpus, one machine, one run. No repetition, so small differences (stub vs
  outline, 2,995 vs 2,743) are not significant.
- `raw ripgrep + read` is a generous baseline: it greps for *every* content term
  in the question, including stop-word-adjacent ones, which finds more files than
  a real agent would guess. A weaker baseline would widen the gap, not narrow it.
- The golden set targets Python symbols. The corpus is 9 languages, but the
  questions are all Python, so per-language representation differences are not
  covered.
- Percentages depend entirely on the raw baseline chosen. Only the absolute
  character counts and the within-ai-db ranking are portable.

## Files

- `eval/context_benchmark.py` — the benchmark
- `eval/results/context_benchmark_32k.json` — per-query data, 32k budget
- `eval/results/context_benchmark_8k.json` — per-query data, 8k budget
