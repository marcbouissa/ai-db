# Context-cost benchmark: no ai-db vs ai-db

Does using ai-db cost less context to answer the same questions, and which
output style costs least?

```bash
uv run python eval/context_benchmark.py --budget 32000 --out eval/results/context_benchmark_32k.json
uv run python eval/context_benchmark.py --budget 8000  --out eval/results/context_benchmark_8k.json
```

## What each column means

Read this before the tables. Four of the six columns measure different things,
and two of them can disagree with each other on purpose.

| Column | What it is | How to read it | How it affects the others |
|---|---|---|---|
| **Hit rate** | Share of the 40 golden questions where the expected symbol was found **and** the answer fitted in the stated budget | The pass/fail gate. This is the only column that depends on the budget. | It gates everything: a condition can be cheap and still score badly if the cheapness pushes the answer *out* of the ranked set. Raising the budget can only help this column, and changes none of the others. |
| **Median ctx** | Characters emitted before the target symbol became visible, over the questions where it was found | The cost of an answer. **Uncapped** — capping made every representation report the same number. | Independent of the budget. This is the cost you pay on a hit, so it is the numerator for everything to its right. |
| **Tokens out** | `Median ctx` ÷ 4, the repo's usual chars-per-token approximation | The same number in token units, for comparison with a model's context window. | A restatement of `Median ctx`, not new information. Tokens are the unit a model actually bills; chars are the unit this benchmark can measure uniformly across a subprocess baseline. |
| **Paired vs raw** | Median, over questions where **both** this condition and the raw baseline found the answer, of `this / raw` **for that same question** | How much of the raw cost each condition uses, per question, then summarised. Lower is better. | Needs the raw row to be meaningful. Pairing is what makes it robust: a plain ratio of medians compares two possibly-different question sets (raw found 39/40, outline 38/40). Not the same as `Median ctx ÷ raw median` — that ratio-of-medians number appeared in an earlier revision of this report and was removed because it is not a per-query quantity. |
| **Token saving** | `1 − (tokens out ÷ tokens in)`, where `tokens in` is the raw file content the representation was built from | What fraction of the source you did not have to send. Higher is better. | Only available for the `analyze`-based rows, which have a well-defined "raw input" (the file they render). `investigate` assembles evidence from many files, so there is no single input to divide by — hence the dash. Do not compare this column to `Paired vs raw`; they answer different questions. |
| *(internal)* `pack_claim` | The token count the pack reports about itself | Kept to compare against `Tokens out`. | The two disagree, and that is a finding — see below. |

## Method

Held constant: the 40 questions in `eval/golden/ai_db.jsonl`, the hit test
(*is the expected symbol literally visible in the context emitted*, in the file
the golden entry names — not "did retrieval return the right chunk id"), and the
budget.

Varies per row: discovery (ripgrep, or ai-db lexical) and output style.

## Results

### Budget 32,000 chars (~8k tokens)

| Condition | Hit rate | Median ctx (ch) | Tokens out | Paired vs raw | Token saving |
|---|---|---|---|---|---|
| raw ripgrep + read  (no ai-db) | 5/40 (12%) | 170,948 | 42,737 | 100.0% | 0% |
| ai-db lexical · outline | 35/40 (88%) | 2,743 | 685 | 4.6% | 80% |
| ai-db lexical · stub | 35/40 (88%) | 2,995 | 748 | 4.9% | 79% |
| ai-db lexical · prose | 35/40 (88%) | 3,528 | 881 | 5.7% | 74% |
| ai-db lexical · sexp | 35/40 (88%) | 3,770 | 941 | 6.0% | 73% |
| ai-db lexical · json | 25/40 (62%) | 9,514 | 2,377 | 12.9% | 41% |
| ai-db investigate `--mode locate` | 37/40 (92%) | 20,490 | 5,122 | 12.4% | — |
| ai-db investigate `--mode flow` | 31/40 (78%) | 18,461 | 4,615 | 12.0% | — |
| ai-db investigate `--mode impact` | 21/40 (52%) | 30,128 | 7,532 | 19.3% | — |
| ai-db investigate `--mode explain` | 13/40 (32%) | 32,438 | 8,109 | 19.2% | — |
| ai-db investigate `explain` · **compact** | 38/40 (95%) | 26,559 | 6,639 | 15.8% | — |
| ai-db investigate `explain` · **stub** | 32/40 (80%) | 3,607 | 901 | 2.6% | — |

### Budget 8,000 chars (~2k tokens)

| Condition | Hit rate | Median ctx (ch) | Tokens out | Paired vs raw | Token saving |
|---|---|---|---|---|---|
| raw ripgrep + read  (no ai-db) | 1/40 (2%) | 170,948 | 42,737 | 100.0% | 0% |
| ai-db lexical · outline | 22/40 (55%) | 2,743 | 685 | 4.6% | 80% |
| ai-db lexical · stub | 22/40 (55%) | 2,995 | 748 | 4.9% | 79% |
| ai-db lexical · prose | 21/40 (52%) | 3,528 | 881 | 5.7% | 74% |
| ai-db lexical · sexp | 21/40 (52%) | 3,770 | 941 | 6.0% | 73% |
| ai-db lexical · json | 15/40 (38%) | 8,135 | 2,033 | 11.5% | 45% |
| ai-db investigate `--mode locate` | 14/40 (35%) | 8,032 | 2,008 | 5.0% | — |
| ai-db investigate `--mode flow` | 14/40 (35%) | 8,022 | 2,005 | 5.6% | — |
| ai-db investigate `--mode impact` | 17/40 (42%) | 7,986 | 1,996 | 5.6% | — |
| ai-db investigate `--mode explain` | 6/40 (15%) | 8,213 | 2,053 | 5.1% | — |
| ai-db investigate `explain` · **compact** | 36/40 (90%) | 6,034 | 1,508 | 3.8% | — |
| ai-db investigate `explain` · **stub** | 32/40 (80%) | 1,167 | 291 | 0.9% | — |

## What the numbers say

**The raw baseline cannot answer these questions in any realistic window.** A
median of **170,948 characters — roughly 43,000 tokens** — and 5/40 even with
32,000 characters available. One Python file in this repo is larger than a
typical agent's whole budget. The honest framing is not "ai-db saves tokens" but
*without it you cannot answer at all*.

**`investigate --mode explain` was overrunning its own budget.** At
`--budget 8000` it reports 15% in-budget, and at 32,000 only 32%. The pack sizes
its evidence selection to fit the budget, then wraps it in `indent=2` JSON —
and the envelope pushes the total over. The pack believes it emitted 8,000
tokens; the caller actually receives 8,109. This is invisible from the pack's own
numbers, which is why the benchmark measures rendered output rather than
`token_count`.

**The two new formats fix that and shrink it further.**

| | 32k hit rate | 8k hit rate | Median ctx | vs json |
|---|---|---|---|---|
| `explain` (json, as before) | 13/40 | 6/40 | 32,438 | 1.00× |
| `explain --format compact` | **38/40** | **36/40** | 26,559 | 0.82× |
| `explain --format stub` | 32/40 | 32/40 | **3,607** | **0.11×** |

`compact` is the same data on one line — 18% smaller, no information lost — and
by removing the indent overhead it stops the budget overrun. `stub` is 9× smaller
again and lands at **2.6% of the raw baseline**, the best figure in the table.

**The stub's lower hit rate than compact is not a regression in quality.** It is
the same pack, rendered without bodies. At 3,607 characters it fits, so 32/40
survive; the 6 that miss are ones where the answer lived only in a body, which
`stub` deliberately does not inline and which remain one `ai-db expand <ref>`
away. That is the intended trade: pay 3.6k, and fetch the bodies you actually
need.

**Among the `analyze` formats, `outline` is the cheapest and `sexp` is fourth.**
All four of outline/stub/prose/sexp score an identical 35/40, because the symbol
name appears in a signature in all of them. `sexp` costs 37% more than `outline`
and buys nothing for locating a symbol; it earns its place when you need the
tree.

**`--depth summary` is the same as `--depth structure`.** Byte-identical medians
(2,995 both), 0–9 character differences on a header line. There is no separate
"Signature Summary" format to document.

## Two accounting problems this surfaced

**1. `meta.tokens_out` ignores `--format`.** For `ai_db/analysis/pack.py` it
reports `tokens_in=447, tokens_out=423` identically for json, stub, sexp,
outline and prose — while the actual emitted output ranges 658 to 2,930
characters. It tracks `--depth` but not the format. So the token figures the
product reports cannot support any comparison between output formats, and
`ai-db telemetry`'s savings line is a depth measurement wearing a format
label.

**2. `investigate` had no format option at all.** No `--format` on the CLI, no
`format` in the dispatcher schema, `indent=2` hardcoded in both the CLI and
`mcp_server.py`. The `analyze` formatters could not be reused because a pack is
a different schema from analyze output, so pack renderers were written rather
than the existing ones reshaped. `json` remains the default so every existing
caller keeps parsing it.

## Caveats

- The `analyze` rows emit a **whole file**, so they measure the containing file,
  not the symbol alone. A span- or symbol-targeted extraction would be cheaper
  for all of them; the ranking would likely hold.
- One corpus, one machine, one run. Differences under ~5% (stub vs outline) are
  not significant.
- `raw ripgrep + read` is a **generous** baseline: it greps every content term in
  the question, finding more files than a real agent would guess. A weaker
  baseline widens the gap.
- The golden set targets Python symbols. Per-language representation differences
  are not covered.
- Only the absolute character counts and the within-ai-db ranking are portable;
  every percentage depends on the raw baseline chosen.

## Files

- `eval/context_benchmark.py` — the benchmark
- `eval/results/context_benchmark_32k.json`, `context_benchmark_8k.json` — per-query
- `eval/results/BENCHMARKS.md` — the retrieval/device/vec0 benchmarks
