# Token cost and latency: no ai-db vs ai-db

40 questions from `eval/golden/ai_db.jsonl`, each asking where
something in this repo is implemented. Every arm answers the same question with
the same scorer.

## What is being compared

| Arm | What it does |
|---|---|
| no ai-db: read whole files | `rg -l` for candidates, then reads the first 5 matching files end to end |
| no ai-db: read matching lines | `rg -n -C3`, first 200 lines of matches. A careful agent without ai-db |
| ai-db `query` | one lexical query |
| ai-db `locate` | one locate call |
| ai-db `investigate` | one investigate, explain mode |
| workflow: locate + analyze | locate, then analyze the symbol it found |
| workflow: full | locate, outline, analyze, then investigate |

The two baselines are both included on purpose. Reporting only the naive one
would flatter ai-db; the line-window baseline is the strongest thing an agent
can do with a shell, and it is the comparison that matters.

## Results

| Arm | Median tokens | vs baseline | p90 tokens | Median latency | Hit rate | Symbol rate |
|---|---|---|---|---|---|---|
| no ai-db: read whole files | 157.6k | baseline | 200.9k | 7 ms | 95% | 98% |
| no ai-db: read matching lines | 33.1k | baseline | 66.3k | 7 ms | 95% | 92% |
| ai-db `query` | 636 | **52x less** | 701 | 238 ms | 72% | 60% |
| ai-db `locate` | 304 | **109x less** | 349 | 250 ms | 72% | 12% |
| ai-db `investigate` | 7,612 | **4x less** | 8,780 | 249 ms | 82% | 80% |
| workflow: locate + analyze | 1,322 | **25x less** | 7,129 | 487 ms | 72% | 62% |
| workflow: locate + outline + analyze + investigate | 9,271 | **4x less** | 17.9k | 880 ms | 82% | 85% |

## What the numbers say

1. **Every ai-db arm finds the golden file less often than the baseline.** Best ai-db recall is 82%, against 95% for the line-window baseline and 95% for whole-file reads. The baseline has a structural advantage worth naming: it *reads* files, so once ripgrep ranks a file into its top 5 the filename is guaranteed to appear in the output. It is brute force, and brute force is exactly why it costs 33.1k tokens. So the honest reading is not "ai-db is 52x cheaper", it is "ai-db is far cheaper and somewhat less exhaustive". Closing the recall gap is the real work: ai-db `query` is the weakest arm here at 72%.

2. **On the 29 questions where both arms found the file**, `query` reads 626 median tokens against the baseline's 30.9k -- a **49x reduction at matched recall**. This is the defensible version of the headline; the unrestricted 52x overstates it by quietly crediting ai-db with questions it never answered.

3. **Combining features is not automatically better.** The full workflow costs 9,271 tokens against `query`'s 636; the minimal workflow 1,322. Chaining re-reads code `query` already summarised, and the agent pays for the concatenation. It does buy recall (82% vs 72%), so the trade is tokens for recall, not free extra context. The lesson is "use `query` unless you need a body only `analyze` can give you", not "use more features".

4. **ai-db loses on wall clock, by a lot.** `query` takes 238 ms against the baseline's 7 ms: every CLI invocation pays ~200 ms of interpreter start and import before doing ~1 ms of work. The exchange rate is explicit -- about 231 ms of wall clock to avoid re-reading 32.5k tokens. Over a session that is a clear win; for a single one-shot grep it is not, and ripgrep will always win that case.

Across all 40 questions the line-window baseline reads 1.5M tokens in total; `query` reads 25.2k, 1.5M fewer (98%). That total is the figure to quote for context-window pressure, and unlike the per-query medians it does not depend on the recall caveat -- it is what both arms cost when asked all 40 questions.

### What this does not measure

- **It is not a measurement of agent skill.** The baselines are a stated, mechanical
  model of shell search, reproducible to the byte. A real agent that reads a file,
  understands it and answers without re-reading would beat both arms.
- **Tokens out, not tokens in.** Query text, tool schemas and retries are uncounted.
  That favours neither side strongly, but it does favour repeated baseline calls,
  which re-send the query every time.
- **The hit test is a filename mention**, matched on either the full relative path
  or the basename. It is deliberately generous to the baseline, which names every
  file it read, and it does not check that the *right part* of the file was surfaced.
- **Lexical config only.** The hybrid config was not measured because it costs
  ~10.5 s per invocation, which would dominate the latency column for reasons
  unrelated to token cost.

## Reproducing

```bash
uv run python eval/matrix_configs.py /tmp/feature_bench /tmp/feature_bench/db
uv run python eval/token_budget_benchmark.py
uv run python eval/render_token_table.py
```
