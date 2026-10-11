"""Cut noisy command output down to what an agent needs.

An agent that runs ``pytest``, a build or a linter usually needs a few lines of a long
output: the failure, its traceback, the summary. Pasting the whole output into its
context costs tokens and buries the signal. ``filter_output`` keeps:

- blocks relevant to the agent's stated ``need`` (BM25 over the output's blocks),
- error/failure blocks (tracebacks, ``error:`` lines, ``FAILED``),
- summary lines (``3 failed, 10 passed``, ``test result:``) and the final block,

within a token budget, in original order, with ``[lines a-b omitted]`` markers. Repeated
same-shape lines (progress, downloads) collapse into one marker; ANSI colour codes and
carriage-return progress bars are removed first.

With ``semantic`` (``ai-db filter --rerank``), need-relevance also uses a model score
(the configured cross-encoder reranker, or embedding cosine similarity), so a need worded
differently from the log ("math crash" vs ``ZeroDivisionError``) still finds its block.

Pure: no database; a model only through the injected ``semantic`` scorer. The raw output is stored separately (``output_store``) so
the agent can re-filter with another need, or print an exact line range, without
re-running the command.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ai_db.constants import (
    OUTPUT_BUDGET_TOKENS,
    OUTPUT_COLLAPSE_MIN,
    OUTPUT_MAX_BLOCK_LINES,
    OUTPUT_RERANK_BLOCK_CHARS,
    OUTPUT_RERANK_MAX_BLOCKS,
    OUTPUT_TAIL_LINES,
    OUTPUT_W_NEED,
    OUTPUT_W_SEMANTIC,
    OUTPUT_W_SIGNAL,
    OUTPUT_W_SUMMARY,
    OUTPUT_W_TAIL,
)
from ai_db.search.query_builder import split_identifier
from ai_db.tokens import count_tokens

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_SIGNAL = re.compile(
    r"(?i)\b(error|errors|exception|traceback|failed|failure|fatal|panicked|panic|"
    r"assertionerror|segmentation fault|undefined reference|cannot find|permission denied|"
    r"timed out|timeout)\b|^E\s{2,}|^\s*FAIL\b|:\d+(:\d+)?: (error|fatal)|"
    r"^_{3,} \S.* _{3,}$")  # pytest names each failure in a ____ header
_SUMMARY = re.compile(
    r"(?i)(\b\d+ (passed|failed|errors?|skipped|warnings?|tests?)\b|^test result:|"
    r"^Tests?:\s|\bbuild (succeeded|failed)\b|^ok\s+\S+|^FAIL\s+\S+|exit (code|status)\s*\d+)")
# Lines that open a new block: tracebacks, pytest/unittest section headers, go/rust/gcc errors.
_BLOCK_START = re.compile(
    r"^(Traceback \(most recent call last\)|_{3,} .* _{3,}$|={3,}.*={3,}$|-{3,} .* -{3,}$|"
    r"(FAIL|PASS|---|ok)\b|error(\[E\d+\])?:|warning:|\S[^:\s]*:\d+(:\d+)?: (error|warning))")
_SHAPE = re.compile(r"0x[0-9a-fA-F]+|\d+(\.\d+)?|\s+")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


@dataclass
class _Line:
    no: int  # 1-based line number in the cleaned output (what --lines refers to)
    text: str
    collapsed: int = 0  # >0: this line stands for this many further same-shape lines


@dataclass
class _Block:
    lines: list[_Line]
    score: float = 0.0
    signal: bool = False
    tokens: int = 0
    terms: list[str] = field(default_factory=list)

    @property
    def first(self) -> int:
        return self.lines[0].no

    @property
    def last(self) -> int:
        return self.lines[-1].no + self.lines[-1].collapsed


@dataclass
class FilterResult:
    text: str  # the filtered output, ready to show the agent
    total_lines: int
    kept_lines: int  # original lines shown verbatim (collapsed runs count as their 2 shown lines)
    tokens_in: int
    tokens_out: int
    dropped_signal_blocks: int  # error blocks that did not fit the budget
    semantic: bool = False  # need-relevance included a model score


# (need, block texts) -> one relevance score per text, any scale (min-max normalised here).
SemanticScorer = Callable[[str, list[str]], list[float]]


def _normalise(scores: list[float]) -> list[float]:
    lo, hi = min(scores, default=0.0), max(scores, default=0.0)
    return [0.0] * len(scores) if hi <= lo else [(s - lo) / (hi - lo) for s in scores]


def clean(text: str) -> list[str]:
    """Strip ANSI codes and resolve carriage-return progress bars to their final state."""
    lines = []
    for raw in _ANSI.sub("", text).split("\n"):
        if "\r" in raw:
            parts = [p for p in raw.split("\r") if p.strip()]
            raw = parts[-1] if parts else ""
        lines.append(raw.rstrip())
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _shape(line: str) -> str:
    """A line with numbers, hex and whitespace runs normalised: progress and counters
    (``[  9%]`` vs ``[ 10%]``, ``item 41 of 300``) compare equal."""
    return _SHAPE.sub(lambda m: " " if m.group().isspace() else "#", line)


def _collapse(lines: list[str]) -> list[_Line]:
    """Fold runs of same-shape lines (digits/hex ignored) into first + count + last."""
    out: list[_Line] = []
    i = 0
    while i < len(lines):
        shape = _shape(lines[i])
        j = i + 1
        while j < len(lines) and lines[j] and _shape(lines[j]) == shape:
            j += 1
        run = j - i
        if run >= OUTPUT_COLLAPSE_MIN and lines[i]:
            out.append(_Line(i + 1, lines[i], collapsed=run - 2))
            out.append(_Line(j, lines[j - 1]))
        else:
            out.extend(_Line(k + 1, lines[k]) for k in range(i, j))
        i = j
    return out


def _segment(lines: list[_Line]) -> list[_Block]:
    blocks: list[_Block] = []
    current: list[_Line] = []
    in_traceback = False

    def close() -> None:
        nonlocal current
        while current and not current[-1].text:
            current.pop()
        while current:
            blocks.append(_Block(current[:OUTPUT_MAX_BLOCK_LINES]))
            current = current[OUTPUT_MAX_BLOCK_LINES:]

    for line in lines:
        text = line.text
        if in_traceback:
            current.append(line)
            if text and not text[0].isspace() and not text.startswith("Traceback"):
                in_traceback = False  # the exception line ends a Python traceback
                close()
            continue
        if not text:
            close()
            continue
        if _BLOCK_START.match(text):
            close()
            in_traceback = text.startswith("Traceback")
        current.append(line)
    close()
    return blocks


def _stem(word: str) -> str:
    """Light suffix stripping so "parsing" meets ``parse_price`` and "commas" meets "comma"."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[:-len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) > 3 else word


def _terms(text: str) -> list[str]:
    terms: list[str] = []
    for word in _WORD.findall(text):
        terms.append(_stem(word.lower()))
        parts = split_identifier(word)
        if len(parts) > 1:
            terms.extend(_stem(p) for p in parts)
    return terms


def _bm25(query: list[str], blocks: list[_Block]) -> list[float]:
    if not query or not blocks:
        return [0.0] * len(blocks)
    k1, b = 1.2, 0.75
    avg = sum(len(bl.terms) for bl in blocks) / len(blocks) or 1.0
    df = {t: sum(1 for bl in blocks if t in set(bl.terms)) for t in set(query)}
    n = len(blocks)
    scores = []
    for bl in blocks:
        tf: dict[str, int] = {}
        for t in bl.terms:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in set(query):
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(bl.terms) / avg))
        scores.append(s)
    return scores


def _render(line: _Line) -> str:
    if line.collapsed:
        return f"{line.text}\n  ... x{line.collapsed} similar lines"
    return line.text


def _fit(block: _Block, budget: int) -> list[_Line] | None:
    """The whole block if it fits, else its head and tail within ``budget``."""
    if block.tokens <= budget:
        return block.lines
    head: list[_Line] = []
    tail: list[_Line] = []
    used = 0
    lines = block.lines
    lo, hi = 0, len(lines) - 1
    take_head = True
    while lo <= hi:
        line = lines[lo] if take_head else lines[hi]
        cost = count_tokens(_render(line)) + 1
        if used + cost > budget - 8:  # leave room for the omission marker
            break
        used += cost
        if take_head:
            head.append(line)
            lo += 1
        else:
            tail.insert(0, line)
            hi -= 1
        take_head = not take_head
    if not head:
        # Not even the first line fits (minified JSON, one huge log line): keep its
        # start and end rather than dropping the block. ~3 chars per token, conservatively.
        if budget <= 24:
            return None
        return [_truncate_line(lines[0], (budget - 16) * 3)]
    marker = _Line(-1, f"  ... [lines {lines[lo].no}-{lines[hi].no} of this block omitted]") \
        if lo <= hi else None
    return head + ([marker] if marker else []) + tail


def _truncate_line(line: _Line, max_chars: int) -> _Line:
    text = line.text
    keep_head, keep_tail = (max_chars * 2) // 3, max_chars // 3
    cut = len(text) - keep_head - keep_tail
    return _Line(line.no, f"{text[:keep_head]} ... [{cut} characters of this line omitted] ... "
                          f"{text[len(text) - keep_tail:]}")


def _omitted(first: int, last: int) -> str:
    return f"... [line {first} omitted]" if first == last else f"... [lines {first}-{last} omitted]"


def filter_output(text: str, need: str | None = None,
                  budget_tokens: int = OUTPUT_BUDGET_TOKENS,
                  semantic: SemanticScorer | None = None) -> FilterResult:
    """Keep what matters in ``text`` for ``need`` (free text: "why did test_x fail").

    ``semantic`` adds a model relevance score for ``need`` (requires a need)."""
    if semantic is not None and not (need and need.strip()):
        raise ValueError("semantic rerank needs a 'need' to rank against")
    cleaned = clean(text)
    tokens_in = count_tokens("\n".join(cleaned))
    if tokens_in <= budget_tokens:
        body = "\n".join(cleaned)
        return FilterResult(body, len(cleaned), len(cleaned), tokens_in, tokens_in, 0)

    blocks = _segment(_collapse(cleaned))
    for bl in blocks:
        joined = "\n".join(_render(line) for line in bl.lines)
        bl.terms = _terms(joined)
        bl.tokens = count_tokens(joined) + 1
        bl.signal = any(_SIGNAL.search(line.text) for line in bl.lines)

    relevance = _normalise(_bm25(_terms(need or ""), blocks))
    tail_from = max(1, len(cleaned) - OUTPUT_TAIL_LINES + 1)
    signal_counts = [sum(1 for line in bl.lines if _SIGNAL.search(line.text)) for bl in blocks]
    if semantic is not None and need:
        relevance = _blend_semantic(semantic, need, blocks, relevance, signal_counts)
    for bl, rel, signal_lines in zip(blocks, relevance, signal_counts):
        summary = any(_SUMMARY.search(line.text) for line in bl.lines)
        bl.score = (OUTPUT_W_NEED * rel
                    + OUTPUT_W_SIGNAL * min(signal_lines, 3) / 3
                    + (OUTPUT_W_SUMMARY if summary else 0.0)
                    + (OUTPUT_W_TAIL if bl.last >= tail_from else 0.0))

    chosen: dict[int, list[_Line]] = {}
    remaining = budget_tokens
    for idx in sorted(range(len(blocks)), key=lambda i: blocks[i].score, reverse=True):
        bl = blocks[idx]
        if bl.score <= 0 or remaining <= 16:
            continue
        fitted = _fit(bl, remaining)
        if fitted is None:
            continue
        chosen[idx] = fitted
        remaining -= sum(count_tokens(_render(line)) + 1 for line in fitted)

    out: list[str] = []
    cursor = 1
    kept = 0
    for idx in sorted(chosen):
        bl = blocks[idx]
        if bl.first > cursor:
            out.append(_omitted(cursor, bl.first - 1))
        out.extend(_render(line) for line in chosen[idx])
        kept += sum(1 for line in chosen[idx] if line.no > 0)
        cursor = bl.last + 1
    if cursor <= len(cleaned):
        out.append(_omitted(cursor, len(cleaned)))
    body = "\n".join(out)
    dropped = sum(1 for i, bl in enumerate(blocks) if bl.signal and i not in chosen)
    return FilterResult(body, len(cleaned), kept, tokens_in, count_tokens(body), dropped,
                        semantic=semantic is not None)


def _blend_semantic(semantic: SemanticScorer, need: str, blocks: list[_Block],
                    lexical: list[float], signal_counts: list[int]) -> list[float]:
    """Mix a model score into need-relevance. Only the OUTPUT_RERANK_MAX_BLOCKS most
    promising blocks (by keyword relevance + error signal) are sent to the model."""
    order = sorted(range(len(blocks)),
                   key=lambda i: lexical[i] + min(signal_counts[i], 3) / 3, reverse=True)
    picked = order[:OUTPUT_RERANK_MAX_BLOCKS]
    texts = ["\n".join(_render(line) for line in blocks[i].lines)[:OUTPUT_RERANK_BLOCK_CHARS]
             for i in picked]
    scores = semantic(need, texts)
    if len(scores) != len(texts):
        raise ValueError(f"semantic scorer returned {len(scores)} scores for {len(texts)} blocks")
    model = [0.0] * len(blocks)
    for i, s in zip(picked, _normalise([float(x) for x in scores])):
        model[i] = s
    return [OUTPUT_W_SEMANTIC * m + (1 - OUTPUT_W_SEMANTIC) * lex for m, lex in zip(model, lexical)]


def line_range(text: str, first: int, last: int) -> str:
    """Exact lines ``first..last`` (1-based, inclusive) of the cleaned output."""
    lines = clean(text)
    return "\n".join(f"{n:>6}  {lines[n - 1]}" for n in range(max(1, first), min(last, len(lines)) + 1))
