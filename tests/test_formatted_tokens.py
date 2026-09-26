"""`meta.tokens_out_formatted`: the token count of the string a caller receives.

`meta.tokens_out` is a depth-derived estimate of how much *content* was
selected. It is identical for every output format -- for `ai_db/device.py` it
reports 817 whether you asked for json, stub, sexp, outline or prose -- so it
could never support a comparison between them, which is what the README's token
table claims to do. `tokens_out_formatted` counts the rendered string, and the
two together show the overhead each format adds.
"""

from __future__ import annotations

import copy
import json

import pytest

from ai_db.analyzer.formatters import (
    ANALYZE_FORMATS,
    annotate_formatted_tokens,
    format_analyze,
)
from ai_db.errors import AiDbConfigError


@pytest.fixture()
def analysis() -> dict:
    return {
        "file": "ai_db/device.py",
        "meta": {"tokens_in": 1818, "tokens_out": 817, "cached": True, "conf": 0.95},
        "symbols": [
            {"name": "resolve_dtype", "kind": "function", "sig": "def resolve_dtype(device: str)",
             "span": [70, 100], "ref": "ref:aaaa1111"},
            {"name": "_probe_cpu_dtypes", "kind": "function",
             "sig": "def _probe_cpu_dtypes(candidates: tuple[str, ...])",
             "span": [75, 108], "ref": "ref:bbbb2222"},
        ],
    }


# ==============================================================================
# the point of the field
# ==============================================================================

def test_tokens_out_is_format_blind_but_the_new_field_is_not(analysis):
    """The regression this field exists for, stated as a test."""
    estimates, actuals = set(), {}
    for style in sorted(ANALYZE_FORMATS):
        d = copy.deepcopy(analysis)
        annotate_formatted_tokens(d, style)
        estimates.add(d["meta"]["tokens_out"])
        actuals[style] = d["meta"]["tokens_out_formatted"]

    assert len(estimates) == 1, "tokens_out must be format-blind; that is the bug"
    assert len(set(actuals.values())) > 1, "tokens_out_formatted must vary by format"
    # and it must vary in the direction that matters: json is the big one
    assert actuals["json"] == max(actuals.values())
    assert actuals["outline"] < actuals["json"]


def test_field_is_actually_the_size_of_the_returned_string(analysis):
    from ai_db.parser.chunker import count_tokens

    for style in sorted(ANALYZE_FORMATS):
        d = copy.deepcopy(analysis)
        rendered = annotate_formatted_tokens(d, style)
        assert d["meta"]["tokens_out_formatted"] == count_tokens(rendered), style
        assert d["meta"]["format"] == style


def test_annotate_returns_exactly_what_it_counted(analysis):
    """The caller must print the string the recorded count describes.

    This used to assert equality with a render of the *pre-annotation* data,
    which is only true if the annotation happens after rendering -- and that is
    precisely the bug: it meant the json payload never contained the field. The
    correct invariant is the fixed point, asserted in
    `TestAnnotatedCountIsAFixedPoint`. Here we only require that a re-render of
    the annotated dict reproduces what was returned, i.e. nothing is mutated
    after the last render.
    """
    for style in sorted(ANALYZE_FORMATS):
        d = copy.deepcopy(analysis)
        rendered = annotate_formatted_tokens(d, style, count=True)
        assert rendered == format_analyze(d, style), style


def test_existing_meta_is_preserved(analysis):
    d = copy.deepcopy(analysis)
    annotate_formatted_tokens(d, "stub")
    assert d["meta"]["tokens_in"] == 1818
    assert d["meta"]["tokens_out"] == 817, "the estimate must not be overwritten"
    assert d["meta"]["cached"] is True
    assert d["meta"]["conf"] == 0.95


# ==============================================================================
# shape handling
# ==============================================================================

def test_batch_results_keep_their_own_per_file_meta(analysis):
    batch = {"meta": {"tokens_in": 2265, "tokens_out": 1240},
             "results": {"ai_db/device.py": copy.deepcopy(analysis)}}
    annotate_formatted_tokens(batch, "outline")
    assert batch["meta"]["tokens_out_formatted"] > 0
    per_file = batch["results"]["ai_db/device.py"]["meta"]
    assert per_file["tokens_in"] == 1818
    assert "tokens_out_formatted" not in per_file, (
        "the batch total is annotated; per-file counts are the caller's to request")


def test_json_payload_does_contain_its_own_count(analysis):
    """A json payload carries the count, and the count describes that payload.

    This test previously asserted the opposite -- that the field could *not* be
    in the payload -- and gave a rationale: "counting a re-render would be worse,
    that string is longer than the one counted, so the number would describe
    something nobody receives." The rationale does not hold. A fixed point
    resolves it: set the value, re-render, recount, repeat until stable, and
    return the string whose cost is the number recorded. The earlier behaviour
    meant the field was unreachable in the only format whose output is read by a
    machine.
    """
    d = copy.deepcopy(analysis)
    rendered = annotate_formatted_tokens(d, "json", count=True)
    parsed = json.loads(rendered)
    assert parsed["meta"]["tokens_out"] == 817
    assert parsed["meta"]["tokens_out_formatted"] > 0
    from ai_db.parser.chunker import count_tokens

    assert parsed["meta"]["tokens_out_formatted"] == count_tokens(rendered)
    # The dict the MCP and HTTP transports deliver carries the same number.
    assert d["meta"]["tokens_out_formatted"] == parsed["meta"]["tokens_out_formatted"]
    from ai_db.parser.chunker import count_tokens
    assert d["meta"]["tokens_out_formatted"] == count_tokens(rendered)


@pytest.mark.parametrize("style", sorted(ANALYZE_FORMATS))
def test_every_declared_format_renders(analysis, style):
    assert format_analyze(analysis, style).strip()


def test_unknown_format_is_rejected_with_the_valid_set(analysis):
    with pytest.raises(AiDbConfigError) as exc:
        format_analyze(analysis, "xml")
    msg = str(exc.value)
    assert "xml" in msg
    for style in ANALYZE_FORMATS:
        assert style in msg


def test_a_tokenizer_failure_does_not_lose_the_output(analysis, monkeypatch):
    """A count is telemetry; the rendered text is the product."""
    from ai_db.parser import chunker

    def boom(_text):
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr(chunker, "count_tokens", boom)
    d = copy.deepcopy(analysis)
    rendered = annotate_formatted_tokens(d, "stub")
    assert rendered.strip(), "the text must survive a failed count"
    assert d["meta"]["tokens_out_formatted"] is None
    assert d["meta"]["format"] == "stub"


class TestAnnotatedCountIsAFixedPoint:
    """The recorded count must be the cost of the string the caller receives.

    The previous implementation rendered first and set `meta` afterwards, so the
    field never appeared in the `json` payload -- the one format whose output a
    machine reads `meta` from. Asserting on the mutated *dict* (as the tests
    above do) cannot catch that; the assertion has to be on the rendered string.
    """

    @staticmethod
    def _count(text: str) -> int:
        from ai_db.parser.chunker import count_tokens

        return count_tokens(text)

    def test_json_output_actually_contains_the_field(self, analysis):
        d = copy.deepcopy(analysis)
        rendered = annotate_formatted_tokens(d, "json", count=True)
        assert "tokens_out_formatted" in rendered, (
            "the field must be visible in the json payload, not only in the dict")
        assert json.loads(rendered)["meta"]["format"] == "json"

    def test_recorded_equals_true_cost_of_returned_string(self, analysis):
        d = copy.deepcopy(analysis)
        rendered = annotate_formatted_tokens(d, "json", count=True)
        recorded = json.loads(rendered)["meta"]["tokens_out_formatted"]
        assert recorded == self._count(rendered), (
            f"recorded {recorded} but the string costs {self._count(rendered)}")

    def test_converges_in_a_bounded_number_of_passes(self, analysis):
        """A fixed point that oscillates would make the number meaningless."""
        d = copy.deepcopy(analysis)
        rendered = annotate_formatted_tokens(d, "json", count=True)
        recorded = json.loads(rendered)["meta"]["tokens_out_formatted"]
        again = annotate_formatted_tokens(d, "json", count=True)
        assert json.loads(again)["meta"]["tokens_out_formatted"] == recorded

    def test_holds_for_every_analyze_format(self, analysis):
        for style in ANALYZE_FORMATS:
            d = copy.deepcopy(analysis)
            rendered = annotate_formatted_tokens(d, style, count=True)
            assert d["meta"]["format"] == style
            assert isinstance(d["meta"]["tokens_out_formatted"], int)
            assert rendered.strip()

    def test_count_false_skips_the_count_but_keeps_the_format(self, analysis):
        d = copy.deepcopy(analysis)
        rendered = annotate_formatted_tokens(d, "stub", count=False)
        assert d["meta"]["format"] == "stub"
        assert d["meta"].get("tokens_out_formatted") is None
        assert rendered.strip()
