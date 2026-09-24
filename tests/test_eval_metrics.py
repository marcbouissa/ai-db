"""Hand-computed checks for the retrieval metrics."""

import json
import math

import pytest

from ai_db.eval.harness import compare_to_baseline, load_golden
from ai_db.eval.metrics import mrr, ndcg_at_k, recall_at_k, relevance_vector

EXP = [("a.py", "foo"), ("b.py", None)]


def test_relevance_vector_credits_each_expected_once():
    ranked = [("a.py", "def foo"), ("a.py", "foo2"), ("c.py", "x"), ("b.py", "y")]
    assert relevance_vector(ranked, EXP) == [1, 0, 0, 1]


def test_recall_at_k():
    ranked = [("c.py", "x"), ("a.py", "Cls.foo"), ("b.py", "z")]
    assert recall_at_k(ranked, EXP, 1) == 0.0
    assert recall_at_k(ranked, EXP, 2) == 0.5
    assert recall_at_k(ranked, EXP, 3) == 1.0


def test_mrr():
    assert mrr([("c.py", "x"), ("b.py", "q")], EXP) == 0.5
    assert mrr([("c.py", "x")], EXP) == 0.0


def test_ndcg_at_k():
    ranked = [("c.py", "x"), ("a.py", "foo")]
    expected_dcg = 1 / math.log2(3)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, EXP, 2) == pytest.approx(expected_dcg / ideal)
    assert ndcg_at_k([("a.py", "foo"), ("b.py", "x")], EXP, 2) == pytest.approx(1.0)


def test_symbol_mismatch_is_not_relevant():
    assert recall_at_k([("a.py", "bar")], [("a.py", "foo")], 5) == 0.0


def test_load_golden_rejects_bad_kind(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"query": "q", "expected": [{"filepath": "a"}], "kind": "nope"}))
    with pytest.raises(ValueError):
        load_golden(str(p))


def test_compare_to_baseline():
    assert compare_to_baseline({"k": 10, "recall@10": 0.5}, {"recall@10": 0.51}) is None
    assert compare_to_baseline({"k": 10, "recall@10": 0.4}, {"recall@10": 0.5}) is not None
