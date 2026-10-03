"""C2: exact-lookup eval suites (symbols, callers) on the refs_py fixture."""

import os

import pytest

from ai_db import VectorDB
from ai_db.eval.harness import compare_refs_to_baseline, run_callers, run_symbols

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "tests", "fixtures", "refs_py")


@pytest.mark.parametrize("runner,golden", [
    (run_symbols, "eval/golden/symbols.jsonl"),
    (run_callers, "eval/golden/callers.jsonl"),
])
def test_refs_suites_are_exact(tmp_path, runner, golden):
    db = VectorDB(str(tmp_path / "e.db"))
    try:
        result = runner(os.path.join(REPO, golden), ROOT, db)
    finally:
        db.close()
    bad = [q for q in result["per_query"] if q["missed"] or q["wrong"]]
    assert not bad, bad
    assert result["precision"] == 1.0 and result["recall"] == 1.0


def test_refs_baseline_comparison():
    base = {"precision": 1.0, "recall": 1.0}
    assert compare_refs_to_baseline({"suite": "callers", "precision": 1.0, "recall": 1.0}, base) is None
    err = compare_refs_to_baseline({"suite": "callers", "precision": 1.0, "recall": 0.9}, base)
    assert err and "recall" in err
    with pytest.raises(ValueError):
        compare_refs_to_baseline({"suite": "x", "precision": 1.0, "recall": 1.0}, {})
