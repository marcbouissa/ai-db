"""Hand-computed checks for the retrieval metrics."""

import json
import math

import pytest

from ai_db.eval.harness import compare_pack_to_baseline, compare_to_baseline, load_golden
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


def test_symbol_match_is_exact_not_substring():
    """G10: `run` must not match `prune`; a dotted expectation matches the qualified name."""
    assert relevance_vector([("a.py", "def prune"), ("a.py", "def Cls.run (part 1/2)")],
                            [("a.py", "run")]) == [0, 1]
    assert relevance_vector([("a.py", "Other.run"), ("a.py", "Outer.Cls.run")],
                            [("a.py", "Cls.run")]) == [0, 1]
    assert relevance_vector([("a.py", "MyCls.run")], [("a.py", "Cls.run")]) == [0]


def test_git_worktree_materializes_head_as_a_checkout(tmp_path):
    import subprocess

    from ai_db.eval.harness import git_worktree

    repo = tmp_path / "repo"
    repo.mkdir()
    env_args = ["-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(repo), *env_args, "commit", "-qm", "c"], check=True)
    (repo / "untracked.py").write_text("y = 2\n")
    dest = tmp_path / "wt"
    with git_worktree(str(repo), str(dest)) as tree:
        assert (dest / "a.py").exists() and not (dest / "untracked.py").exists()
        out = subprocess.run(["git", "-C", tree, "rev-parse", "--is-inside-work-tree"],
                             capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "true"  # diff mode can run git here
    assert not dest.exists()


def test_load_golden_rejects_bad_kind(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"query": "q", "expected": [{"filepath": "a"}], "kind": "nope"}))
    with pytest.raises(ValueError):
        load_golden(str(p))


def test_compare_to_baseline():
    assert compare_to_baseline({"k": 10, "recall@10": 0.5}, {"recall@10": 0.51}) is None
    assert compare_to_baseline({"k": 10, "recall@10": 0.4}, {"recall@10": 0.5}) is not None


def test_compare_pack_to_baseline():
    assert compare_pack_to_baseline({"pack_recall": 0.5}, {"pack_recall": 0.51}) is None
    assert compare_pack_to_baseline({"pack_recall": 0.4}, {"pack_recall": 0.5}) is not None


def test_compare_pack_to_baseline_missing_key():
    try:
        compare_pack_to_baseline({"pack_recall": 0.5}, {})
    except ValueError as e:
        assert "baseline has no 'pack_recall' entry" in str(e)
    else:
        raise AssertionError("expected ValueError")
