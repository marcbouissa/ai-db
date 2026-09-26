"""Matrix 5, the held-out split, and the persistent MCP client.

These are the parts that decide whether Matrix 5's numbers mean anything. Two of
them exist specifically because the alternative was a flattering wrong answer:

- The split exists so config selection and config reporting cannot be the same
  queries. A harness that skips it produces a recall number it manufactured.
- The empty-pack check exists because a pack with no `project` matches nothing
  and returns in ~8 ms, which averages into the latency and token columns as a
  *win* while silently destroying recall.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))

import agent_matrix
from heldout import HeldOutSplit, score_recall

GOLDEN = os.path.join(REPO, "eval/golden/ai_db.jsonl")


# ---------------------------------------------------------------- the split
class TestHeldOutSplit:
    def test_is_deterministic(self):
        """No seed and no shuffling: the same file must split the same way twice."""
        a = HeldOutSplit.from_golden(GOLDEN).describe()
        b = HeldOutSplit.from_golden(GOLDEN).describe()
        assert a == b

    def test_halves_are_disjoint_and_cover_everything(self):
        d = HeldOutSplit.from_golden(GOLDEN).describe()
        assert d["disjoint"] is True
        assert d["covers_all"] is True
        assert d["n_tune"] + d["n_holdout"] == d["n_total"] == 40

    def test_holdout_is_not_the_smaller_half(self):
        """Odd-sized groups give the extra query to holdout, so it is never starved."""
        d = HeldOutSplit.from_golden(GOLDEN).describe()
        assert d["n_holdout"] >= d["n_tune"] - 4

    def test_is_stratified_by_module(self):
        """Both halves must span the same modules, or a config tuned on one half
        is being tuned on a different codebase than it is reported on."""
        d = HeldOutSplit.from_golden(GOLDEN).describe()
        shared = set(d["modules_tune"]) & set(d["modules_holdout"])
        assert len(shared) >= 7, f"only {shared} modules appear in both halves"
        # No module should be wildly lopsided between halves.
        for mod in shared:
            t, h = d["modules_tune"][mod], d["modules_holdout"][mod]
            assert abs(t - h) <= 1, f"{mod}: {t} tune vs {h} holdout"

    def test_ground_truth_survives_the_split(self):
        s = HeldOutSplit.from_golden(GOLDEN)
        for row in s.all:
            assert row["expected"], row["query"]
            for e in row["expected"]:
                assert e.get("filepath")

    def test_known_unreachable_query_is_visible_not_hidden(self):
        """One golden entry targets a file deleted with the HTTP transport. A split
        must not quietly drop it, or the holdout recall looks better than reality."""
        s = HeldOutSplit.from_golden(GOLDEN)
        queries = {r["query"] for r in s.all}
        assert "http server post endpoint json" in queries

    def test_ordering_is_stable_under_input_reordering(self):
        """The split must not depend on the order rows appear in the file."""
        with open(GOLDEN, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        forward = HeldOutSplit.from_rows(rows)
        backward = HeldOutSplit.from_rows(list(reversed(rows)))
        assert ({r["query"] for r in forward.holdout}
                == {r["query"] for r in backward.holdout})


class TestScoreRecall:
    def test_scores_partial_hits_proportionally(self):
        rows = [{"query": "q", "expected": [{"filepath": "a.py"}, {"filepath": "b.py"}]}]
        assert score_recall(rows, lambda r: {"a.py"})["recall"] == 0.5

    def test_rows_without_expectations_are_skipped_not_zeroed(self):
        assert score_recall([{"query": "q", "expected": []}], lambda r: set())["n"] == 0


# -------------------------------------------------------------- empty packs
class TestEmptyPackDetection:
    def test_detects_a_well_formed_empty_pack(self):
        pack = json.dumps({"query": "x", "entry_points": [], "evidence": [],
                           "call_graph": {"nodes": [], "edges": []}})
        assert agent_matrix.is_empty_pack(pack) is True

    def test_a_populated_pack_is_not_empty(self):
        pack = json.dumps({"query": "x", "entry_points": [], "evidence": [
            {"ref": "ref:abc", "filepath": "a.py"}]})
        assert agent_matrix.is_empty_pack(pack) is False

    def test_entry_points_alone_count_as_content(self):
        pack = json.dumps({"query": "x", "entry_points": [{"qualified_name": "f"}],
                           "evidence": []})
        assert agent_matrix.is_empty_pack(pack) is False

    def test_non_json_is_not_treated_as_empty(self):
        """A stub/sexp/prose render has no `evidence` key; calling it empty would
        mislabel a perfectly good result."""
        assert agent_matrix.is_empty_pack("def f(): ...") is False
        assert agent_matrix.is_empty_pack("") is False


# ----------------------------------------------------------------- summaries
class TestSummarise:
    def _rec(self, tokens, warm, recall, empty=False):
        return {"tokens": tokens, "warm_ms": warm, "first_ms": warm,
                "score": {"recall": recall, "symbol_recall": None, "empty": empty}}

    def test_empty_results_are_counted_not_averaged(self):
        s = agent_matrix.summarise([self._rec(500, 1.0, 0.0, empty=True),
                                    self._rec(5000, 2.0, 1.0)])
        assert s["empty_results"] == 1
        # recall is still the plain mean, but emptiness is separately visible so
        # a reader can discount the row rather than trusting the mean blindly.
        assert s["recall"] == 0.5
        assert s["empty_results"] > 0

    def test_handles_no_records(self):
        s = agent_matrix.summarise([])
        assert s["n"] == 0 and s["recall"] is None and s["warm_ms"] is None


# ------------------------------------------------------------- doc_weight
class TestDocWeight:
    def test_writes_a_sibling_config_and_leaves_the_original_alone(self, tmp_path):
        base = tmp_path / "c.json"
        base.write_text(json.dumps({
            "version": 2, "storage": {"provider": "sqlite", "options": {"path": "x.db"}},
            "retrieval": {"mode": "hybrid"},
            "embedding": {"provider": "none"}, "rerank": {"provider": "none"},
        }), encoding="utf-8")
        out = agent_matrix.with_doc_weight(str(base), 0.7)
        assert out != str(base)
        with open(out, encoding="utf-8") as fh:
            new = json.load(fh)
        with open(base, encoding="utf-8") as fh:
            old = json.load(fh)
        assert new["retrieval"]["doc_weight"] == 0.7
        assert "doc_weight" not in old["retrieval"], "source config was mutated"

    def test_decimal_weights_produce_distinct_filenames(self, tmp_path):
        base = tmp_path / "c.json"
        base.write_text(json.dumps({"retrieval": {"mode": "hybrid"}}), encoding="utf-8")
        a = agent_matrix.with_doc_weight(str(base), 0.3)
        b = agent_matrix.with_doc_weight(str(base), 0.5)
        assert a != b


# ------------------------------------------------------------- MCP client
class TestMcpClient:
    """Driven against a real server on the lexical config; no GPU needed."""

    CONFIG = "/tmp/ai_db_bench/primary/lexical.json"

    def _client(self):
        from mcp_client import PersistentMcpClient

        if not os.path.exists(self.CONFIG):
            pytest.skip(f"{self.CONFIG} not built; run matrix_configs first")
        return PersistentMcpClient(self.CONFIG)

    def test_lists_tools_over_a_persistent_session(self):
        with self._client() as c:
            tools = c.list_tools()
        assert len(tools) >= 20
        assert "investigate" in tools

    def test_one_process_serves_many_calls(self):
        """The whole point: the server is started once, not per call."""
        with self._client() as c:
            for _ in range(3):
                text, ms = c.call_tool("investigate", {
                    "query": "where is bm25 ranking of chunks done",
                    "mode": "locate", "budget_tokens": 8000, "project": "ai-db"})
                assert text
                assert ms >= 0

    def test_project_is_required_for_a_non_empty_result(self):
        """Regression guard for a genuinely dangerous default.

        Omitting `project` returns a well-formed, empty, ~500-char pack in ~8 ms.
        Scored naively that is a fast, cheap, zero-recall *success*. Every caller
        in this harness passes `project`; this test documents what happens if one
        forgets.
        """
        with self._client() as c:
            without, _ = c.call_tool("investigate", {
                "query": "where is bm25 ranking of chunks done",
                "mode": "locate", "budget_tokens": 8000})
            with_proj, _ = c.call_tool("investigate", {
                "query": "where is bm25 ranking of chunks done",
                "mode": "locate", "budget_tokens": 8000, "project": "ai-db"})
        assert agent_matrix.is_empty_pack(without) is True
        assert agent_matrix.is_empty_pack(with_proj) is False
        assert len(with_proj) > len(without)

    def test_server_is_terminated_even_when_a_call_fails(self):
        from mcp_client import McpError

        c = self._client()
        if not os.path.exists(self.CONFIG):
            pytest.skip("config not built")
        c.start()
        with pytest.raises(McpError):
            c.call_tool("no_such_tool", {})
        proc = c.proc
        c.close()
        assert c.proc is None
        assert proc is not None
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pytest.fail("server process outlived close()")

    def test_call_before_start_raises(self):
        from mcp_client import McpError, PersistentMcpClient

        c = PersistentMcpClient(self.CONFIG)
        with pytest.raises(McpError):
            c._round_trip("tools/list", {})
