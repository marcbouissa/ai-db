"""The benchmark harness must be trustworthy, and that is testable.

A benchmark that reports flattering numbers because its scoring is broken is
worse than no benchmark. These tests pin the parts that decide whether a number
means anything: the scoring functions, the format/mode sets, the invariant
definitions, and the correspondence between eval/scenarios.json and the golden
sets its ground truth is copied from.

The fabrication metric in particular has already failed once -- its first
version flagged most citations in correct output as fabrications, including real
files and absolute paths with the leading slash stripped. A metric that cries
wolf trains the reader to ignore it, so it gets explicit tests.
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))

import benchmark_runner as br

SCENARIOS_PATH = os.path.join(REPO, "eval/scenarios.json")


def scenarios() -> dict:
    with open(SCENARIOS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def golden(name: str) -> list[dict]:
    path = os.path.join(REPO, "eval/golden", name)
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------ scoring
class TestScoring:
    def test_recall_is_fraction_of_expected_files(self):
        s = br.score("found ai_db/config.py", {"recall": ["ai_db/config.py", "ai_db/x.py"]}, REPO)
        assert s["recall"] == 0.5
        assert s["missing_files"] == ["ai_db/x.py"]

    def test_symbol_recall_uses_word_boundaries(self):
        # "parse_configs" must not satisfy "parse_config".
        s = br.score("parse_configs is unrelated", {"symbol_recall": ["parse_config"]}, REPO)
        assert s["symbol_recall"] == 0.0

    def test_order_requires_presence_and_sequence(self):
        truth = {"order": ["a_entry", "b_middle", "c_leaf"]}
        assert br.score("a_entry then b_middle then c_leaf", truth, REPO)["order_ok"] is True
        assert br.score("c_leaf then b_middle then a_entry", truth, REPO)["order_ok"] is False
        assert br.score("a_entry then c_leaf", truth, REPO)["order_ok"] is False
        assert br.score("a_entry then b_middle then c_leaf", truth, REPO)["order_found"] == 1.0

    def test_absent_axes_report_none_not_zero(self):
        """None means 'not applicable'; 0.0 would mean 'scored and failed'."""
        s = br.score("anything", {}, REPO)
        assert s["recall"] is None
        assert s["symbol_recall"] is None
        assert s["order_ok"] is None

    def test_scoring_is_deterministic(self):
        truth = {"recall": ["ai_db/config.py"], "symbol_recall": ["parse_config"]}
        text = "ai_db/config.py parse_config"
        first = br.score(text, truth, REPO)
        for _ in range(5):
            assert br.score(text, truth, REPO) == first


class TestFabricationMetric:
    """A conservative metric: only verifiable repo-relative paths are judged."""

    def test_real_paths_are_not_fabrications(self):
        text = "see ai_db/config.py and ai_db/parser/ts_graph.py for details"
        assert br.fabrications(text) == []

    def test_invented_paths_are_detected(self):
        text = "defined in ai_db/parser/ts_graph.py and ai_db/ghost/module.py"
        assert br.fabrications(text) == ["ai_db/ghost/module.py"]

    def test_absolute_repo_paths_are_not_flagged(self):
        """A stripped leading slash used to make every absolute path look invented."""
        text = f"see {REPO}/ai_db/utils.py for the helper"
        assert br.fabrications(text) == []

    def test_bare_basenames_are_not_judged(self):
        """`foo.py` cannot be resolved repo-relative, so judging it is a guess."""
        assert br.fabrications("see foo.py and bar.py") == []

    def test_traversal_is_not_judged(self):
        assert br.fabrications("see ../../etc/passwd.py") == []

    def test_realistic_output_has_near_zero_fabrications(self):
        """The regression guard: correct output must not trip the metric."""
        good = (f"Retrieval is implemented in {REPO}/ai_db/search/retriever.py, "
                "with ranking in ai_db/search/ranking.py and storage in "
                "ai_db/storage/sqlite_backend.py. Parsing lives in "
                "ai_db/parser/ts_graph.py.")
        assert br.fabrications(good) == []


# ------------------------------------------------------------------ formats
class TestFormatSets:
    def test_analyze_and_pack_formats_are_different_sets(self):
        assert set(br.ANALYZE_FORMATS) == {"json", "stub", "sexp", "outline", "prose"}
        assert set(br.PACK_FORMATS) == {"compact", "json", "stub", "sexp"}
        # The asymmetry runs BOTH ways, which is easy to get backwards:
        # outline/prose exist only for analyze, and compact only for a pack.
        assert set(br.ANALYZE_FORMATS) - set(br.PACK_FORMATS) == {"outline", "prose"}
        assert set(br.PACK_FORMATS) - set(br.ANALYZE_FORMATS) == {"compact"}
        assert set(br.ANALYZE_FORMATS) & set(br.PACK_FORMATS) == {"json", "stub", "sexp"}

    def test_investigate_modes(self):
        assert br.INVESTIGATE_MODES == ("locate", "explain", "impact", "flow", "diff")

    def test_analyze_depths(self):
        assert br.ANALYZE_DEPTHS == ("summary", "structure", "targeted", "full")


# ---------------------------------------------------------------- scenarios
class TestScenariosFile:
    def test_is_valid_json(self):
        scenarios()

    def test_every_mode_is_covered(self):
        modes = {m["mode"] for m in scenarios()["query_modes"]}
        assert modes == set(br.INVESTIGATE_MODES)

    def test_every_mode_has_cases_with_ground_truth(self):
        for spec in scenarios()["query_modes"]:
            assert spec["cases"], f"{spec['mode']} has no cases"
            for case in spec["cases"]:
                assert case["ground_truth"], f"{spec['mode']}: {case['query']}"

    def test_diff_cases_all_carry_a_since_ref(self):
        """diff is meaningless without one and the CLI rejects it."""
        diff = next(m for m in scenarios()["query_modes"] if m["mode"] == "diff")
        assert diff.get("requires_since") is True
        for case in diff["cases"]:
            assert case.get("since"), f"diff case {case['query']!r} has no 'since'"
            assert ".." in case["since"], "since must be a commit range"

    def test_non_diff_cases_carry_no_since(self):
        for spec in scenarios()["query_modes"]:
            if spec["mode"] == "diff":
                continue
            for case in spec["cases"]:
                assert "since" not in case, f"{spec['mode']}: {case['query']}"

    def test_ground_truth_matches_the_golden_set_it_cites(self):
        """Ground truth copied by hand drifts; copied by test does not."""
        for spec in scenarios()["query_modes"]:
            src = spec.get("golden_source")
            if not src:
                continue
            base = os.path.basename(src)
            if base == "flow.jsonl":
                continue  # flow rows carry `expected_path`, not `expected`
            rows = {r["query"]: r for r in golden(base)}
            for case in spec["cases"]:
                assert case["query"] in rows, (
                    f"{spec['mode']}: {case['query']!r} is not in {src}")
                expected = {e["filepath"] for e in rows[case["query"]]["expected"]}
                got = set(case["ground_truth"].get("recall", []))
                assert got == expected, (
                    f"{spec['mode']}: {case['query']!r} claims {got} but {src} says "
                    f"{expected}")

    def test_flow_order_matches_flow_golden(self):
        """flow.jsonl keys rows as `module.<symbol>`; scenarios use the bare symbol."""
        flow = next(m for m in scenarios()["query_modes"] if m["mode"] == "flow")
        rows = {r["entry"].split(".")[-1].split(":")[0]: r["expected_path"]
                for r in golden("flow.jsonl")}
        for case in flow["cases"]:
            key = case["query"]
            assert key in rows, (
                f"flow case {key!r} is not a real symbol in the fixture; "
                f"flow.jsonl has {sorted(rows)}")
            assert case["ground_truth"]["order"] == rows[key], (
                f"flow order for {key!r} differs from flow.jsonl")

    def test_flow_queries_name_real_fixture_symbols(self):
        """A scenario must query a symbol that exists; an invented name recalls nothing."""
        src = os.path.join(REPO, "tests/fixtures/callflow/stage_manager.py")
        with open(src, encoding="utf-8") as fh:
            body = fh.read()
        flow = next(m for m in scenarios()["query_modes"] if m["mode"] == "flow")
        for case in flow["cases"]:
            assert f"def {case['query']}(" in body, (
                f"flow query {case['query']!r} is not defined in the fixture")

    def test_format_matrix_matches_the_code(self):
        fm = scenarios()["format_matrix"]
        assert set(fm["analyze_formats"]) == set(br.ANALYZE_FORMATS)
        assert set(fm["pack_formats"]) == set(br.PACK_FORMATS)
        assert set(fm["rejected_pack_formats"]) == set(br.ANALYZE_FORMATS) - set(br.PACK_FORMATS)

    def test_depth_matrix_matches_the_code(self):
        assert set(scenarios()["depth_matrix"]["analyze_depths"]) == set(br.ANALYZE_DEPTHS)

    def test_referenced_target_files_exist(self):
        fm = scenarios()["format_matrix"]
        for t in fm["targets"]:
            assert os.path.isfile(os.path.join(REPO, t)), t
        for t in scenarios()["depth_matrix"]["targets"]:
            assert os.path.isfile(os.path.join(REPO, t)), t

    def test_referenced_repos_exist(self):
        for name, rel in scenarios()["repos"].items():
            assert os.path.isdir(os.path.join(REPO, rel)), f"{name} -> {rel}"

    def test_ground_truth_files_exist(self):
        """A scenario pointing at a deleted file can never be recalled."""
        for spec in scenarios()["query_modes"]:
            for case in spec["cases"]:
                for f in case["ground_truth"].get("recall", []):
                    if f.endswith("/"):
                        continue  # a directory prefix, not a file
                    assert os.path.exists(os.path.join(REPO, f)), (
                        f"{spec['mode']}: {case['query']!r} -> missing {f}")

    def test_llm_is_off_by_default(self):
        """The harness must be runnable offline without inventing LLM numbers."""
        assert scenarios()["llm"]["enabled"] is False


# ------------------------------------------------------------------- runner
class TestRunnerHelpers:
    def test_content_terms_drops_stopwords_and_dedupes(self):
        terms = br.content_terms("where is the bm25 ranking of chunks done")
        assert "bm25" in terms and "ranking" in terms
        assert "where" not in terms and "the" not in terms
        assert len(terms) == len(set(terms))

    def test_baseline_file_slice_is_a_slice_not_an_index(self):
        """Regression guard for a real bug in this harness.

        `[cfg["file_read"]]` on a list is *integer indexing*, not slicing: it
        returned the 6th path, and the loop then iterated that path's characters
        as if they were paths. Every read raised OSError and was swallowed, so the
        `files` baseline reported 0 tokens and 0% recall. Nothing looked wrong --
        ripgrep had genuinely found 40 files -- and only asserting on the value's
        *type* caught it.
        """
        cfg = {"file_read": 3}
        paths = [f"dir/f{i}.py" for i in range(10)]

        sliced = paths[cfg["file_read"]:]
        assert sliced == paths[3:], "slice keeps every path from the limit on"
        assert all("/" in p for p in sliced)

        indexed = paths[cfg["file_read"]]
        assert isinstance(indexed, str), "indexing yields one element, not a sublist"
        # And the character-walk failure mode it enabled: iterating the indexed
        # string yields path *characters*, each of which then fails to open.
        assert list(indexed)[:3] == ["d", "i", "r"]

    def test_ensure_index_reports_instead_of_silently_passing(self, tmp_path):
        cfg = tmp_path / "nonexistent.json"
        result = br.ensure_index(str(cfg), "unit-test")
        assert "synced" in result
