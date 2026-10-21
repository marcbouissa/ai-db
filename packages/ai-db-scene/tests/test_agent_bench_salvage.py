"""strata_agent._salvage_call keeps call-level keys as arguments (qwen3-14b aidb8 round_lobby,
2026-10-21: the model closed ``arguments`` after the spec, wrote ``"brief"`` beside it and added a
surplus brace; the salvaged build_scene call dropped the brief)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1] / "eval" / "agent_bench"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bench_aidb8" / "qwen_round_lobby_calls.json"


def _agent():
    sys.path.insert(0, str(BENCH_DIR))
    try:
        return importlib.import_module("strata_agent")
    finally:
        sys.path.remove(str(BENCH_DIR))


def test_brief_beside_arguments_is_merged_into_them():
    spec = json.loads(FIXTURE.read_text())["calls"][0]["spec"]
    brief = "A round hotel lobby with a reception desk, four armchairs around a low table."
    content = ("<tool_call>\n" + json.dumps({"name": "aidb__build_scene", "arguments": {"spec": spec}})[:-1]
               + ', "brief": ' + json.dumps(brief) + "}}\n</tool_call>")
    call = _agent()._salvage_call(content, 3)
    assert call is not None and call["function"]["name"] == "aidb__build_scene"
    args = json.loads(call["function"]["arguments"])
    assert args["brief"] == brief and args["spec"] == spec


def test_arguments_win_over_call_level_duplicates_and_plain_salvage_is_unchanged():
    sa = _agent()
    call = sa._salvage_call('<tool_call>{"name": "t", "arguments": {"x": 1}, "x": 2}}</tool_call>', 1)
    assert json.loads(call["function"]["arguments"]) == {"x": 1}
    call = sa._salvage_call('<tool_call>{"name": "t", "arguments": {"a": {"b": 1}}}}}</tool_call>', 2)
    assert json.loads(call["function"]["arguments"]) == {"a": {"b": 1}}
    assert sa._salvage_call('<tool_call>{"name": "t", "arguments": {"a": 1}} trailing</tool_call>', 3) is None
