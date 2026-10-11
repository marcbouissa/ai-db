"""Teacher loop (eval/teacher.py): failing runs' specs corrected and stored as exemplars."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from ai_db_scene.memory import SceneMemory
from ai_db_scene.models import RunRecord
from ai_db_scene.tools import scene_tools

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


@pytest.fixture(scope="module")
def teacher() -> Any:
    sys.path.insert(0, str(EVAL_DIR))
    try:
        return importlib.import_module("teacher")
    finally:
        sys.path.remove(str(EVAL_DIR))


@pytest.fixture
def mem(tmp_path, monkeypatch):
    m = SceneMemory("json_file", {"path": str(tmp_path / "m")})
    monkeypatch.setattr(scene_tools, "_scene_memory", m)
    return m


_BROKEN = {
    "setting": {"kind": "indoor", "bounds_m": [4, 4, 2.7], "mood": "cosy"},
    "objects": [{"id": "sofa", "category": "sofaa", "colour": "red"},
                {"id": "lamp", "category": "floor_lamp"},
                {"id": "blob", "category": "zzzzqqq"}],
    "relations": [{"a": "lamp", "rel": "beside", "b": "sofa"}, {"a": "lamp", "rel": "next_to", "b": "sofa"},
                  {"a": "blob", "rel": "beside", "b": "lamp"}],
}


def test_correct_spec_applies_the_validators_suggestions(teacher):
    spec, notes = teacher.correct_spec(_BROKEN)
    assert spec is not None, notes
    assert [o["id"] for o in spec["objects"]] == ["sofa", "lamp"] and spec["objects"][0]["category"] == "sofa"
    assert [(r["a"], r["rel"], r["b"]) for r in spec["relations"]] == [("lamp", "beside", "sofa")]
    assert any("dropped object blob" in n for n in notes)
    assert teacher.correct_spec({"setting": {"kind": "spaceship"}, "objects": []})[0] is None


def test_teach_is_a_dry_run_unless_asked(teacher, mem, tmp_path):
    rows = tmp_path / "bench.jsonl"
    rows.write_text("\n".join(json.dumps(r) for r in [
        {"brief": "a cosy corner", "stage": "layout", "spec": _BROKEN, "spec_errors": ["x"]},
        {"brief": "fine", "stage": "done", "blocking_issues": 0, "spec": _BROKEN}]), encoding="utf-8")
    mem.record_run(RunRecord(brief="a dim study", spec=_BROKEN, outcome="partial"))
    failures = teacher.failures_from_jsonl(str(rows)) + teacher.failures_from_memory(mem)
    assert [f["brief"] for f in failures] == ["a cosy corner", "a dim study"]
    dry = teacher.teach(failures, write=False)
    assert all(r["corrected"] and "run_id" not in r for r in dry)
    assert not mem.search_exemplars("cosy corner")
    done = teacher.teach(failures, write=True)
    stored = mem.get_run(done[0]["run_id"])
    assert stored.outcome == "success" and stored.score == 3.5 and stored.spec["objects"][0]["category"] == "sofa"
    assert not mem.search_exemplars("a cosy corner")  # unjudged fixes stay out of exemplar search
    promoted = teacher.teach(failures[:1], write=True, score=4.0)
    assert mem.get_run(promoted[0]["run_id"]).score == 4.0
    assert [r.brief for r in mem.search_exemplars("a cosy corner", top_k=1)] == ["a cosy corner"]
