"""build_scene tells the model when it is done (bench mining 2026-10-16, item 10 of the findings).

mimo aidb4 timed out in 3 of 6 scenes after a good build (21-57 calls of renders and bounds
checks); others took built_with_issues as done. The build result now carries a one-line
``verdict`` and a ``next`` instruction, and the tool descriptions lead with build_scene.
"""

from __future__ import annotations

import pytest
from ai_db_scene.tools import scene_pipeline as sp

SPEC = {"setting": {"kind": "indoor", "bounds_m": [5.0, 4.0, 2.7]},
        "objects": [{"id": "sofa", "category": "sofa"}, {"id": "table", "category": "coffee_table"}],
        "relations": [{"a": "sofa", "rel": "against_wall", "b": "wall_north"},
                      {"a": "table", "rel": "in_front_of", "b": "sofa"}]}


@pytest.fixture
def fake_blender(monkeypatch):
    class Fake:
        def batch(self, cmds, timeout=None):
            return [{"ok": 1, "status": "success"} for _ in cmds]

        def request(self, handler, params, timeout=None):
            return {}

    monkeypatch.setattr(sp, "BlenderClient", Fake)
    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})


def test_a_clean_build_says_done_and_stop(fake_blender):
    out = sp._handle_build_scene({"spec": SPEC, "snapshot": False, "exposure_check": False})
    assert out["verdict"].startswith("done: 2 objects built")
    assert "Stop here and reply DONE" in out["next"]


def test_a_build_with_issues_names_one_thing(fake_blender):
    spec = {**SPEC, "relations": [*SPEC["relations"], {"a": "table", "rel": "centered_in", "b": "room"},
                                  {"a": "sofa", "rel": "centered_in", "b": "room"}]}
    out = sp._handle_build_scene({"spec": spec, "snapshot": False, "exposure_check": False,
                                  "brief": "a sofa, a coffee table and two armchairs"})
    assert out["status"] == "built_with_issues"
    assert out["verdict"].startswith("one thing to fix: relation 'sofa centered_in room' could not be kept")
    assert " more)" in out["verdict"]  # the brief's missing armchairs come after
    assert "edit_scene" in out["next"] or "build_scene" in out["next"]


def test_verdict_rules():
    out = {"objects": 3, "issues_left": [{"check": "realism", "severity": "warning", "message": "bare walls"}],
           "scene_id": "s1"}
    verdict, nxt = sp.build_verdict(out)
    assert verdict == "done: 3 objects built, nothing blocking; 1 advisory warning(s) you may ignore"
    assert "saved as s1" in nxt
    out = {"objects": 3, "issues_left": [{"check": "floating", "severity": "blocking", "message": "lamp floats"}],
           "snapshot_error": "no Blender capture"}
    verdict, nxt = sp.build_verdict(out)
    assert verdict == "one thing to fix: lamp floats" and "snapshot_scene" in nxt


def test_descriptions_lead_with_build_scene():
    tools = [(name, desc) for name, desc, _s, _h in sp.PIPELINE_TOOLS]
    assert tools[0][0] == "build_scene" and tools[0][1].startswith("START HERE")
    desc = dict(tools)
    assert "do not loop solve_layout" in desc["solve_layout"]
    assert desc["apply_layout"].startswith("Low-level") and desc["submit_scene_spec"].startswith("Optional")
