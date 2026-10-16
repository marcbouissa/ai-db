"""How a weak model hands a SceneSpec to the pipeline tools (bench mining 2026-10-16, items 1-2).

The shapes are those of real failing calls (docs/WEAK_MODEL_FAILURES_2026-10-16.md), on a
neutral spec: ``build_scene({setting, objects})`` without the ``spec`` wrapper, ``spec`` as a
JSON string, and ``build_scene()`` in a later sandbox call where the spec variable is gone.
"""

from __future__ import annotations

import json

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


def test_no_pipeline_tool_requires_the_spec_wrapper():
    tools = {name: schema for name, _d, schema, _h in sp.PIPELINE_TOOLS}
    for name in ("submit_scene_spec", "solve_layout", "build_scene", "apply_layout"):
        assert "spec" not in tools[name].get("required", []), name
        assert {"setting", "objects", "relations"} <= set(tools[name]["properties"]), name


def test_spec_fields_at_the_top_level():
    # tools.aidb.build_scene({setting: {...}, objects: [...]}) was "spec: Missing key"
    out = sp._handle_submit_scene_spec({"setting": SPEC["setting"], "objects": SPEC["objects"],
                                        "relations": SPEC["relations"]})
    assert out["status"] == "ok", out
    # an empty object list is now a spec error, not a missing argument
    out = sp._handle_submit_scene_spec({"setting": SPEC["setting"], "objects": []})
    assert out["status"] == "invalid" and any(e.startswith("objects:") for e in out["errors"])
    out = sp._handle_solve_layout(dict(SPEC))
    assert out["status"] == "ok" and len(out["placements"]) == 2


def test_spec_as_a_json_string():
    assert sp._handle_submit_scene_spec({"spec": json.dumps(SPEC)})["status"] == "ok"
    assert sp._handle_submit_scene_spec({"spec": {"spec": SPEC}})["status"] == "ok"  # wrapped twice
    bad = sp._handle_submit_scene_spec({"spec": json.dumps(SPEC)[:-3]})
    assert bad["status"] == "invalid" and "line 1 column" in bad["errors"][0]


def test_build_scene_takes_the_unwrapped_spec(fake_blender):
    out = sp._handle_build_scene({**SPEC, "snapshot": False, "exposure_check": False, "poly_budget": None})
    assert out["status"] in ("built", "built_with_issues"), out
    assert out["objects"] == 2
