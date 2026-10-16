"""build_scene's result as a weak model reads it (qwen3-14b-local-aidb5, 2026-10-17).

The Strata agent shows the first 2,500 characters of a tool result. The classroom build opened
with 27 created names and 28 material notes that each repeated the whole vocabulary: verdict,
next and scene_id never reached the model, which invented ``scene_001`` for edit_scene and
rebuilt the scene seven times until its context ran out. Inputs are that run's real call and
notes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ai_db_scene.build_report import compact_names, compact_notes, present_build
from ai_db_scene.tools import scene_pipeline as sp

CLASSROOM = json.loads((Path(__file__).parent / "fixtures" / "bench_qwen_classroom_aidb5.json").read_text())["args"]
_KINDS = ("(kinds: ['brass', 'ceramic', 'chrome', 'color', 'concrete', 'fabric', 'foliage', 'glass', 'gold', "
          "'leather', 'marble', 'metal', 'paint', 'plaster', 'plastic', 'rubber', 'stone', 'velvet', 'wood'], "
          "e.g. wood:walnut, fabric:linen:#6b7f99)")
#: the aidb5 classroom's repairs, as the run returned them
AIDB5_NOTES = (
    [f"setting.wall_material: material 'drywall' is not in the vocabulary; used the category default {_KINDS}"]
    + [f"objects[{i}].material: material '{m}' is not in the vocabulary; used the category default {_KINDS}"
       for i, m in enumerate(["wood_oak", "fabric_black", "glossy_white"]
                             + ["laminate_brown", "plastic_blue"] * 12)])


@pytest.fixture
def fake_blender(monkeypatch):
    class Fake:
        def batch(self, cmds, timeout=None):
            return [{"ok": 1, "status": "success"} for _ in cmds]

        def request(self, handler, params, timeout=None):
            return {}

        def run_code(self, code, timeout=None):
            return {}

    class Memory:
        def record_run(self, run):
            return "run-1"

        def snapshot_scene(self, **kw):
            return "0f6c2b8e-5d1a-4c4e-9a43-2a7a4f1d9c11"

    monkeypatch.setattr(sp, "BlenderClient", Fake)
    monkeypatch.setattr(sp, "_memory", lambda: Memory())
    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})


def test_classroom_result_leads_with_verdict_and_next(fake_blender):
    res = sp._handle_build_scene({**CLASSROOM, "exposure_check": False})
    text = json.dumps(res, indent=2)  # as the MCP transport sends it
    keys = list(res)
    assert keys[:4] == ["verdict", "next", "status", "scene_id"], keys
    head = text[:500]
    assert json.dumps(res["verdict"]) in head and json.dumps(res["next"]) in head, head
    assert res["scene_id"] in text[:700]
    assert "spec" not in res  # saved with the version; edit_scene starts from it
    assert len(text) < 2500, len(text)  # the whole result fits what the weak agent sees


def test_material_notes_fold_to_one_line_per_coercion():
    out = compact_notes(AIDB5_NOTES)
    assert len(out) == 7, out  # drywall, 5 distinct materials, the vocabulary once
    assert sum("kinds:" in n for n in out) == 1 and out[-1].startswith("material kinds: ['brass'")
    blue = next(n for n in out if "'plastic_blue'" in n)
    assert blue.startswith("objects[4,6,8,10,12,14 +6 more].material: material 'plastic_blue' is not in the vocabulary")
    assert len("\n".join(out)) < 1000
    assert compact_notes(out) == out  # idempotent: edit_scene presents a presented build again


def test_other_notes_keep_their_order_and_text():
    notes = ["dropped relation 'pallet1 on rack1'", "enlarged the setting to 60.0 x 36.0 m",
             "relations[3].b: 'desk_1' read as 'desk' (relations name the declared object id)",
             "relations[4].b: 'desk_1' read as 'desk' (relations name the declared object id)"]
    assert compact_notes(notes) == [notes[0], notes[1],
                                    "relations[3,4].b: 'desk_1' read as 'desk' (relations name the declared object id)"]


def test_created_runs_are_shortened_only_when_long():
    few = ["room_shell", "sofa", "table_1", "table_2", "table_3"]
    assert compact_names(few) == few
    names = ["room_shell", "teacher_desk_001", "teacher_chair_001", "whiteboard_001"] + [
        f"student_{k}_{i:03d}" for i in range(1, 13) for k in ("desk", "chair")]
    assert compact_names(names) == ["room_shell", "teacher_desk_001", "teacher_chair_001", "whiteboard_001",
                                    "student_desk_001 .. student_desk_012 (12)",
                                    "student_chair_001 .. student_chair_012 (12)"]


def test_unsaved_build_keeps_its_spec_and_empty_issue_fields_go():
    out = present_build({"status": "built", "spec": {"objects": []}, "verdict": "done", "next": "stop",
                         "issues_left": [{"check": "camera", "severity": "warning", "fix": None, "objects": [],
                                          "message": "m"}]})
    assert list(out)[:3] == ["verdict", "next", "status"] and out["spec"] == {"objects": []}
    assert out["issues_left"] == [{"check": "camera", "severity": "warning", "message": "m"}]
