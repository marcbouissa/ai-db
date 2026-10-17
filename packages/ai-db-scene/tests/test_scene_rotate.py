"""edit_scene can do what the verdict asks: "rotate it 180 deg".

qwen3-14b-local-aidb6's classroom verdict said "teacher_desk faces Room_Walls 0.43 m away ...:
rotate it 180 deg"; next listed ops without any rotation, and the model's
``update_object {set: {rotation: [0, 0, 180]}}`` was refused ("fields ['rotation'] not
editable"). Its living room had the validator turn the TV in Blender only, so every rebuild had
it facing the wall again. Now: ObjectSpec.turn_deg (quarter turns on top of the yaw the
relations give), edit_scene ``rotate``, the refused form read as rotate, the verdict's next
names the exact step, and an applied faces_wall fix is kept in the stored spec.
"""

from __future__ import annotations

import json
import re

import pytest
from ai_db_scene.edits import OPS, apply_steps
from ai_db_scene.layout import check_layout, solve_layout
from ai_db_scene.spec import validate_spec
from ai_db_scene.tools import scene_pipeline as sp

ROOM = {"setting": {"kind": "indoor", "bounds_m": [5.0, 4.0, 3.0]},
        "objects": [{"id": "teacher_desk", "category": "desk"}, {"id": "shelf", "category": "bookshelf"}],
        "relations": [{"a": "teacher_desk", "rel": "against_wall", "b": "wall_north"},
                      {"a": "shelf", "rel": "against_wall", "b": "wall_west"}]}


@pytest.mark.parametrize(("step", "turn"), [
    ({"op": "rotate", "id": "teacher_desk", "deg": 180}, 180),
    ({"op": "update_object", "id": "teacher_desk", "set": {"rotation": [0.0, 0.0, 180.0]}}, 180),  # qwen aidb6
    ({"op": "turn", "target": "teacher_desk", "degrees": "90 deg"}, 90),
    ({"op": "flip", "id": "teacher_desk"}, 180),
    ({"op": "rotate", "id": "teacher_desk", "deg": -90}, 270),
    ({"op": "rotate", "id": "teacher_desk", "deg": 100}, 90),
])
def test_rotate_steps(step, turn):
    data, done, errors = apply_steps(ROOM, [step])
    assert not errors and data is not None, errors
    assert data["objects"][0]["turn_deg"] == turn and done[0].startswith("turned teacher_desk")
    spec, errs = validate_spec(data)
    assert spec is not None and spec.objects[0].turn_deg == turn, errs


def test_rotate_without_degrees_says_how():
    _data, _done, errors = apply_steps(ROOM, [{"op": "rotate", "id": "teacher_desk"}])
    assert errors and "deg: 180" in errors[0]


def test_turn_deg_is_applied_on_top_of_the_relations():
    plain = {p.id: p for p in solve_layout(validate_spec(ROOM)[0])}
    data, _done, _e = apply_steps(ROOM, [{"op": "rotate", "id": "teacher_desk", "deg": 180},
                                         {"op": "rotate", "id": "shelf", "deg": 90}])
    spec, errors = validate_spec(data)
    assert spec is not None, errors
    pl = {p.id: p for p in solve_layout(spec)}
    desk, shelf = pl["teacher_desk"], pl["shelf"]
    assert desk.rotation_deg_z == pytest.approx((plain["teacher_desk"].rotation_deg_z + 180) % 360)
    assert desk.location[1] == pytest.approx(plain["teacher_desk"].location[1])  # still flush with the wall
    assert shelf.rotation_deg_z == pytest.approx((plain["shelf"].rotation_deg_z + 90) % 360)
    assert shelf.dims == plain["shelf"].dims  # its own dims ...
    assert shelf.location[0] == pytest.approx(shelf.dims[0] / 2)  # ... its width now runs out from the wall
    assert not check_layout(spec, list(pl.values()))


def test_turn_deg_values():
    for raw, want, noted in ((-90, 270, False), (175, 180, True), (450, 90, False)):
        spec, errors = validate_spec({**ROOM, "objects": [{"id": "teacher_desk", "category": "desk", "turn_deg": raw},
                                                          ROOM["objects"][1]]})
        assert spec is not None and spec.objects[0].turn_deg == want, errors
        assert any("turn_deg" in n for n in spec.notes) == noted
        assert spec.to_dict()["objects"][0]["turn_deg"] == want
    _spec, errors = validate_spec({**ROOM, "objects": [{"id": "teacher_desk", "category": "desk", "turn_deg": "left"},
                                                       ROOM["objects"][1]]})
    assert any("turn_deg" in e for e in errors)


_FACES = {"check": "faces_wall", "severity": "blocking", "objects": ["teacher_desk", "Room_Walls"],
          "message": ("teacher_desk faces Room_Walls 0.43 m away (front_axis -Y); its back has open space: "
                      "rotate it 180 deg"),
          "fix": {"handler": "modify_object", "params": {"name": "teacher_desk", "rotation": [0, 0, 0]}}}


def test_verdict_names_the_rotate_step_and_only_existing_ops():
    out = {"objects": 2, "scene_id": "blender-1", "issues_left": [_FACES], "failed_commands": []}
    verdict, nxt = sp.build_verdict(out)
    assert "rotate it 180 deg" in verdict
    assert json.dumps({"op": "rotate", "id": "teacher_desk", "deg": 180}) in nxt
    ops = re.search(r"\(ops: ([a-z_, ]+)\)", nxt)
    assert ops and all(o in OPS for o in ops.group(1).split(", "))


@pytest.fixture
def fake_blender(monkeypatch):
    saved = {}

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
            saved.update(kw)
            return "scene-1"

    reports = iter([[_FACES]])
    monkeypatch.setattr(sp, "BlenderClient", Fake)
    monkeypatch.setattr(sp, "_memory", lambda: Memory())
    monkeypatch.setattr(sp, "_handle_validate_scene",
                        lambda args: {"status": "success", "result": {"issues": next(reports, [])}})
    return saved


def test_an_applied_faces_wall_fix_is_kept_in_the_stored_spec(fake_blender):
    out = sp._handle_build_scene({"spec": ROOM, "exposure_check": False})
    stored = fake_blender["metadata"]["spec"]
    desk = next(o for o in stored["objects"] if o["id"] == "teacher_desk")
    assert desk["turn_deg"] == 180
    assert any("turned 180 deg" in r for r in out["repairs"]), out["repairs"]
