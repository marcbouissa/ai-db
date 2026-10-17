"""Round rooms (Marc's review of the aidb5 round lobbies, 2026-10-17).

qwen3-14b-local-aidb5 built "a round hotel lobby" as a rectangle: its spec had no
``setting.shape`` and the aidb5 runner passed no brief, so nothing said "round". Its armchairs
(four objects, each ``around low_table``) were four rings of one, each at a random angle and
yaw; lightning aidb5 had the same armchairs randomly turned and its reception desk skewed in the
middle of the round room. Fixtures: the specs those runs (and qwen aidb6) built.

Now: a round brief gives an ellipse room, all objects ``around`` one target share one ring
(evenly spaced, each facing the table's centre, the table square in the open middle), and a
desk or counter left free in a round room stands with its back tangent to the curved wall.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ai_db_scene.layout import check_layout
from ai_db_scene.normalize import room_shape_from_brief
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.shapes import RoomShape
from ai_db_scene.spec import validate_spec
from ai_db_scene.tools import scene_pipeline as sp

SPECS = json.loads((Path(__file__).parent / "fixtures" / "bench_round_lobbies.json").read_text())
BRIEF = ("A round hotel lobby with a reception desk, four armchairs around a low table, two potted plants "
         "and a pendant lamp.")


def _solve(name, seed=0):
    raw, _note = room_shape_from_brief(SPECS[name], BRIEF)
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), seed, 6)
    assert fixed.error is None and fixed.placements and fixed.spec is not None, fixed.error
    assert not fixed.dropped, fixed.dropped
    return fixed.spec, {p.id: p for p in fixed.placements}


def _front(p):
    yaw = math.radians(p.rotation_deg_z)
    return math.sin(yaw), -math.cos(yaw)


def test_round_brief_gives_an_ellipse_room():
    assert "shape" not in SPECS["qwen_aidb5_round_lobby"]["setting"]  # why it came out rectangular
    spec, _pl = _solve("qwen_aidb5_round_lobby")
    assert spec.setting.shape == "ellipse"
    # its wall_north became the curved wall's north
    assert [(r.b, r.at_deg) for r in spec.relations if r.a == "reception_desk"] == [("wall", 90.0)]


@pytest.mark.parametrize("name", sorted(SPECS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_armchairs_ring_the_table_evenly_and_face_it(name, seed):
    spec, pl = _solve(name, seed)
    table = next(p for p in pl.values() if p.category == "coffee_table")
    chairs = [p for p in pl.values() if p.category == "armchair"]
    assert len(chairs) == 4
    angles = sorted(math.degrees(math.atan2(c.location[1] - table.location[1], c.location[0] - table.location[0])) % 360
                    for c in chairs)
    steps = [(b - a) % 360 for a, b in zip(angles, angles[1:] + angles[:1], strict=True)]
    assert all(abs(s - 90) < 1 for s in steps), angles  # evenly spaced
    for c in chairs:  # each faces the table's centre
        f = _front(c)
        to = (table.location[0] - c.location[0], table.location[1] - c.location[1])
        assert (f[0] * to[0] + f[1] * to[1]) / math.hypot(*to) > math.cos(math.radians(3))
    assert table.rotation_deg_z % 90 == pytest.approx(0, abs=1e-6)  # square, not skewed
    w, d, _h = spec.setting.bounds_m
    assert math.dist(table.location[:2], (w / 2, d / 2)) < 0.25 * max(w, d)  # in the open middle
    assert not [i for i in check_layout(spec, list(pl.values())) if i.startswith("hard")]


@pytest.mark.parametrize("name", sorted(SPECS))
@pytest.mark.parametrize("seed", [0, 1])
def test_reception_desk_is_tangent_to_the_curved_wall(name, seed):
    spec, pl = _solve(name, seed)
    desk = pl["reception_desk"]
    w, d, _h = spec.setting.bounds_m
    shape = RoomShape("ellipse", w, d)
    p, n = shape.wall_ref("wall", None, desk.location[0], desk.location[1])
    f = _front(desk)
    assert f[0] * n[0] + f[1] * n[1] > 0.99  # back to the wall, facing into the room
    back = (desk.location[0] - f[0] * desk.dims[1] / 2, desk.location[1] - f[1] * desk.dims[1] / 2)
    assert math.dist(back, p) < 0.3  # flush with the curve (a flat back on a curve: a small sagitta)


def test_uneven_ring_falls_back_when_the_room_is_too_small():
    raw = {"setting": {"kind": "indoor", "bounds_m": [3.2, 3.2, 3]},
           "objects": [{"id": "table", "category": "dining_table"}, {"id": "chair", "category": "dining_chair", "count": 4},
                       {"id": "shelf", "category": "bookshelf"}],
           "relations": [{"a": "chair", "rel": "around", "b": "table"}, {"a": "shelf", "rel": "against_wall", "b": "wall_north"}]}
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    assert solve_with_repairs(spec.to_dict(), 0, 6).placements


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
            return "scene-1"

    monkeypatch.setattr(sp, "BlenderClient", Fake)
    monkeypatch.setattr(sp, "_memory", lambda: Memory())
    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})


def test_build_scene_reads_round_from_the_brief(fake_blender):
    out = sp._handle_build_scene({"spec": SPECS["qwen_aidb5_round_lobby"], "brief": BRIEF, "exposure_check": False})
    assert any("round room" in r for r in out["repairs"]), out["repairs"]
    assert out["status"] == "built", out
