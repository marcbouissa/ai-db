"""The build camera stands on the open side (Marc's review of the aidb5 round lobby, 2026-10-17).

"The camera in the lobby is flipped so sofas are facing the wall, preventing the open feel":
ai-db's build camera (not the model's: no bench run called set_camera, and spec.camera
positions were ignored) took the clearest room corner farthest from the heroes and aimed at
them, often from behind the seating. camera.py scores every spot by the fronts of the seating
and features it sees, the heroes in frame and the room to step back. Fixtures: the lobbies and
living rooms the bench built (round lobbies with the brief's round room).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ai_db_scene.camera import SEAT_NEAR_COVER, best_views, near_seating, view_issue
from ai_db_scene.edits import apply_steps
from ai_db_scene.normalize import room_shape_from_brief
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.spec import validate_spec
from ai_db_scene.tools import scene_pipeline as sp
from ai_db_scene.tools import scene_tools

FIX = Path(__file__).parent / "fixtures"
LOBBIES = json.loads((FIX / "bench_round_lobbies.json").read_text())
LIVING = {**{f"aidb5_{k}": v for k, v in json.loads((FIX / "bench_living_rooms_aidb5.json").read_text()).items()
             if k != "source"},
          **json.loads((FIX / "bench_qwen_living_room_aidb6.json").read_text())}


def _solve(raw, brief=""):
    raw, _note = room_shape_from_brief(raw, brief)
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), 0, 6)
    assert fixed.placements and fixed.spec is not None, fixed.error
    return fixed.spec, fixed.placements


def _sees_front(v, p):
    """dot(view ray to p, p's forward) < 0: the camera looks at p's front."""
    yaw = math.radians(p.rotation_deg_z)
    f = (math.sin(yaw), -math.cos(yaw))
    return (p.location[0] - v.x) * f[0] + (p.location[1] - v.y) * f[1] < 0


def _in_frustum(v, p):
    ax, ay = v.aim[0] - v.x, v.aim[1] - v.y
    ox, oy = p.location[0] - v.x, p.location[1] - v.y
    ang = abs(math.atan2(ax * oy - ay * ox, ax * ox + ay * oy))
    return math.degrees(ang) <= v.hfov_deg / 2


@pytest.mark.parametrize("name", sorted(LIVING))
def test_living_room_camera_faces_the_sofa_front(name):
    spec, pls = _solve(LIVING[name])
    v = best_views(spec, pls, 1.4)[0]
    by = {p.id: p for p in pls}
    sofa, table = by["sofa"], by["coffee_table"]
    assert _in_frustum(v, sofa) and _sees_front(v, sofa)
    assert _in_frustum(v, table)
    assert v.fronts > 0 and 50 <= v.hfov_deg <= 100 and v.z == pytest.approx(1.4)


@pytest.mark.parametrize("name", sorted(LOBBIES))
def test_lobby_camera_looks_from_the_open_side(name):
    spec, pls = _solve(LOBBIES[name], "A round hotel lobby")
    views = best_views(spec, pls, 1.4)
    v = views[0]
    by = {p.id: p for p in pls}
    desk = by["reception_desk"]
    chairs = [p for p in pls if p.category == "armchair"]

    def clean(u):  # no armchair side filling the bottom of the picture (aidb7 lobby review)
        return all(f <= SEAT_NEAR_COVER for _k, f in near_seating(pls, u.x, u.y, u.z, u.aim, u.hfov_deg))

    def open_side(u):
        return (_in_frustum(u, desk) and _sees_front(u, desk) and all(_in_frustum(u, c) for c in chairs)
                and sum(_sees_front(u, c) for c in chairs) >= 2)

    assert clean(v)
    if any(clean(u) and open_side(u) for u in views):  # a room big enough for both
        assert open_side(v)  # the feature from its front, not from behind the ring
    else:  # a small round room: the clean view across the ring, still seeing fronts
        assert sum(_sees_front(v, c) for c in chairs if c.id in v.seen) >= 1
    w, d, _h = spec.setting.bounds_m
    assert ((v.x - w / 2) / (w / 2)) ** 2 + ((v.y - d / 2) / (d / 2)) ** 2 < 1  # inside the round wall


def test_a_camera_behind_the_sofa_or_at_a_wall_gets_a_hint():
    spec, pls = _solve(LIVING["qwen_aidb6_living_room"])
    sofa = next(p for p in pls if p.id == "sofa")
    # behind the sofa, looking past its back at the TV wall (yaw 0 looks along +Y)
    behind = view_issue(spec, pls, [sofa.location[0], 0.3, 1.4], [90, 0, 0], 24)
    assert behind and "backs of the seating" in behind and "rotation [90, 0," in behind
    wall = view_issue(spec, pls, [3.0, 2.5, 1.4], [90, 0, 90], 50)  # yaw 90: looks along -X, at the west wall
    assert wall and "blank wall" in wall
    best = best_views(spec, pls, 1.4)[0]
    assert view_issue(spec, pls, [best.x, best.y, 1.4], [90, 0, best.yaw_deg], best.lens_mm) is None


def test_set_camera_returns_the_hint_for_the_last_build(monkeypatch):
    spec, pls = _solve(LIVING["qwen_aidb6_living_room"])
    monkeypatch.setitem(sp._LAST_BUILD, "spec", spec)
    monkeypatch.setitem(sp._LAST_BUILD, "placements", pls)
    monkeypatch.setattr(scene_tools, "_forward", lambda tool, args: {"status": "success", "result": {}})
    off = sp.frame_offset(spec)
    sofa = next(p for p in pls if p.id == "sofa")
    out = scene_tools._handle_set_camera({"location": [sofa.location[0] + off[0], 0.3 + off[1], 1.4],
                                          "rotation": [90, 0, 0], "lens": 24})
    assert "backs of the seating" in out.get("hint", "")


def test_camera_edits_and_spec_positions_are_explained():
    data, done, errors = apply_steps(LIVING["qwen_aidb6_living_room"],
                                     [{"op": "update_camera", "set": {"position": [4, 4, 3], "fov": 60}}])
    assert not errors and data is not None and "set_camera" in done[0]
