"""The build camera on the aidb7 scenes (visual review, 2026-10-17).

* Foreground occluders: a potted plant (cafe), an armchair (living room) filled a corner of the
  picture from under a metre away. No object within NEAR_M of the camera may cover more than
  COVER_MAX of the frame.
* Classroom: the camera stood at the board looking at the class with the board edge-on; it now
  stands at the back and looks over the class at the board.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ai_db_scene.camera import COVER_MAX, SENSOR_MM, best_views, foreground_cover
from ai_db_scene.normalize import room_shape_from_brief, table_shape_from_brief
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

FIX = json.loads((Path(__file__).parent / "fixtures" / "bench_specs_aidb7.json").read_text())
KEYS = ["raw/qwen/living_room", "raw/lightning/living_room", "last/qwen/cafe", "raw/lightning/cafe",
        "raw/qwen/classroom", "raw/lightning/classroom", "last/qwen/round_lobby", "raw/lightning/round_lobby",
        "last/qwen/warehouse"]


def _solve(key):
    raw = dict(FIX[key]["spec"])
    raw.pop("brief", None)
    brief = FIX["briefs"][key.split("/")[-1]]
    raw, _ = room_shape_from_brief(raw, brief)
    raw, _ = table_shape_from_brief(raw, brief)
    spec, errors = sp._spec_from_args({"spec": raw})
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), 0, 6)
    assert fixed.placements, fixed.error
    return fixed.spec, fixed.placements


def _pose(spec, placements):
    pose = sp.camera_poses(spec, placements, (0.0, 0.0, 0.0))[0]
    x, y, z = pose["location"]
    yaw = math.radians(pose["rotation"][2])
    hfov = math.degrees(2 * math.atan(SENSOR_MM / 2 / pose["lens"]))
    return x, y, z, (-math.sin(yaw), math.cos(yaw)), hfov, pose["shift_y"] * SENSOR_MM / pose["lens"]


def _cover(placements, view):
    x, y, z, (dx, dy), hfov, tilt = view  # tilt: the lens shift as a tangent
    return foreground_cover(placements, x, y, z, (x + dx * 5, y + dy * 5, z + tilt * 5), hfov)


@pytest.mark.parametrize("key", KEYS)
def test_no_foreground_occluder(key):
    spec, placements = _solve(key)
    cover = _cover(placements, _pose(spec, placements))
    assert all(f <= COVER_MAX for _k, f in cover), cover


@pytest.mark.parametrize("key", ["raw/qwen/classroom", "raw/lightning/classroom"])
def test_classroom_view_looks_over_the_class_at_the_board(key):
    spec, placements = _solve(key)
    board = next(p for p in placements if p.category == "whiteboard")
    x, y, _z, (dx, dy), _hfov, _t = _pose(spec, placements)
    to = (board.location[0] - x, board.location[1] - y)
    d = math.hypot(*to)
    assert d > 0.6 * spec.setting.bounds_m[1]  # from the back half of the room
    yaw = math.radians(board.rotation_deg_z)
    front = (math.sin(yaw), -math.cos(yaw))
    assert -(dx * front[0] + dy * front[1]) > 0.5  # the board face-on-ish (aidb7: ~85 deg, edge-on)
    assert board.id in best_views(spec, placements, 1.4)[0].seen


@pytest.mark.parametrize("key", ["raw/qwen/living_room", "last/qwen/cafe", "raw/lightning/cafe"])
@pytest.mark.parametrize("category,dims", [("potted_plant", (0.5, 0.5, 1.1)), ("armchair", (0.85, 0.9, 0.9))])
def test_a_near_plant_or_armchair_moves_the_camera(key, category, dims):
    """Put a plant / an armchair just in front of the lens of the view the build picked (the
    aidb7 renders: a bush filling the cafe's left third, an armchair's back the living room's
    bottom-right corner): the camera now picks a view that does not have it in the foreground."""
    from ai_db_scene.layout import Placement

    spec, placements = _solve(key)
    view = _pose(spec, placements)
    x, y, _z, (dx, dy), _hfov, _t = view
    fwd = 0.9 + max(dims[:2]) / 2  # its near face 0.9 m from the lens
    ox, oy = x + dx * fwd - dy * 0.3, y + dy * fwd + dx * 0.3
    near = Placement("intruder", (ox, oy, dims[2] / 2), 0.0, dims, category)
    if not any(f > COVER_MAX for _k, f in _cover([near], view)):
        pytest.skip("from this view the intruder does not fill the foreground")
    placements = [*placements, near]
    cover = _cover(placements, _pose(spec, placements))
    assert all(f <= COVER_MAX for _k, f in cover), cover
