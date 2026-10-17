"""Outdoor views fill the frame with the group (bench park_corner / beach_cove, aidb7 review).

Park: the camera stood 13 m back with a 93 deg lens (the canopies needed the height), the group
filled ~40 % of the width over a foreground of empty grass, and a ring tree's trunk stood in front
of the fountain. Beach: a 2.5 m eye saw the sea as a grey reflection, a bush filled the bottom
corner and a palm trunk the side. Inputs: the models' stored specs (bench_specs_outdoor_aidb7).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
from ai_db_scene import camera, terrain
from ai_db_scene.constants import BUILD_MAX_REPAIRS, SCENE_CAMERA_WATER_EYE_M
from ai_db_scene.layout import Placement
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.terrain import apply_brief_defaults, water_from_brief
from ai_db_scene.tools import scene_pipeline as sp

FX = json.loads((Path(__file__).parent / "fixtures" / "bench_specs_outdoor_aidb7.json").read_text())
PARKS = ("last/qwen/park_corner", "last/lightning/park_corner")
BEACHES = ("last/qwen/beach_cove", "last/lightning/beach_cove")


def _built(key: str):
    e = FX[key]
    raw, _note = water_from_brief(copy.deepcopy(e["spec"]), e["brief"])
    spec, errors = sp._spec_from_args({"spec": raw}, remembered=False)
    assert spec is not None, errors
    spec, _notes = apply_brief_defaults(spec, e["brief"])
    fixed = solve_with_repairs(spec.to_dict(), 0, BUILD_MAX_REPAIRS)
    return fixed.spec, terrain.settle(fixed.spec, fixed.placements)


def _width_share(v: camera.View, pls: list[Placement], ids: set[str]) -> float:
    """Share of the frame's width the objects ``ids`` span (with their footprints)."""
    vx, vy = v.aim[0] - v.x, v.aim[1] - v.y
    n = math.hypot(vx, vy)
    vx, vy = vx / n, vy / n
    lo = hi = 0.0
    for p in pls:
        if p.id not in ids:
            continue
        off = camera._off_angle(p, v.x, v.y, vx, vy)
        w = math.atan2(max(p.dims[0], p.dims[1]) / 2, math.hypot(p.location[0] - v.x, p.location[1] - v.y))
        lo, hi = min(lo, off - w), max(hi, off + w)
    return (hi - lo) / math.radians(v.hfov_deg)


@pytest.mark.parametrize("key", PARKS)
def test_the_park_group_fills_the_frame(key):
    spec, pls = _built(key)
    v = camera.best_views(spec, pls, 2.5)[0]
    group = {p.id for p in pls if p.category in ("bench", "fountain", "tree", "trash_can")}
    assert _width_share(v, pls, group) >= 0.6, (v, _width_share(v, pls, group))
    assert v.hfov_deg <= 75.0  # not a 93 deg lens for the canopies


@pytest.mark.parametrize("key", PARKS)
def test_no_trunk_in_front_of_the_fountain(key):
    spec, pls = _built(key)
    v = camera.best_views(spec, pls, 2.5)[0]
    by = {p.id: p for p in pls}
    vx, vy = v.aim[0] - v.x, v.aim[1] - v.y
    n = math.hypot(vx, vy)
    weights = camera.focus_weights(spec, pls)
    for t in pls:
        if t.category in camera.TRUNKS:
            r = math.hypot(t.location[0] - v.x, t.location[1] - v.y)
            off = camera._off_angle(t, v.x, v.y, vx / n, vy / n)
            assert camera._trunk_hides(t, v.x, v.y, r, off, by, weights, v.seen, vx / n, vy / n) == 0.0, t.id


def test_a_trunk_in_front_of_the_fountain_costs_the_view():
    spec, pls = _built("last/qwen/park_corner")
    f = next(p for p in pls if p.category == "fountain")
    x, y = f.location[0], f.location[1] - 12.0
    aim = (f.location[0], f.location[1], 2.0)
    clear = camera.score_view(spec, pls, x, y, 2.5, aim=aim, hfov_deg=60)
    trunk = Placement("tree_x", (x, f.location[1] - 3.0, 3.0), 0.0, (3.0, 3.0, 6.0), "tree")
    hidden = camera.score_view(spec, [*pls, trunk], x, y, 2.5, aim=aim, hfov_deg=60)
    assert hidden.score < clear.score - 1.5


@pytest.mark.parametrize("key", BEACHES)
def test_the_beach_camera_looks_across_the_group_at_the_water(key):
    spec, pls = _built(key)
    ter = terrain.terrain_of(spec)
    assert ter is not None
    v = camera.best_views(spec, pls, SCENE_CAMERA_WATER_EYE_M)[0]
    n = ter.to_water(v.x, v.y)
    vx, vy = v.aim[0] - v.x, v.aim[1] - v.y
    d = math.hypot(vx, vy)
    assert (vx * n[0] + vy * n[1]) / d >= 0.5  # toward the water
    assert d >= camera.WATER_MIN_DIST_M - 1e-6  # not a drone shot of the loungers
    assert ter.signed(v.x, v.y) < ter.dry()  # standing on the land
    assert {"parasol", "parasol1"} & set(v.seen)
    # no palm trunk near the lens in the frame
    for p in pls:
        if p.category == "palm_tree":
            r = math.hypot(p.location[0] - v.x, p.location[1] - v.y)
            off = abs(camera._off_angle(p, v.x, v.y, vx / d, vy / d))
            assert r >= 0.5 * d or off > math.radians(v.hfov_deg) / 2 + math.atan2(1.8, r), p.id


def test_the_beach_frame_puts_the_horizon_in_the_upper_part():
    spec, pls = _built("last/qwen/beach_cove")
    v = camera.best_views(spec, pls, SCENE_CAMERA_WATER_EYE_M)[0]
    d = math.hypot(v.aim[0] - v.x, v.aim[1] - v.y)
    centre = (v.aim[2] - v.z) / d  # tan of the frame centre's elevation (the lens shift)
    half_v = math.tan(math.radians(v.hfov_deg) / 2) / (16 / 9)
    assert -half_v < centre <= 0.0  # the horizon (tan 0) in the upper half: water, not sky


def test_water_scenes_shoot_from_a_dune_top():
    spec, pls = _built("last/qwen/beach_cove")
    (pose, *_rest) = sp.camera_poses(spec, pls, sp.frame_offset(spec))
    assert pose["location"][2] == pytest.approx(SCENE_CAMERA_WATER_EYE_M)
    park, ppls = _built("last/qwen/park_corner")
    assert sp.camera_poses(park, ppls, sp.frame_offset(park))[0]["location"][2] == pytest.approx(2.5)
