"""The beach camera frames the shore group large, the water behind it (lead review of the f7e0879 demo).

The demo's beach view took in the whole cove with an 85-96 deg lens: the loungers under the parasol
were a speck at the right edge. By the water a view now scores the shore group's share of the frame's
width (SHORE_FILL wanted) and _shore_views adds views from inland looking across the group at the
water, the lens just wide enough for the group and for the band from its near edge up past the
horizon, skipping spots with a trunk between the lens and the group. Fixtures: the demo's spec and
the aidb7 beaches.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
from ai_db_scene import camera, terrain
from ai_db_scene.constants import BUILD_MAX_REPAIRS, SCENE_CAMERA_MAX_SHIFT, SCENE_CAMERA_WATER_EYE_M
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.terrain import apply_brief_defaults, water_from_brief
from ai_db_scene.tools import scene_pipeline as sp

FIX = Path(__file__).parent / "fixtures"
OUT = json.loads((FIX / "bench_specs_outdoor_aidb7.json").read_text())
DEMO = json.loads((FIX / "bench_qwen_demo_f7e0879.json").read_text())
BEACHES = {"demo/beach_cove": DEMO["beach_cove"], "last/qwen/beach_cove": OUT["last/qwen/beach_cove"],
           "last/lightning/beach_cove": OUT["last/lightning/beach_cove"]}


def _built(e):
    raw, _note = water_from_brief(copy.deepcopy(e["spec"]), e["brief"])
    spec, errors = sp._spec_from_args({"spec": raw}, remembered=False)
    assert spec is not None, errors
    spec, _notes = apply_brief_defaults(spec, e["brief"])
    fixed = solve_with_repairs(spec.to_dict(), 0, BUILD_MAX_REPAIRS)
    return fixed.spec, terrain.settle(fixed.spec, fixed.placements)


@pytest.mark.parametrize("key", sorted(BEACHES))
def test_shore_group_fills_a_good_share_with_the_water_behind(key):
    spec, pls = _built(BEACHES[key])
    group = camera.shore_group(pls)
    assert len(group) >= 2
    v = camera.best_views(spec, pls, SCENE_CAMERA_WATER_EYE_M)[0]
    vx, vy = v.aim[0] - v.x, v.aim[1] - v.y
    d = math.hypot(vx, vy)
    lo, hi = camera._span(group, v.x, v.y, vx / d, vy / d)
    half = math.radians(v.hfov_deg) / 2
    assert -half <= lo and hi <= half, key  # the group wholly in frame
    assert (hi - lo) / (2 * half) >= 0.22, (key, (hi - lo) / (2 * half), v)  # the f7e0879 view: ~0.08
    ter = terrain.terrain_of(spec)
    n = ter.to_water(v.x, v.y)
    assert (vx * n[0] + vy * n[1]) / d >= 0.5  # across the group toward the water
    # the horizon inside the frame, under its top: the water is the background, not cut away
    th = math.tan(half)
    t0 = max(-2 * SCENE_CAMERA_MAX_SHIFT * th, min(2 * SCENE_CAMERA_MAX_SHIFT * th, (v.aim[2] - v.z) / d))
    top = t0 + th / (16 / 9)
    assert top >= 0.05, (key, top)


def test_the_old_wide_view_scores_lower():
    spec, pls = _built(BEACHES["demo/beach_cove"])
    best = camera.best_views(spec, pls, SCENE_CAMERA_WATER_EYE_M)
    wide = [u for u in best if u.hfov_deg >= 80]
    assert wide and best[0].score > wide[0].score and best[0].hfov_deg < 70


def test_shore_group_is_the_cluster_round_the_parasol():
    from ai_db_scene.layout import Placement

    par = Placement("parasol", (0.0, 0.0, 1.2), 0.0, (2.8, 2.8, 2.45), "parasol")
    a = Placement("l1", (1.0, 0.0, 0.4), 0.0, (0.7, 1.95, 0.85), "sun_lounger")
    far = Placement("l2", (20.0, 0.0, 0.4), 0.0, (0.7, 1.95, 0.85), "sun_lounger")
    assert {p.id for p in camera.shore_group([par, a, far])} == {"parasol", "l1"}
    assert camera.shore_group([a]) == []
