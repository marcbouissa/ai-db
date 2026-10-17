"""Beach cove and outdoor daylight (bench beach_cove, aidb7 visual review): dusky light and grey
sand (the sky swamped the sun, the exposure probe took 2 stops off), a camera looking inland,
bushes and ferns scattered over the sand instead of along the back.

Inputs are the models' own stored specs (tests/fixtures/bench_specs_outdoor_aidb7.json).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
from ai_db_scene import camera, constants, terrain
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.terrain import apply_brief_defaults, lift_terrain
from ai_db_scene.tools import scene_pipeline as sp

FX = json.loads((Path(__file__).parent / "fixtures" / "bench_specs_outdoor_aidb7.json").read_text())
BEACHES = ("last/qwen/beach_cove", "last/lightning/beach_cove")


def _solved(key: str):
    e = FX[key]
    spec, errors = sp._spec_from_args({"spec": copy.deepcopy(e["spec"])}, remembered=False)
    assert spec is not None, errors
    spec, _notes = apply_brief_defaults(spec, e["brief"])
    fixed = solve_with_repairs(spec.to_dict(), 0, 3)
    assert fixed.placements is not None and fixed.spec is not None
    return fixed.spec, fixed.placements


@pytest.mark.parametrize("key", BEACHES)
def test_the_beach_camera_looks_across_the_shore_group_at_the_water(key):
    spec, pls = _solved(key)
    ter = terrain.terrain_of(spec)
    assert ter is not None and ter.water is not None
    v = camera.best_views(spec, pls, 2.5)[0]
    dx, dy = v.aim[0] - v.x, v.aim[1] - v.y
    n = ter.to_water(v.x, v.y)
    assert (dx * n[0] + dy * n[1]) / math.hypot(dx, dy) > 0.3  # toward the water, not inland
    group = {p.id for p in pls if p.category in ("sun_lounger", "parasol")}
    assert group <= set(v.seen)
    assert set(sp.coverage_targets(spec, pls)) >= group  # the Blender check asks for the same objects


@pytest.mark.parametrize("key", BEACHES)
def test_bushes_and_ferns_go_along_the_back(key):
    data, notes = lift_terrain(copy.deepcopy(FX[key]["spec"]))
    plants = {o["id"] for o in data["objects"] if o["category"] in terrain.BACK_PLANTS}
    assert plants
    for pid in plants:
        mine = [r for r in data["relations"] if r.get("a") == pid]
        assert mine == [{"a": pid, "rel": "back_of", "b": "water"}], mine
    assert any("goes along the back" in n for n in notes)
    spec, pls = _solved(key)
    ter = terrain.terrain_of(spec)
    assert ter is not None
    groups = [p for p in pls if p.category in ("sun_lounger", "parasol")]
    for p in pls:
        if p.category in terrain.BACK_PLANTS:  # behind the shore group, away from the water
            assert ter.signed(p.location[0], p.location[1]) < min(ter.signed(g.location[0], g.location[1]) for g in groups)


def test_a_bush_placed_by_a_real_relation_keeps_it():
    spec = {"setting": {"kind": "outdoor", "bounds_m": [20, 15, 5], "ground": "sand", "water": {"kind": "sea"}},
            "objects": [{"id": "bush", "category": "tropical_bush"}, {"id": "parasol", "category": "parasol"}],
            "relations": [{"a": "bush", "rel": "in_front_of", "b": "parasol"}]}
    data, _notes = lift_terrain(copy.deepcopy(spec))
    assert data["relations"] == spec["relations"]


def test_outdoor_daylight_is_a_warm_high_sun_behind_the_camera_and_a_weak_sky():
    spec, pls = _solved("last/qwen/beach_cove")
    off = sp.frame_offset(spec)
    cmds = sp.lighting_commands(spec, pls, off)
    rig = cmds[0]["params"]
    assert cmds[0]["type"] == "daylight_rig"
    assert rig["strength"] == constants.SCENE_SUN_ENERGY_OUTDOOR > 4.0
    assert rig["sky_strength"] == constants.SCENE_SKY_STRENGTH_OUTDOOR < 0.8
    assert rig["sun_elevation"] >= 45 and not rig["window_portals"]
    assert "AgX - Medium High Contrast" in cmds[1]["params"]["code"] and "0.76" in cmds[1]["params"]["code"]
    pose = sp.camera_poses(spec, pls, off)[0]
    view = math.radians(90.0 + pose["rotation"][2])
    sun = math.radians(rig["sun_azimuth"])
    assert math.cos(view) * math.cos(sun) + math.sin(view) * math.sin(sun) < -0.5  # the sun behind the camera


def test_indoor_lighting_is_unchanged():
    spec, errors = sp._spec_from_args({"spec": {"setting": {"kind": "indoor", "bounds_m": [5, 4, 2.8]},
                                                "objects": [{"id": "sofa", "category": "sofa"}], "relations": []}},
                                      remembered=False)
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), 0, 3)
    assert fixed.spec is not None and fixed.placements is not None
    rig = sp.lighting_commands(fixed.spec, fixed.placements, sp.frame_offset(fixed.spec))[0]["params"]
    assert rig["sun_azimuth"] == 135.0 and rig["sky_strength"] == 0.8 and rig["strength"] == constants.SCENE_SUN_ENERGY


def test_outdoor_exposure_band_is_brighter():
    lo, hi = constants.BUILD_EXPOSURE_BAND_OUTDOOR
    assert hi > constants.BUILD_EXPOSURE_BAND[1] and lo <= constants.BUILD_EXPOSURE_TARGET_OUTDOOR <= hi
