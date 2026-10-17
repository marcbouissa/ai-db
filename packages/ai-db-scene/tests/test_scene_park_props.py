"""Park and stock props read as the real thing (aidb7 visual review, renders bench/_after_*).

* a park's plaza in asphalt / dark paving is paved in light stone (a light paving stays);
* a park bench: wooden slats on cast-iron side frames; an outdoor bin: a lidded, slatted litter
  bin; street lamps keep clear of the benches (lightning: lamp1 'beside bench1' 0.3 m from it);
* a warehouse's crates are cardboard cartons unless the spec makes them wooden;
* a potted plant: a terracotta pot with soil and leaves of varied size on stalks (was a stack of
  foliage balls with no pot to be seen).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from ai_db_scene import camera, terrain, vegetation
from ai_db_scene.layout import PAIR_CLEARANCE_M
from ai_db_scene.recipes import recipe_spec
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.spec import SceneSpec
from ai_db_scene.terrain import apply_brief_defaults
from ai_db_scene.tools import scene_pipeline as sp

FIX = Path(__file__).parent / "fixtures"
OUT = json.loads((FIX / "bench_specs_outdoor_aidb7.json").read_text())
IN = json.loads((FIX / "bench_specs_aidb7.json").read_text())


def _solve(fx, key):
    e = fx[key]
    spec, errors = sp._spec_from_args({"spec": copy.deepcopy(e["spec"])}, remembered=False)
    assert spec is not None, errors
    spec, _notes = apply_brief_defaults(spec, e.get("brief") or IN["briefs"].get(key.split("/")[-1], ""))
    fixed = solve_with_repairs(spec.to_dict(), 0, 3)
    return spec, fixed


@pytest.mark.parametrize("mat", ["asphalt", "tarmac", "concrete:#2a2a2a", "stone:dark", "paving:black"])
def test_a_dark_plaza_is_paved_in_light_stone(mat):
    pave, note = terrain.plaza_material(mat)
    assert pave == terrain.PLAZA_MATERIAL and note and "too dark" in note


@pytest.mark.parametrize("mat", ["concrete", "stone:#c8c0b0", "brick", "granite"])
def test_a_light_paving_stays(mat):
    assert terrain.plaza_material(mat) == (mat, None)


def test_an_asphalt_park_says_so():
    e = copy.deepcopy(OUT["last/qwen/park_corner"])
    e["spec"]["setting"]["floor_material"] = "asphalt"
    spec, errors = sp._spec_from_args({"spec": e["spec"]}, remembered=False)
    assert spec is not None, errors
    spec, notes = apply_brief_defaults(spec, e["brief"])
    assert spec.setting.ground == "grass" and any("too dark for a park's plaza" in n for n in notes)
    fixed = solve_with_repairs(spec.to_dict(), 0, 3)
    cmds, _ = sp.build_layout_commands(fixed.spec, fixed.placements)
    pave = [c for c in cmds if c["type"] == "apply_material" and c["params"]["name"].startswith("plaza_")]
    assert pave and pave[0]["params"]["material"].startswith(terrain.PLAZA_MATERIAL)


def test_the_park_bench_is_slats_on_cast_iron_frames():
    parts = recipe_spec("bench", "b", (1.6, 0.5, 0.85))["parts"]
    seat = [p for p in parts if p["id"].startswith("slat_seat_")]
    back = [p for p in parts if p["id"].startswith("slat_back_")]
    frames = [p for p in parts if p["id"].startswith("frame_")]
    assert len(seat) >= 4 and len(back) >= 2 and all(p["material"].startswith("wood") for p in seat + back)
    ys = sorted(p["at"][1] for p in seat)
    assert all(b - a > p["size"][1] for a, b, p in zip(ys, ys[1:], seat, strict=False))  # gaps between the slats
    assert {p["material"] for p in frames} == {"rubber:#1c1c1c"}  # cast iron
    assert {"frame_l_front_leg", "frame_r_rear_leg", "frame_l_arm"} <= {p["id"] for p in frames}
    assert sp._KEEP_PARTS["bench"] == ("frame_",)  # a "wood" bench repaints the slats, never the iron


def test_an_outdoor_bin_is_a_slatted_litter_bin_with_a_lid():
    _spec, fixed = _solve(OUT, "last/qwen/park_corner")
    assert sp.context_recipe(fixed.spec, "trash_can", None) == "litter_bin"
    parts = recipe_spec("trash_can", "t", (0.5, 0.5, 0.9), kind="litter_bin")["parts"]
    ids = {p["id"] for p in parts}
    assert {"lid", "liner", "mouth"} <= ids and sum(i.startswith("slat_") for i in ids) >= 12
    room = SceneSpec.from_dict({"setting": {"kind": "indoor", "bounds_m": [5, 4, 3]},
                                "objects": [{"id": "bin", "category": "trash_can"}]})
    assert sp.context_recipe(room, "trash_can", None) is None  # an office bin stays the steel one
    cmds, _ = sp.build_layout_commands(fixed.spec, fixed.placements)
    (spec_cmd,) = [c for c in cmds if c["type"] == "build_asset" and c["params"]["spec"]["asset"] == "trash_can"]
    assert any(p["id"] == "lid" for p in spec_cmd["params"]["spec"]["parts"])


def test_street_lamps_keep_clear_of_the_benches():
    _spec, fixed = _solve(OUT, "last/lightning/park_corner")
    by = {p.id: p for p in fixed.placements}
    need = PAIR_CLEARANCE_M[frozenset(("street_lamp", "bench"))]
    for lamp in ("lamp1", "lamp2"):
        for b in ("bench1", "bench2", "bench3"):
            l, p = by[lamp], by[b]
            gap = camera._footprint_dist(l.location[0], l.location[1], p) - l.dims[0] / 2
            assert gap >= need - 0.05, (lamp, b, gap)


def test_warehouse_crates_are_cartons_unless_wooden():
    spec, fixed = _solve(IN, "last/qwen/warehouse")
    crate = next(o for o in fixed.spec.objects if o.category == "crate")
    assert crate.material == "wood"
    assert sp.context_recipe(fixed.spec, "crate", crate) == "cardboard_box"
    cmds, _ = sp.build_layout_commands(fixed.spec, fixed.placements, variation=sp.variation_key(spec))
    built = [c["params"]["spec"] for c in cmds if c["type"] == "build_asset" and c["params"]["spec"]["category"] == "crate"]
    assert built and all({"body", "tape_top", "label"} <= {p["id"] for p in b["parts"]} for b in built)
    oak = type(crate)(**{**crate.__dict__, "material": "wood:oak"})
    assert sp.context_recipe(fixed.spec, "crate", oak) is None
    named = type(crate)(**{**crate.__dict__, "id": "wooden_crate"})
    assert sp.context_recipe(fixed.spec, "crate", named) is None
    home = SceneSpec.from_dict({"setting": {"kind": "indoor", "bounds_m": [5, 4, 3]},
                                "objects": [{"id": "crate", "category": "crate"}]})
    assert sp.context_recipe(home, "crate", home.objects[0]) is None


def test_a_potted_plant_is_grown_in_a_pot():
    assert vegetation.has_plant("potted_plant")
    cmds = vegetation.plant_commands("potted_plant", "plant1", (0.6, 0.6, 1.2), [0, 0, 0], 0, 3, "ceramic:proj=box")
    assert '"kind": "potted"' in cmds[0]["params"]["code"]
    mats = {c["params"]["name"]: c["params"]["material"] for c in cmds[1:]}
    assert mats["plant1_pot"].startswith("ceramic") and mats["plant1_leaves"].startswith("foliage")
    green = vegetation.plant_commands("potted_plant", "p", (0.6, 0.6, 1.2), [0, 0, 0], 0, 3, "foliage:#4a7a3a:proj=box")
    gm = {c["params"]["name"]: c["params"]["material"] for c in green[1:]}
    assert gm["p_leaves"].startswith("foliage:#4a7a3a") and gm["p_pot"].startswith("ceramic:#b5603a")


def test_lobby_plants_are_grown_not_scattered_over():
    spec, fixed = _solve(IN, "last/qwen/round_lobby")
    cmds, _ = sp.build_layout_commands(fixed.spec, fixed.placements, variation=sp.variation_key(spec))
    code = [c["params"]["code"] for c in cmds if c["type"] == "run_code"]
    assert sum('"kind": "potted"' in c for c in code) == 2
    assert not any("plant1_1" in c and "max_leaves" in c for c in code)  # no leaf scatter on the grown plant
