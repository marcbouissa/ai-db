"""A park corner: benches round the fountain, a lawn with a paved plaza, an outdoor bin, a
camera that shows the fountain and the trees whole (bench park_corner, aidb7 visual review).

Inputs are the models' own stored specs (tests/fixtures/bench_specs_outdoor_aidb7.json: the
last build_scene version of qwen3-14b-local-aidb7 and nemotron-3.5-lightning-free-aidb7): three
benches ``facing fountain`` stood 6-15 m from it, side-on; ``floor_material: concrete`` made a
flat grey slab; a trunk filled the middle of the qwen picture.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
from ai_db_scene import camera, terrain
from ai_db_scene.layout import OUTDOOR_DIMS, Placement
from ai_db_scene.normalize import normalize_spec
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.terrain import apply_brief_defaults
from ai_db_scene.tools import scene_pipeline as sp

FX = json.loads((Path(__file__).parent / "fixtures" / "bench_specs_outdoor_aidb7.json").read_text())
PARKS = ("last/qwen/park_corner", "last/lightning/park_corner")


def _solved(key: str):
    e = FX[key]
    spec, errors = sp._spec_from_args({"spec": copy.deepcopy(e["spec"])}, remembered=False)
    assert spec is not None, errors
    spec, notes = apply_brief_defaults(spec, e["brief"])
    fixed = solve_with_repairs(spec.to_dict(), 0, 3)
    assert fixed.placements is not None and fixed.spec is not None
    return fixed.spec, fixed.placements, notes


@pytest.mark.parametrize("key", PARKS)
def test_benches_facing_a_fountain_ring_it_at_a_sensible_radius(key):
    _spec, pls, _notes = _solved(key)
    by = {p.id: p for p in pls}
    f = by["fountain"]
    for b in ("bench1", "bench2", "bench3"):
        p = by[b]
        dx, dy = f.location[0] - p.location[0], f.location[1] - p.location[1]
        assert 2.5 <= math.hypot(dx, dy) <= 4.0, (b, math.hypot(dx, dy))
        front = camera._front(p)  # local -Y
        assert (front[0] * dx + front[1] * dy) / math.hypot(dx, dy) > math.cos(math.radians(20)), b


def test_trees_beside_the_fountain_stand_in_the_ring_between_the_benches():
    data, notes = normalize_spec(copy.deepcopy(FX["last/qwen/park_corner"]["spec"]))
    ring = [r["a"] for r in data["relations"] if r.get("rel") == "around" and r.get("b") == "fountain"]
    assert ring == ["bench1", "tree1", "bench2", "tree2", "bench3"]
    assert any("stands in the ring" in n for n in notes)


def test_a_facing_and_an_around_of_one_bench_make_one_ring_member():
    spec = {"setting": {"kind": "outdoor", "bounds_m": [12, 12, 5]},
            "objects": [{"id": "fountain", "category": "fountain"}, {"id": "bench", "category": "bench", "count": 3}],
            "relations": [{"a": "bench", "rel": "around", "b": "fountain", "gap_m": 9.0},
                          {"a": "bench", "rel": "facing", "b": "fountain"}]}
    data, _notes = normalize_spec(spec)
    assert data["relations"] == [{"a": "bench", "rel": "around", "b": "fountain", "gap_m": 2.55}]  # 4 m at most


def test_a_bench_placed_by_something_else_keeps_its_relations():
    spec = {"setting": {"kind": "outdoor", "bounds_m": [12, 12, 5]},
            "objects": [{"id": "fountain", "category": "fountain"}, {"id": "bench", "category": "bench"},
                        {"id": "tree", "category": "tree"}],
            "relations": [{"a": "bench", "rel": "beside", "b": "tree"}, {"a": "bench", "rel": "facing", "b": "fountain"}]}
    data, _notes = normalize_spec(copy.deepcopy(spec))
    assert data["relations"] == spec["relations"]


def test_a_park_with_a_concrete_floor_is_a_lawn_with_a_paved_plaza():
    spec, pls, notes = _solved("last/qwen/park_corner")
    assert spec.setting.ground == "grass" and spec.setting.floor_material == "concrete"
    assert any("paves the plaza" in n for n in notes)
    (pz,) = terrain.plaza_layout(spec, pls)
    by = {p.id: p for p in pls}
    for b in ("bench1", "bench2", "bench3"):  # the benches stand on the paving
        p = by[b]
        assert math.hypot(p.location[0] - pz["centre"][0], p.location[1] - pz["centre"][1]) + 0.8 <= pz["radius"]
    assert pz["path"] is not None and pz["path"]["length"] > pz["radius"]
    cmds = terrain.plaza_commands(spec, pls, (0.0, 0.0, 0.0), "concrete:proj=box")
    assert [c["type"] for c in cmds] == ["run_code", "apply_material"]
    assert cmds[1]["params"] == {"name": "plaza_fountain", "material": "concrete:proj=box"}


def test_no_plaza_on_a_beach_or_without_a_fountain():
    spec, pls, _notes = _solved("last/qwen/park_corner")
    assert terrain.plaza_layout(spec, [p for p in pls if p.category != "fountain"]) == []
    beach, bpls, _n = _solved("last/qwen/beach_cove")
    assert terrain.plaza_layout(beach, bpls) == []


def test_brief_grass_does_not_override_a_non_paving_floor():
    assert terrain.paving("concrete") and terrain.paving("stone:#a59e92") and not terrain.paving("wood")
    assert not terrain.paving(None)


def test_an_outdoor_trash_can_is_park_sized():
    _spec, pls, _notes = _solved("last/lightning/park_corner")
    (bin_,) = [p for p in pls if p.category == "trash_can"]
    assert bin_.dims == OUTDOOR_DIMS["trash_can"] == (0.5, 0.5, 0.9)


def test_a_trunk_in_the_middle_of_the_picture_costs_the_view():
    spec, pls, _notes = _solved("last/lightning/park_corner")
    by = {p.id: p for p in pls}
    f = by["fountain"]
    x, y = f.location[0], f.location[1] - 9.0
    clear = camera.score_view(spec, pls, x, y, 2.5, aim=(f.location[0], f.location[1], 0.6), hfov_deg=70)
    trunk = Placement("tree_x", (x, y + 3.0, 3.0), 0.0, (3.0, 3.0, 6.0), "tree")
    blocked = camera.score_view(spec, [*pls, trunk], x, y, 2.5, aim=(f.location[0], f.location[1], 0.6), hfov_deg=70)
    assert blocked.score < clear.score - 1.0


@pytest.mark.parametrize("key", PARKS)
def test_the_park_camera_sees_the_fountain_and_the_trees(key):
    spec, pls, _notes = _solved(key)
    best = camera.best_views(spec, pls, 2.5)[0]
    assert "fountain" in best.seen
    assert {"tree1", "tree2"} <= set(best.seen)
