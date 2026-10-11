"""Layout solver: 20 synthetic specs, unsatisfiable specs, determinism."""

from __future__ import annotations

import random

import pytest
from ai_db_scene.layout import LayoutError, check_layout, solve_layout
from ai_db_scene.spec import SceneSpec


def _spec(d: dict) -> SceneSpec:
    return SceneSpec.from_dict(d)


def _indoor(rng: random.Random, i: int) -> dict:
    w, d = rng.uniform(5.0, 9.0), rng.uniform(4.5, 8.0)
    kind = i % 4
    objs: list[dict] = []
    rels: list[dict] = []
    if kind == 0:  # living room
        objs += [
            {"id": "sofa", "category": "sofa", "hero": True},
            {"id": "table", "category": "coffee_table"},
            {"id": "lamp", "category": "floor_lamp"},
            {"id": "rug", "category": "rug"},
            {"id": "armchair", "category": "armchair", "count": rng.randint(1, 2)},
            {"id": "vase", "category": "vase"},
            {"id": "tv_stand", "category": "tv_stand"},
            {"id": "plant", "category": "potted_plant"},
        ]
        rels += [
            {"a": "sofa", "rel": "against_wall", "b": rng.choice(["wall_north", "wall_east"])},
            {"a": "table", "rel": "in_front_of", "b": "sofa", "gap_m": round(rng.uniform(0.3, 0.6), 2)},
            {"a": "lamp", "rel": "beside", "b": "sofa"},
            {"a": "rug", "rel": "centered_in", "b": "room"},
            {"a": "vase", "rel": "on", "b": "table"},
            {"a": "armchair", "rel": "facing", "b": "table"},
            {"a": "tv_stand", "rel": "against_wall", "b": "wall_south"},
            {"a": "plant", "rel": "against_wall", "b": "wall_west"},
        ]
    elif kind == 1:  # dining + kitchen
        objs += [
            {"id": "dtable", "category": "dining_table", "hero": True},
            {"id": "chair", "category": "dining_chair", "count": rng.randint(4, 6)},
            {"id": "fridge", "category": "refrigerator"},
            {"id": "counter", "category": "kitchen_counter"},
            {"id": "bowl", "category": "vase", "count": 2},
            {"id": "light", "category": "pendant_lamp"},
        ]
        rels += [
            {"a": "dtable", "rel": "centered_in", "b": "room"},
            {"a": "chair", "rel": "around", "b": "dtable", "gap_m": 0.05},
            {"a": "fridge", "rel": "against_wall", "b": "wall_north"},
            {"a": "counter", "rel": "against_wall", "b": "wall_north"},
            {"a": "bowl", "rel": "on", "b": "dtable"},
            {"a": "light", "rel": "against_wall", "b": "wall_south"} if False else {"a": "light", "rel": "centered_in", "b": "dtable"},
        ]
    elif kind == 2:  # bedroom/office
        objs += [
            {"id": "bed", "category": "bed", "hero": True},
            {"id": "ns", "category": "nightstand", "count": 2},
            {"id": "wardrobe", "category": "wardrobe"},
            {"id": "desk", "category": "desk"},
            {"id": "stool", "category": "stool"},
            {"id": "laptop", "category": "laptop"},
            {"id": "window", "category": "window"},
            {"id": "door", "category": "door"},
        ]
        rels += [
            {"a": "bed", "rel": "against_wall", "b": "wall_west"},
            {"a": "ns", "rel": "beside", "b": "bed"},
            {"a": "wardrobe", "rel": "against_wall", "b": "wall_east"},
            {"a": "desk", "rel": "against_wall", "b": "wall_north"},
            {"a": "stool", "rel": "under", "b": "desk"},
            {"a": "laptop", "rel": "on", "b": "desk"},
            {"a": "window", "rel": "against_wall", "b": "wall_north"},
            {"a": "door", "rel": "against_wall", "b": "wall_south"},
        ]
    else:  # warehouse
        w, d = rng.uniform(12, 20), rng.uniform(10, 16)
        objs += [
            {"id": "rack", "category": "pallet_rack", "count": rng.randint(2, 3)},
            {"id": "pallet", "category": "pallet", "count": 3},
            {"id": "crate", "category": "crate", "count": 3},
            {"id": "forklift", "category": "forklift", "hero": True},
            {"id": "bench", "category": "workbench"},
            {"id": "barrier", "category": "safety_barrier", "count": 2},
            {"id": "barrel", "category": "barrel", "count": 2},
        ]
        rels += [
            {"a": "rack", "rel": "against_wall", "b": "wall_north"},
            {"a": "pallet", "rel": "in_front_of", "b": "rack", "gap_m": 0.8},
            {"a": "crate", "rel": "on", "b": "pallet"},
            {"a": "bench", "rel": "against_wall", "b": "wall_east"},
            {"a": "barrier", "rel": "beside", "b": "forklift", "gap_m": 1.0},
            {"a": "barrel", "rel": "beside", "b": "bench", "gap_m": 0.3},
        ]
    return {"setting": {"kind": "indoor", "bounds_m": [round(w, 2), round(d, 2), 3.0 if kind != 3 else 6.0]}, "objects": objs, "relations": rels}


def _outdoor(rng: random.Random, i: int) -> dict:
    w, d = rng.uniform(15, 30), rng.uniform(15, 30)
    return {
        "setting": {"kind": "outdoor", "bounds_m": [round(w, 1), round(d, 1), 20]},
        "objects": [
            {"id": "tree", "category": "tree", "count": rng.randint(2, 4)},
            {"id": "bench", "category": "bench", "hero": True},
            {"id": "lamp", "category": "street_lamp"},
            {"id": "rock", "category": "rock", "count": rng.randint(2, 5)},
            {"id": "bush", "category": "bush", "count": 3},
            {"id": "table", "category": "picnic_table"},
            {"id": "vase", "category": "vase"},
        ],
        "relations": [
            {"a": "lamp", "rel": "beside", "b": "bench", "gap_m": 0.5},
            {"a": "rock", "rel": "around", "b": "tree", "gap_m": 0.2},
            {"a": "bush", "rel": "beside", "b": "bench", "gap_m": 0.3},
            {"a": "vase", "rel": "on", "b": "table"},
            {"a": "bench", "rel": "facing", "b": "table"},
        ],
    }


def synthetic_specs() -> list[dict]:
    rng = random.Random(1234)
    return [(_indoor(rng, i) if i < 14 else _outdoor(rng, i)) for i in range(20)]


@pytest.mark.parametrize("idx", range(20))
def test_synthetic_specs_satisfy_all_constraints(idx):
    spec = _spec(synthetic_specs()[idx])
    placements = solve_layout(spec, seed=idx)
    assert len(placements) == sum(o.count for o in spec.objects)
    issues = check_layout(spec, placements)
    assert issues == [], issues  # overlaps, bounds, support, every hard relation, facing


def test_unsatisfiable_too_many_objects_names_object():
    spec = _spec({
        "setting": {"kind": "indoor", "bounds_m": [4, 3, 2.8]},
        "objects": [{"id": "bed", "category": "bed", "count": 4}],
        "relations": [],
    })
    with pytest.raises(LayoutError) as ei:
        solve_layout(spec)
    assert "cannot place 'bed_" in str(ei.value)
    assert any(c.startswith("bed_") for c in ei.value.conflicts)


def test_contradictory_walls_name_the_relations():
    spec = _spec({
        "setting": {"kind": "indoor", "bounds_m": [6, 5, 2.8]},
        "objects": [{"id": "sofa", "category": "sofa"}],
        "relations": [
            {"a": "sofa", "rel": "against_wall", "b": "wall_north"},
            {"a": "sofa", "rel": "against_wall", "b": "wall_south"},
        ],
    })
    with pytest.raises(LayoutError) as ei:
        solve_layout(spec)
    assert "sofa against_wall wall_north" in ei.value.conflicts or "sofa against_wall wall_south" in ei.value.conflicts


def test_surface_object_without_on_is_an_error():
    spec = _spec({"setting": {"kind": "indoor", "bounds_m": [5, 5, 3]}, "objects": [{"id": "l", "category": "laptop"}], "relations": []})
    with pytest.raises(LayoutError, match="must rest on a surface"):
        solve_layout(spec)


def test_on_target_that_cannot_support():
    spec = _spec({
        "setting": {"kind": "indoor", "bounds_m": [5, 5, 3]},
        "objects": [{"id": "l", "category": "laptop"}, {"id": "s", "category": "sofa"}],
        "relations": [{"a": "l", "rel": "on", "b": "s"}],
    })
    with pytest.raises(LayoutError) as ei:
        solve_layout(spec)
    assert ei.value.conflicts == ["l on s"]


def test_circular_relations_named():
    spec = _spec({
        "setting": {"kind": "outdoor", "bounds_m": [20, 20, 5]},
        "objects": [{"id": "a", "category": "bench"}, {"id": "b", "category": "bench"}],
        "relations": [{"a": "a", "rel": "beside", "b": "b"}, {"a": "b", "rel": "beside", "b": "a"}],
    })
    with pytest.raises(LayoutError) as ei:
        solve_layout(spec)
    assert set(ei.value.conflicts) == {"a beside b (gap 0.15 m)", "b beside a (gap 0.15 m)"}


def test_measured_dims_are_used():
    spec = _spec({"setting": {"kind": "indoor", "bounds_m": [8, 8, 3]}, "objects": [{"id": "sofa", "category": "sofa"}], "relations": [{"a": "sofa", "rel": "against_wall", "b": "wall_south"}]})
    (p,) = solve_layout(spec, measured_dims={"sofa": (3.0, 1.2, 0.9)})
    assert p.dims == (3.0, 1.2, 0.9)
    assert p.location[1] == pytest.approx(0.6, abs=1e-3)  # flush with y=0
    assert p.location[2] == pytest.approx(0.45)


def test_deterministic_for_seed_and_differs_across_seeds():
    spec = _spec(synthetic_specs()[0])
    a = solve_layout(spec, seed=7)
    assert a == solve_layout(spec, seed=7)
    assert a != solve_layout(spec, seed=8)


def test_centered_in_a_rug_means_on_it_not_its_exact_center():
    """Benchmark case: table centered_in the rug *and* in front of the sofa (wall-backed)."""
    from ai_db_scene.layout import check_layout, solve_layout
    from ai_db_scene.spec import validate_spec
    spec, errors = validate_spec({
        "setting": {"kind": "indoor", "bounds_m": [5.0, 4.0, 2.7]},
        "objects": [{"id": "sofa", "category": "sofa"}, {"id": "tv_stand", "category": "tv_stand"},
                    {"id": "coffee_table", "category": "coffee_table"}, {"id": "rug", "category": "rug"}],
        "relations": [{"a": "sofa", "rel": "against_wall", "b": "wall_north"},
                      {"a": "tv_stand", "rel": "against_wall", "b": "wall_south"},
                      {"a": "coffee_table", "rel": "in_front_of", "b": "sofa"},
                      {"a": "rug", "rel": "centered_in", "b": "room"},
                      {"a": "coffee_table", "rel": "centered_in", "b": "rug"}]})
    assert spec is not None, errors
    placements = solve_layout(spec)
    assert check_layout(spec, placements) == []
    table, rug = (next(p for p in placements if p.id == i) for i in ("coffee_table", "rug"))
    assert abs(table.location[1] - rug.location[1]) <= rug.dims[1] / 2  # on the rug


def test_grid_places_rows_of_desks_as_one_block():
    """Benchmark case: 12 student desks in rows (chained in_front_of between desks failed)."""
    from ai_db_scene.layout import check_layout, solve_layout
    from ai_db_scene.spec import validate_spec
    raw = {
        "setting": {"kind": "indoor", "bounds_m": [10.0, 8.0, 3.0]},
        "objects": [{"id": "whiteboard", "category": "whiteboard"},
                    {"id": "teacher_desk", "category": "desk"},
                    {"id": "desk", "category": "desk", "grid": {"rows": 3, "cols": 4}}],
        "relations": [{"a": "whiteboard", "rel": "against_wall", "b": "wall_north"},
                      {"a": "teacher_desk", "rel": "in_front_of", "b": "whiteboard", "gap_m": 1.0},
                      {"a": "desk", "rel": "facing", "b": "whiteboard"}]}
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    assert spec.objects[2].count == 12
    placements = solve_layout(spec)
    desks = [p for p in placements if p.id.startswith("desk_")]
    assert len(desks) == 12 and len({p.rotation_deg_z for p in desks}) == 1
    assert all(p.facing_target is None for p in desks)  # the block faces the board, not each cell
    assert check_layout(spec, placements) == []
    moved = [p if p.id != "desk_5" else type(p)(p.id, (p.location[0] + 0.5, *p.location[1:]), p.rotation_deg_z,
                                                p.dims, p.category) for p in placements]
    assert any("desk_5 is off its grid position" in i for i in check_layout(spec, moved))


def test_grid_count_must_match():
    from ai_db_scene.spec import validate_spec
    _spec, errors = validate_spec({"setting": {"kind": "indoor", "bounds_m": [6, 6, 3]},
                                   "objects": [{"id": "t", "category": "desk", "count": 5,
                                                "grid": {"rows": 2, "cols": 2}}]})
    assert any("does not match grid 2x2" in e for e in errors)


def test_facing_relations_hold_across_seeds():
    """Benchmark spec (local model): armchairs against side walls facing the coffee table.
    The solver kept a layout 48 deg off, which validate_scene blocks."""
    import json
    from pathlib import Path

    from ai_db_scene.layout import check_layout, solve_layout
    from ai_db_scene.spec import validate_spec
    raw = json.loads((Path(__file__).parent / "fixtures" / "living_room_spec.json").read_text())
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    for seed in range(5):
        assert check_layout(spec, solve_layout(spec, seed=seed)) == [], seed
