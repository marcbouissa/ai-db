"""Relation words weak models write, read as the relation they mean (not refused).

qwen3-14b-local-aidb6 (the fixture: every build_scene call of that arm refused over a relation
word or wall name) wrote near, in_row, opposite, corner / at_corner, aligned_with, connects,
at_edge_of, evenly_spaced, ``on ceiling`` and walls as north_wall / south / front_wall. Each
refusal cost a retry; normalize_spec now rewrites them, with a note per rewrite.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from ai_db_scene.normalize import corner_walls, room_shape_from_brief, wall_side
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.spec import validate_spec

SPECS = json.loads((Path(__file__).parent / "fixtures" / "bench_qwen_aidb6_relation_vocab.json").read_text())
_VOCAB = re.compile(r"rel: '[a-z_ ]*' is not allowed|is not a wall|not a valid target for 'on'|"
                    r"cannot be related to itself|against_wall needs")


@pytest.mark.parametrize("name", sorted(SPECS))
def test_bench_relation_words_are_read_not_refused(name):
    spec, errors = validate_spec(SPECS[name])
    assert not [e for e in errors if _VOCAB.search(e)], errors
    if spec is not None:
        assert solve_with_repairs(spec.to_dict(), 0, 6).error is None


def _rels(name):
    spec, errors = validate_spec(SPECS[name])
    assert spec is not None, errors
    return spec, [(r.a, r.rel, r.b, r.at_deg) for r in spec.relations]


def test_corner_is_against_both_walls():
    spec, rels = _rels("qwen3-14b-local-aidb6/cafe#1")
    assert ("plant1", "against_wall", "wall_north", None) in rels and ("plant1", "against_wall", "wall_west", None) in rels
    assert ("plant2", "against_wall", "wall_south", None) in rels and ("plant2", "against_wall", "wall_east", None) in rels
    fixed = solve_with_repairs(spec.to_dict(), 0, 6)
    assert fixed.spec is not None and fixed.placements
    pl = {p.id: p for p in fixed.placements}
    w, d, _h = fixed.spec.setting.bounds_m  # the repair loop may have grown the room
    p1, p2 = pl["plant1"], pl["plant2"]
    assert p1.location[0] == pytest.approx(p1.dims[0] / 2) and p1.location[1] == pytest.approx(d - p1.dims[1] / 2)
    assert p2.location[0] == pytest.approx(w - p2.dims[0] / 2) and p2.location[1] == pytest.approx(p2.dims[1] / 2)


def test_wall_names_near_and_self_relations():
    _spec, rels = _rels("qwen3-14b-local-aidb6/cafe#2")
    assert ("plant1", "against_wall", "wall_north", None) in rels
    spec, rels = _rels("qwen3-14b-local-aidb6/warehouse#1")
    assert not any(a == b for a, _r, b, _d in rels)
    rack = next(o for o in spec.objects if o.id == "pallet_rack")
    assert rack.grid is not None and rack.grid[:2] == (1, 4)  # "four pallet racks in_row": one row
    assert any("evenly_spaced" in n and "dropped" in n for n in spec.notes)
    spec, rels = _rels("qwen3-14b-local-aidb6/warehouse#2")
    assert not any(b == "ceiling" for _a, _r, b, _d in rels)
    spec, rels = _rels("qwen3-14b-local-aidb6/park_corner#1")
    assert ("trash_can", "beside", "bench1", None) in rels or any(r[1] == "beside" for r in rels)
    assert any("no walls outdoors" in n for n in spec.notes)
    spec, rels = _rels("qwen3-14b-local-aidb6/round_lobby#1")
    assert not any(b == "entrance" for _a, _r, b, _d in rels)
    assert any("'entrance' is not an object" in n for n in spec.notes)


def test_opposite_is_facing_and_curved_corner_is_an_angle():
    raw = {"setting": {"kind": "indoor", "bounds_m": [8, 8, 3], "shape": "round"},
           "objects": [{"id": "plant", "category": "potted_plant"}, {"id": "chair_a", "category": "armchair"},
                       {"id": "chair_b", "category": "armchair"}],
           "relations": [{"a": "plant", "rel": "at_corner", "b": "south_east"},
                         {"a": "chair_a", "rel": "opposite", "b": "chair_b"}]}
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    assert spec.setting.shape == "ellipse"
    assert [(r.a, r.rel, r.b, r.at_deg) for r in spec.relations] == [
        ("plant", "against_wall", "wall", 315.0), ("chair_a", "facing", "chair_b", None)]


@pytest.mark.parametrize(("b", "walls"), [("north_west", ["wall_north", "wall_west"]), ("nw", ["wall_north", "wall_west"]),
                                          ("corner_se", ["wall_south", "wall_east"]), ("south", ["wall_south"]),
                                          ("entrance", None)])
def test_corner_words(b, walls):
    assert corner_walls(b) == walls


@pytest.mark.parametrize(("b", "wall"), [("north_wall", "wall_north"), ("front_wall", "wall_north"), ("south", "wall_south"),
                                         ("the east side", "wall_east"), ("wall_west", "wall_west"), ("sofa", None)])
def test_wall_words(b, wall):
    assert wall_side(b) == wall


def test_round_brief_gives_a_round_room():
    raw = {"setting": {"kind": "indoor", "bounds_m": [8, 8, 4]}, "objects": [{"id": "desk", "category": "reception_desk"}]}
    out, note = room_shape_from_brief(raw, "A round hotel lobby with a reception desk")
    assert out["setting"]["shape"] == "ellipse" and note and "round" in note
    assert room_shape_from_brief(raw, "A lobby with a round table")[1] is None
    assert room_shape_from_brief({**raw, "setting": {**raw["setting"], "shape": "rect"}}, "a round lobby")[1] is None
