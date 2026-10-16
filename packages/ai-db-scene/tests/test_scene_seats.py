"""Chairs at their desks and tables (bench mining 2026-10-16, item 7).

Judges in 7+ scenes: "chairs clustered apart from the desks", "chairs are scattered and
unpaired". The spec shapes are those of the failing bench calls (a grid of chairs
``in_front_of`` a grid of desks, chairs ``around`` tables) and the new ``seats`` field.
"""

from __future__ import annotations

import math

from ai_db_scene import ergonomics
from ai_db_scene.edits import apply_steps
from ai_db_scene.layout import check_layout, solve_layout
from ai_db_scene.spec import validate_spec


def _solve(raw):
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    pls = solve_layout(spec)
    assert check_layout(spec, pls) == []
    return spec, pls


def _gap(a, b) -> float:
    return ergonomics._gap(a, b)


def test_a_chair_grid_in_front_of_a_desk_grid_becomes_seats():
    # big-pickle-aidb4/classroom#15: "cannot place 'student_chairs' (chair, 3.9x3.3x0.9 m) ... 2300x out of
    # bounds. relations of 'student_chairs': ['student_chairs in_front_of student_desks (gap 0.1 m)']"
    raw = {"setting": {"kind": "indoor", "bounds_m": [10.0, 8.0, 3.2]},
           "objects": [{"id": "board", "category": "whiteboard"},
                       {"id": "student_desks", "category": "desk", "grid": {"rows": 3, "cols": 4}},
                       {"id": "student_chairs", "category": "chair", "grid": {"rows": 3, "cols": 4}}],
           "relations": [{"a": "board", "rel": "against_wall", "b": "wall_north"},
                         {"a": "student_desks", "rel": "facing", "b": "board"},
                         {"a": "student_chairs", "rel": "in_front_of", "b": "student_desks", "gap_m": 0.1}]}
    spec, pls = _solve(raw)
    assert spec.objects[1].seats == 1 and [o.id for o in spec.objects] == ["board", "student_desks"]
    desks = [p for p in pls if p.category == "desk"]
    chairs = [p for p in pls if p.category == "chair"]
    assert len(desks) == len(chairs) == 12
    assert all(min(_gap(c, d) for c in chairs) <= ergonomics.SEAT_REACH_M for d in desks)
    assert not [i for i in ergonomics.check(spec, pls) if "no chair" in i["message"]]
    # the desks face the board, so each chair is behind its desk: the student looks at the board
    board = next(p for p in pls if p.id == "board")
    for k, d in enumerate(desks, 1):
        c = next(p for p in chairs if p.id == f"student_desks_seat_{k}")
        assert math.dist(c.location[:2], board.location[:2]) > math.dist(d.location[:2], board.location[:2])


def test_seats_on_tables():
    # cafe: "four round tables with two chairs each" as one object
    raw = {"setting": {"kind": "indoor", "bounds_m": [9.0, 7.0, 3.0]},
           "objects": [{"id": "table", "category": "dining_table", "count": 4, "seats": 2}]}
    _spec, pls = _solve(raw)
    tables = [p for p in pls if p.category == "dining_table"]
    chairs = [p for p in pls if p.category == "dining_chair"]
    assert len(tables) == 4 and len(chairs) == 8
    for t in tables:  # two chairs each, facing it
        mine = [c for c in chairs if c.facing_target == t.id]
        assert len(mine) == 2 and all(_gap(c, t) <= 0.1 for c in mine)


def test_chairs_around_tables_with_matching_counts_become_seats():
    # ultra-aidb4/cafe: 8 dining chairs around 4 tables, "scattered and unpaired"
    raw = {"setting": {"kind": "indoor", "bounds_m": [9.0, 7.0, 3.0]},
           "objects": [{"id": "tables", "category": "dining_table", "count": 4},
                       {"id": "chairs", "category": "dining_chair", "count": 8}],
           "relations": [{"a": "chairs", "rel": "around", "b": "tables"}]}
    spec, _pls = _solve(raw)
    assert spec.objects[0].seats == 2 and "became seats" in spec.notes[0]


def test_things_on_a_seated_desk_stand_on_the_desk():
    raw = {"setting": {"kind": "indoor", "bounds_m": [5.0, 4.0, 3.0]},
           "objects": [{"id": "desk", "category": "desk", "seats": 1, "seat_category": "office_chair"},
                       {"id": "screen", "category": "monitor"}],
           "relations": [{"a": "desk", "rel": "against_wall", "b": "wall_north"},
                         {"a": "screen", "rel": "on", "b": "desk"}]}
    _spec, pls = _solve(raw)
    by = {p.id: p for p in pls}
    desk, screen, chair = by["desk"], by["screen"], by["desk_seat_1"]
    assert chair.category == "office_chair" and chair.location[1] < desk.location[1]  # in front, into the room
    assert abs(screen.location[0] - desk.location[0]) < desk.dims[0] / 2
    assert abs(screen.location[1] - desk.location[1]) < desk.dims[1] / 2
    assert desk.location[1] + desk.dims[1] / 2 > 3.99  # the desk itself is flush with the wall


def test_seats_round_trip_and_edit():
    raw = {"setting": {"kind": "indoor", "bounds_m": [5.0, 4.0, 3.0]},
           "objects": [{"id": "desk", "category": "desk"}]}
    data, _done, errors = apply_steps(raw, [{"op": "update_object", "id": "desk", "set": {"seats": 2}}])
    assert not errors
    spec, _ = validate_spec(data)
    assert spec is not None and spec.objects[0].seats == 2
    assert validate_spec(spec.to_dict())[0] == spec
    bad = {**raw, "objects": [{"id": "desk", "category": "desk", "seats": -1}]}
    assert any("seats" in e for e in validate_spec(bad)[1])
