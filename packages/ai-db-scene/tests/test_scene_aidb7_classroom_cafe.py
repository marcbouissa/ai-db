"""aidb7 classrooms and cafes (visual review, 2026-10-17).

Classroom: both models left the teacher's chair without a relation (it stood stranded by a wall),
lightning put the teacher's desk against the back wall behind the students, and qwen's 10 x 8 m
room was grown to 12 x 9.6 m because the desk block's candidates fell out of the room. Now the
chair is the desk's seat, the desk stands before the board with the teacher between them facing
the class, and the room keeps its size.

Cafe: "four round tables with two chairs each, a counter with two bar stools": the tables came
out rectangular (lightning even as coffee tables, chairs in a loose ring), the counter was a
kitchen counter or a free-standing island with stools at its ends. Now the tables are round
dining tables with their two chairs as seats, the counter a bar counter at a wall with the stools
along its front.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ai_db_scene.normalize import room_shape_from_brief, table_shape_from_brief
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

FIX = json.loads((Path(__file__).parent / "fixtures" / "bench_specs_aidb7.json").read_text())


def _solve(key, seed=0):
    raw = FIX[key]["spec"]
    brief = FIX["briefs"][key.split("/")[-1]]
    raw, _ = room_shape_from_brief(raw, brief)
    raw, _ = table_shape_from_brief(raw, brief)
    spec, errors = sp._spec_from_args({"spec": raw})
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), seed, 6)
    assert fixed.error is None and fixed.placements, fixed.error
    return spec, fixed, {p.id: p for p in fixed.placements}


def _front(p):
    yaw = math.radians(p.rotation_deg_z)
    return math.sin(yaw), -math.cos(yaw)


@pytest.mark.parametrize("key", ["raw/qwen/classroom", "raw/lightning/classroom"])
def test_teacher_sits_behind_the_desk_at_the_front(key):
    _spec, fixed, pl = _solve(key)
    assert fixed.data["setting"]["bounds_m"][:2] == FIX[key]["spec"]["setting"]["bounds_m"][:2], fixed.repairs
    assert not fixed.dropped, fixed.dropped
    assert "teacher_chair" not in pl  # it is the desk's seat
    desk, chair, board = pl["teacher_desk"], pl["teacher_desk_seat_1"], pl["whiteboard"]
    students = [p for p in pl.values() if p.id.startswith("student_desk_") and "_seat_" not in p.id]
    d = lambda p: math.dist(p.location[:2], board.location[:2])
    assert d(chair) < d(desk) < min(d(s) for s in students)  # board, teacher, desk, class
    assert d(desk) < 2.0
    f = _front(chair)  # the teacher looks at the class, away from the board
    away = (chair.location[0] - board.location[0], chair.location[1] - board.location[1])
    assert f[0] * away[0] + f[1] * away[1] > 0.9 * math.hypot(*away)


@pytest.mark.parametrize("key", ["raw/lightning/cafe", "last/qwen/cafe"])
def test_cafe_round_tables_with_their_chairs_and_a_counter_at_the_wall(key):
    spec, _fixed, pl = _solve(key)
    tables = [o for o in spec.objects if o.id.startswith("table")]
    assert len(tables) == 4
    for t in tables:
        assert (t.category, t.shape, t.seats) == ("dining_table", "round", 2), t
        p = pl[t.id]
        assert p.dims[0] == p.dims[1]  # a round top on a square footprint
        seats = [pl[f"{t.id}_seat_{k}"] for k in (1, 2)]
        for s in seats:
            assert math.dist(s.location[:2], p.location[:2]) < 0.9
    assert not [o for o in spec.objects if o.category in ("dining_chair", "chair", "bar_stool")]  # no loose chairs
    for a in tables:  # tables are not pushed together
        for b in tables:
            if a.id < b.id:
                assert math.dist(pl[a.id].location[:2], pl[b.id].location[:2]) > 1.5
    counter = next(o for o in spec.objects if o.category == "bar_counter")
    assert any(r.a == counter.id and r.rel == "against_wall" for r in spec.relations)
    c = pl[counter.id]
    f = _front(c)
    for k in (1, 2):  # both stools along the counter's front, on the room side
        s = pl[f"{counter.id}_seat_{k}"]
        rel = (s.location[0] - c.location[0], s.location[1] - c.location[1])
        assert rel[0] * f[0] + rel[1] * f[1] > c.dims[1] / 2
    s1, s2 = pl[f"{counter.id}_seat_1"], pl[f"{counter.id}_seat_2"]
    assert math.dist(s1.location[:2], s2.location[:2]) > 0.9  # spread along it, not huddled


def test_table_shape_from_brief_names_only_the_kind_asked():
    data = {"objects": [{"id": "t", "category": "dining_table"}, {"id": "c", "category": "coffee_table"}]}
    out, note = table_shape_from_brief(data, "a round coffee table and a dining table")
    assert [o.get("shape") for o in out["objects"]] == [None, "round"] and note
    out, _ = table_shape_from_brief(data, "two round tables")
    assert [o.get("shape") for o in out["objects"]] == ["round", None]
    assert table_shape_from_brief(data, "a round lobby")[1] is None


@pytest.mark.parametrize("marked", [False, True])
def test_round_tables_build_round_tops(marked):
    from ai_db_scene import furniture_variants as fv

    spec, fixed, _pl = _solve("raw/lightning/cafe")
    key = sp.variation_key(spec)
    cmds, _ = sp.build_layout_commands(fixed.spec, fixed.placements,
                                       variation=[*key, fv.variation_marker(0)] if marked else key)
    gen = {c["params"]["name"]: c["params"] for c in cmds if c["type"] == "generate_asset"}
    for t in ("table1", "table2", "table3", "table4"):
        assert gen[t]["family"] == "table" and gen[t]["params"]["top_shape"] == "round"
