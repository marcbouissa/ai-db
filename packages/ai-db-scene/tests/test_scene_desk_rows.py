"""Student desks in rows facing the board (bench classroom, qwen3-14b-local aidb4..aidb6).

qwen aidb4/aidb5 declared student_desk_001 .. student_desk_012 chained by ``beside`` (chairs
beside each desk, or with no relation at all) and the solver could not place the chain; qwen
aidb6 wrote one desk object with count 12 ``in_front_of`` the whiteboard and the copies stood
scattered over the room (bench/qwen3-14b-local-aidb6/classroom/probe_plan.png). Both now build
a grid of rows facing the board, a chair behind every desk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ai_db_scene.layout import check_layout, solve_layout
from ai_db_scene.spec import validate_spec

AIDB5 = json.loads((Path(__file__).parent / "fixtures" / "bench_qwen_classroom_aidb5.json").read_text())["args"]["spec"]
#: qwen3-14b-local-aidb6/classroom: the spec of its first build (transcript line 7)
AIDB6 = {"setting": {"bounds_m": [10, 8, 3], "style": "modern", "floor_material": "carpet", "kind": "indoor"},
         "objects": [{"id": "teacher_desk", "category": "desk", "material": "wood", "hero": True},
                     {"id": "teacher_chair", "category": "chair", "material": "fabric", "hero": True},
                     {"id": "whiteboard", "category": "whiteboard", "material": "white", "hero": True},
                     {"id": "student_desk", "category": "desk", "material": "wood", "count": 12},
                     {"id": "student_chair", "category": "chair", "material": "fabric", "count": 12}],
         "relations": [{"a": "whiteboard", "rel": "against_wall", "b": "wall_south"},
                       {"a": "teacher_desk", "rel": "in_front_of", "b": "whiteboard"},
                       {"a": "teacher_chair", "rel": "beside", "b": "teacher_desk"},
                       {"a": "student_desk", "rel": "in_front_of", "b": "whiteboard"},
                       {"a": "student_chair", "rel": "beside", "b": "student_desk"}]}


def _rows(raw, board_id):
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    pls = solve_layout(spec)
    assert check_layout(spec, pls) == []
    by = {p.id: p for p in pls}
    block = next(o for o in spec.objects if o.grid is not None)
    desks = [p for p in pls if p.id.startswith(f"{block.id}_") and p.category == "desk"]
    chairs = [p for p in pls if p.id.startswith(f"{block.id}_seat_")]
    return spec, block, desks, chairs, by[board_id]


def _check_rows(desks, chairs, board):
    assert len(desks) == len(chairs) == 12
    # rows: the desks stand on 3 or 4 lines parallel to the board's wall, all turned alike
    ys = sorted({round(p.location[1], 2) for p in desks})
    assert len(ys) in (3, 4)
    assert len({round(p.rotation_deg_z % 360, 1) for p in desks}) == 1
    # facing the board: every chair is behind its desk, seen from the board
    for k, d in enumerate(desks, 1):
        c = next(p for p in chairs if p.id.endswith(f"_seat_{k}"))
        assert math.dist(c.location[:2], board.location[:2]) > math.dist(d.location[:2], board.location[:2])


def test_a_chain_of_desks_becomes_rows_facing_the_board():
    spec, block, desks, chairs, board = _rows(AIDB5, "whiteboard_001")
    assert block.id == "student_desk" and block.grid[:2] == (3, 4) and block.seats == 1
    assert [o.id for o in spec.objects] == ["teacher_desk_001", "teacher_chair_001", "whiteboard_001", "student_desk"]
    assert any("12 desks chained by beside laid out as 'student_desk'" in n for n in spec.notes)
    _check_rows(desks, chairs, board)


def test_desk_copies_in_front_of_the_board_become_rows():
    spec, block, desks, chairs, board = _rows(AIDB6, "whiteboard")
    assert block.id == "student_desk" and block.seats == 1
    assert ("student_desk", "facing", "whiteboard") in [(r.a, r.rel, r.b) for r in spec.relations]
    _check_rows(desks, chairs, board)
    # the teacher desk stays its own object, between the board and the rows
    teacher = next(o for o in spec.objects if o.id == "teacher_desk")
    assert teacher.grid is None


def test_desk_copies_in_a_row_by_the_board_become_rows():
    # qwen3-14b-local-aidb7 classroom: every build a conflict (the room grown to 57 x 43 m)
    raw = {**AIDB6, "relations": [{"a": "whiteboard", "rel": "on", "b": "front_wall"},
                                  {"a": "teacher_desk", "rel": "against_wall", "b": "wall_north"},
                                  {"a": "teacher_chair", "rel": "beside", "b": "teacher_desk", "gap_m": 0.6},
                                  {"a": "student_desk", "rel": "in_row", "b": "whiteboard", "gap_m": 1.0},
                                  {"a": "student_chair", "rel": "beside", "b": "student_desk", "gap_m": 0.5}]}
    spec, block, desks, chairs, board = _rows(raw, "whiteboard")
    assert block.id == "student_desk" and block.seats == 1 and spec.setting.bounds_m[:2] == (10, 8)
    _check_rows(desks, chairs, board)


def test_per_desk_chairs_beside_a_chain_merge_into_seats():
    # qwen aidb4 classroom: student_chair_i beside student_desk_i
    objs = [{"id": "board", "category": "whiteboard"}]
    rels = [{"a": "board", "rel": "against_wall", "b": "wall_north"}]
    for i in range(1, 9):
        objs += [{"id": f"student_desk_{i}", "category": "desk"}, {"id": f"student_chair_{i}", "category": "chair"}]
        rels.append({"a": f"student_chair_{i}", "rel": "beside", "b": f"student_desk_{i}", "gap_m": 0.2})
        if i < 8:
            rels.append({"a": f"student_desk_{i}", "rel": "beside", "b": f"student_desk_{i + 1}", "gap_m": 0.5})
    spec, errors = validate_spec({"setting": {"kind": "indoor", "bounds_m": [9.0, 7.0, 3.0]},
                                  "objects": objs, "relations": rels})
    assert spec is not None, errors
    assert [(o.id, o.count, o.seats) for o in spec.objects] == [("board", 1, 0), ("student_desk", 8, 1)]
    pls = solve_layout(spec)
    assert check_layout(spec, pls) == []
    assert sum(p.category == "chair" for p in pls) == 8


def test_a_few_desks_or_an_office_without_a_board_are_left_alone():
    # three desks beside each other: a short row, not a classroom
    raw = {"setting": {"kind": "indoor", "bounds_m": [8.0, 6.0, 3.0]},
           "objects": [{"id": f"desk_{i}", "category": "desk"} for i in range(1, 4)],
           "relations": [{"a": "desk_1", "rel": "beside", "b": "desk_2"}, {"a": "desk_2", "rel": "beside", "b": "desk_3"}]}
    spec, _ = validate_spec(raw)
    assert spec is not None and all(o.grid is None for o in spec.objects)
    # six desk copies with no board: no rows to face
    raw = {"setting": {"kind": "indoor", "bounds_m": [8.0, 6.0, 3.0]},
           "objects": [{"id": "desk", "category": "desk", "count": 6}]}
    spec, _ = validate_spec(raw)
    assert spec is not None and spec.objects[0].grid is None
