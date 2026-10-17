"""A room the model sized far bigger than its content is composed, not filled (qwen aidb7
round_lobby: a 10 m round lobby for a desk, four armchairs round a table and two plants; the group
a speck in the middle, the plants anywhere by the wall). Nothing is added; the verdict suggests a
smaller room or more content."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ai_db_scene.normalize import SPARSE_FILL, normalize_spec
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

FX = json.loads((Path(__file__).parent / "fixtures" / "bench_specs_aidb7.json").read_text())


def _solved(key: str):
    spec, errors = sp._spec_from_args({"spec": copy.deepcopy(FX[key]["spec"])}, remembered=False)
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), 0, 3)
    assert fixed.placements is not None
    return spec, fixed


def test_the_lounge_group_stands_across_the_room_from_the_desk():
    spec, fixed = _solved("last/qwen/round_lobby")
    assert any("the room is sparse" in n for n in spec.notes)
    by = {p.id: p for p in fixed.placements}
    w, d, _h = fixed.spec.setting.bounds_m
    desk, table = by["reception_desk"], by["low_table"]
    cx, cy = w / 2, d / 2
    # the desk on one side of the middle, the table past the middle on the other
    v = (cx - desk.location[0], cy - desk.location[1])
    t = (table.location[0] - cx, table.location[1] - cy)
    assert v[0] * t[0] + v[1] * t[1] > 0.2  # off-centre, away from the desk
    assert {p.id for p in fixed.placements} == {"reception_desk", "armchair1_1", "armchair1_2", "armchair1_3",
                                                 "armchair1_4", "low_table", "plant1_1", "plant1_2", "pendant_lamp"}


def test_free_plants_frame_the_desk():
    _spec, fixed = _solved("last/qwen/round_lobby")
    by = {p.id: p for p in fixed.placements}
    desk = by["reception_desk"]
    for pid in ("plant1_1", "plant1_2"):
        p = by[pid]
        assert abs(p.location[0] - desk.location[0]) + abs(p.location[1] - desk.location[1]) < 2.6
    xs = sorted(by[p].location[0] - desk.location[0] for p in ("plant1_1", "plant1_2"))
    assert xs[0] < 0 < xs[1]  # one each side


def test_a_room_with_enough_content_or_its_own_placing_is_left_alone():
    data = copy.deepcopy(FX["last/lightning/round_lobby"]["spec"])  # 8 m, coffee table centred by the model
    _out, notes = normalize_spec(data)
    assert not any("sparse" in n for n in notes)
    living = copy.deepcopy(FX["last/qwen/living_room"]["spec"])
    _out, notes = normalize_spec(living)
    assert not any("sparse" in n for n in notes)
    assert SPARSE_FILL < 0.2


def test_no_object_is_added_and_no_rug_invented():
    spec, _fixed = _solved("last/qwen/round_lobby")
    assert [o.category for o in spec.objects].count("rug") == 0
    assert len(spec.objects) == len(FX["last/qwen/round_lobby"]["spec"]["objects"])


def test_the_verdict_suggests_a_smaller_room():
    _spec, fixed = _solved("last/qwen/round_lobby")
    hint = sp.sparse_hint(fixed.spec, fixed.placements)
    assert hint and "update_setting" in hint and "mostly empty" in hint
    _v, nxt = sp.build_verdict({"objects": 9, "scene_id": "s1", "sparse": hint})
    assert nxt.startswith("The scene is finished") and "Optional, once" in nxt and "otherwise stop" in nxt
    _s, ok = _solved("last/qwen/living_room")
    assert sp.sparse_hint(ok.spec, ok.placements) is None
