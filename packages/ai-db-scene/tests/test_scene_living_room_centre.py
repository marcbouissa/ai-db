"""The living-room group centred in the room (lead's check of qwen aidb6, 2026-10-17).

qwen3-14b-local-aidb6's living room (fixture: its built spec plus the TV its edit added) was a
symmetric seating group flush with the south wall: 2.4 m of empty floor between the armchairs
and the TV stand, and the TV on the stand faced the wall (the validator's faces_wall fix turned
it in Blender, so every rebuild had it backwards again). Now the sofa stands off its wall by
the gap that centres the group between that wall and the TV stand (never a slot narrower than
a walkway, never closer than 2 m to the TV), and an object on a support faces the way it does.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ai_db_scene.layout import check_layout
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.spec import validate_spec

FIX = Path(__file__).parent / "fixtures"
SPEC = json.loads((FIX / "bench_qwen_living_room_aidb6.json").read_text())["qwen_aidb6_living_room"]


def _solve(raw, seed=0):
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), seed, 6)
    assert fixed.error is None and fixed.placements and fixed.spec is not None, fixed.error
    return spec, fixed.spec, {p.id: p for p in fixed.placements}


def _front(p):
    yaw = math.radians(p.rotation_deg_z)
    return math.sin(yaw), -math.cos(yaw)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_group_is_centred_between_the_sofa_wall_and_the_tv(seed):
    asked, spec, pl = _solve(SPEC, seed)
    assert any("stands" in n and "off wall_south" in n for n in asked.notes), asked.notes
    sofa, table, stand, tv = pl["sofa"], pl["coffee_table"], pl["tv_stand"], pl["tv"]
    chairs = [pl["armchair1"], pl["armchair2"]]
    behind = sofa.location[1] - sofa.dims[1] / 2  # the sofa stands against wall_south (y = 0)
    group_far = max(table.location[1] + table.dims[1] / 2, *(c.location[1] + c.dims[0] / 2 for c in chairs))
    in_front = (stand.location[1] - stand.dims[1] / 2) - group_far
    assert behind >= 0.6 and abs(behind - in_front) < 0.15, (behind, in_front)
    assert (stand.location[1] - stand.dims[1] / 2) - (sofa.location[1] + sofa.dims[1] / 2) >= 2.0
    # the TV faces the sofa, in the middle of its stand
    f = _front(tv)
    to = (sofa.location[0] - tv.location[0], sofa.location[1] - tv.location[1])
    assert (f[0] * to[0] + f[1] * to[1]) / math.hypot(*to) > 0.99
    assert abs(tv.location[0] - stand.location[0]) < 0.1
    assert not [i for i in check_layout(spec, list(pl.values())) if i.startswith("hard")]


@pytest.mark.parametrize("depth", [3.6, 3.8])
def test_shallow_rooms_keep_the_sofa_at_the_wall(depth):
    raw = json.loads(json.dumps(SPEC))
    raw["setting"]["bounds_m"][1] = depth  # less than a walkway behind it, or the TV closer than 2 m
    asked, _spec, pl = _solve(raw)
    assert not any("off wall" in n for n in asked.notes)
    sofa = pl["sofa"]
    assert sofa.location[1] - sofa.dims[1] / 2 == pytest.approx(0, abs=1e-6)


def test_against_wall_gap_and_flush_wall_mounts():
    raw = {"setting": {"kind": "indoor", "bounds_m": [5, 4, 3]},
           "objects": [{"id": "desk", "category": "desk"}, {"id": "art", "category": "picture_frame"}],
           "relations": [{"a": "desk", "rel": "against_wall", "b": "wall_north", "gap_m": 0.5},
                         {"a": "art", "rel": "against_wall", "b": "wall_east", "gap_m": 0.5}]}
    _asked, _spec, pl = _solve(raw)
    desk, art = pl["desk"], pl["art"]
    assert desk.location[1] + desk.dims[1] / 2 == pytest.approx(4 - 0.5)
    assert art.location[0] + art.dims[1] / 2 == pytest.approx(5)  # a picture hangs on the wall itself
