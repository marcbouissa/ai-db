"""qwen3-14b-local-aidb7 living room (visual review, 2026-10-17).

The model wrote ``armchair (count 2) beside sofa`` and ``floor_lamp beside sofa``: the
seating-group rewrite only read single armchairs, so both copies stood at the sofa's ends facing
the way it does and the floor lamp ended up behind the floated sofa's centre. The pair is now
split into armchair_1 / armchair_2 at the coffee table's short ends, facing it; the lamp stands at
a sofa end; the sofa floats far enough forward that the TV is not across an empty room.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

SPECS = json.loads((Path(__file__).parent / "fixtures" / "bench_specs_aidb7.json").read_text())


def _solve(key, seed):
    spec, errors = sp._spec_from_args({"spec": SPECS[key]["spec"]})
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), seed, 6)
    assert fixed.error is None and fixed.placements, fixed.error
    return spec, fixed, {p.id: p for p in fixed.placements}


def _facing_deg(p, target):
    yaw = math.radians(p.rotation_deg_z)
    f = (math.sin(yaw), -math.cos(yaw))
    to = (target.location[0] - p.location[0], target.location[1] - p.location[1])
    return math.degrees(math.acos(max(-1.0, min(1.0, (f[0] * to[0] + f[1] * to[1]) / math.hypot(*to)))))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_armchair_pair_sits_at_the_table_ends(seed):
    spec, _fixed, pl = _solve("raw/qwen/living_room", seed)
    assert any("split into 'armchair_1' and 'armchair_2'" in n for n in spec.notes), spec.notes
    sofa, table, stand, lamp = pl["sofa"], pl["coffee_table"], pl["tv_stand"], pl["floor_lamp"]
    a1, a2 = pl["armchair_1"], pl["armchair_2"]
    for a in (a1, a2):
        assert _facing_deg(a, table) < 15
        assert abs(a.location[1] - table.location[1]) < 0.1  # level with the table, not the sofa
    assert (a1.location[0] - sofa.location[0]) * (a2.location[0] - sofa.location[0]) < 0  # one each side
    # the lamp stands at one of the sofa's ends, not behind it
    end = sofa.dims[0] / 2
    assert abs(lamp.location[1] - sofa.location[1]) < sofa.dims[1] / 2
    assert abs(lamp.location[0] - sofa.location[0]) > end
    # sofa front to TV stand front: a viewing distance, not across an empty room
    view = abs(stand.location[1] - sofa.location[1]) - sofa.dims[1] / 2 - stand.dims[1] / 2
    assert 2.0 <= view <= 3.0, view
