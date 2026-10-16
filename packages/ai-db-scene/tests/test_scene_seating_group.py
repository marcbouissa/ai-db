"""A living-room seating group (Marc's review of the aidb5 living rooms, 2026-10-17).

qwen3-14b-local-aidb5 and nemotron-3.5-lightning-free-aidb5 both wrote ``armchair beside
sofa``: the armchairs sat at the sofa's ends facing the way the sofa does, the coffee table was
off the sofa's axis, the rug ran half under one end of the sofa (qwen) or floated mid-room
(lightning), the TV stand was off the axis. The specs below are the last stored versions of
those runs. The solver now reads them as a seating group: sofa against a wall, table centred in
front, rug centred under the table, armchairs at the table's short ends facing it, TV stand
against the opposite wall facing the sofa.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

SPECS = json.loads((Path(__file__).parent / "fixtures" / "bench_living_rooms_aidb5.json").read_text())


def _solve(name, seed):
    spec, errors = sp._spec_from_args({"spec": SPECS[name]})
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), seed, 6)
    assert fixed.error is None and fixed.placements, fixed.error
    return spec, {p.id: p for p in fixed.placements}


def _facing_deg(p, target):
    yaw = math.radians(p.rotation_deg_z)
    f = (math.sin(yaw), -math.cos(yaw))  # front = -Y at yaw 0
    to = (target.location[0] - p.location[0], target.location[1] - p.location[1])
    cos = (f[0] * to[0] + f[1] * to[1]) / math.hypot(*to)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


@pytest.mark.parametrize("name", ["qwen", "lightning"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_seating_group_is_symmetric_about_the_sofa(name, seed):
    spec, pl = _solve(name, seed)
    assert any(n.startswith("seating group: 'armchair1'") and "instead of ['armchair1 beside sofa']" in n
               for n in spec.notes), spec.notes
    sofa, table, rug, stand = pl["sofa"], pl["coffee_table"], pl["rug"], pl["tv_stand"]
    a1, a2 = pl["armchair1"], pl["armchair2"]
    sx = sofa.location[0]  # the sofa stands against wall_south: its axis is x = sx
    assert abs(table.location[0] - sx) < 0.05 and abs(stand.location[0] - sx) < 0.05
    assert abs((a1.location[0] - sx) + (a2.location[0] - sx)) < 0.1  # mirrored about the axis
    assert abs(a1.location[1] - a2.location[1]) < 0.1 and abs(a1.location[1] - table.location[1]) < 0.1
    assert abs(abs(a1.location[0] - sx) - (table.dims[0] / 2 + 0.35 + a1.dims[1] / 2)) < 0.06  # at the short ends
    for a in (a1, a2):
        assert _facing_deg(a, table) < 15
    assert _facing_deg(stand, sofa) < 5 and _facing_deg(sofa, stand) < 5
    centre = ((a1.location[0] + a2.location[0]) / 2, table.location[1])
    assert math.dist(rug.location[:2], centre) < 0.1
    # the rug runs along the sofa and the sofa's front edge is either fully on it or fully off
    assert rug.rotation_deg_z % 180 == pytest.approx(0, abs=1e-6)
    edge_y = sofa.location[1] + sofa.dims[1] / 2
    on = abs(edge_y - rug.location[1]) < rug.dims[1] / 2
    assert not on or rug.dims[0] / 2 >= sofa.dims[0] / 2 + abs(rug.location[0] - sx)
    for p in (sofa, table, rug, stand, a1, a2):
        assert p.rotation_deg_z % 90 == pytest.approx(0, abs=1e-6), (p.id, p.rotation_deg_z)


def test_explicit_relations_are_kept():
    raw = {"setting": {"kind": "indoor", "bounds_m": [6, 5, 3]},
           "objects": [{"id": "sofa", "category": "sofa"}, {"id": "table", "category": "coffee_table"},
                       {"id": "chair", "category": "armchair"}, {"id": "tv", "category": "tv_stand"}],
           "relations": [{"a": "sofa", "rel": "against_wall", "b": "wall_east"},
                         {"a": "table", "rel": "in_front_of", "b": "sofa", "gap_m": 0.5},
                         {"a": "chair", "rel": "against_wall", "b": "wall_north"},
                         {"a": "tv", "rel": "beside", "b": "sofa"}]}
    spec, errors = sp._spec_from_args({"spec": raw})
    assert spec is not None, errors
    assert not any(n.startswith("seating group") for n in spec.notes), spec.notes
