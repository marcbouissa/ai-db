"""Furniture square to the room in rect rooms (bench mining 2026-10-16, item 8).

Judges in 8 scenes: "racks scattered at odd angles", "desks on a diagonal", "desks rotated
haphazardly", "skewed sofa". Spec shapes as in those scenes (objects with no orienting relation,
a desk grid facing the board); several seeds, since the skew came from random yaw jitter.
"""

from __future__ import annotations

import pytest
from ai_db_scene.layout import Placement, check_layout, solve_layout
from ai_db_scene.realism import off_axis
from ai_db_scene.spec import validate_spec


def _off(p) -> float:
    return abs((p.rotation_deg_z + 45.0) % 90.0 - 45.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_free_furniture_is_square_to_the_walls(seed):
    raw = {"setting": {"kind": "indoor", "bounds_m": [20.0, 30.0, 8.0], "style": "industrial"},
           "objects": [{"id": "racks", "category": "pallet_rack", "count": 6}, {"id": "lift", "category": "forklift"},
                       {"id": "sofa", "category": "sofa"}, {"id": "table", "category": "coffee_table"}],
           "relations": [{"a": "table", "rel": "in_front_of", "b": "sofa"}]}
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    pls = solve_layout(spec, seed=seed)
    assert check_layout(spec, pls) == []
    assert [p.id for p in pls if _off(p) > 0.5] == []


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_desk_grid_facing_the_board_is_square(seed):
    raw = {"setting": {"kind": "indoor", "bounds_m": [10.0, 8.0, 3.0]},
           "objects": [{"id": "board", "category": "whiteboard"}, {"id": "teacher", "category": "desk"},
                       {"id": "desks", "category": "desk", "grid": {"rows": 3, "cols": 4}}],
           "relations": [{"a": "board", "rel": "against_wall", "b": "wall_north"},
                         {"a": "teacher", "rel": "in_front_of", "b": "board", "gap_m": 1.0},
                         {"a": "desks", "rel": "facing", "b": "board"}]}
    spec, _ = validate_spec(raw)
    pls = solve_layout(spec, seed=seed)
    assert check_layout(spec, pls) == []
    assert [p.id for p in pls if _off(p) > 0.5] == []


def test_off_axis_warning():
    spec, _ = validate_spec({"setting": {"kind": "indoor", "bounds_m": [6.0, 5.0, 3.0]},
                             "objects": [{"id": "sofa", "category": "sofa"}, {"id": "chair", "category": "armchair"},
                                         {"id": "tv", "category": "tv_stand"}],
                             "relations": [{"a": "chair", "rel": "facing", "b": "tv"}]})
    pls = [Placement("sofa", (3.0, 2.0, 0.4), 20.0, (2.0, 0.9, 0.85), "sofa"),
           Placement("chair", (1.0, 1.0, 0.4), 30.0, (0.8, 0.8, 0.9), "armchair"),
           Placement("tv", (3.0, 4.5, 0.3), 180.0, (1.6, 0.4, 0.5), "tv_stand")]
    issues = off_axis(spec, pls)
    assert len(issues) == 1 and issues[0]["check"] == "off_axis" and issues[0]["objects"] == ["sofa"]
