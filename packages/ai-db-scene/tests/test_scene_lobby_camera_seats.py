"""The auto camera stands clear of the seating (lead review of the qwen aidb7 round lobby).

The camera stood in the gap of the armchair ring by the wall: the side of an armchair 1.5 m off
filled the bottom-left of the picture. Now a spot inside a ring of seating, and big seating near
the lens (< 2.5 m) seen side- or back-on in the lower third of the frame, count against a view.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ai_db_scene.camera import SEAT_NEAR_COVER, SEAT_NEAR_M, best_views, near_seating, seating_rings
from ai_db_scene.layout import Placement
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.spec import validate_spec

FIX = Path(__file__).parent / "fixtures"
REVIEW = json.loads((FIX / "bench_qwen_review_aidb7.json").read_text())
AIDB7 = json.loads((FIX / "bench_specs_aidb7.json").read_text())
SPECS = {"review/round_lobby": REVIEW["round_lobby"]["spec"], "review/living_room": REVIEW["living_room"]["spec"],
         **{k: AIDB7[k]["spec"] for k in ("last/qwen/round_lobby", "last/lightning/round_lobby",
                                          "last/qwen/living_room")}}


def _solve(raw):
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), 0, 6)
    assert fixed.placements and fixed.spec is not None, fixed.error
    return fixed.spec, fixed.placements


@pytest.mark.parametrize("key", sorted(SPECS))
def test_camera_outside_rings_and_no_seat_filling_the_bottom(key):
    spec, pls = _solve(SPECS[key])
    v = best_views(spec, pls, 1.4)[0]
    for cx, cy, r in seating_rings(pls):
        assert math.hypot(v.x - cx, v.y - cy) >= r, (key, (v.x, v.y), (cx, cy, r))
    near = near_seating(pls, v.x, v.y, v.z, v.aim, v.hfov_deg)
    assert all(f <= SEAT_NEAR_COVER + 0.02 for _k, f in near), (key, near)  # at most the edge of an arm
    assert v.seen  # still a picture of the room's furniture


def test_review_lobby_is_a_ring_and_the_old_spot_is_penalised():
    spec, pls = _solve(SPECS["review/round_lobby"])
    rings = seating_rings(pls)
    assert len(rings) == 1
    # the render the lead reviewed: the camera by the wall in the gap of the ring
    best = best_views(spec, pls, 1.4)
    old = min(best, key=lambda u: math.hypot(u.x - 2.23, u.y - 0.93))
    assert old.score < best[0].score
    assert any(f > SEAT_NEAR_COVER for _k, f in near_seating(pls, old.x, old.y, old.z, old.aim, old.hfov_deg))


def test_near_seating_ignores_fronts_far_seats_and_seats_behind():
    chair = Placement("a", (0.0, 1.5, 0.45), 0.0, (0.85, 0.9, 0.9), "armchair")  # front faces -y: the camera
    assert near_seating([chair], 0.0, 0.0, 1.4, (0.0, 5.0, 0.6), 80.0) == []
    side = Placement("a", (0.0, 1.5, 0.45), 90.0, (0.85, 0.9, 0.9), "armchair")
    assert near_seating([side], 0.0, 0.0, 1.4, (0.0, 5.0, 0.6), 80.0)
    far = Placement("a", (0.0, SEAT_NEAR_M + 1.0, 0.45), 90.0, (0.85, 0.9, 0.9), "armchair")
    assert near_seating([far], 0.0, 0.0, 1.4, (0.0, 9.0, 0.6), 80.0) == []
    behind = Placement("a", (0.0, -1.5, 0.45), 90.0, (0.85, 0.9, 0.9), "armchair")
    assert near_seating([behind], 0.0, 0.0, 1.4, (0.0, 5.0, 0.6), 80.0) == []
    stool = Placement("a", (0.0, 1.5, 0.45), 90.0, (0.4, 0.4, 0.75), "chair")
    assert near_seating([stool], 0.0, 0.0, 1.4, (0.0, 5.0, 0.6), 80.0) == []
