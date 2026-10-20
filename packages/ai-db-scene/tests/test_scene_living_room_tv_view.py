"""The living-room camera shows the sofa's front AND the TV it faces (lead review of the f7e0879 demo).

From the open side of the seating U the TV stood beside the lens, cut by the frame's edge. When the
sofa or armchairs face a TV / fireplace, the first candidate height (SCENE_CAMERA_FACED_EYE_M)
scores views with _faced_term: the feature wholly in frame (not beside the lens) with a seat that
faces it seen from its front, a moderate lens, and armchair backs near the lens judged by the share
of the frame they cover. Fixture: the stored specs of the demo (bench_qwen_demo_f7e0879.json).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ai_db_scene.camera import (
    COVER_MAX,
    NEAR_M,
    SENSOR_MM,
    _footprint_dist,
    _front,
    best_views,
    faced_features,
    foreground_cover,
)
from ai_db_scene.constants import SCENE_CAMERA_FACED_EYE_M
from ai_db_scene.layout import Placement
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.spec import validate_spec
from ai_db_scene.tools import scene_pipeline as sp

FIX = Path(__file__).parent / "fixtures"
DEMO = json.loads((FIX / "bench_qwen_demo_f7e0879.json").read_text())
AIDB7 = json.loads((FIX / "bench_specs_aidb7.json").read_text())
LIVING = {"demo/living_room": DEMO["living_room"]["spec"], "last/qwen/living_room": AIDB7["last/qwen/living_room"]["spec"]}
#: a 5 x 4 m room: from no spot are the TV and the sofa's front both clear in frame
CRAMPED = AIDB7["last/lightning/living_room"]["spec"]


def _solve(raw):
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), 0, 6)
    assert fixed.placements and fixed.spec is not None, fixed.error
    return fixed.spec, fixed.placements


def _pose(spec, pls):
    pose = sp.camera_poses(spec, pls, (0.0, 0.0, 0.0))[0]
    x, y, z = pose["location"]
    yaw = math.radians(pose["rotation"][2])
    hfov = math.degrees(2 * math.atan(SENSOR_MM / 2 / pose["lens"]))
    return x, y, z, (-math.sin(yaw), math.cos(yaw)), hfov, pose["shift_y"] * SENSOR_MM / pose["lens"]


def _off_deg(x, y, d, p):
    ox, oy = p.location[0] - x, p.location[1] - y
    return math.degrees(abs(math.atan2(d[0] * oy - d[1] * ox, d[0] * ox + d[1] * oy)))


@pytest.mark.parametrize("key", sorted(LIVING))
def test_tv_wholly_in_frame_with_the_sofa_front(key):
    spec, pls = _solve(LIVING[key])
    faced = faced_features(pls)
    assert faced, "the sofa faces a TV"
    x, y, z, d, hfov, tilt = _pose(spec, pls)
    assert z == pytest.approx(SCENE_CAMERA_FACED_EYE_M)
    for f, seats in faced:
        r = math.hypot(f.location[0] - x, f.location[1] - y)
        half_w = math.degrees(math.atan2(max(f.dims[:2]) / 2, r))
        assert _off_deg(x, y, d, f) + half_w <= hfov / 2, (key, f.id)  # not cut by the frame's edge
        assert _footprint_dist(x, y, f) >= NEAR_M  # not beside the lens
        sofa = next(s for s in pls if s.category == "sofa")
        assert _off_deg(x, y, d, sofa) <= hfov / 2
        fr = _front(sofa)
        assert (sofa.location[0] - x) * fr[0] + (sofa.location[1] - y) * fr[1] < 0  # its front
    aim = (x + d[0] * 5, y + d[1] * 5, z + tilt * 5)
    assert all(c <= COVER_MAX for _k, c in foreground_cover(pls, x, y, z, aim, hfov)), key


def test_open_side_view_without_the_tv_loses_in_faced_mode():
    spec, pls = _solve(LIVING["demo/living_room"])
    plain = best_views(spec, pls, SCENE_CAMERA_FACED_EYE_M)[0]  # the old pick: the TV beside the lens
    tv = next(p for p in pls if p.category == "tv")
    assert "tv" not in plain.seen
    best = best_views(spec, pls, SCENE_CAMERA_FACED_EYE_M, faced=True)[0]
    assert tv.id in best.seen and best.score > 0


def test_cramped_room_without_such_a_view_keeps_the_old_camera():
    spec, pls = _solve(CRAMPED)
    assert faced_features(pls)
    assert sp.camera_poses(spec, pls, (0.0, 0.0, 0.0))[0]["location"][2] == pytest.approx(1.4)


def test_no_faced_feature_keeps_the_old_heights():
    spec, pls = _solve(DEMO["round_lobby"]["spec"])
    assert not faced_features(pls)
    assert sp.camera_poses(spec, pls, (0.0, 0.0, 0.0))[0]["location"][2] == pytest.approx(1.4)


def test_seat_facing_away_from_tv_is_not_a_faced_group():
    tv = Placement("tv", (0.0, 3.0, 0.9), 0.0, (1.2, 0.08, 0.7), "tv")
    sofa = Placement("sofa", (0.0, 0.0, 0.42), 0.0, (2.1, 0.95, 0.85), "sofa")  # front -y: away from the TV
    assert faced_features([tv, sofa]) == []
    sofa_t = Placement("sofa", (0.0, 0.0, 0.42), 180.0, (2.1, 0.95, 0.85), "sofa")
    assert faced_features([tv, sofa_t])[0][0].id == "tv"
