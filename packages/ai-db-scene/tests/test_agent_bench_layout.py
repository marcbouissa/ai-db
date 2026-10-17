"""Layout sanity term of the bench quality score (eval/agent_bench/layout_sanity.py, formula v3) on
synthetic inventories: each check passes, fails or does not apply, with its reason."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "eval" / "agent_bench"


@pytest.fixture(scope="module")
def mods() -> tuple[Any, Any]:
    sys.path.insert(0, str(BENCH_DIR))
    try:
        return importlib.import_module("bench"), importlib.import_module("layout_sanity")
    finally:
        sys.path.remove(str(BENCH_DIR))


def _root(x: float, y: float, w: float, d: float, h: float, yaw: float = 0.0, z0: float = 0.0) -> dict[str, Any]:
    return {"box": [[x - w / 2, y - d / 2, z0], [x + w / 2, y + d / 2, z0 + h]], "loc": [x, y, z0], "yaw_deg": yaw,
            "terrain": False}


def _inv(roots: dict[str, dict[str, Any]], cats: dict[str, str], camera: dict[str, Any] | None = None) -> dict[str, Any]:
    objs = [{"name": n, "type": "EMPTY", "category": cats.get(n, ""), "parent": None, "mats": 0, "tris": 0,
             "faces": 0, "size_m": None} for n in roots]
    return {"objects": objs, "camera": "cam", "lights": 1, "layout": {"v": 2, "roots": roots, "camera": camera}}


def _room() -> dict[str, dict[str, Any]]:
    """A 5 x 4 m room: front of an object = local -Y; yaw 180 turns it to +Y."""
    return {"Room_Floor": _root(0, 0, 5, 4, 0.02, z0=-0.02), "Room_Walls": _root(0, 0, 5.2, 4.2, 2.8)}


def _living(sofa_yaw: float = 180.0, table_xy: tuple[float, float] = (0.0, 0.0)) -> dict[str, dict[str, Any]]:
    # the sofa at the south wall facing north (+Y) to the coffee table and the TV
    return {**_room(), "sofa": _root(0, -1.4, 2.0, 0.9, 0.85, sofa_yaw),
            "coffee_table": _root(*table_xy, 1.1, 0.6, 0.42), "tv_stand": _root(0, 1.75, 1.5, 0.45, 0.55, 0.0),
            "armchair": _root(-1.6, 0, 0.85, 0.85, 0.9, 90.0)}


CATS = {"sofa": "sofa", "coffee_table": "coffee_table", "tv_stand": "tv_stand", "armchair": "armchair"}
CAM_FRONT = {"loc": [0.0, 1.9, 1.4], "dir": [0, -1, 0], "rays": 336, "median_m": 3.0,
             "hits": {"Room_Walls": [200, 3.5, 90], "sofa": [60, 3.2, 40], "coffee_table": [20, 2.0, 15]}}


def test_a_good_living_room_passes(mods: tuple[Any, Any]) -> None:
    _bench, ls = mods
    checks = ls.check(_inv(_living(), CATS, CAM_FRONT))
    assert checks["facing_focus"]["pass"] and checks["seat_groups"]["pass"]
    assert checks["chairs_at_tables"]["pass"] is None  # no chairs: does not apply
    assert checks["in_bounds"]["pass"] and checks["grounded"]["pass"] and checks["no_occluder"]["pass"]
    assert checks["seating_fronts"]["pass"]
    assert checks["fill_ratio"]["pass"] is True, checks["fill_ratio"]["why"]
    assert ls.score(checks) == 100.0


def test_a_sofa_turned_away_fails_facing_and_fronts(mods: tuple[Any, Any]) -> None:
    _bench, ls = mods
    checks = ls.check(_inv(_living(sofa_yaw=0.0), CATS, CAM_FRONT))
    assert checks["facing_focus"]["pass"] is False and "sofa->" in checks["facing_focus"]["why"]
    assert checks["seating_fronts"]["pass"] is False


def test_benches_far_from_the_fountain_fail_seat_groups(mods: tuple[Any, Any]) -> None:
    _bench, ls = mods
    roots = {"ground": {**_root(0, 0, 60, 60, 0.1, z0=-0.1), "terrain": True}, "fountain": _root(0, 0, 2.4, 2.4, 1.2),
             "bench1": _root(0, -3.0, 1.6, 0.5, 0.85, 180.0), "bench2": _root(0, 12.0, 1.6, 0.5, 0.85, 0.0)}
    checks = ls.check(_inv(roots, {"fountain": "fountain", "bench1": "bench", "bench2": "bench"}))
    assert checks["seat_groups"]["pass"] is False and "bench2" in checks["seat_groups"]["why"]
    assert checks["fill_ratio"]["pass"] is None and checks["in_bounds"]["pass"] is None  # outdoors
    assert checks["no_occluder"]["pass"] is None  # no camera


def test_loose_chairs_floating_objects_and_an_empty_hall_fail(mods: tuple[Any, Any]) -> None:
    _bench, ls = mods
    roots = {"Room_Floor": _root(0, 0, 30, 20, 0.02, z0=-0.02), "Room_Walls": _root(0, 0, 30.2, 20.2, 4),
             "table": _root(0, 0, 1.6, 0.9, 0.75), "chair_1": _root(0, -0.8, 0.5, 0.5, 0.9, 180.0),
             "chair_2": _root(10, 5, 0.5, 0.5, 0.9), "crate": _root(5, 5, 0.6, 0.6, 0.6, z0=1.2)}
    checks = ls.check(_inv(roots, {}))  # categories from the names
    assert checks["chairs_at_tables"]["pass"] is False and "chair_2" in checks["chairs_at_tables"]["why"]
    assert checks["grounded"]["pass"] is False and "crate" in checks["grounded"]["why"]
    assert checks["fill_ratio"]["pass"] is False  # 600 m2 for a table and two chairs


def test_an_object_on_another_does_not_float(mods: tuple[Any, Any]) -> None:
    _bench, ls = mods
    roots = {**_room(), "desk": _root(0, 0, 1.4, 0.7, 0.75), "monitor": _root(0, 0.1, 0.6, 0.2, 0.4, z0=0.75)}
    assert ls.check(_inv(roots, {}))["grounded"]["pass"] is True


def test_a_near_object_in_front_of_the_subject_is_an_occluder(mods: tuple[Any, Any]) -> None:
    _bench, ls = mods
    cam = {**CAM_FRONT, "hits": {**CAM_FRONT["hits"], "armchair": [40, 0.9, 20]}}
    checks = ls.check(_inv(_living(), CATS, cam))
    assert checks["no_occluder"]["pass"] is False and "armchair" in checks["no_occluder"]["why"]


def test_v3_score_mixes_layout_in_and_old_inventories_stay_v2(mods: tuple[Any, Any]) -> None:
    bench, _ls = mods
    inv = _inv(_living(sofa_yaw=0.0), CATS, CAM_FRONT)
    q = bench.score("living_room", inv, {"issues": []})
    assert q["formula"] == "v3" and q["layout_pct"] is not None and q["layout_pct"] < 100
    assert q["score"] == pytest.approx(q["score_v2"] * (1 - bench.W_LAYOUT * (1 - q["layout_pct"] / 100)), abs=0.1)
    assert q["score"] < q["score_v2"]
    old = {k: v for k, v in inv.items() if k != "layout"}
    q2 = bench.score("living_room", old, {"issues": []})
    assert q2["formula"] == "v2" and q2["layout_pct"] is None and q2["score"] == q2["score_v2"]


def test_category_from_the_name(mods: tuple[Any, Any]) -> None:
    _bench, ls = mods
    assert ls.category("dining_chair_3", "") == "dining_chair"
    assert ls.category("Chair.001", "") == "chair"
    assert ls.category("benches", "") == "bench"
    assert ls.category("x", "sofa") == "sofa"
