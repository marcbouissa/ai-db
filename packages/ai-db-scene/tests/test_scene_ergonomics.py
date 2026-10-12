"""Furniture-use checks: reach, heights, room to pull a chair out."""

from __future__ import annotations

from ai_db_scene import ergonomics
from ai_db_scene.layout import Placement
from ai_db_scene.spec import validate_spec


def _p(pid, cat, x, y, dims, rot=0.0):
    return Placement(id=pid, category=cat, location=(x, y, dims[2] / 2), rotation_deg_z=rot, dims=dims)


def _spec(w=6.0, d=5.0):
    spec, errs = validate_spec({"setting": {"kind": "indoor", "bounds_m": [w, d, 2.7]},
                                "objects": [{"id": "x", "category": "sofa"}], "relations": []})
    assert spec is not None, errs
    return spec


def test_good_living_room_has_no_warnings():
    pls = [_p("sofa", "sofa", 0, 1.5, (2.1, 0.95, 0.85)), _p("ct", "coffee_table", 0, 0.37, (1.1, 0.6, 0.42)),
           _p("st", "side_table", 1.4, 1.5, (0.5, 0.5, 0.55))]
    assert ergonomics.check(_spec(), pls) == []


def test_far_tall_coffee_table_and_cramped_dining_are_flagged():
    pls = [_p("sofa", "sofa", 0, 1.5, (2.1, 0.95, 0.85)), _p("ct", "coffee_table", 0, -0.6, (1.1, 0.6, 0.7)),
           _p("dt", "dining_table", 2.1, -1.9, (1.6, 0.9, 0.75)), _p("ch", "dining_chair", 2.1, -2.4, (0.45, 0.5, 0.9))]
    msgs = " ".join(i["message"] for i in ergonomics.check(_spec(), pls))
    assert "from sofa" in msgs and "below seat height" in msgs and "pulling a chair out" in msgs


def test_desks_without_chairs_at_them_are_flagged():
    desks = [_p(f"desk_{i}", "desk", x, 0, (1.2, 0.6, 0.75)) for i, x in enumerate((-2.0, 0.0, 2.0))]
    block = [_p(f"chair_{i}", "chair", x, -2.2, (0.45, 0.5, 0.9)) for i, x in enumerate((-0.5, 0.0, 0.5))]
    msgs = " ".join(i["message"] for i in ergonomics.check(_spec(), desks + block))
    assert "3 desk(s) with no chair" in msgs
    seated = [_p(f"chair_{i}", "chair", x, -0.6, (0.45, 0.5, 0.9)) for i, x in enumerate((-2.0, 0.0, 2.0))]
    assert not [i for i in ergonomics.check(_spec(), desks + seated) if "no chair" in i["message"]]
