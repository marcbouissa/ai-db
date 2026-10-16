"""build_scene's repair loop (bench mining 2026-10-16, item 6).

A bigger room only for space problems, growth capped by area, a second seed before a facing /
in_front_of relation is dropped, and every dropped relation reported as an issue. Spec shapes
follow the bench calls named in docs/WEAK_MODEL_FAILURES_2026-10-16.md.
"""

from __future__ import annotations

import pytest
from ai_db_scene import repair
from ai_db_scene.layout import LayoutError
from ai_db_scene.repair import relax, solve_with_repairs


def test_a_relation_conflict_never_grows_the_room():
    # round_lobby#11: a picture frame with no wall -> "enlarged the setting to 14.4 x 14.4 m", 17.28,
    # 20.74, then a conflict anyway
    spec = {"setting": {"kind": "indoor", "bounds_m": [12.0, 12.0, 3.5]},
            "objects": [{"id": "art", "category": "picture_frame"}], "relations": []}
    err = LayoutError("'art' (picture_frame) is wall-mounted: add a relation {...}", ["art"])
    assert relax(spec, err, [12.0, 12.0, 3.5]) is None
    assert "bigger room cannot fix" in repair.explain(err, grown=False)


def test_growth_is_capped_by_area():
    err = LayoutError("cannot place 'x_9' (...); main blockers: 300x out of bounds", ["x"])
    spec = {"setting": {"kind": "indoor", "bounds_m": [10.0, 10.0, 3.0]},
            "objects": [{"id": "x", "category": "crate", "count": 9}], "relations": []}
    sizes = []
    while (fix := relax(spec, err, [10.0, 10.0, 3.0])) is not None and "enlarged" in fix[1]:
        spec = fix[0]
        sizes.append(spec["setting"]["bounds_m"][:2])
    assert sizes == [[12.0, 12.0], [14.4, 14.4]]  # 2.07x the area; the next step would be 2.99x


def test_ceiling_lights_centered_in_room_spread_instead_of_growing_the_room():
    # warehouse#15-#19: six ceiling lights centered_in room -> room grown to 69 x 104 m
    spec = {"setting": {"kind": "indoor", "bounds_m": [20.0, 30.0, 6.0]},
            "objects": [{"id": "lights", "category": "ceiling_light", "count": 6},
                        {"id": "rack", "category": "pallet_rack", "count": 4}],
            "relations": [{"a": "lights", "rel": "centered_in", "b": "room"},
                          {"a": "rack", "rel": "against_wall", "b": "wall_west"}]}
    out = solve_with_repairs(spec)
    assert out.error is None and out.repairs == [] and len(out.placements) == 10
    assert out.data["setting"]["bounds_m"] == [20.0, 30.0, 6.0]
    assert any("spread as a" in n for n in out.spec.notes)


def test_dropped_relations_are_listed():
    # round_lobby#17: "dropped relation 'plant_1 in_front_of low_table'", then DONE as if satisfied
    spec = {"setting": {"kind": "indoor", "bounds_m": [5.0, 4.0, 2.7]},
            "objects": [{"id": "sofa", "category": "sofa"}, {"id": "table", "category": "coffee_table"}],
            "relations": [{"a": "sofa", "rel": "against_wall", "b": "wall_north"},
                          {"a": "table", "rel": "centered_in", "b": "room"},
                          {"a": "sofa", "rel": "centered_in", "b": "room"}]}
    out = solve_with_repairs(spec)
    assert out.placements and out.dropped == ["sofa centered_in room"]


def test_a_second_seed_before_dropping_facing(monkeypatch):
    spec = {"setting": {"kind": "indoor", "bounds_m": [5.0, 4.0, 2.7]},
            "objects": [{"id": "sofa", "category": "sofa"}, {"id": "tv", "category": "tv_stand"}],
            "relations": [{"a": "sofa", "rel": "facing", "b": "tv"}]}
    real = repair.solve_layout

    def flaky(s, seed=0):
        if seed == 0:
            raise LayoutError("cannot place 'sofa'", ["sofa facing tv"])
        return real(s, seed=seed)

    monkeypatch.setattr(repair, "solve_layout", flaky)
    out = solve_with_repairs(spec)
    assert out.dropped == [] and out.placements and "second seed" in out.repairs[0]


@pytest.fixture
def fake_blender(monkeypatch):
    from ai_db_scene.tools import scene_pipeline as sp

    class Fake:
        def batch(self, cmds, timeout=None):
            return [{"ok": 1, "status": "success"} for _ in cmds]

        def request(self, handler, params, timeout=None):
            return {}

    monkeypatch.setattr(sp, "BlenderClient", Fake)
    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})
    return sp


def test_build_reports_a_dropped_relation_as_an_issue(fake_blender):
    sp = fake_blender
    spec = {"setting": {"kind": "indoor", "bounds_m": [5.0, 4.0, 2.7]},
            "objects": [{"id": "sofa", "category": "sofa"}, {"id": "table", "category": "coffee_table"}],
            "relations": [{"a": "sofa", "rel": "against_wall", "b": "wall_north"},
                          {"a": "table", "rel": "centered_in", "b": "room"},
                          {"a": "sofa", "rel": "centered_in", "b": "room"}]}
    out = sp._handle_build_scene({"spec": spec, "snapshot": False, "exposure_check": False})
    assert out["status"] == "built_with_issues"
    assert any(i["check"] == "dropped_relation" and "sofa centered_in room" in i["message"] for i in out["issues_left"])


def test_build_conflict_says_what_to_change(fake_blender):
    sp = fake_blender
    spec = {"setting": {"kind": "indoor", "bounds_m": [2.0, 2.0, 2.7]},
            "objects": [{"id": "bed", "category": "bed", "count": 6, "grid": {"rows": 2, "cols": 3}}]}
    out = sp._handle_build_scene({"spec": spec, "snapshot": False})
    assert out["status"] == "conflict" and out["hint"]
