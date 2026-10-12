"""Realism warnings: flat colours on big surfaces, unlit rooms, bare walls."""

from __future__ import annotations

from ai_db_scene import realism
from ai_db_scene.tools import scene_pipeline as sp


def _run(raw):
    spec, errs = sp._parse_spec(raw)
    assert spec is not None, errs
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": raw})["placements"])
    return [i["message"] for i in realism.check(spec, pls)]


def test_flat_unlit_bare_room_gets_all_three_warnings():
    raw = {"setting": {"kind": "indoor", "bounds_m": [5.0, 4.0, 2.7], "floor_material": "color:#777777"},
           "objects": [{"id": "sofa", "category": "sofa", "material": "color:#aa2222"},
                       {"id": "table", "category": "coffee_table"}, {"id": "chair", "category": "armchair"}],
           "relations": []}
    msgs = " | ".join(_run(raw))
    assert "floor are a flat colour" in msgs and "no lamp and no window" in msgs and "bare walls" in msgs
    assert "sofa is a large object in a flat colour" in msgs


def test_dressed_room_is_clean():
    raw = {"setting": {"kind": "indoor", "bounds_m": [5.0, 4.0, 2.7]},
           "objects": [{"id": "sofa", "category": "sofa"}, {"id": "lamp", "category": "floor_lamp"},
                       {"id": "art", "category": "picture_frame"}],
           "relations": [{"a": "art", "rel": "against_wall", "b": "wall_north"}]}
    assert _run(raw) == []


def test_brief_counts_and_missing_objects():
    from ai_db_scene.layout import Placement

    brief = "A cozy living room: a sofa facing a TV on a TV stand, a coffee table on a rug, two armchairs and a floor lamp."
    counts = realism.brief_counts(brief, {"sofa", "tv", "tv_stand", "coffee_table", "rug", "armchair", "floor_lamp", "table"})
    assert counts == {"sofa": 1, "tv": 1, "tv_stand": 1, "coffee_table": 1, "rug": 1, "armchair": 2, "floor_lamp": 1}

    def pl(pid, cat):
        return Placement(id=pid, category=cat, location=(0.0, 0.0, 0.4), rotation_deg_z=0.0, dims=(1.0, 1.0, 0.8))

    scene = [pl("sofa", "sofa"), pl("coffee_table", "coffee_table"), pl("rug", "rug"), pl("tv_stand", "tv_stand"),
             pl("tv", "tv"), pl("lamp", "floor_lamp")] + [pl(f"armchair_{i}", "armchair") for i in range(4)]
    msgs = [i["message"] for i in realism.check_brief(brief, scene)]
    assert msgs == ["the brief asks for 2 armchair, the scene has 4"]
