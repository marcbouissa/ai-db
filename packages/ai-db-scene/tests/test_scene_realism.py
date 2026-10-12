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
