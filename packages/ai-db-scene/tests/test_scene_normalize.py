"""Relations weak models wrote that the spec used to refuse (bench mining 2026-10-16, item 5).

Each case has the argument shape of a real refused call (docs/WEAK_MODEL_FAILURES_2026-10-16.md,
item 5 table) on a neutral spec; now it is rewritten with a note, and the result solves cleanly.
"""

from __future__ import annotations

import math

from ai_db_scene.layout import check_layout, solve_layout
from ai_db_scene.spec import validate_spec


def _solve(raw):
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    pls = solve_layout(spec)
    assert check_layout(spec, pls) == []
    return spec, {p.id: p for p in pls}


def _rect(objects, relations, bounds=(6.0, 5.0, 3.0)):
    return {"setting": {"kind": "indoor", "bounds_m": list(bounds)}, "objects": objects, "relations": relations}


def test_compass_walls_in_a_round_room():
    # round_lobby#16: "'wall_north' is not a wall of this ellipse room; allowed values: ['wall']"
    raw = {"setting": {"kind": "indoor", "bounds_m": [10.0, 10.0, 3.5], "shape": "ellipse"},
           "objects": [{"id": "desk", "category": "desk"}, {"id": "sofa", "category": "sofa"}],
           "relations": [{"a": "desk", "rel": "against_wall", "b": "wall_north"},
                         {"a": "sofa", "rel": "against_wall", "b": "wall_south"}]}
    spec, pl = _solve(raw)
    assert [(r.b, r.at_deg) for r in spec.relations] == [("wall", 90.0), ("wall", 270.0)]
    assert pl["desk"].location[1] > 7 and pl["sofa"].location[1] < 3  # north and south
    assert any("wall_north" in n for n in spec.notes)


def test_pendant_above_a_table_hangs_over_it():
    # round_lobby#9 / cafe: "'above' places a wall-mounted object over another; 'pendant_lamp' ... is ceiling-supported"
    _spec, pl = _solve(_rect([{"id": "table", "category": "dining_table"}, {"id": "pendant", "category": "pendant_lamp"}],
                            [{"a": "pendant", "rel": "above", "b": "table", "gap_m": 1.2}]))
    t, p = pl["table"].location, pl["pendant"].location
    assert math.hypot(t[0] - p[0], t[1] - p[1]) < 0.06
    assert p[2] + pl["pendant"].dims[2] / 2 > 2.99  # hangs from the 3 m ceiling


def test_wall_mounted_objects_get_a_wall():
    # classroom: "'whiteboard' (whiteboard) is wall-mounted: add a relation {...}"; picture frames, windows
    spec, _pl = _solve(_rect([{"id": "board", "category": "whiteboard"}, {"id": "sofa", "category": "sofa"},
                             {"id": "art", "category": "picture_frame", "count": 2}],
                            [{"a": "sofa", "rel": "against_wall", "b": "wall_north"}]))
    walls = {r.a: r.b for r in spec.relations if r.rel == "against_wall"}
    assert walls["board"] != "wall_north" and walls["art"] not in ("wall_north", walls["board"])
    assert sum("is wall-mounted: added" in n for n in spec.notes) == 2
    # in a round room: the free-est angle on its one wall
    raw = {"setting": {"kind": "indoor", "bounds_m": [10.0, 10.0, 3.5], "shape": "ellipse"},
           "objects": [{"id": "art", "category": "picture_frame"}, {"id": "desk", "category": "desk"}],
           "relations": [{"a": "desk", "rel": "against_wall", "b": "wall", "at_deg": 90}]}
    spec, _ = _solve(raw)
    art = next(r for r in spec.relations if r.a == "art")
    assert art.b == "wall" and art.at_deg == 270


def test_coffee_table_on_the_rug():
    # living_room: "category 'rug' cannot support other objects" (coffee_table on rug)
    objs = [{"id": "sofa", "category": "sofa"}, {"id": "table", "category": "coffee_table"},
            {"id": "rug", "category": "rug"}]
    base = [{"a": "sofa", "rel": "against_wall", "b": "wall_north"}, {"a": "table", "rel": "in_front_of", "b": "sofa"}]
    spec, pl = _solve(_rect(objs, [*base, {"a": "table", "rel": "on", "b": "rug"}]))
    assert ("rug", "centered_in", "table") in [(r.a, r.rel, r.b) for r in spec.relations]
    assert math.dist(pl["rug"].location[:2], pl["table"].location[:2]) < 0.06
    # "rug under coffee_table" (not within the footprint) reads the same
    spec, _ = _solve(_rect(objs, [*base, {"a": "rug", "rel": "under", "b": "table"}]))
    assert ("rug", "centered_in", "table") in [(r.a, r.rel, r.b) for r in spec.relations]
    # a rug with a place of its own: the table stands within it
    spec, _ = _solve(_rect(objs, [{"a": "rug", "rel": "centered_in", "b": "room"},
                                  {"a": "table", "rel": "on", "b": "rug"}]))
    assert ("table", "centered_in", "rug") in [(r.a, r.rel, r.b) for r in spec.relations]


def test_gap_ranges_floor_targets_rect_at_deg_and_copy_ids():
    raw = _rect([{"id": "sofa", "category": "sofa"}, {"id": "table", "category": "coffee_table"},
                 {"id": "plant1", "category": "potted_plant"}],
                [{"a": "sofa", "rel": "against_wall", "b": "wall_north", "at_deg": 90},  # rect: at_deg ignored
                 {"a": "table", "rel": "in_front_of", "b": "sofa", "gap_m": [0.3, 0.5]},  # gap range
                 {"a": "table", "rel": "on", "b": "floor"},  # on room / on floor
                 {"a": "plant1_1", "rel": "beside", "b": "sofa"}])  # an expanded copy id
    spec, _pl = _solve(raw)
    assert [(r.a, r.rel, r.b, r.gap_m, r.at_deg) for r in spec.relations] == [
        ("sofa", "against_wall", "wall_north", None, None), ("table", "in_front_of", "sofa", 0.4, None),
        ("plant1", "beside", "sofa", None, None)]
    assert len(spec.notes) == 4
    assert validate_spec(spec.to_dict())[0].notes == ()  # normalised once, clean after


def test_what_has_no_single_reading_is_still_refused():
    raw = _rect([{"id": "lamp", "category": "floor_lamp"}, {"id": "sofa", "category": "sofa"}],
                [{"a": "lamp", "rel": "on", "b": "nowhere"}])
    spec, errors = validate_spec(raw)
    assert spec is None and any("'nowhere'" in e for e in errors)
