"""Wall words outdoors mean sides of the beach (bench beach_cove, 2026-10-17).

"dense green bushes and ferns along the back": weak models write ``ferns along back_wall`` or
``bushes against north_wall`` outdoors, and both were dropped (no walls outdoors). With water,
the back is away from it and the front toward it; the water's own compass side is its front.
"""

from __future__ import annotations

from ai_db_scene.layout import check_layout, solve_layout
from ai_db_scene.spec import validate_spec
from ai_db_scene.terrain import terrain_of


def _beach(relations, water=None):
    return {"setting": {"kind": "outdoor", "bounds_m": [20, 16, 5], "ground": "sand",
                        "water": water or {"kind": "lagoon"}},
            "objects": [{"id": "ferns", "category": "fern", "count": 4},
                        {"id": "bushes", "category": "tropical_bush", "count": 3},
                        {"id": "rocks", "category": "rock", "count": 2}],
            "relations": relations}


def _rels(raw):
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    return spec, sorted((r.a, r.rel, r.b) for r in spec.relations)


def test_back_and_front_words_relate_to_the_water():
    spec, rels = _rels(_beach([{"a": "ferns", "rel": "along", "b": "back_wall"},
                               {"a": "bushes", "rel": "against", "b": "the rear"},
                               {"a": "rocks", "rel": "at", "b": "front"}]))
    assert rels == [("bushes", "back_of", "water"), ("ferns", "back_of", "water"), ("rocks", "near", "water")]
    assert any("'ferns along back_wall' read as 'ferns back_of water'" in n for n in spec.notes)


def test_compass_walls_relative_to_the_waters_side():
    # the sea north (the default side): north is the front, south the back, east/west across the shore
    _spec, rels = _rels(_beach([{"a": "ferns", "rel": "against_wall", "b": "wall_south"},
                                {"a": "bushes", "rel": "against", "b": "north_wall"},
                                {"a": "rocks", "rel": "along", "b": "east_wall"}]))
    assert rels == [("bushes", "near", "water"), ("ferns", "back_of", "water")]
    _spec, rels = _rels(_beach([{"a": "ferns", "rel": "along", "b": "west side"}], {"kind": "sea", "side": "east"}))
    # bushes with no place of their own go along the back too (aidb7 beach_cove)
    assert sorted(rels) == [("bushes", "back_of", "water"), ("ferns", "back_of", "water")]
    # a pond has no side: only back / front mean something
    _spec, rels = _rels(_beach([{"a": "ferns", "rel": "along", "b": "back_wall"},
                                {"a": "bushes", "rel": "along", "b": "north_wall"}], {"kind": "pond"}))
    assert rels == [("ferns", "back_of", "water")]


def test_ferns_along_the_back_stand_behind_the_beach():
    spec, _ = _rels(_beach([{"a": "ferns", "rel": "along", "b": "back_wall"},
                            {"a": "rocks", "rel": "at", "b": "water_edge"}]))
    pls = solve_layout(spec)
    assert check_layout(spec, pls) == []
    ter = terrain_of(spec)
    assert ter is not None
    ferns = [ter.signed(*p.location[:2]) for p in pls if p.category == "fern"]
    rocks = [ter.signed(*p.location[:2]) for p in pls if p.category == "rock"]
    assert len(ferns) == 4 and max(ferns) < min(rocks) - 6  # far up the beach, at its back


def test_without_water_wall_words_are_still_dropped():
    raw = {"setting": {"kind": "outdoor", "bounds_m": [12, 10, 4], "ground": "grass"},
           "objects": [{"id": "bench", "category": "bench"}],
           "relations": [{"a": "bench", "rel": "against", "b": "back_wall"}]}
    spec, rels = _rels(raw)
    assert rels == [] and any("no walls outdoors" in n for n in spec.notes)
