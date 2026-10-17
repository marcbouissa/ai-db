"""The kind of water a brief names, when the spec leaves it implicit (bench beach_cove, 2026-10-17).

qwen3-14b-local-aidb6's beach_cove spec had no setting.water; 'rocks at water_edge' or a water
object (category ``water``, material ``turquoise``) added the open sea, so "a tropical beach cove
... calm turquoise water" was built with deep blue waves. The brief now picks the kind.
"""

from __future__ import annotations

from ai_db_scene import terrain
from ai_db_scene.spec import validate_spec

BRIEF = ("A tropical beach cove outdoors: a sandy beach sloping into calm turquoise water, five palm trees, "
         "dense green bushes and ferns along the back, a wooden boardwalk from the bushes to the water, two sun "
         "loungers under a parasol, and a few rocks at the water's edge.")
#: qwen3-14b-local-aidb6 beach_cove, the demo build (no setting.water; the water's edge in relations)
DEMO = {"setting": {"kind": "outdoor", "floor_material": "sand", "bounds_m": [20, 20, 5], "style": "tropical"},
        "objects": [{"id": "palm1", "category": "palm_tree", "count": 5, "material": "tropical_leaf", "hero": True},
                    {"id": "rock1", "category": "rock", "count": 4, "material": "weathered_stone"}],
        "relations": [{"a": "rock1", "rel": "at", "b": "water_edge"}]}


def _water(raw, brief):
    raw, note = terrain.water_from_brief(raw, brief)
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    return spec.setting.water, note


def test_brief_words_pick_the_kind_of_water():
    assert terrain.brief_terrain(BRIEF)[1] == {"kind": "lagoon", "side": "north"}
    assert terrain.brief_terrain("a quiet cove with a jetty")[1] == {"kind": "lagoon", "side": "north"}
    assert terrain.brief_terrain("turquoise sea and white sand")[1] == {"kind": "lagoon", "side": "north"}
    assert terrain.brief_terrain("a lake with a small beach")[1] == {"kind": "lake", "side": "north"}
    assert terrain.brief_terrain("a duck pond near the shore")[1] == {"kind": "pond"}
    assert terrain.brief_terrain("a beach with a bar")[1] == {"kind": "sea", "side": "north"}


def test_water_named_only_by_the_waters_edge_takes_the_briefs_kind():
    water, note = _water(DEMO, BRIEF)
    assert water is not None and water.kind == "lagoon" and water.side == "north"
    assert note and "lagoon" in note
    # without the brief the edge still adds the sea (as before)
    water, note = _water(DEMO, None)
    assert water is not None and water.kind == "sea" and note is None


def test_a_generic_water_object_takes_the_briefs_kind():
    raw = {**DEMO, "objects": [*DEMO["objects"], {"id": "water", "category": "turquoise_water"}]}
    assert _water(raw, BRIEF)[0].kind == "lagoon"
    assert _water(raw, "a garden pond with two benches")[0].kind == "pond"
    # water: true / {} / {side} without a kind
    for have in (True, {}, {"side": "east"}):
        raw = {**DEMO, "setting": {**DEMO["setting"], "water": have}}
        water = _water(raw, "a lakeside picnic spot")[0]
        assert water.kind == "lake" and water.side == ("east" if have == {"side": "east"} else "north")


def test_a_kind_the_model_wrote_stays():
    raw = {**DEMO, "setting": {**DEMO["setting"], "water": {"kind": "sea"}}}
    assert _water(raw, BRIEF) == (terrain.Water("sea", "north", 0.3, None), None)
    raw = {**DEMO, "objects": [*DEMO["objects"], {"id": "o", "category": "ocean"}]}
    assert _water(raw, BRIEF)[0].kind == "sea"
    # indoors, or a brief without water: nothing
    indoor = {"setting": {"kind": "indoor", "bounds_m": [5, 4, 3]}, "objects": [{"id": "r", "category": "rock"}]}
    assert terrain.water_from_brief(indoor, BRIEF) == (indoor, None)
    assert terrain.water_from_brief(DEMO, "a park corner with benches") == (DEMO, None)
