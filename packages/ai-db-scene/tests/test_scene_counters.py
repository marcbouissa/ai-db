"""bar_counter, reception_desk and bar_stool (bench mining 2026-10-16, item 12 of the findings).

Judges in 4 cafe scenes: "the counter is a low sideboard" (built as kitchen_counter); in 4 round
lobbies: "the reception desk is a thin console table" (built as desk). The category names are
the ones models wrote.
"""

from __future__ import annotations

from ai_db_scene.layout import check_layout, solve_layout
from ai_db_scene.materials import CATEGORY_MATERIALS, material_error
from ai_db_scene.priors import load_priors
from ai_db_scene.recipes import has_recipe, recipe_spec
from ai_db_scene.room_kits import kits_for
from ai_db_scene.spec import validate_spec

NEW = ("bar_counter", "reception_desk", "bar_stool")


def test_new_categories_have_priors_recipes_and_materials():
    priors = load_priors()
    for cat in NEW:
        assert cat in priors and has_recipe(cat) and material_error(CATEGORY_MATERIALS[cat]) is None
        spec = recipe_spec(cat, f"x_{cat}", priors[cat].typical_m)
        assert spec["bbox_m"] == list(priors[cat].typical_m)
        assert all(material_error(p["material"].split(":scale=")[0]) is None for p in spec["parts"]), cat
    # a counter you stand at is taller than a kitchen run, and a bar stool's seat suits it
    assert priors["bar_counter"].typical_m[2] >= 1.05 > priors["kitchen_counter"].typical_m[2]
    assert priors["reception_desk"].typical_m[2] >= 1.05
    assert priors["bar_counter"].typical_m[2] - priors["bar_stool"].typical_m[2] < 0.35


def test_names_models_wrote_map_to_the_categories():
    raw = {"setting": {"kind": "indoor", "bounds_m": [8.0, 6.0, 3.0]},
           "objects": [{"id": "counter", "category": "service_counter", "seats": 3},
                       {"id": "desk", "category": "front_desk"}],
           "relations": [{"a": "counter", "rel": "against_wall", "b": "wall_north"}]}
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    assert [o.category for o in spec.objects] == ["bar_counter", "reception_desk"]
    pls = solve_layout(spec)
    assert check_layout(spec, pls) == []
    assert sum(p.category == "bar_stool" for p in pls) == 3  # seats at a bar counter are bar stools


def test_kits_use_them():
    cafe = kits_for("a small cafe")[0]["typical"]
    assert any(i["category"] == "bar_counter" for i in cafe)
    lobby = kits_for("a round hotel lobby with a reception desk")[0]["typical"]
    assert any(i["category"] == "reception_desk" for i in lobby)
