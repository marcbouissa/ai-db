"""SceneSpec validation: exact, actionable error strings."""

from __future__ import annotations

import copy

import pytest
from ai_db_scene.priors import categories
from ai_db_scene.spec import RELATIONS, SETTING_KINDS, SceneSpec, spec_schema, validate_spec

VALID = {
    "setting": {"kind": "indoor", "bounds_m": [6, 5, 2.8], "style": "scandinavian"},
    "objects": [
        {"id": "sofa", "category": "sofa", "count": 1, "hero": True},
        {"id": "table", "category": "coffee_table"},
        {"id": "lamp", "category": "floor_lamp"},
    ],
    "relations": [
        {"a": "sofa", "rel": "against_wall", "b": "wall_north"},
        {"a": "table", "rel": "in_front_of", "b": "sofa", "gap_m": 0.45},
        {"a": "lamp", "rel": "beside", "b": "sofa"},
    ],
    "lighting": {"mood": "warm evening"},
    "camera": {"focus": "sofa"},
}


def _bad(mutate):
    d = copy.deepcopy(VALID)
    mutate(d)
    spec, errors = validate_spec(d)
    assert spec is None
    return errors


def test_valid_spec_roundtrip():
    spec, errors = validate_spec(VALID)
    assert errors == [] and spec is not None
    assert spec.objects[0].hero and spec.relations[1].gap_m == 0.45
    again, errs = validate_spec(spec.to_dict())
    assert errs == [] and again == spec
    assert SceneSpec.from_dict(VALID) == spec


def test_vocabularies_in_schema():
    schema = spec_schema()
    assert schema["properties"]["setting"]["properties"]["kind"]["enum"] == list(SETTING_KINDS)
    assert schema["properties"]["relations"]["items"]["properties"]["rel"]["enum"] == list(RELATIONS)
    assert set(RELATIONS) == {"on", "against_wall", "in_front_of", "beside", "facing", "centered_in", "under", "around",
                              "above"}


def test_root_not_object():
    assert validate_spec([1])[1] == ["<root>: expected object, got list"]


def test_missing_required():
    spec, errors = validate_spec({})
    assert spec is None
    assert errors == ["setting: required field is missing", "objects: required field is missing"]


def test_unknown_category_suggests():
    errors = _bad(lambda d: d["objects"][0].update(category="sofaa"))
    assert len(errors) == 1
    assert errors[0].startswith("objects[0].category: 'sofaa' is not a known category. Did you mean 'sofa'? allowed values: [")
    assert str(categories()) in errors[0]


def test_bad_relation_name():
    errors = _bad(lambda d: d["relations"][1].update(rel="near"))
    assert errors[0] == (
        "relations[1].rel: 'near' is not allowed; allowed values: "
        "['on', 'against_wall', 'in_front_of', 'beside', 'facing', 'centered_in', 'under', 'around', 'above']"
    )


def test_bad_setting_kind_and_bounds():
    errors = _bad(lambda d: d["setting"].update(kind="space", bounds_m=[6, -1, 3]))
    assert "setting.kind: 'space' is not allowed; allowed values: ['indoor', 'outdoor', 'tabletop', 'single_asset']" in errors
    assert "setting.bounds_m[1]: expected number in (0, 500], got -1" in errors


def test_unknown_object_reference():
    errors = _bad(lambda d: d["relations"][2].update(b="couch"))
    assert errors == ["relations[2].b: 'couch' is not a valid target for 'beside'. allowed values: ['sofa', 'table', 'lamp']"]


def test_against_wall_needs_wall_and_indoor():
    errors = _bad(lambda d: d["relations"][0].update(b="sofa2"))
    assert errors[0].startswith("relations[0].b: 'sofa2' is not a wall; allowed values: ['wall_north'")
    errors = _bad(lambda d: d["setting"].update(kind="outdoor"))
    assert errors == ["relations[0]: against_wall needs setting.kind 'indoor' (walls exist only indoors), got 'outdoor'"]


def test_duplicate_reserved_ids_and_self_relation():
    errors = _bad(lambda d: d["objects"].append({"id": "sofa", "category": "chair"}))
    assert errors == ["objects[3].id: duplicate id 'sofa'"]
    errors = _bad(lambda d: d["objects"][1].update(id="wall_north"))
    assert errors[0].startswith("objects[1].id: 'wall_north' is reserved")
    errors = _bad(lambda d: d["relations"][1].update(b="table"))
    assert errors == ["relations[1]: object 'table' cannot be related to itself"]


def test_count_gap_hero_types_and_unknown_fields():
    errors = _bad(lambda d: d["objects"][0].update(count=0, hero="yes", color="red"))
    assert "objects[0].count: expected integer in [1, 50], got 0" in errors
    assert "objects[0].hero: expected boolean, got 'yes'" in errors
    assert "objects[0].color: unknown field; allowed fields: ['id', 'category', 'count', 'hero', 'detail', 'grid', 'material', 'dynamic', 'room']" in errors
    errors = _bad(lambda d: d["relations"][1].update(gap_m=-2))
    assert errors == ["relations[1].gap_m: expected number >= 0, got -2"]


def test_centered_in_room_and_camera_focus():
    d = copy.deepcopy(VALID)
    d["relations"].append({"a": "table", "rel": "centered_in", "b": "room"})
    assert validate_spec(d)[1] == []
    d["camera"] = {"focus": "ghost"}
    assert validate_spec(d)[1] == ["camera.focus: 'ghost' is not a declared object id; declared ids: ['sofa', 'table', 'lamp']"]


def test_from_dict_raises_with_all_errors():
    with pytest.raises(ValueError, match="objects: required field is missing"):
        SceneSpec.from_dict({"setting": VALID["setting"]})


def test_materials_are_validated_and_round_trip():
    from ai_db_scene.materials import CATEGORY_MATERIALS, material_error
    from ai_db_scene.priors import categories
    base = {"setting": {"kind": "indoor", "bounds_m": [5, 4, 2.8], "floor_material": "marble:carrara",
                        "wall_material": "plaster:#e8e0d0"},
            "objects": [{"id": "sofa", "category": "sofa", "material": "fabric:linen:#6b7f99"}]}
    spec, errors = validate_spec(base)
    assert spec is not None, errors
    assert spec.objects[0].material == "fabric:linen:#6b7f99"
    assert validate_spec(spec.to_dict())[0] == spec
    odd = {**base, "objects": [{"id": "sofa", "category": "sofa", "material": "wood:teak"}]}
    spec, errors = validate_spec(odd)  # coerced with a note, never a refused spec
    assert spec is not None and spec.objects[0].material == "wood:walnut" and "wood:teak" in spec.notes[0]
    assert "unknown wood variant 'teak'" in material_error("wood:teak")
    assert material_error("color") and material_error("wood:oak:#12345") and not material_error("ceramic")
    # every category has a valid default
    assert set(categories()) <= set(CATEGORY_MATERIALS)
    assert all(material_error(m) is None for m in CATEGORY_MATERIALS.values())
