"""Material strings as weak models wrote them (bench mining 2026-10-16, item 4).

One odd material used to refuse the whole spec (46 errors in 22 scenes); now it is coerced and
the result says so. Inputs are the material strings of real failing calls.
"""

from __future__ import annotations

import pytest
from ai_db_scene.materials import coerce_material, material_error
from ai_db_scene.spec import validate_spec


@pytest.mark.parametrize(("raw", "want"), [
    ("metal:brass", "brass"), ("metal:gold", "gold"), ("stone:marble", "marble"),
    ("fabric:blue", "fabric:#2f5d9e"), ("wood:beech", "wood:oak"), ("wool", "fabric:weave"),
    ("terracotta", "ceramic:#b5603a"), ("grass", "foliage"), ("cardboard", "paint:#b08a5a"),
    ("brick", "stone:#8a4b3a"), ("tree", "foliage"), ("metal:brass:#aa8800", "brass:#aa8800"),
])
def test_bench_materials_are_coerced(raw, want):
    got, note = coerce_material(raw)
    assert got == want and note and raw in note
    assert material_error(got) is None


def test_valid_materials_pass_untouched_and_nonsense_falls_back():
    assert coerce_material("fabric:linen:#6b7f99") == ("fabric:linen:#6b7f99", None)
    got, note = coerce_material("unobtainium")
    assert got is None and "category default" in (note or "")


def test_no_error_says_allowed_empty():
    # park_corner#10: "unknown stone variant 'marble'; allowed: []"
    for raw in ("stone:marble", "brass:shiny", "plaster:rough"):
        err = material_error(raw) or ""
        assert "allowed: []" not in err and "takes no variant" in err


def test_one_bad_material_no_longer_refuses_the_spec():
    # round_lobby#14: objects[8].material 'metal:brass' refused a 9-object spec
    raw = {"setting": {"kind": "indoor", "bounds_m": [8.0, 8.0, 3.0], "floor_material": "stone:marble"},
           "objects": [{"id": "desk", "category": "desk", "material": "wood:oak"},
                       {"id": "lamp", "category": "floor_lamp", "material": "metal:brass"},
                       {"id": "rug", "category": "rug", "material": "carpet:navy"}]}
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    assert [o.material for o in spec.objects] == ["wood:oak", "brass", "fabric:weave:#1f2f55"]
    assert spec.setting.floor_material == "marble"
    assert len(spec.notes) == 3 and all("read as" in n for n in spec.notes)
    again, _ = validate_spec(spec.to_dict())
    assert again is not None and again.notes == ()  # the coerced spec is clean


def test_build_and_submit_report_the_coercion():
    from ai_db_scene.tools import scene_pipeline as sp
    out = sp._handle_submit_scene_spec({"spec": {"setting": {"kind": "indoor", "bounds_m": [4.0, 4.0, 2.7]},
                                                 "objects": [{"id": "lamp", "category": "floor_lamp",
                                                              "material": "metal:brass"}]}})
    assert out["status"] == "ok" and out["notes"] == ["objects[0].material: material 'metal:brass' read as 'brass'"]


@pytest.mark.parametrize(("raw", "want"), [
    # qwen3-14b-local-aidb5 classroom: 28 of these fell back to the category default
    ("wood_oak", "wood:oak"), ("fabric_black", "fabric:#151515"), ("plastic_blue", "plastic:#2f5d9e"),
    ("glossy_white", "paint:#f2f2f0"), ("laminate_brown", "paint:#6b4a2f"), ("drywall", "plaster"),
])
def test_kind_underscore_variant_and_colour_are_read(raw, want):
    got, note = coerce_material(raw)
    assert got == want and note == f"material {raw!r} read as {want!r}"
    assert material_error(got) is None


def test_known_underscore_aliases_keep_their_reading():
    assert coerce_material("stainless_steel")[0] == "metal:brushed"
    assert coerce_material("unobtainium_metal")[0] is not None  # 'metal' is a kind: the kind itself
    assert coerce_material("foo_bar")[0] is None
