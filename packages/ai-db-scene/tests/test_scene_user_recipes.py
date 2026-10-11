"""User recipes from text: create_recipe / draft_recipe, ShapeAssembly, dimension extraction."""

from __future__ import annotations

import pytest
from ai_db_scene import user_recipes as ur
from ai_db_scene.recipes import has_recipe, recipe_spec

TABLE = [{"id": f"leg_{i}", "shape": "box", "size": [0.06, 0.06, 0.94], "at": [x, y, 0], "material": "wood:oak"}
         for i, (x, y) in enumerate(((-0.45, -0.45), (0.45, -0.45), (-0.45, 0.45), (0.45, 0.45)))] + [
    {"id": "top", "shape": "box", "size": [1.0, 1.0, 0.06], "at": [0, 0, 0.94], "material": "wood:oak"}]


def test_recipes_dir_is_isolated(tmp_path):
    assert ur.recipes_dir().startswith(str(tmp_path))


def test_create_recipe_used_by_recipe_spec(tmp_path):
    assert not has_recipe("dining_table")
    out = ur.create_recipe("dining_table", "trestle", TABLE, dims=[1.8, 0.9, 0.75],
                           source={"url": "https://example.org/t", "license": "CC0"})
    assert out["status"] == "created" and "warning" not in out
    assert has_recipe("dining_table")
    spec = recipe_spec("dining_table", "t1", [1.8, 0.9, 0.75])
    assert spec["parts"] == TABLE and spec["style"] == "ai-db user recipe" and spec["bbox_m"] == [1.8, 0.9, 0.75]
    with pytest.raises(ValueError, match="overwrite"):
        ur.create_recipe("dining_table", "other", TABLE)
    assert ur.list_user_recipes()[0]["source"]["license"] == "CC0"


def test_builtin_wins_unless_overwrite():
    builtin = recipe_spec("stool", "s", [0.4, 0.4, 0.7])["parts"]
    with pytest.raises(ValueError, match="built-in"):
        ur.create_recipe("stool", "mine", TABLE)
    ur.create_recipe("stool", "mine", TABLE, overwrite=True)
    assert recipe_spec("stool", "s", [0.4, 0.4, 0.7])["parts"] == TABLE != builtin


@pytest.mark.parametrize("parts, msg", [
    ([], "non-empty"),
    ([{**TABLE[4], "at": [0, 0, 0.0]}, {**TABLE[4], "id": "lid", "at": [0, 0, 0.5]}], "not connected"),
    ([{**TABLE[0], "material": "unobtainium"}], "material"),
    ([{**TABLE[0], "shape": "torus"}], "shape"),
    ([{**TABLE[0], "size": [3, 1, 1]}], "size"),
    ([*TABLE, {**TABLE[4], "id": "top"}], "duplicate"),
    ([{**TABLE[4], "size": [2.0, 1.0, 0.06], "at": [0, 0, 0]}], "outside"),
    ([{**TABLE[4], "at": [0, 0, 0]}], "fill the unit box"),
])
def test_create_recipe_rejects(parts, msg):
    with pytest.raises(ValueError, match=msg):
        ur.create_recipe("dining_table", "bad", parts)


def test_create_recipe_rejects_unknown_category_and_bad_source():
    with pytest.raises(ValueError, match="unknown category"):
        ur.create_recipe("spaceship", "x", TABLE)
    with pytest.raises(ValueError, match="http"):
        ur.create_recipe("dining_table", "x", TABLE, source={"url": "file:///etc/passwd"})


SA_CHAIR = """
@root_assembly
def root_assembly():
    bbox = Cuboid(1.0, 1.6, 1.0, True)
    seat = Cuboid(1.0, 0.1, 1.0, True)
    back = Cuboid(1.0, 0.8, 0.1, True)
    leg = Cuboid(0.1, 0.7, 0.1, True)
    attach(seat, bbox, 0.5, 0.0, 0.5, 0.5, 0.4375, 0.5)
    attach(back, seat, 0.5, 0.0, 0.5, 0.5, 1.0, 0.05)
    squeeze(leg, bbox, seat, bot, 0.05, 0.05)
    reflect(leg, X)
    translate(leg, Z, 1, 0.9)
    rotate(back, 10)
def child():
    x = Cuboid(1, 1, 1, True)
"""


def test_shapeassembly_chair_becomes_connected_parts():
    parts, warnings = ur.parse_shapeassembly(SA_CHAIR)
    ids = [p["id"] for p in parts]
    assert ids == ["seat", "back", "leg", "leg_r", "leg_t1"]
    assert any("rotate" in w for w in warnings) and any("sub-assembly" in w for w in warnings)
    seat = parts[0]
    assert seat["size"] == [1.0, 1.0, 0.0625] and seat["at"] == [0.0, 0.0, 0.4375]
    leg = parts[2]
    assert leg["size"][2] == 0.4375 and leg["at"][2] == 0.0  # squeezed floor -> seat bottom
    assert parts[3]["at"][0] == -leg["at"][0]  # reflected across X
    assert ur.recipe_errors(parts) == []


@pytest.mark.parametrize("text, dims", [
    ("Dimensions: W 180 cm x D 90 cm x H 75 cm. Solid oak.", [1.8, 0.9, 0.75]),
    ("Overall 72 x 36 x 30 in", [1.8288, 0.9144, 0.762]),
    ("Width: 600 mm, Depth: 400mm, Height: 1200 mm", [0.6, 0.4, 1.2]),
    ("It is 45 cm wide, 50 cm deep and 85 cm high", [0.45, 0.5, 0.85]),
    ("W 180 x D 90 x H 75 cm", [1.8, 0.9, 0.75]),
])
def test_extract_dims(text, dims):
    got, _ = ur.extract_dims(text)
    assert got == pytest.approx(dims)


def test_draft_recipe_from_spec_sheet_is_valid_skeleton():
    d = ur.draft_recipe("dining_chair", "Oak dining chair with four legs, a padded seat and a tall back. "
                                        "Seat height 46 cm. W 45 cm x D 52 cm x H 90 cm.")
    assert d["dims_m"] == pytest.approx([0.45, 0.52, 0.9])
    assert d["part_words"]["legs"] == 4 and "seat" in d["part_words"] and "back" in d["part_words"]
    assert d["issues"] == [] and {p["id"] for p in d["parts"]} >= {"leg_0", "seat", "back"}
    shelf = ur.draft_recipe("bookshelf", "Bookcase with five shelves, 80 x 30 x 180 cm")
    assert shelf["part_words"]["shelves"] == 5 and shelf["issues"] == []
    assert len([p for p in shelf["parts"] if p["id"].startswith("shelf_")]) == 5
    sa = ur.draft_recipe("chair", SA_CHAIR)
    assert sa["notes"] == ["ShapeAssembly program"] and sa["issues"] == []


def test_tools_registered():
    from ai_db.dispatcher import ServiceDispatcher

    names = {t["name"] for t in ServiceDispatcher(db_path=":memory:").list_tools()}
    assert {"create_recipe", "draft_recipe", "list_recipes"} <= names


def test_case_goods_skeleton_has_every_front_and_stays_valid():
    from ai_db_scene.user_recipes import draft_recipe, recipe_errors

    r = draft_recipe("cabinet", "Oak sideboard. W 160 cm x D 45 cm x H 80 cm. 3 doors, 2 drawers, 4 legs, 2 shelves.")
    ids = [p["id"] for p in r["parts"]]
    assert sum(i.startswith("door_") and not i.endswith("handle") for i in ids) == 3
    assert sum(i.startswith("drawer_") and not i.endswith("handle") for i in ids) == 2
    assert sum(i.startswith("leg_") for i in ids) == 4 and sum(i.endswith("_handle") for i in ids) == 5
    assert recipe_errors(r["parts"]) == []
