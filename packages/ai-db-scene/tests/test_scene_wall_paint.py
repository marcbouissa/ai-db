"""Interior walls get plausible paints (lead review of the qwen lobby / living room: salmon-red walls).

Root cause: the model wrote ``beige_paint``; coercion dropped the beige and kept ``paint``, whose
addon default is red clear-coated car paint. Walls now resolve through ``interior_wall``.
"""

from __future__ import annotations

import pytest
from ai_db_scene.materials import (
    COLOUR_NAMES,
    STYLE_PALETTES,
    WALL_CLADDING,
    WALL_MATERIAL,
    WALL_MAX_SATURATION,
    coerce_material,
    interior_wall,
    style_palette,
    wall_saturation,
)
from ai_db_scene.spec import validate_spec


def test_default_and_every_style_wall_is_a_muted_paint() -> None:
    assert wall_saturation(interior_wall(None)) <= WALL_MAX_SATURATION
    for name, pal in STYLE_PALETTES.items():
        wall = interior_wall(pal["wall"])
        assert wall_saturation(wall) <= WALL_MAX_SATURATION, name
        assert wall_saturation(pal["wall"]) <= WALL_MAX_SATURATION, f"{name} palette itself is too saturated"


@pytest.mark.parametrize("colour", sorted(COLOUR_NAMES))
def test_no_colour_word_makes_a_saturated_wall(colour: str) -> None:
    for written in (colour, f"{colour}_paint", f"paint:{colour}", f"plaster:{colour}"):
        mat, _ = coerce_material(written)
        assert wall_saturation(interior_wall(mat)) <= WALL_MAX_SATURATION + 0.005, (written, mat)


def test_beige_paint_keeps_its_beige_and_becomes_plaster() -> None:
    mat, _ = coerce_material("beige_paint")
    assert mat == f"paint:{COLOUR_NAMES['beige']}"
    assert interior_wall(mat) == f"plaster:{COLOUR_NAMES['beige']}"


def test_untinted_paint_wall_is_warm_white_plaster_not_car_paint() -> None:
    assert interior_wall("paint") == WALL_MATERIAL
    assert interior_wall("color:#a8322d").startswith("plaster:#")
    assert WALL_MATERIAL.startswith("plaster:#")


def test_cladding_and_textured_walls_kept() -> None:
    assert interior_wall(WALL_CLADDING) == WALL_CLADDING
    assert interior_wall("concrete:#a7a49c") == "concrete:#a7a49c"
    assert interior_wall("pbr:dusty_plaster") == "pbr:dusty_plaster"


def test_style_words_pick_a_palette() -> None:
    assert style_palette("cozy") == "cozy"
    assert style_palette("modern hotel lobby") == "modern"
    assert style_palette("hotel lobby") == "luxury"
    assert style_palette("minimalist") == "modern"
    assert style_palette("whatever") is None


def test_room_walls_command_uses_the_muted_wall() -> None:
    from ai_db_scene.layout import solve_layout
    from ai_db_scene.tools.scene_pipeline import build_layout_commands

    raw = {"setting": {"kind": "indoor", "bounds_m": [6, 5, 3], "style": "cozy", "wall_material": "paint"},
           "objects": [{"id": "sofa", "category": "sofa"}], "relations": []}
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    cmds, _ = build_layout_commands(spec, solve_layout(spec))
    walls = [c["params"]["material"] for c in cmds
             if c["type"] == "apply_material" and c["params"]["name"] == "Room_Walls"]
    assert walls and walls[0].startswith("plaster:#ece6dc")
