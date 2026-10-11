"""Surface materials for the scene pipeline: a closed vocabulary the small model can pick from,
and a default per category so every placed object is textured even when it picks nothing.

The vocabulary mirrors the Blender addon's procedural library (``LIB_KINDS`` in its
``materials.py``): ``kind`` or ``kind:variant``, optionally ``:#rrggbb`` to tint the base
colour, e.g. ``wood:walnut``, ``fabric:linen:#6b7f99``, ``metal:brushed``, ``plaster``.
"""

from __future__ import annotations

import re
from typing import TypedDict

#: kind -> variants ("" = the kind takes no variant). Keep in sync with the addon.
MATERIAL_KINDS: dict[str, tuple[str, ...]] = {
    "wood": ("oak", "walnut", "mahogany", "cherry", "pine", "dark"),
    "brass": ("",), "gold": ("",), "chrome": ("",), "plaster": ("",),
    "marble": ("carrara", "calacatta", "verde"),
    "leather": ("",), "velvet": ("",), "ceramic": ("",),
    "fabric": ("weave", "linen", "denim", "canvas", "velvet", "leather"),
    "glass": ("clear", "frosted", "tinted"),
    "metal": ("steel", "copper", "chrome", "rust", "brushed"),
    # matte/painted surfaces with colour, roughness and bump variation (addon >= this version;
    # an older addon gets legacy_fallback's flat colour)
    "concrete": ("",), "stone": ("",), "foliage": ("",), "plastic": ("",), "paint": ("",), "rubber": ("",),
    "color": ("",),
}

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

#: ``pbr:<preset>`` -> (library base material, wear amount 0..1, wear scale): the base gets the
#: procedural grunge layer of the pbr-shader-builder skill (``add_grunge``), on a per-object copy.
PBR_PRESETS: dict[str, tuple[str, float, float]] = {
    "worn_metal": ("metal:steel", 0.6, 6.0), "rusty_steel": ("metal:rust", 0.7, 5.0),
    "aged_brass": ("brass", 0.4, 10.0), "weathered_wood": ("wood:pine", 0.5, 4.0),
    "scuffed_leather": ("leather", 0.4, 8.0), "dusty_plaster": ("plaster", 0.3, 12.0),
    "worn_concrete": ("color:#8a8782", 0.6, 3.0),
}

#: Material of each category when the spec names none (box projection: placeholders have no UVs).
CATEGORY_MATERIALS: dict[str, str] = {
    "armchair": "fabric:linen", "sofa": "fabric:linen", "bed": "fabric:weave", "rug": "fabric:weave",
    "chair": "wood:oak", "dining_chair": "wood:walnut", "office_chair": "fabric:weave", "stool": "wood:oak",
    "bench": "wood:pine", "picnic_table": "wood:pine",
    "coffee_table": "wood:walnut", "side_table": "wood:walnut", "dining_table": "wood:oak",
    "conference_table": "wood:walnut", "desk": "wood:oak", "nightstand": "wood:cherry",
    "dresser": "wood:cherry", "wardrobe": "wood:oak", "cabinet": "wood:oak", "bookshelf": "wood:walnut",
    "shelving_unit": "metal:steel", "tv_stand": "wood:dark", "kitchen_counter": "marble:carrara",
    "kitchen_island": "marble:carrara", "workbench": "wood:pine", "crate": "wood:pine",
    "pallet": "wood:pine", "door": "wood:oak", "window": "glass:clear", "stairs": "wood:oak",
    "refrigerator": "metal:brushed", "stove": "metal:steel", "sink": "ceramic", "toilet": "ceramic",
    "bathtub": "ceramic", "shower_stall": "glass:frosted", "mirror": "glass:clear",
    "tv": "plastic:#111111", "monitor": "plastic:#111111", "laptop": "metal:brushed", "printer": "plastic:#d8d8d8",
    "filing_cabinet": "metal:steel", "pallet_rack": "metal:steel", "conveyor": "metal:steel",
    "forklift": "paint:#e0a000", "car": "paint:#7a1f1f", "safety_barrier": "paint:#e0a000",
    "trash_can": "metal:steel", "street_lamp": "metal:steel", "fence_section": "wood:pine",
    "floor_lamp": "brass", "table_lamp": "brass", "pendant_lamp": "brass", "ceiling_light": "color:#f2f2f2",
    "wall_sconce": "brass", "fireplace": "marble:carrara", "column": "marble:carrara",
    "potted_plant": "ceramic", "vase": "ceramic", "tree": "foliage", "bush": "foliage:#4a7a35",
    "rock": "stone", "barrel": "wood:dark", "book_stack": "color:#7a3b2e",
    "picture_frame": "wood:dark", "wall_clock": "wood:dark", "whiteboard": "plastic:#f4f4f4",
    "wall_partition": "plaster", "niche": "plaster", "fountain": "stone:#b7b1a6",
    # outdoor / industrial
    "playground_slide": "plastic:#c8372d", "swing_set": "metal:steel", "gazebo": "wood:pine",
    "planter": "concrete", "hedge": "foliage:#3f6b33", "signpost": "paint:#1f5a3a", "bollard": "metal:steel",
    "parking_bay": "concrete:#3a3a3c", "loading_dock": "concrete", "shipping_container": "metal:rust",
    "traffic_cone": "plastic:#ff6a13", "lamp_post_double": "metal:steel",
}
DEFAULT_MATERIAL = "plaster"
FLOOR_MATERIALS = {"indoor": "wood:oak", "outdoor": "foliage:#4f6b3a", "tabletop": "wood:walnut"}
WALL_MATERIAL = "plaster"

#: ``setting.style`` -> palette: ``floor``/``wall`` for indoor rooms, ``categories`` per-category
#: overrides. Explicit spec materials (objects[].material, floor_material, wall_material) win.
class StylePalette(TypedDict):
    floor: str
    wall: str
    categories: dict[str, str]


STYLE_PALETTES: dict[str, StylePalette] = {
    "modern": {"floor": "wood:oak", "wall": "plaster:#f2f2f0", "categories": {
        "sofa": "fabric:weave:#5a5f66", "armchair": "fabric:weave:#5a5f66", "coffee_table": "glass:tinted",
        "side_table": "metal:chrome", "dining_table": "wood:walnut", "dining_chair": "metal:brushed",
        "chair": "metal:brushed", "bookshelf": "metal:brushed", "tv_stand": "color:#1c1c1c",
        "floor_lamp": "chrome", "table_lamp": "chrome", "pendant_lamp": "chrome", "rug": "fabric:weave:#8c8c8c"}},
    "scandinavian": {"floor": "wood:pine", "wall": "plaster:#f4f1ea", "categories": {
        "sofa": "fabric:linen:#d9d4c7", "armchair": "fabric:linen:#c8bfae", "bed": "fabric:linen:#e8e4da",
        "coffee_table": "wood:oak", "side_table": "wood:oak", "dining_table": "wood:oak", "chair": "wood:oak",
        "dining_chair": "wood:oak", "bookshelf": "wood:oak", "tv_stand": "wood:oak", "nightstand": "wood:oak",
        "floor_lamp": "color:#f0f0f0", "table_lamp": "ceramic", "pendant_lamp": "color:#f0f0f0",
        "rug": "fabric:weave:#e6e1d6"}},
    "industrial": {"floor": "color:#6e6e6a", "wall": "plaster:#8f8a84", "categories": {
        "sofa": "leather", "armchair": "leather", "coffee_table": "wood:dark", "side_table": "metal:steel",
        "dining_table": "wood:dark", "chair": "metal:steel", "dining_chair": "metal:steel", "stool": "metal:steel",
        "bookshelf": "metal:steel", "tv_stand": "metal:steel", "desk": "wood:dark",
        "floor_lamp": "metal:steel", "table_lamp": "metal:steel", "pendant_lamp": "metal:steel",
        "rug": "fabric:weave:#5b5048"}},
    "rustic": {"floor": "wood:dark", "wall": "plaster:#e8dcc4", "categories": {
        "sofa": "fabric:canvas", "armchair": "fabric:canvas", "coffee_table": "wood:pine", "side_table": "wood:pine",
        "dining_table": "wood:pine", "chair": "wood:pine", "dining_chair": "wood:pine", "bookshelf": "wood:pine",
        "tv_stand": "wood:pine", "bed": "fabric:weave:#a08060", "floor_lamp": "metal:copper",
        "table_lamp": "metal:copper", "pendant_lamp": "metal:copper", "rug": "fabric:weave:#8a5a3c"}},
    "classic": {"floor": "wood:mahogany", "wall": "plaster:#efe6d2", "categories": {
        "sofa": "velvet", "armchair": "velvet", "coffee_table": "wood:mahogany", "side_table": "wood:mahogany",
        "dining_table": "wood:mahogany", "chair": "wood:mahogany", "dining_chair": "wood:mahogany",
        "bookshelf": "wood:mahogany", "desk": "wood:mahogany", "fireplace": "marble:carrara",
        "floor_lamp": "brass", "table_lamp": "brass", "pendant_lamp": "brass", "rug": "fabric:weave:#7a2e2e"}},
}


def style_error(value: object) -> str | None:
    """Why ``value`` is not a usable ``setting.style``, or None. Free text is allowed (stored specs
    say e.g. "cozy living room"); only a text naming a known style picks a palette."""
    if isinstance(value, str) and value.strip():
        return None
    return f"expected a style string, e.g. one of {sorted(STYLE_PALETTES)}, got {value!r}"


def style_palette(style: str | None) -> str | None:
    """The known style a ``setting.style`` text names ("Industrial loft" -> industrial), else None."""
    words = re.findall(r"[a-z]+", (style or "").lower())
    return next((w for w in words if w in STYLE_PALETTES), None)


def style_material(style: str | None, key: str) -> str | None:
    """The style's material for a category, or for ``floor`` / ``wall``; None = no override."""
    name = style_palette(style)
    palette = STYLE_PALETTES[name] if name else None
    if not palette:
        return None
    if key == "floor":
        return palette["floor"]
    if key == "wall":
        return palette["wall"]
    return palette["categories"].get(key)


def material_vocabulary() -> list[str]:
    """Every allowed ``kind[:variant]`` (tint suffix ``:#rrggbb`` allowed on any but ``pbr:``)."""
    return [k if v == "" else f"{k}:{v}" for k, vs in MATERIAL_KINDS.items() for v in vs] + [
        f"pbr:{p}" for p in PBR_PRESETS]


def pbr_preset(value: str | None) -> tuple[str, float, float] | None:
    """(base material, wear amount, wear scale) of a ``pbr:<preset>`` material, else None."""
    if not value or not value.startswith("pbr:"):
        return None
    return PBR_PRESETS.get(value[4:])


def material_error(value: object) -> str | None:
    """Why ``value`` is not a valid material string, or None."""
    if not isinstance(value, str) or not value:
        return f"expected a material string like 'wood:oak', got {value!r}"
    if value.startswith("pbr:"):
        if value[4:] not in PBR_PRESETS:
            return f"unknown pbr preset {value[4:]!r}; allowed: {sorted(PBR_PRESETS)}"
        return None
    parts = value.split(":")
    kind, rest = parts[0], parts[1:]
    if kind not in MATERIAL_KINDS:
        return f"unknown material kind {kind!r}; allowed: {sorted(MATERIAL_KINDS)}"
    variants = MATERIAL_KINDS[kind]
    if rest and not _HEX.match(rest[0]):
        if rest[0] not in variants or rest[0] == "":
            return f"unknown {kind} variant {rest[0]!r}; allowed: {[v for v in variants if v]}"
        rest = rest[1:]
    if kind == "color" and not rest:
        return "color needs a tint: 'color:#rrggbb'"
    if len(rest) > 1 or (rest and not _HEX.match(rest[0])):
        return f"after kind[:variant] only one ':#rrggbb' tint is allowed, got {value!r}"
    return None


#: flat colour for each newer kind when the connected addon predates it
_LEGACY = {"concrete": "#9a9890", "stone": "#7d7a72", "foliage": "#3d6b2f", "plastic": "#b0b0b0",
           "paint": "#7a1f1f", "rubber": "#111111"}


def legacy_fallback(material: str) -> str | None:
    """``color:#...`` for a material of a kind an older addon rejects (keeps its tint), else None."""
    parts = material.split(":")
    if parts[0] not in _LEGACY:
        return None
    tint = next((p for p in parts[1:] if _HEX.match(p)), _LEGACY[parts[0]])
    return f"color:{tint}"


def material_for(category: str, chosen: str | None = None) -> str:
    """The addon material string for an object: the spec's choice, else the category default.
    Box projection: placeholders and simple meshes have no UV map."""
    base = chosen or CATEGORY_MATERIALS.get(category, DEFAULT_MATERIAL)
    pbr = pbr_preset(base)
    base = pbr[0] if pbr else base  # the wear layer goes on in a later step (scene_pipeline)
    return base if base.startswith("color") else f"{base}:proj=box"
