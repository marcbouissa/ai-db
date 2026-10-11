"""Surface materials for the scene pipeline: a closed vocabulary the small model can pick from,
and a default per category so every placed object is textured even when it picks nothing.

The vocabulary mirrors the Blender addon's procedural library (``LIB_KINDS`` in its
``materials.py``): ``kind`` or ``kind:variant``, optionally ``:#rrggbb`` to tint the base
colour, e.g. ``wood:walnut``, ``fabric:linen:#6b7f99``, ``metal:brushed``, ``plaster``.
"""

from __future__ import annotations

import re

#: kind -> variants ("" = the kind takes no variant). Keep in sync with the addon.
MATERIAL_KINDS: dict[str, tuple[str, ...]] = {
    "wood": ("oak", "walnut", "mahogany", "cherry", "pine", "dark"),
    "brass": ("",), "gold": ("",), "chrome": ("",), "plaster": ("",),
    "marble": ("carrara", "calacatta", "verde"),
    "leather": ("",), "velvet": ("",), "ceramic": ("",),
    "fabric": ("weave", "linen", "denim", "canvas", "velvet", "leather"),
    "glass": ("clear", "frosted", "tinted"),
    "metal": ("steel", "copper", "chrome", "rust", "brushed"),
    "color": ("",),
}

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

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
    "tv": "color:#111111", "monitor": "color:#111111", "laptop": "metal:brushed", "printer": "color:#d8d8d8",
    "filing_cabinet": "metal:steel", "pallet_rack": "metal:steel", "conveyor": "metal:steel",
    "forklift": "color:#e0a000", "car": "color:#7a1f1f", "safety_barrier": "color:#e0a000",
    "trash_can": "metal:steel", "street_lamp": "metal:steel", "fence_section": "wood:pine",
    "floor_lamp": "brass", "table_lamp": "brass", "pendant_lamp": "brass", "ceiling_light": "color:#f2f2f2",
    "wall_sconce": "brass", "fireplace": "marble:carrara", "column": "marble:carrara",
    "potted_plant": "ceramic", "vase": "ceramic", "tree": "color:#3d6b2f", "bush": "color:#4a7a35",
    "rock": "color:#7d7a72", "barrel": "wood:dark", "book_stack": "color:#7a3b2e",
    "picture_frame": "wood:dark", "wall_clock": "wood:dark", "whiteboard": "color:#f4f4f4",
    "wall_partition": "plaster", "niche": "plaster", "fountain": "color:#b7b1a6",
    # outdoor / industrial
    "playground_slide": "color:#c8372d", "swing_set": "metal:steel", "gazebo": "wood:pine",
    "planter": "color:#9a9890", "hedge": "color:#3f6b33", "signpost": "color:#1f5a3a", "bollard": "metal:steel",
    "parking_bay": "color:#3a3a3c", "loading_dock": "color:#9a9890", "shipping_container": "metal:rust",
    "traffic_cone": "color:#ff6a13", "lamp_post_double": "metal:steel",
}
DEFAULT_MATERIAL = "plaster"
FLOOR_MATERIALS = {"indoor": "wood:oak", "outdoor": "color:#6f7d5a", "tabletop": "wood:walnut"}
WALL_MATERIAL = "plaster"


def material_vocabulary() -> list[str]:
    """Every allowed ``kind[:variant]`` (tint suffix ``:#rrggbb`` allowed on any)."""
    return [k if v == "" else f"{k}:{v}" for k, vs in MATERIAL_KINDS.items() for v in vs]


def material_error(value: object) -> str | None:
    """Why ``value`` is not a valid material string, or None."""
    if not isinstance(value, str) or not value:
        return f"expected a material string like 'wood:oak', got {value!r}"
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


def material_for(category: str, chosen: str | None = None) -> str:
    """The addon material string for an object: the spec's choice, else the category default.
    Box projection: placeholders and simple meshes have no UV map."""
    base = chosen or CATEGORY_MATERIALS.get(category, DEFAULT_MATERIAL)
    return base if base.startswith("color") else f"{base}:proj=box"
