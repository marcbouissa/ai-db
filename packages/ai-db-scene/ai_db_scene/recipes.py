"""Part recipes for categories without a furniture generator: real silhouettes instead of boxes.

Each recipe is a ``build_asset`` spec (the Blender addon's declarative parts format): parts are
laid out in relative units and ``bbox_m`` scales the whole asset to the placement's exact
[w, d, h], so a recipe never changes the solved layout. Front is -Y, base at z=0, XY centred.
The judges' fourth most common complaint about ai-db scenes was "boxes instead of objects".
"""

from __future__ import annotations

from typing import Any

_DARK = "color:#141414"
_STONE = "color:#b7b1a6"
_GREEN = "color:#3f6b33"
_GREEN2 = "color:#4f7d3c"
_BARK = "color:#5a4030"
_YELLOW = "color:#e0a000"
_WHITE = "color:#f2f2f2"


def _box(pid: str, size: list[float], at: list[float] | str, material: str, **kw: Any) -> dict[str, Any]:
    return {"id": pid, "shape": "box", "size": size, "at": at, "material": material, **kw}


def _cyl(pid: str, size: list[float], at: list[float] | str, material: str, **kw: Any) -> dict[str, Any]:
    return {"id": pid, "shape": "cylinder", "size": size, "at": at, "material": material, **kw}


def _sphere(pid: str, size: list[float], at: list[float] | str, material: str, **kw: Any) -> dict[str, Any]:
    return {"id": pid, "shape": "sphere", "size": size, "at": at, "material": material, **kw}


def _screen() -> list[dict[str, Any]]:
    return [_box("foot", [0.35, 0.8, 0.03], [0, 0, 0], "metal:chrome"),
            _box("neck", [0.06, 0.1, 0.13], [0, 0.05, 0.03], "metal:chrome"),
            _box("panel", [1.0, 0.12, 0.84], [0, 0, 0.16], _DARK)]


_RECIPES: dict[str, Any] = {
    "tv": _screen,
    "monitor": _screen,
    "floor_lamp": lambda: [
        _cyl("base", [0.6, 0.6, 0.02], [0, 0, 0], "metal:steel"),
        _cyl("pole", [0.05, 0.05, 0.76], [0, 0, 0.02], "metal:chrome"),
        _cyl("shade", [1.0, 1.0, 0.22], [0, 0, 0.78], "fabric:linen")],
    "table_lamp": lambda: [
        _cyl("base", [0.5, 0.5, 0.08], [0, 0, 0], "metal:gold"),
        _cyl("stem", [0.12, 0.12, 0.45], [0, 0, 0.08], "metal:gold"),
        _cyl("shade", [1.0, 1.0, 0.47], [0, 0, 0.53], "fabric:linen")],
    "pendant_lamp": lambda: [
        _cyl("shade", [1.0, 1.0, 0.3], [0, 0, 0], "metal:copper"),
        _cyl("cord", [0.02, 0.02, 0.7], [0, 0, 0.3], _DARK)],
    "ceiling_light": lambda: [_cyl("disc", [1.0, 1.0, 1.0], [0, 0, 0], _WHITE)],
    "wall_sconce": lambda: [
        _box("plate", [0.4, 0.12, 0.6], [0, 0.44, 0.2], "metal:gold"),
        _box("arm", [0.08, 0.5, 0.06], [0, 0.17, 0.42], "metal:gold"),  # plate -> shade
        _cyl("shade", [0.9, 0.9, 0.5], [0, -0.05, 0.25], "fabric:linen")],
    "street_lamp": lambda: [
        _cyl("foot", [0.5, 0.5, 0.05], [0, 0, 0], _DARK),
        _cyl("pole", [0.2, 0.2, 0.88], [0, 0, 0.05], _DARK),
        _box("head", [1.0, 0.6, 0.07], [0, -0.2, 0.93], _DARK)],
    "potted_plant": lambda: [
        _cyl("pot", [0.55, 0.55, 0.3], [0, 0, 0], "color:#a0522d"),
        _sphere("leaves", [1.0, 1.0, 0.55], [0, 0, 0.3], _GREEN),
        _sphere("leaves_top", [0.6, 0.6, 0.3], [0.05, -0.05, 0.7], _GREEN2)],
    "tree": lambda: [
        _cyl("trunk", [0.12, 0.12, 0.45], [0, 0, 0], _BARK),
        _sphere("crown", [1.0, 1.0, 0.5], [0, 0, 0.4], _GREEN),
        _sphere("crown_top", [0.65, 0.65, 0.3], [0.05, 0, 0.7], _GREEN2)],
    "bush": lambda: [
        _sphere("body", [1.0, 0.9, 0.8], [0, 0, 0], _GREEN),
        _sphere("lobe", [0.6, 0.6, 0.6], [0.25, -0.1, 0.35], _GREEN2)],
    "whiteboard": lambda: [
        _box("frame", [1.0, 1.0, 1.0], [0, 0, 0], "metal:steel"),
        _box("board", [0.96, 0.2, 0.92], [0, -0.45, 0.04], _WHITE)],
    "kitchen_counter": lambda: [
        _box("cabinet", [1.0, 0.95, 0.94], [0, 0.025, 0], "color:#e9e5dc"),
        _box("top", [1.0, 1.0, 0.06], [0, 0, 0.94], "color:#d8d6d0"),
        _box("doors_line", [0.004, 0.02, 0.8], [0, -0.48, 0.07], _DARK)],
    "kitchen_island": lambda: [
        _box("cabinet", [0.94, 0.9, 0.94], [0, 0, 0], "color:#e9e5dc"),
        _box("top", [1.0, 1.0, 0.06], [0, 0, 0.94], "color:#d8d6d0")],
    "refrigerator": lambda: [
        _box("body", [1.0, 1.0, 1.0], [0, 0, 0], "metal:steel"),
        _box("split", [1.0, 0.02, 0.01], [0, -0.505, 0.62], _DARK),
        _box("handle", [0.03, 0.04, 0.3], [0.38, -0.52, 0.65], "metal:chrome"),
        _box("handle_low", [0.03, 0.04, 0.2], [0.38, -0.52, 0.35], "metal:chrome")],
    "pallet_rack": lambda: [
        *[_box(f"post_{i}", [0.04, 0.06, 1.0], [sx * 0.48, sy * 0.47, 0], "color:#1f4fa0")
          for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))],
        *[_box(f"beam_{k}", [1.0, 1.0, 0.03], [0, 0, z], "color:#e06a00") for k, z in enumerate((0.05, 0.36, 0.67, 0.97))]],
    "bench": lambda: [
        # two side frames carry the seat; posts rise from them and carry the backrest
        _box("frame_l", [0.05, 0.55, 0.42], [-0.47, 0.02, 0], _DARK),
        _box("frame_r", [0.05, 0.55, 0.42], [0.47, 0.02, 0], _DARK),
        _box("seat", [1.0, 0.5, 0.05], [0, 0.0, 0.42], "wood:pine"),
        _box("post_l", [0.05, 0.05, 0.45], [-0.47, 0.265, 0.42], _DARK),
        _box("post_r", [0.05, 0.05, 0.45], [0.47, 0.265, 0.42], _DARK),
        _box("backrest", [1.0, 0.04, 0.24], [0, 0.265, 0.63], "wood:pine")],
    "trash_can": lambda: [
        _cyl("bin", [1.0, 1.0, 0.92], [0, 0, 0], "metal:steel"),
        _cyl("lid", [1.04, 1.04, 0.08], [0, 0, 0.92], _DARK)],
    "safety_barrier": lambda: [
        _box("post_l", [0.06, 1.0, 1.0], [-0.47, 0, 0], _YELLOW),
        _box("post_r", [0.06, 1.0, 1.0], [0.47, 0, 0], _YELLOW),
        _box("rail_top", [1.0, 0.8, 0.08], [0, 0, 0.9], _YELLOW),
        _box("rail_mid", [1.0, 0.8, 0.08], [0, 0, 0.5], _DARK)],
    "forklift": lambda: [
        _box("body", [1.0, 0.62, 0.45], [0, 0.18, 0.1], _YELLOW),
        _box("counterweight", [0.9, 0.18, 0.35], [0, 0.42, 0.1], _DARK),
        _box("roof", [0.9, 0.45, 0.03], [0, 0.2, 0.95], _DARK),
        _box("pillar_l", [0.05, 0.05, 0.45], [-0.42, 0.0, 0.5], _DARK),
        _box("pillar_r", [0.05, 0.05, 0.45], [0.42, 0.0, 0.5], _DARK),
        _box("mast", [0.7, 0.05, 0.95], [0, -0.17, 0.05], _DARK),
        _box("fork_l", [0.08, 0.32, 0.03], [-0.2, -0.36, 0.05], "metal:steel"),
        _box("fork_r", [0.08, 0.32, 0.03], [0.2, -0.36, 0.05], "metal:steel"),
        *[_cyl(f"wheel_{i}", [0.25, 0.25, 0.12], [x, y, 0.0], _DARK, rotation_deg=[0, 90, 0])
          for i, (x, y) in enumerate(((-0.45, -0.1), (0.45, -0.1), (-0.45, 0.35), (0.45, 0.35)))]],
    "fountain": lambda: [
        _cyl("basin", [1.0, 1.0, 0.25], [0, 0, 0], _STONE),
        _cyl("water", [0.92, 0.92, 0.02], [0, 0, 0.22], "glass:tinted"),
        _cyl("pedestal", [0.14, 0.14, 0.45], [0, 0, 0.25], _STONE),
        _cyl("bowl", [0.45, 0.45, 0.08], [0, 0, 0.7], _STONE),
        _cyl("spout", [0.05, 0.05, 0.22], [0, 0, 0.78], _STONE)],
}


_WOOD = "wood:oak"
_WALNUT = "wood:walnut"
_WHITE_C = "color:#f4f3ef"  # ceramic white
_STEEL = "metal:steel"
_FABRIC = "fabric:linen"


def _cabinet(doors: int, drawers: int = 0, legs: bool = False, wood: str = _WOOD) -> list[dict[str, Any]]:
    """A carcass with door/drawer lines and handles on its front (-Y)."""
    z0 = 0.08 if legs else 0.0
    parts = [_box("body", [1.0, 0.96, 1.0 - z0], [0, 0.02, z0], wood)]
    if legs:
        parts += [_box(f"leg_{i}", [0.05, 0.05, z0], [x, y, 0], _DARK)
                  for i, (x, y) in enumerate(((-0.45, -0.42), (0.45, -0.42), (-0.45, 0.45), (0.45, 0.45)))]
    h = 1.0 - z0
    for k in range(drawers):
        z = z0 + h * (1 - (k + 1) / (drawers + doors / 2 + 0.001))
        parts += [_box(f"drawer_line_{k}", [0.94, 0.02, 0.006], [0, -0.48, z], _DARK),
                  _box(f"drawer_pull_{k}", [0.16, 0.03, 0.02], [0, -0.5, z + 0.04], _STEEL)]
    for k in range(doors):
        x = -0.5 + (k + 0.5) / doors
        parts += [_box(f"door_line_{k}", [0.006, 0.02, h * 0.9], [x + 0.5 / doors - 0.003, -0.48, z0 + h * 0.05], _DARK),
                  _box(f"handle_{k}", [0.02, 0.03, 0.14], [x + 0.4 / doors * (1 if k % 2 else -1), -0.5, z0 + h * 0.45], _STEEL)]
    return parts


_RECIPES.update({
    "bed": lambda: [
        _box("frame", [1.0, 1.0, 0.3], [0, 0, 0.06], _WALNUT),
        *[_box(f"leg_{i}", [0.05, 0.04, 0.06], [x, y, 0], _WALNUT)
          for i, (x, y) in enumerate(((-0.46, -0.47), (0.46, -0.47), (-0.46, 0.47), (0.46, 0.47)))],
        _box("mattress", [0.96, 0.94, 0.2], [0, -0.02, 0.36], _WHITE),
        _box("duvet", [0.97, 0.62, 0.05], [0, -0.17, 0.56], "fabric:weave"),
        _box("pillow_l", [0.4, 0.14, 0.08], [-0.24, 0.33, 0.56], _WHITE),
        _box("pillow_r", [0.4, 0.14, 0.08], [0.24, 0.33, 0.56], _WHITE),
        _box("headboard", [1.0, 0.04, 0.64], [0, 0.48, 0.36], _WALNUT)],
    "wardrobe": lambda: _cabinet(doors=2, drawers=0),
    "dresser": lambda: _cabinet(doors=0, drawers=3, legs=True),
    "nightstand": lambda: _cabinet(doors=0, drawers=2, legs=True, wood=_WALNUT),
    "cabinet": lambda: _cabinet(doors=2),
    "filing_cabinet": lambda: _cabinet(doors=0, drawers=4, wood=_STEEL),
    "tv_stand": lambda: [
        _box("top", [1.0, 1.0, 0.08], [0, 0, 0.92], _WALNUT),
        _box("body", [0.98, 0.96, 0.76], [0, 0.02, 0.16], _WALNUT),
        _box("shelf_gap", [0.5, 0.02, 0.4], [0.2, -0.47, 0.35], _DARK),
        *[_box(f"leg_{i}", [0.04, 0.04, 0.16], [x, y, 0], _DARK)
          for i, (x, y) in enumerate(((-0.46, -0.42), (0.46, -0.42), (-0.46, 0.45), (0.46, 0.45)))]],
    "office_chair": lambda: [
        _cyl("base", [0.9, 0.9, 0.05], [0, 0, 0.04], _DARK),
        *[_sphere(f"caster_{i}", [0.08, 0.08, 0.08], [x, y, 0], _DARK)
          for i, (x, y) in enumerate(((0.4, 0), (-0.4, 0), (0, 0.4), (0, -0.4), (0.28, 0.28)))],
        _cyl("gas_lift", [0.08, 0.08, 0.3], [0, 0, 0.09], "metal:chrome"),
        _box("seat", [0.85, 0.8, 0.09], [0, -0.02, 0.39], "fabric:weave"),
        _box("back_post", [0.08, 0.06, 0.2], [0, 0.4, 0.45], _DARK),
        _box("back", [0.8, 0.08, 0.38], [0, 0.42, 0.62], "fabric:weave")],
    "stool": lambda: [
        _cyl("seat", [1.0, 1.0, 0.07], [0, 0, 0.93], _WOOD),
        *[_box(f"leg_{i}", [0.07, 0.07, 0.93], [x, y, 0], _WOOD)
          for i, (x, y) in enumerate(((-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3)))],
        _box("rung", [0.66, 0.05, 0.04], [0, -0.3, 0.3], _WOOD)],
    "toilet": lambda: [
        _box("tank", [0.9, 0.3, 0.45], [0, 0.35, 0.5], _WHITE_C),
        _cyl("pedestal", [0.5, 0.6, 0.4], [0, 0.05, 0], _WHITE_C),
        _cyl("bowl", [0.8, 0.85, 0.12], [0, -0.05, 0.4], _WHITE_C),
        _cyl("seat", [0.82, 0.87, 0.03], [0, -0.05, 0.52], _WHITE_C)],
    "sink": lambda: [
        _box("cabinet", [0.96, 0.92, 0.8], [0, 0.04, 0], _WALNUT),
        _box("door_line", [0.006, 0.02, 0.7], [0, -0.43, 0.05], _DARK),
        _box("counter", [1.0, 1.0, 0.06], [0, 0, 0.8], _WHITE_C),
        _box("basin", [0.6, 0.55, 0.05], [0, -0.05, 0.84], "color:#c9ccd0"),  # the bowl, seen from above
        _box("faucet_base", [0.06, 0.08, 0.1], [0, 0.32, 0.86], "metal:chrome"),
        _box("spout", [0.04, 0.18, 0.04], [0, 0.2, 0.96], "metal:chrome")],
    "bathtub": lambda: [
        _box("outer", [1.0, 1.0, 1.0], [0, 0, 0], _WHITE_C),
        _box("rim_water", [0.88, 0.8, 0.02], [0, 0, 0.99], "glass:tinted"),
        _cyl("faucet", [0.04, 0.06, 0.15], [0.42, 0.44, 0.85], "metal:chrome")],
    "shower_stall": lambda: [
        _box("tray", [1.0, 1.0, 0.05], [0, 0, 0], _WHITE_C),
        _box("glass_front", [1.0, 0.02, 0.95], [0, -0.49, 0.05], "glass:frosted"),
        _box("glass_side", [0.02, 1.0, 0.95], [-0.49, 0, 0.05], "glass:frosted"),
        _box("head_arm", [0.04, 0.2, 0.03], [0.3, 0.4, 0.9], "metal:chrome"),
        _cyl("head", [0.12, 0.12, 0.03], [0.3, 0.3, 0.88], "metal:chrome")],
    "mirror": lambda: [
        _box("frame", [1.0, 1.0, 1.0], [0, 0, 0], _WALNUT),
        _box("glass", [0.92, 0.2, 0.92], [0, -0.45, 0.04], "metal:chrome")],
    "picture_frame": lambda: [
        _box("frame", [1.0, 1.0, 1.0], [0, 0, 0], _WALNUT),
        _box("canvas", [0.86, 0.2, 0.86], [0, -0.45, 0.07], "color:#8aa7c4")],
    "wall_clock": lambda: [
        _cyl("rim", [1.0, 1.0, 1.0], [0, 0, 0], _DARK, rotation_deg=[90, 0, 0]),
        _cyl("face", [0.9, 0.9, 0.2], [0, -0.45, 0.05], _WHITE, rotation_deg=[90, 0, 0])],
    "stove": lambda: [
        _box("body", [1.0, 1.0, 0.94], [0, 0, 0], _STEEL),
        _box("top", [1.0, 1.0, 0.06], [0, 0, 0.94], _DARK),
        *[_cyl(f"burner_{i}", [0.22, 0.22, 0.01], [x, y, 0.995], "metal:rust")
          for i, (x, y) in enumerate(((-0.22, -0.2), (0.22, -0.2), (-0.22, 0.2), (0.22, 0.2)))],
        _box("oven_door", [0.8, 0.02, 0.5], [0, -0.5, 0.2], _DARK),
        _box("oven_handle", [0.6, 0.04, 0.03], [0, -0.52, 0.74], "metal:chrome")],
    "laptop": lambda: [
        _box("base", [1.0, 1.0, 0.1], [0, 0, 0], _STEEL),
        _box("screen", [1.0, 0.06, 0.9], [0, 0.47, 0.1], _DARK)],
    "printer": lambda: [
        _box("body", [1.0, 1.0, 0.8], [0, 0, 0], "color:#d8d8d8"),
        _box("tray", [0.7, 0.3, 0.05], [0, -0.6, 0.35], "color:#c8c8c8"),
        _box("lid", [1.0, 1.0, 0.2], [0, 0, 0.8], "color:#bdbdbd")],
    "book_stack": lambda: [
        _box(f"book_{i}", [w, d, 0.24], [x, 0, 0.25 * i], c)
        for i, (w, d, x, c) in enumerate(((1.0, 1.0, 0, "color:#7a3b2e"), (0.9, 0.9, 0.04, "color:#2e4f7a"),
                                          (0.95, 0.85, -0.02, "color:#3b6b45"), (0.8, 0.8, 0.05, "color:#b08a3c")))],
    "vase": lambda: [
        _cyl("foot", [0.6, 0.6, 0.1], [0, 0, 0], "color:#3d5a80"),
        _sphere("body", [1.0, 1.0, 0.6], [0, 0, 0.08], "color:#3d5a80"),
        _cyl("neck", [0.4, 0.4, 0.34], [0, 0, 0.66], "color:#3d5a80")],
    "fireplace": lambda: [
        _box("hearth", [1.0, 1.0, 0.08], [0, 0, 0], _STONE),
        _box("surround_l", [0.2, 0.7, 0.75], [-0.4, 0.15, 0.08], _STONE),
        _box("surround_r", [0.2, 0.7, 0.75], [0.4, 0.15, 0.08], _STONE),
        _box("firebox", [0.6, 0.6, 0.6], [0, 0.2, 0.08], _DARK),
        _box("lintel", [1.0, 0.7, 0.17], [0, 0.15, 0.83], _STONE)],
    "column": lambda: [
        _box("base", [1.0, 1.0, 0.08], [0, 0, 0], _STONE),
        _cyl("shaft", [0.8, 0.8, 0.84], [0, 0, 0.08], _STONE),
        _box("capital", [1.0, 1.0, 0.08], [0, 0, 0.92], _STONE)],
    "stairs": lambda: [
        _box(f"step_{i}", [1.0, 1.0 - i / 8, (i + 1) / 8], [0, i / 16, 0], _WOOD) for i in range(8)],
    "workbench": lambda: [
        _box("top", [1.0, 1.0, 0.08], [0, 0, 0.92], "wood:pine"),
        *[_box(f"leg_{i}", [0.06, 0.08, 0.92], [x, y, 0], _STEEL)
          for i, (x, y) in enumerate(((-0.46, -0.42), (0.46, -0.42), (-0.46, 0.42), (0.46, 0.42)))],
        _box("shelf", [0.98, 0.9, 0.04], [0, 0, 0.2], "wood:pine"),
        _box("vice", [0.12, 0.12, 0.1], [0.38, -0.4, 1.0], _DARK)],
    "shelving_unit": lambda: [
        *[_box(f"post_{i}", [0.04, 0.06, 1.0], [sx * 0.48, sy * 0.47, 0], _STEEL)
          for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))],
        *[_box(f"shelf_{k}", [1.0, 1.0, 0.02], [0, 0, z], _STEEL) for k, z in enumerate((0.05, 0.33, 0.62, 0.95))],
        *[_box(f"box_{k}", [0.3, 0.6, 0.15], [x, 0, z + 0.02], "color:#b08a5a")
          for k, (x, z) in enumerate(((-0.25, 0.05), (0.2, 0.33), (-0.1, 0.62)))]],
    "crate": lambda: [
        _box("body", [0.96, 0.96, 0.96], [0, 0, 0.02], "wood:pine"),
        *[_box(f"slat_{k}", [1.0, 1.0, 0.06], [0, 0, z], "wood:dark") for k, z in enumerate((0.0, 0.47, 0.94))]],
    "pallet": lambda: [
        _box("deck", [1.0, 1.0, 0.25], [0, 0, 0.75], "wood:pine"),
        *[_box(f"block_{i}", [0.15, 1.0, 0.5], [x, 0, 0.25], "wood:pine") for i, x in enumerate((-0.42, 0, 0.42))],
        _box("bottom", [1.0, 1.0, 0.25], [0, 0, 0], "wood:pine")],
    "barrel": lambda: [
        _cyl("body", [1.0, 1.0, 1.0], [0, 0, 0], "color:#2f5d8a"),
        *[_cyl(f"rib_{k}", [1.03, 1.03, 0.03], [0, 0, z], _DARK) for k, z in enumerate((0.25, 0.72))]],
    "conveyor": lambda: [
        _box("belt", [1.0, 1.0, 0.1], [0, 0, 0.82], _DARK),
        _box("side_l", [1.0, 0.04, 0.18], [0, -0.48, 0.78], _STEEL),
        _box("side_r", [1.0, 0.04, 0.18], [0, 0.48, 0.78], _STEEL),
        *[_box(f"leg_{i}", [0.04, 0.04, 0.78], [x, y, 0], _STEEL)
          for i, (x, y) in enumerate(((-0.45, -0.45), (0.45, -0.45), (-0.45, 0.45), (0.45, 0.45)))]],
    "fence_section": lambda: [
        _box("post_l", [0.06, 1.0, 1.0], [-0.47, 0, 0], "wood:pine"),
        _box("post_r", [0.06, 1.0, 1.0], [0.47, 0, 0], "wood:pine"),
        *[_box(f"rail_{k}", [1.0, 0.5, 0.08], [0, 0, z], "wood:pine") for k, z in enumerate((0.25, 0.75))],
        *[_box(f"picket_{k}", [0.06, 0.4, 0.9], [-0.4 + k * 0.16, 0, 0.05], "wood:pine") for k in range(6)]],
    "picnic_table": lambda: [
        _box("top", [1.0, 0.45, 0.06], [0, 0, 0.94], "wood:pine"),
        _box("bench_f", [1.0, 0.2, 0.05], [0, -0.38, 0.55], "wood:pine"),
        _box("bench_b", [1.0, 0.2, 0.05], [0, 0.38, 0.55], "wood:pine"),
        _box("frame_l", [0.06, 1.0, 0.55], [-0.4, 0, 0], "wood:pine"),
        _box("frame_r", [0.06, 1.0, 0.55], [0.4, 0, 0], "wood:pine"),
        _box("stem_l", [0.06, 0.12, 0.39], [-0.4, 0, 0.55], "wood:pine"),
        _box("stem_r", [0.06, 0.12, 0.39], [0.4, 0, 0.55], "wood:pine")],
    "car": lambda: [
        # lower body with bonnet and boot, a narrower cabin with dark windows, roof, wheels
        _box("body", [1.0, 1.0, 0.33], [0, 0, 0.17], "color:#7a1f1f"),
        _box("cabin", [0.86, 0.48, 0.3], [0, 0.04, 0.5], "color:#22303c"),
        _box("pillar_f", [0.86, 0.04, 0.3], [0, -0.2, 0.5], "color:#7a1f1f"),
        _box("pillar_b", [0.86, 0.04, 0.3], [0, 0.28, 0.5], "color:#7a1f1f"),
        _box("roof", [0.86, 0.52, 0.04], [0, 0.04, 0.8], "color:#7a1f1f"),
        _box("lights_f", [0.8, 0.02, 0.05], [0, -0.5, 0.38], "color:#f2f2e0"),
        # size is taken before rotation_deg: [height diameter, length diameter, tyre width],
        # then [0, 90, 0] turns the axle across the car (relative units: the car is long in y)
        *[_cyl(f"wheel_{i}", [0.43, 0.15, 0.14], [x, y, 0], _DARK, rotation_deg=[0, 90, 0])
          for i, (x, y) in enumerate(((-0.44, -0.3), (0.44, -0.3), (-0.44, 0.32), (0.44, 0.32)))]],
    "rock": lambda: [
        _sphere("mass", [1.0, 0.85, 0.75], [0, 0, 0], "color:#7d7a72"),
        _sphere("lump", [0.6, 0.55, 0.5], [0.2, 0.1, 0.4], "color:#8a867c")],
})
# kitchen counter: cabinet doors, handles and a top instead of a plain block
_RECIPES["kitchen_counter"] = lambda: [
    _box("cabinet", [1.0, 0.95, 0.94], [0, 0.025, 0], "color:#e9e5dc"),
    *[_box(f"door_line_{k}", [0.006, 0.02, 0.8], [-0.5 + (k + 1) / 3, -0.46, 0.07], _DARK) for k in range(2)],
    *[_box(f"handle_{k}", [0.12, 0.03, 0.02], [-0.5 + (k + 0.5) / 3, -0.48, 0.8], _STEEL) for k in range(3)],
    _box("top", [1.0, 1.0, 0.06], [0, 0, 0.94], "color:#d8d6d0")]


def has_recipe(category: str) -> bool:
    return category in _RECIPES


def recipe_spec(category: str, name: str, dims: tuple[float, float, float] | list[float]) -> dict[str, Any]:
    """The build_asset spec of ``category``, named ``name``, scaled to exactly ``dims``."""
    return {"asset": name, "category": category, "style": "ai-db recipe",
            "bbox_m": [float(dims[0]), float(dims[1]), float(dims[2])], "parts": _RECIPES[category]()}


def recipe_categories() -> list[str]:
    return sorted(_RECIPES)
