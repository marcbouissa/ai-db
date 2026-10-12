"""Part recipes for categories without a furniture generator: real silhouettes instead of boxes.

Each recipe is a ``build_asset`` spec (the Blender addon's declarative parts format): parts are
laid out in relative units and ``bbox_m`` scales the whole asset to the placement's exact
[w, d, h], so a recipe never changes the solved layout. Front is -Y, base at z=0, XY centred.
The judges' fourth most common complaint about ai-db scenes was "boxes instead of objects".
"""

from __future__ import annotations

import math
from typing import Any

from ai_db_scene.constants import (
    RECIPE_EDGE_BEVEL_M,
    RECIPE_EDGE_BEVEL_MAX_SHARE,
    RECIPE_EDGE_BEVEL_SEGMENTS,
    RECIPE_WOOD_GRAIN_SCALE,
)

_DARK = "color:#141414"
_STONE = "color:#b7b1a6"
_GREEN = "foliage:#3f6b33"
_GREEN2 = "foliage:#4f7d3c"
_BARK = "wood:dark"
_YELLOW = "paint:#e0a000"
_BLUE = "paint:#2f5d8a"
_ORANGE = "paint:#e8661a"
_CARDBOARD = "plaster:#a8835a"
_CAST_IRON = "rubber:#1c1c1c"  # matte black
_VENEER = "wood:walnut:scale=2.5"  # fronts: a finer figure; the default reads as zebra bands
_WHITE = "color:#f2f2f2"


def _box(pid: str, size: list[float], at: list[float] | str, material: str, **kw: Any) -> dict[str, Any]:
    return {"id": pid, "shape": "box", "size": size, "at": at, "material": material, **kw}


def _cyl(pid: str, size: list[float], at: list[float] | str, material: str, **kw: Any) -> dict[str, Any]:
    return {"id": pid, "shape": "cylinder", "size": size, "at": at, "material": material, **kw}


def _sphere(pid: str, size: list[float], at: list[float] | str, material: str, **kw: Any) -> dict[str, Any]:
    return {"id": pid, "shape": "sphere", "size": size, "at": at, "material": material, **kw}


def _lobes(pid: str, z0: float, height: float, spread: float, n: int, mat: str, mat2: str) -> list[dict[str, Any]]:
    """A foliage mass of ``n`` overlapping spheres between z0 and z0 + height: a large core and
    smaller lobes around it at fixed, uneven offsets (deterministic), all touching the core."""
    core = 0.62 * height
    parts = [_sphere(f"{pid}_core", [2 * spread * 0.8, 2 * spread * 0.8, core], [0, 0, z0], mat)]  # rests on z0
    ring = [(0.0, 1.0, 0.55), (1.3, 0.8, 0.62), (2.4, 0.95, 0.5), (3.5, 0.75, 0.66), (4.4, 0.9, 0.58),
            (5.4, 0.7, 0.6), (0.7, 0.6, 0.78), (3.0, 0.55, 0.8)]
    import math

    for k, (ang, r, zf) in enumerate(ring[:max(0, n - 1)]):
        size = round(spread * (0.75 if k % 2 else 0.9), 4)
        x, y = round(math.cos(ang) * spread * 0.45 * r, 4), round(math.sin(ang) * spread * 0.45 * r, 4)
        top = z0 + height - size * (0.9 if k >= 6 else 1.0)
        z = round(min(z0 + zf * height - size / 2, top), 4)
        parts.append(_sphere(f"{pid}_{k}", [size, size, round(size * 0.85, 4)], [x, y, max(z0, z)],
                             mat2 if k % 3 == 1 else mat))
    return parts


def _brace(pid: str, x: float, y0: float, z0: float, y1: float, z1: float, tx: float, t: float,
           mat: str) -> dict[str, Any]:
    """A straight strut from (y0, z0) to (y1, z1) in the plane x (unit space), e.g. the zig-zag
    bracing of a rack frame: a box of length |d| turned about X; ``at`` is its rotated bottom centre."""
    import math

    dy, dz = y1 - y0, z1 - z0
    length = math.hypot(dy, dz)
    ang = math.degrees(math.atan2(dz, dy))
    ext_z = abs(length * math.sin(math.radians(ang))) + abs(t * math.cos(math.radians(ang)))
    return _box(pid, [tx, round(length, 4), t], [x, round((y0 + y1) / 2, 4), round((z0 + z1) / 2 - ext_z / 2, 4)],
                mat, rotation_deg=[round(ang, 2), 0, 0])


def _rack() -> list[dict[str, Any]]:
    """Selective pallet rack (2.7 x 1.1 x 4.5 m): two upright frames (front and back posts with
    zig-zag bracing), orange step beams at three levels, wire decks, a loaded pallet per bay level."""
    parts: list[dict[str, Any]] = []
    levels = (0.02, 0.35, 0.68)
    for s, x in (("l", -0.485), ("r", 0.485)):
        for f, y in (("f", -0.464), ("b", 0.464)):
            parts.append(_box(f"post_{s}{f}", [0.03, 0.072, 1.0], [x, y, 0], _BLUE))
            parts.append(_box(f"foot_{s}{f}", [0.05, 0.1, 0.006], [x, y, 0], _STEEL))
        zs = [0.05 + k * 0.1125 for k in range(9)]
        for k in range(8):
            ya, yb = (-0.43, 0.43) if k % 2 == 0 else (0.43, -0.43)
            parts.append(_brace(f"brace_{s}_{k}", x, ya, zs[k], yb, zs[k + 1], 0.012, 0.01, _BLUE))
        parts.append(_box(f"tie_{s}", [0.012, 0.86, 0.01], [x, 0, 0.05], _BLUE))
    for k, z in enumerate(levels[1:] + (0.97,), 1):
        for f, y in (("f", -0.464), ("b", 0.464)):
            parts.append(_box(f"beam_{k}{f}", [1.0, 0.045, 0.027], [0, y, z], _ORANGE))
        if z < 0.9:
            parts.append(_box(f"deck_{k}", [0.94, 0.89, 0.004], [0, 0, z + 0.027], "metal:brushed"))
    for k, z in enumerate(levels):
        base = 0.0 if k == 0 else z + 0.031  # ground level: pallets on the floor
        for i, x in enumerate((-0.24, 0.24)):
            parts.append(_box(f"load_pallet_{k}_{i}", [0.36, 0.85, 0.033], [x, 0, base], "wood:pine"))
            parts.append(_box(f"load_{k}_{i}", [0.34, 0.8, 0.2], [x, 0, base + 0.033], _CARDBOARD))
    return parts


def _panel(pid: str, w: float, h: float, x: float, z: float, mat: str) -> dict[str, Any]:
    """A shaker cabinet front (recessed field) on a carcass whose face is at y = -0.45."""
    return {"id": pid, "shape": "raised_panel", "size": [round(w, 4), 0.03, round(h, 4)], "at": [round(x, 4), -0.465, round(z, 4)],
            "mode": "recessed", "border": round(0.16 * min(w, h), 4), "moulding": round(0.03 * min(w, h), 4),
            "field_depth": 0.008, "material": mat}


def _kitchen_run() -> list[dict[str, Any]]:
    """Base cabinet run (1.8 x 0.6 x 0.9 m): recessed plinth, carcass, a row of drawers over doors
    with shaker fronts and bar pulls, an overhanging worktop."""
    front, n = "plastic:#ece8df", 4
    parts = [_box("plinth", [0.98, 0.85, 0.11], [0, 0.07, 0], _DARK),
             _box("carcass", [1.0, 0.95, 0.845], [0, 0.025, 0.11], front),
             _box("worktop", [1.0, 1.0, 0.045], [0, 0, 0.955], "marble:carrara")]
    w = 1.0 / n
    for k in range(n):
        x = -0.5 + w * (k + 0.5)
        parts += [_panel(f"drawer_{k}", w - 0.008, 0.17, x, 0.775, front),
                  _box(f"drawer_pull_{k}", [w * 0.45, 0.02, 0.012], [x, -0.49, 0.855], _STEEL),
                  _panel(f"door_{k}", w - 0.008, 0.645, x, 0.12, front),
                  _box(f"door_pull_{k}", [0.012, 0.02, 0.12], [x + (w / 2 - 0.035) * (1 if k % 2 else -1), -0.49, 0.6], _STEEL)]
    return parts


def _frame(pid: str, w: float, mat: str) -> list[dict[str, Any]]:
    """Four moulding bars round the edge of a wall piece (unit space, face toward -Y)."""
    return [_box(f"{pid}_top", [1.0, 1.0, w], [0, 0, 1 - w], mat), _box(f"{pid}_bottom", [1.0, 1.0, w], [0, 0, 0], mat),
            _box(f"{pid}_l", [w * 0.7, 1.0, 1 - 2 * w], [-0.5 + w * 0.35, 0, w], mat),
            _box(f"{pid}_r", [w * 0.7, 1.0, 1 - 2 * w], [0.5 - w * 0.35, 0, w], mat)]


def _screen() -> list[dict[str, Any]]:
    """Monitor / TV: an oval foot and neck, a thin black body, and the glass screen inset in its
    bezel (a reflective dark face reads as a screen, a matte black box does not)."""
    return [_cyl("foot", [0.35, 0.8, 0.025], [0, 0.05, 0], "metal:chrome"),
            _box("neck", [0.06, 0.1, 0.15], [0, 0.1, 0.025], "metal:chrome"),
            _box("body", [1.0, 0.1, 0.84], [0, 0.0, 0.16], "plastic:#0b0b0b"),
            _box("screen", [0.96, 0.02, 0.78], [0, -0.06, 0.19], "glass:tinted")]


_RECIPES: dict[str, Any] = {
    "tv": _screen,
    "monitor": _screen,
    # floor lamp: weighted disc base, slim pole, a hollow tapered linen shade (open below, a bulb
    # inside) - a solid cylinder shade is the clearest placeholder tell
    "floor_lamp": lambda: [
        _cyl("base", [0.55, 0.55, 0.025], [0, 0, 0], "metal:steel"),
        _cyl("pole", [0.045, 0.045, 0.76], [0, 0, 0.025], "metal:brushed"),
        _sphere("bulb", [0.2, 0.2, 0.06], [0, 0, 0.8], "plastic:#fff4dc"),
        {"id": "shade", "shape": "tapered_cylinder", "size": [1.0, 1.0, 0.22], "at": [0, 0, 0.78],
         "radius": 1.0, "radius_top": 0.78, "material": "fabric:linen:#e9e1cf",
         "ops": [{"op": "shell", "thickness": 0.006, "open_faces": "bottom"}]}],
    # table lamp: a ceramic body on a small foot, brass stem, hollow tapered shade with a bulb
    "table_lamp": lambda: [
        _cyl("foot", [0.5, 0.5, 0.04], [0, 0, 0], "metal:gold"),
        _sphere("body", [0.55, 0.55, 0.36], [0, 0, 0.04], "ceramic"),
        _cyl("stem", [0.07, 0.07, 0.16], [0, 0, 0.4], "metal:gold"),
        _sphere("bulb", [0.22, 0.22, 0.1], [0, 0, 0.55], "plastic:#fff4dc"),
        {"id": "shade", "shape": "tapered_cylinder", "size": [1.0, 1.0, 0.44], "at": [0, 0, 0.56],
         "radius": 1.0, "radius_top": 0.72, "material": "fabric:linen:#efe8d8",
         "ops": [{"op": "shell", "thickness": 0.008, "open_faces": "bottom"}]}],
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
    # Foliage as clusters of overlapping lobes: one ball per crown is the clearest "placeholder"
    # look; uneven lobes give a broken silhouette that reads as leaves at a distance.
    "potted_plant": lambda: [
        _cyl("pot", [0.55, 0.55, 0.3], [0, 0, 0], "concrete:#a0522d"),
        *_lobes("leaves", 0.3, 0.7, 0.42, 7, _GREEN, _GREEN2)],
    "tree": lambda: [
        _cyl("trunk", [0.12, 0.12, 0.5], [0, 0, 0], _BARK),
        *_lobes("crown", 0.42, 0.58, 0.5, 9, _GREEN, _GREEN2)],
    "bush": lambda: _lobes("body", 0.0, 1.0, 0.5, 7, _GREEN, _GREEN2),
    # wall whiteboard: aluminium frame bars round a glossy white board, marker tray, markers
    "whiteboard": lambda: [
        _box("board", [0.98, 0.7, 0.92], [0, 0.15, 0.06], "plastic:#f7f7f5"),
        _box("frame_top", [1.0, 1.0, 0.02], [0, 0, 0.98], "metal:brushed"),
        _box("frame_bottom", [1.0, 1.0, 0.02], [0, 0, 0.04], "metal:brushed"),
        *[_box(f"frame_{s}", [0.012, 1.0, 0.96], [x, 0, 0.04], "metal:brushed") for s, x in (("l", -0.494), ("r", 0.494))],
        _box("tray", [0.6, 1.0, 0.04], [0, 0, 0.0], "metal:brushed"),
        *[_box(f"marker_{k}", [0.08, 0.8, 0.012], [x, 0, 0.04], c)
          for k, (x, c) in enumerate(((-0.2, "plastic:#1d4fbf"), (-0.08, "plastic:#c62828"), (0.04, "plastic:#111111")))]],
    "kitchen_counter": _kitchen_run,
    "kitchen_island": lambda: [
        _box("cabinet", [0.94, 0.9, 0.94], [0, 0, 0], "color:#e9e5dc"),
        _box("top", [1.0, 1.0, 0.06], [0, 0, 0.94], "color:#d8d6d0")],
    # fridge-freezer (0.7 x 0.7 x 1.8 m): cabinet, two separate doors with a gap (fridge over
    # freezer), bar handles, dark kick plate; clean brushed steel, not a raw sheet
    "refrigerator": lambda: [
        _box("cabinet", [1.0, 0.93, 1.0], [0, 0.035, 0], "metal:brushed"),
        _box("kick", [0.96, 0.03, 0.04], [0, -0.45, 0], _DARK),
        _box("door_freezer", [0.99, 0.04, 0.33], [0, -0.45, 0.045], "metal:brushed"),
        _box("door_fridge", [0.99, 0.04, 0.615], [0, -0.45, 0.383], "metal:brushed"),
        *[p for k, (z, hh) in enumerate(((0.12, 0.2), (0.45, 0.32))) for p in (
            _box(f"handle_{k}", [0.03, 0.02, hh], [0.4, -0.48, z], "metal:chrome"),)]],
    "pallet_rack": _rack,
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
    # steel safety barrier: two round posts on bolted base plates, a top rail and a lower
    # kick rail in safety yellow, black end caps
    "safety_barrier": lambda: [
        *[p for s, x in (("l", -0.45), ("r", 0.45)) for p in (
            _box(f"plate_{s}", [0.12, 1.0, 0.02], [x, 0, 0], _STEEL),
            _cyl(f"post_{s}", [0.067, 0.5, 0.96], [x, 0, 0.02], _YELLOW),
            _cyl(f"cap_{s}", [0.07, 0.52, 0.02], [x, 0, 0.98], _DARK))],
        _box("rail_top", [0.9, 0.4, 0.09], [0, 0, 0.82], _YELLOW),
        _box("rail_kick", [0.9, 0.4, 0.09], [0, 0, 0.25], _YELLOW)],
    # counterbalance forklift (1.2 x 2.8 x 2.2 m), front (forks) at -Y: chassis and rear
    # counterweight, seat and steering column, overhead guard on four pillars, a two-channel mast
    # with carriage and L-forks, big drive wheels at the front, small steer wheels at the back
    "forklift": lambda: [
        _box("chassis", [0.92, 0.56, 0.2], [0, 0.12, 0.09], _YELLOW),
        _box("counterweight", [0.96, 0.17, 0.28], [0, 0.41, 0.07], _DARK),
        _box("hood", [0.8, 0.26, 0.12], [0, 0.24, 0.29], _YELLOW),
        _box("seat", [0.36, 0.11, 0.05], [0, 0.2, 0.41], "rubber"),
        _box("seat_back", [0.36, 0.03, 0.13], [0, 0.265, 0.41], "rubber"),
        _box("column", [0.05, 0.03, 0.2], [0, -0.06, 0.29], _DARK),
        _cyl("steering", [0.22, 0.09, 0.012], [0, -0.06, 0.49], _DARK),
        *[_box(f"pillar_{i}", [0.04, 0.018, 0.6], [x, y, 0.29], _DARK)
          for i, (x, y) in enumerate(((-0.42, -0.1), (0.42, -0.1), (-0.42, 0.33), (0.42, 0.33)))],
        *[_box(f"guard_{k}", [0.88, 0.02, 0.014], [0, round(-0.1 + k * 0.0717, 4), 0.89], _DARK) for k in range(7)],
        *[_box(f"guard_side_{s}", [0.04, 0.45, 0.02], [x, 0.115, 0.88], _DARK) for s, x in (("l", -0.42), ("r", 0.42))],
        _cyl("beacon", [0.06, 0.05, 0.03], [0.3, 0.3, 0.9], "plastic:#ff8c1a"),
        *[_box(f"mast_{s}", [0.06, 0.03, 0.92], [x, -0.17, 0.04], _DARK) for s, x in (("l", -0.27), ("r", 0.27))],
        *[_box(f"mast_tie_{k}", [0.6, 0.02, 0.03], [0, -0.17, z], _DARK) for k, z in enumerate((0.3, 0.93))],
        _box("carriage", [0.62, 0.02, 0.14], [0, -0.195, 0.05], _DARK),
        *[p for s, x in (("l", -0.17), ("r", 0.17)) for p in (
            _box(f"fork_{s}_shank", [0.07, 0.016, 0.2], [x, -0.213, 0.03], _STEEL),
            _box(f"fork_{s}", [0.07, 0.29, 0.016], [x, -0.355, 0.03], _STEEL))],
        *[_cyl(f"wheel_{s}{i}", [w, w * 2.2 / 2.8, t], [x, y, 0.0], "rubber", rotation_deg=[0, 90, 0])
          for i, (y, w, t) in enumerate(((-0.06, 0.29, 0.2), (0.33, 0.21, 0.15)))
          for s, x in (("l", -0.4), ("r", 0.4))]],
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
    """A carcass on optional legs with shaker (recessed-field) drawer fronts over door fronts and
    pulls, all on its face (y = -0.45): real fronts with gaps read as furniture, drawn lines do not."""
    z0 = 0.08 if legs else 0.0
    parts = [_box("body", [1.0, 0.95, 1.0 - z0], [0, 0.025, z0], wood)]
    if legs:
        parts += [_box(f"leg_{i}", [0.05, 0.05, z0], [x, y, 0], _DARK)
                  for i, (x, y) in enumerate(((-0.45, -0.42), (0.45, -0.42), (-0.45, 0.45), (0.45, 0.45)))]
    h = 1.0 - z0 - 0.02
    dh = h * (0.22 if doors else 1.0 / max(drawers, 1))  # drawer height
    top = 1.0 - 0.01
    for k in range(drawers):
        z = top - (k + 1) * dh
        parts += [_panel(f"drawer_{k}", 0.97, dh - 0.012, 0, z + 0.006, wood),
                  _box(f"drawer_pull_{k}", [0.16, 0.02, 0.02], [0, -0.49, z + dh / 2 - 0.01], _STEEL)]
    if doors:
        dz0, dz1 = z0 + 0.01, top - drawers * dh
        w = 0.97 / doors
        for k in range(doors):
            x = -0.485 + w * (k + 0.5)
            parts += [_panel(f"door_{k}", w - 0.01, dz1 - dz0 - 0.008, x, dz0, wood),
                      _box(f"handle_{k}", [0.02, 0.02, min(0.14, (dz1 - dz0) * 0.3)],
                           [x + (w / 2 - 0.05) * (1 if k % 2 == 0 else -1) * -1, -0.49, dz0 + (dz1 - dz0) * 0.45], _STEEL)]
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
    # task chair: five-star base on casters, gas lift, upholstered seat and back cushions on a
    # back support bar, armrests
    "office_chair": lambda: [
        _cyl("hub", [0.12, 0.12, 0.06], [0, 0, 0.05], _DARK),
        *[_box(f"leg_{k}", [0.42, 0.06, 0.04], [round(0.19 * math.cos(math.radians(90 + 72 * k)), 4),
                                                 round(0.19 * math.sin(math.radians(90 + 72 * k)), 4), 0.06],
               _DARK, rotation_deg=[0, 0, 90 + 72 * k]) for k in range(5)],
        *[_sphere(f"caster_{k}", [0.08, 0.08, 0.08], [round(0.38 * math.cos(math.radians(90 + 72 * k)), 4),
                                                        round(0.38 * math.sin(math.radians(90 + 72 * k)), 4), 0], "rubber")
          for k in range(5)],
        _cyl("gas_lift", [0.06, 0.06, 0.32], [0, 0, 0.1], "metal:chrome"),
        _box("seat_pan", [0.7, 0.66, 0.03], [0, 0, 0.41], _DARK),
        {"id": "seat", "shape": "cushion", "size": [0.86, 0.8, 0.09], "at": [0, -0.02, 0.44], "crown": 0.012,
         "material": "fabric:weave:#33373d"},
        _box("back_bar", [0.08, 0.06, 0.3], [0, 0.41, 0.41], _DARK),
        {"id": "back", "shape": "cushion", "size": [0.78, 0.08, 0.38], "at": [0, 0.44, 0.6], "crown": 0.01,
         "material": "fabric:weave:#33373d"},
        *[p for s, x in (("l", -0.44), ("r", 0.44)) for p in (
            _box(f"arm_post_{s}", [0.05, 0.05, 0.2], [x, 0.05, 0.44], _DARK),
            _box(f"arm_pad_{s}", [0.08, 0.36, 0.035], [x, 0.03, 0.64], "rubber"))]],
    "stool": lambda: [
        _cyl("seat", [1.0, 1.0, 0.07], [0, 0, 0.93], _WOOD),
        *[_box(f"leg_{i}", [0.07, 0.07, 0.93], [x, y, 0], _WOOD)
          for i, (x, y) in enumerate(((-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3)))],
        _box("rung", [0.66, 0.05, 0.04], [0, -0.3, 0.3], _WOOD)],
    # close-coupled toilet (front -Y): pedestal, a bowl widening to the rim, seat and lid,
    # cistern with its lid and a chrome flush button
    "toilet": lambda: [
        {"id": "pedestal", "shape": "tapered_cylinder", "size": [0.5, 0.55, 0.4], "at": [0, 0.02, 0],
         "radius": 0.5, "radius_top": 0.62, "material": _WHITE_C},
        {"id": "bowl", "shape": "tapered_cylinder", "size": [0.92, 0.76, 0.14], "at": [0, -0.06, 0.39],
         "radius": 0.8, "radius_top": 1.0, "material": _WHITE_C},
        _cyl("seat", [0.94, 0.78, 0.035], [0, -0.06, 0.53], "plastic:#f4f4f2"),
        _cyl("lid", [0.9, 0.74, 0.025], [0, -0.07, 0.565], "plastic:#f4f4f2"),
        {"id": "cistern", "shape": "rounded_slab", "size": [0.9, 0.24, 0.42], "at": [0, 0.38, 0.53],
         "corner_radius": 0.03, "material": _WHITE_C},
        {"id": "cistern_lid", "shape": "rounded_slab", "size": [0.94, 0.26, 0.04], "at": [0, 0.38, 0.95],
         "corner_radius": 0.02, "material": _WHITE_C},
        _cyl("flush", [0.12, 0.08, 0.012], [0, 0.38, 0.99], "metal:chrome")],
    # bathroom vanity (0.8 x 0.5 x 0.85 m): two shaker doors, ceramic top with a hollow basin
    # set into it, a mixer tap with spout and lever
    "sink": lambda: [
        _box("carcass", [0.98, 0.95, 0.8], [0, 0.025, 0], _VENEER),
        _panel("door_l", 0.48, 0.74, -0.245, 0.03, _VENEER),
        _panel("door_r", 0.48, 0.74, 0.245, 0.03, _VENEER),
        _box("pull_l", [0.012, 0.02, 0.12], [-0.03, -0.49, 0.55], _STEEL),
        _box("pull_r", [0.012, 0.02, 0.12], [0.03, -0.49, 0.55], _STEEL),
        _box("top", [1.0, 1.0, 0.05], [0, 0, 0.8], _WHITE_C),
        {"id": "basin", "shape": "rounded_slab", "size": [0.62, 0.6, 0.16], "at": [0, -0.06, 0.72], "corner_radius": 0.08,
         "material": _WHITE_C, "ops": [{"op": "shell", "thickness": 0.012, "open_faces": "top"}]},
        _cyl("tap_post", [0.05, 0.08, 0.12], [0, 0.34, 0.85], "metal:chrome"),
        _box("tap_spout", [0.03, 0.2, 0.03], [0, 0.22, 0.94], "metal:chrome"),
        _box("tap_lever", [0.08, 0.02, 0.015], [0.05, 0.34, 0.97], "metal:chrome")],
    "bathtub": lambda: [
        {"id": "body", "shape": "rounded_slab", "size": [1.0, 1.0, 0.9], "at": [0, 0, 0], "corner_radius": 0.06,
         "material": _WHITE_C, "ops": [{"op": "shell", "thickness": 0.04, "open_faces": "top"}]},
        _box("water", [0.9, 0.86, 0.01], [0, 0, 0.42], "glass:tinted"),
        _cyl("mixer_post", [0.025, 0.06, 0.08], [0.44, 0, 0.9], "metal:chrome"),
        _box("spout", [0.06, 0.03, 0.025], [0.415, 0, 0.955], "metal:chrome")],
    "shower_stall": lambda: [
        _box("tray", [1.0, 1.0, 0.05], [0, 0, 0], _WHITE_C),
        _box("glass_front", [1.0, 0.02, 0.95], [0, -0.49, 0.05], "glass:frosted"),
        _box("glass_side", [0.02, 1.0, 0.95], [-0.49, 0, 0.05], "glass:frosted"),
        _box("riser", [0.03, 0.03, 0.88], [0.3, 0.485, 0.05], "metal:chrome"),
        _box("head_arm", [0.04, 0.2, 0.03], [0.3, 0.4, 0.9], "metal:chrome"),
        _cyl("head", [0.12, 0.12, 0.03], [0.3, 0.3, 0.88], "metal:chrome")],
    # wall mirror: moulded frame, mirror glass inset behind it
    "mirror": lambda: [
        *_frame("frame", 0.07, "wood:walnut"),
        _box("back", [0.94, 0.4, 0.88], [0, 0.3, 0.06], "wood:dark"),
        _box("glass", [0.9, 0.2, 0.86], [0, -0.1, 0.07], "metal:chrome")],
    "picture_frame": lambda: [
        *_frame("frame", 0.09, "metal:gold"),
        _box("back", [0.94, 0.4, 0.84], [0, 0.3, 0.08], "wood:dark"),
        _box("mat", [0.88, 0.3, 0.82], [0, -0.05, 0.09], "plaster:#f2efe8"),
        _box("sky", [0.76, 0.2, 0.34], [0, -0.2, 0.56], "fabric:canvas:#9fb8cf"),
        _box("hills", [0.76, 0.2, 0.16], [0, -0.2, 0.4], "fabric:canvas:#5d7a52"),
        _box("field", [0.76, 0.2, 0.25], [0, -0.2, 0.15], "fabric:canvas:#b59a5c")],
    "wall_clock": lambda: [
        _cyl("rim", [1.0, 1.0, 1.0], [0, 0, 0], _DARK, rotation_deg=[90, 0, 0]),
        _cyl("face", [0.9, 0.9, 0.2], [0, -0.45, 0.05], _WHITE, rotation_deg=[90, 0, 0])],
    # freestanding cooker (0.6 x 0.6 x 0.9 m): brushed body, dark glass hob with four burners
    # under cast-iron grates, a control strip with knobs, an oven door with a window and bar handle
    "stove": lambda: [
        _box("body", [1.0, 0.95, 0.9], [0, 0.025, 0], "metal:brushed"),
        _box("hob", [1.0, 1.0, 0.04], [0, 0, 0.9], "plastic:#0e0e0e"),
        *[p for i, (x, y) in enumerate(((-0.23, -0.2), (0.23, -0.2), (-0.23, 0.22), (0.23, 0.22))) for p in (
            _cyl(f"burner_{i}", [0.2, 0.2, 0.02], [x, y, 0.94], "metal:steel"),
            _box(f"grate_{i}_x", [0.36, 0.025, 0.03], [x, y, 0.94], _CAST_IRON),
            _box(f"grate_{i}_y", [0.025, 0.36, 0.03], [x, y, 0.94], _CAST_IRON))],
        _box("controls", [1.0, 0.03, 0.09], [0, -0.465, 0.79], "metal:brushed"),
        *[_cyl(f"knob_{k}", [0.06, 0.05, 0.035], [round(-0.36 + k * 0.18, 4), -0.48, 0.815], _DARK,
               rotation_deg=[90, 0, 0]) for k in range(5)],
        _box("oven_door", [0.92, 0.03, 0.6], [0, -0.465, 0.1], "metal:brushed"),
        _box("oven_window", [0.66, 0.012, 0.3], [0, -0.486, 0.24], "glass:tinted"),
        _box("oven_handle", [0.7, 0.02, 0.025], [0, -0.49, 0.64], "metal:chrome"),
        _box("drawer", [0.92, 0.03, 0.08], [0, -0.465, 0.01], "metal:brushed")],
    "laptop": lambda: [
        _box("base", [1.0, 1.0, 0.1], [0, 0, 0], _STEEL),
        _box("screen", [1.0, 0.06, 0.9], [0, 0.47, 0.1], _DARK)],
    "printer": lambda: [
        _box("body", [1.0, 1.0, 0.8], [0, 0, 0], "color:#d8d8d8"),
        _box("tray", [0.7, 0.3, 0.05], [0, -0.6, 0.35], "color:#c8c8c8"),
        _box("lid", [1.0, 1.0, 0.2], [0, 0, 0.8], "color:#bdbdbd")],
    "book_stack": lambda: [
        _box(f"book_{i}", [w, d, 0.25], [x, 0, 0.25 * i], c)
        for i, (w, d, x, c) in enumerate(((1.0, 1.0, 0, "color:#7a3b2e"), (0.9, 0.9, 0.04, "color:#2e4f7a"),
                                          (0.95, 0.85, -0.02, "color:#3b6b45"), (0.8, 0.8, 0.05, "color:#b08a3c")))],
    # glazed ceramic vase: foot, a full body, shoulder, neck and a slightly flared lip
    "vase": lambda: [
        _cyl("foot", [0.55, 0.55, 0.06], [0, 0, 0], "ceramic:#3d5a80"),
        _sphere("body", [1.0, 1.0, 0.6], [0, 0, 0.04], "ceramic:#3d5a80"),
        _cyl("neck", [0.38, 0.38, 0.3], [0, 0, 0.6], "ceramic:#3d5a80"),
        _cyl("lip", [0.48, 0.48, 0.06], [0, 0, 0.9], "ceramic:#3d5a80")],
    "fireplace": lambda: [
        _box("hearth", [1.0, 1.0, 0.05], [0, 0, 0], "marble:carrara"),
        _box("back", [1.0, 0.45, 0.95], [0, 0.275, 0.05], "marble:carrara"),
        *[p for s, x in (("l", -0.4), ("r", 0.4)) for p in (
            _box(f"plinth_{s}", [0.18, 0.6, 0.08], [x, 0.15, 0.05], "marble:carrara"),
            _box(f"pilaster_{s}", [0.14, 0.55, 0.62], [x, 0.17, 0.13], "marble:carrara"),
            _box(f"capital_{s}", [0.18, 0.6, 0.05], [x, 0.15, 0.75], "marble:carrara"))],
        _box("frieze", [0.98, 0.55, 0.12], [0, 0.17, 0.8], "marble:carrara"),
        _box("mantel", [1.0, 0.95, 0.06], [0, -0.025, 0.92], "marble:carrara"),
        # the opening: a sooty recess (dark back, brick cheeks) set behind the surround's face
        _box("firebox", [0.6, 0.08, 0.6], [0, 0.09, 0.05], "rubber:#0f0c0a"),
        *[_box(f"cheek_{s}", [0.05, 0.3, 0.6], [x, 0.0, 0.05], "stone:#4a2a20") for s, x in (("l", -0.3), ("r", 0.3))],
        # logs lie along x: about Y the part's z (0.22 -> 0.33 m) becomes the length, x (0.073 -> 8 cm) the height
        *[_cyl(f"log_{k}", [0.073, 0.2, 0.22], [0, y, 0.05], "wood:dark", rotation_deg=[0, 90, 0])
          for k, y in enumerate((-0.05, 0.03))]],
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
    # slatted wooden crate: corner posts, three spaced slats per side (the gaps show the dark
    # inside), a closed bottom and lid boards
    "crate": lambda: [
        _box("inside", [0.9, 0.86, 0.94], [0, 0, 0.03], "wood:dark"),
        *[_box(f"post_{i}", [0.07, 0.1, 1.0], [x, y, 0], "wood:pine")
          for i, (x, y) in enumerate(((-0.465, -0.45), (0.465, -0.45), (-0.465, 0.45), (0.465, 0.45)))],
        *[_box(f"slat_{s}_{k}", [0.93, 0.05, 0.22], [0, y, z], "wood:pine")
          for s, y in (("f", -0.475), ("b", 0.475)) for k, z in enumerate((0.04, 0.39, 0.74))],
        *[_box(f"side_{s}_{k}", [0.05, 0.9, 0.22], [x, 0, z], "wood:pine")
          for s, x in (("l", -0.475), ("r", 0.475)) for k, z in enumerate((0.04, 0.39, 0.74))],
        _box("bottom", [1.0, 1.0, 0.03], [0, 0, 0], "wood:pine"),
        *[_box(f"lid_{k}", [1.0, 0.3, 0.04], [0, y, 0.96], "wood:pine") for k, y in enumerate((-0.33, 0.0, 0.33))]],
    "pallet": lambda: [
        *[_box(f"top_{k}", [0.1, 1.0, 0.15], [x, 0, 0.85], "wood:pine") for k, x in enumerate((-0.45, -0.225, 0.0, 0.225, 0.45))],
        *[_box(f"cross_{k}", [1.0, 0.12, 0.15], [0, y, 0.7], "wood:pine") for k, y in enumerate((-0.44, 0.0, 0.44))],
        *[_box(f"block_{i}_{j}", [0.145, 0.12, 0.55], [x, y, 0.15], "wood:pine")
          for i, x in enumerate((-0.4275, 0.0, 0.4275)) for j, y in enumerate((-0.44, 0.0, 0.44))],
        *[_box(f"bottom_{k}", [0.145, 1.0, 0.15], [x, 0, 0], "wood:pine") for k, x in enumerate((-0.4275, 0.0, 0.4275))]],
    "barrel": lambda: [
        _cyl("body", [1.0, 1.0, 1.0], [0, 0, 0], "color:#2f5d8a"),
        *[_cyl(f"rib_{k}", [1.03, 1.03, 0.03], [0, 0, z], _DARK) for k, z in enumerate((0.25, 0.72))]],
    # roller conveyor (3.0 x 0.6 x 0.8 m typical): steel side channels, a bed of rollers between
    # them, leg frames with cross braces and foot plates, a drive unit under the bed
    "conveyor": lambda: [
        *[_box(f"rail_{s}", [1.0, 0.083, 0.15], [0, y, 0.85], _STEEL) for s, y in (("l", -0.4585), ("r", 0.4585))],
        *[_cyl(f"roller_{k}", [0.017, 0.0625, 0.84], [round(-0.48 + k * 0.0331, 4), 0, 0.925], "metal:brushed",
               rotation_deg=[90, 0, 0]) for k in range(30)],
        *[p for i, x in enumerate((-0.46, 0.0, 0.46)) for p in (
            _box(f"leg_{i}_l", [0.017, 0.083, 0.85], [x, -0.4585, 0], _BLUE),
            _box(f"leg_{i}_r", [0.017, 0.083, 0.85], [x, 0.4585, 0], _BLUE),
            _box(f"brace_{i}", [0.017, 0.84, 0.05], [x, 0, 0.3], _BLUE),
            _box(f"foot_{i}_l", [0.04, 0.083, 0.015], [x, -0.4585, 0], _STEEL),
            _box(f"foot_{i}_r", [0.04, 0.083, 0.015], [x, 0.4585, 0], _STEEL))],
        _box("drive", [0.1, 0.84, 0.2], [0.25, 0, 0.65], _BLUE),
        _cyl("motor", [0.06, 0.3, 0.12], [0.25, -0.25, 0.53], "metal:steel", rotation_deg=[90, 0, 0])],
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
        # front at -Y. A rounded painted body over dark sills and bumpers, a glass greenhouse
        # (tinted windows all round, narrower than the body) under a rounded roof, head and tail
        # lights, a grille, door mirrors; rubber tyres with brushed rims standing proud of them.
        {"id": "body", "shape": "rounded_slab", "size": [1.0, 0.98, 0.3], "at": [0, 0, 0.2], "corner_radius": 0.06,
         "material": "paint:#7a1f1f"},
        _box("sill", [0.98, 0.7, 0.06], [0, 0, 0.15], _DARK),
        _box("bumper_f", [0.96, 0.03, 0.1], [0, -0.485, 0.18], "plastic:#1a1a1a"),
        _box("bumper_r", [0.96, 0.03, 0.1], [0, 0.485, 0.18], "plastic:#1a1a1a"),
        {"id": "greenhouse", "shape": "rounded_slab", "size": [0.84, 0.5, 0.27], "at": [0, 0.05, 0.5],
         "corner_radius": 0.04, "material": "glass:tinted"},
        {"id": "roof", "shape": "rounded_slab", "size": [0.86, 0.46, 0.04], "at": [0, 0.06, 0.77],
         "corner_radius": 0.03, "material": "paint:#7a1f1f"},
        _box("grille", [0.4, 0.02, 0.07], [0, -0.495, 0.3], _DARK),
        *[_box(f"headlight_{s}", [0.16, 0.02, 0.05], [x, -0.495, 0.38], "plastic:#f4f2e8") for s, x in (("l", -0.34), ("r", 0.34))],
        *[_box(f"taillight_{s}", [0.16, 0.02, 0.05], [x, 0.495, 0.4], "plastic:#b01818") for s, x in (("l", -0.36), ("r", 0.36))],
        *[_box(f"mirror_{s}", [0.06, 0.03, 0.04], [x, -0.17, 0.5], "paint:#7a1f1f") for s, x in (("l", -0.45), ("r", 0.45))],
        # size before rotation_deg [0, 90, 0]: [height diameter, length diameter, tyre width]
        *[p for i, (x, y) in enumerate(((-0.42, -0.3), (0.42, -0.3), (-0.42, 0.32), (0.42, 0.32))) for p in (
            _cyl(f"wheel_{i}", [0.43, 0.15, 0.14], [x, y, 0], "rubber", rotation_deg=[0, 90, 0]),
            _cyl(f"rim_{i}", [0.27, 0.094, 0.015], [round(x + (0.0725 if x > 0 else -0.0725), 4), y, 0.08],
                 "metal:brushed", rotation_deg=[0, 90, 0]))]],
    "rock": lambda: [
        _sphere("mass", [1.0, 0.85, 0.75], [0, 0, 0], "color:#7d7a72"),
        _sphere("lump", [0.6, 0.55, 0.5], [0.2, 0.1, 0.4], "color:#8a867c")],
})
def _taper(pid: str, size: list[float], at: list[float], material: str, top: float) -> dict[str, Any]:
    """A cone-like cylinder: ``top`` is the top radius as a fraction of the base radius."""
    return {"id": pid, "shape": "tapered_cylinder", "size": size, "at": at, "material": material,
            "radius": 0.5, "radius_top": 0.5 * top}


_RED = "color:#c8372d"
_ORANGE = "color:#ff6a13"
_CONCRETE = "color:#9a9890"
_ASPHALT = "color:#3a3a3c"
#: chute and its rails slope from the platform's front edge (y 0.2, z 0.64) to the ground (y -0.5)
_SLIDE_DEG = 42.4
_SLIDE_LEN = 0.948
#: swing legs lean from the ground (y +-0.45) to the beam (y 0, z 0.97)
_SWING_DEG = 24.9
_SWING_LEN = 1.069

# Outdoor and industrial vocabulary. Every part rests on the ground or on another part (the
# recipe test checks it): a floating rail or roof tier shows up immediately in a render.
_RECIPES.update({
    "playground_slide": lambda: [
        # ladder at the back (+Y), platform on top of it, chute down to the front (-Y)
        _box("ladder_l", [0.04, 0.12, 1.0], [-0.4, 0.42, 0], _STEEL),
        _box("ladder_r", [0.04, 0.12, 1.0], [0.4, 0.42, 0], _STEEL),
        *[_box(f"rung_{k}", [0.76, 0.04, 0.03], [0, 0.42, 0.12 + 0.16 * k], _STEEL) for k in range(3)],
        _box("platform", [0.84, 0.2, 0.04], [0, 0.3, 0.6], _WOOD),
        _box("guard_l", [0.04, 0.2, 0.36], [-0.4, 0.3, 0.64], _RED),
        _box("guard_r", [0.04, 0.2, 0.36], [0.4, 0.3, 0.64], _RED),
        _box("chute", [0.6, _SLIDE_LEN, 0.03], [0, -0.14, 0], _RED, rotation_deg=[_SLIDE_DEG, 0, 0]),
        _box("chute_side_l", [0.04, _SLIDE_LEN, 0.1], [-0.32, -0.117, 0], _RED, rotation_deg=[_SLIDE_DEG, 0, 0]),
        _box("chute_side_r", [0.04, _SLIDE_LEN, 0.1], [0.32, -0.117, 0], _RED, rotation_deg=[_SLIDE_DEG, 0, 0])],
    "swing_set": lambda: [
        # an A-frame at each end carries the top beam; two seats hang from it on chains
        *[_box(f"leg_{i}", [0.06, 0.06, _SWING_LEN], [x, sy * 0.198, 0], _STEEL, rotation_deg=[sy * _SWING_DEG, 0, 0])
          for i, (x, sy) in enumerate(((-0.47, -1), (-0.47, 1), (0.47, -1), (0.47, 1)))],
        _box("beam", [1.0, 0.08, 0.05], [0, 0, 0.95], _STEEL),
        *[part for k, x in enumerate((-0.2, 0.2)) for part in (
            _box(f"chain_{k}_l", [0.01, 0.01, 0.74], [x - 0.1, 0, 0.21], _STEEL),
            _box(f"chain_{k}_r", [0.01, 0.01, 0.74], [x + 0.1, 0, 0.21], _STEEL),
            _box(f"seat_{k}", [0.22, 0.12, 0.02], [x, 0, 0.19], _DARK))]],
    "gazebo": lambda: [
        _box("deck", [1.0, 1.0, 0.05], [0, 0, 0], "wood:pine"),
        *[_box(f"post_{i}", [0.05, 0.05, 0.65], [sx * 0.45, sy * 0.45, 0.05], _WHITE)
          for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))],
        # railings on three sides, the front (-Y) is the entrance
        _box("rail_back", [0.9, 0.03, 0.03], [0, 0.45, 0.3], _WHITE),
        _box("rail_l", [0.03, 0.9, 0.03], [-0.45, 0, 0.3], _WHITE),
        _box("rail_r", [0.03, 0.9, 0.03], [0.45, 0, 0.3], _WHITE),
        _box("eave", [1.0, 1.0, 0.04], [0, 0, 0.7], _WHITE),
        _box("roof_1", [0.8, 0.8, 0.1], [0, 0, 0.74], "color:#5b4636"),
        _box("roof_2", [0.55, 0.55, 0.08], [0, 0, 0.84], "color:#5b4636"),
        _box("roof_3", [0.3, 0.3, 0.06], [0, 0, 0.92], "color:#5b4636"),
        _cyl("finial", [0.04, 0.04, 0.02], [0, 0, 0.98], _DARK)],
    "planter": lambda: [
        _box("box", [1.0, 1.0, 0.7], [0, 0, 0], _CONCRETE),
        _box("soil", [0.92, 0.88, 0.05], [0, 0, 0.66], "color:#4a3526"),
        *[_sphere(f"plant_{k}", [0.3, 0.6, 0.32], [x, 0, 0.68], g)
          for k, (x, g) in enumerate(((-0.3, _GREEN), (0.0, _GREEN2), (0.3, _GREEN)))]],
    "hedge": lambda: [
        _box("body", [1.0, 0.9, 0.85], [0, 0, 0], _GREEN),
        *[_sphere(f"top_{k}", [0.3, 1.0, 0.3], [x, 0, 0.7], _GREEN2) for k, x in enumerate((-0.35, 0.0, 0.35))]],
    "signpost": lambda: [
        _cyl("post", [0.1, 0.6, 0.72], [0, 0.2, 0], _STEEL),
        _box("board", [1.0, 0.4, 0.35], [0, -0.3, 0.65], "color:#1f5a3a"),
        _box("text_band", [0.8, 0.05, 0.08], [0, -0.525, 0.78], _WHITE)],
    "bollard": lambda: [
        _cyl("body", [0.9, 0.9, 0.88], [0, 0, 0], _DARK),
        _cyl("band", [0.92, 0.92, 0.06], [0, 0, 0.7], _YELLOW),
        _sphere("cap", [0.9, 0.9, 0.24], [0, 0, 0.76], _DARK)],
    "parking_bay": lambda: [
        # flat: an asphalt patch with painted lines on it, front (-Y) open
        _box("surface", [1.0, 1.0, 0.6], [0, 0, 0], _ASPHALT),
        _box("line_l", [0.04, 1.0, 0.4], [-0.48, 0, 0.6], _WHITE),
        _box("line_r", [0.04, 1.0, 0.4], [0.48, 0, 0.6], _WHITE),
        _box("line_back", [0.92, 0.04, 0.4], [0, 0.48, 0.6], _WHITE)],
    "loading_dock": lambda: [
        _box("platform", [1.0, 0.9, 0.92], [0, 0.05, 0], _CONCRETE),
        *[_box(f"bumper_{k}", [0.08, 0.1, 0.25], [x, -0.45, 0.55], _DARK) for k, x in enumerate((-0.3, 0.3))],
        _box("edge_stripe", [1.0, 0.04, 0.02], [0, -0.38, 0.92], _YELLOW),
        _box("leveler", [0.4, 0.3, 0.01], [0, -0.2, 0.92], _STEEL)],
    "shipping_container": lambda: [
        _box("base_frame", [1.0, 1.0, 0.04], [0, 0, 0], _DARK),
        _box("body", [0.97, 0.98, 0.92], [0, 0, 0.04], "color:#2f5d8a"),
        _box("top_frame", [1.0, 1.0, 0.04], [0, 0, 0.96], _DARK),
        *[_box(f"rib_{s}_{k}", [0.015, 0.02, 0.92], [sx * 0.4925, -0.45 + 0.1 * k, 0.04], "color:#2a5480")
          for s, sx in (("l", -1), ("r", 1)) for k in range(10)],
        # doors at the back (+Y): a centre split and four locking bars
        _box("door_split", [0.006, 0.02, 0.88], [0, 0.49, 0.06], _DARK),
        *[_box(f"lock_bar_{k}", [0.02, 0.02, 0.86], [x, 0.49, 0.07], _STEEL)
          for k, x in enumerate((-0.35, -0.15, 0.15, 0.35))]],
    "traffic_cone": lambda: [
        _box("base", [1.0, 1.0, 0.06], [0, 0, 0], _DARK),
        _taper("cone", [0.75, 0.75, 0.94], [0, 0, 0.06], _ORANGE, top=0.13),
        _cyl("band", [0.5, 0.5, 0.12], [0, 0, 0.5], _WHITE)],
    "lamp_post_double": lambda: [
        _cyl("foot", [0.15, 0.6, 0.03], [0, 0, 0], _DARK),
        _cyl("pole", [0.08, 0.3, 0.9], [0, 0, 0.03], _DARK),
        _box("crossarm", [0.9, 0.15, 0.02], [0, 0, 0.9], _DARK),
        *[_box(f"head_{k}", [0.12, 0.8, 0.08], [x, 0, 0.84], _DARK) for k, x in enumerate((-0.44, 0.44))],
        *[_box(f"lens_{k}", [0.1, 0.7, 0.01], [x, 0, 0.83], "color:#fff4d6") for k, x in enumerate((-0.44, 0.44))],
        _cyl("finial", [0.05, 0.2, 0.08], [0, 0, 0.92], _DARK)],
})


# --- seeded variation -----------------------------------------------------------------------
# The idea (one generator per category, its proportions drawn from ranges by a seed so two
# instances differ) follows Infinigen Indoors (Princeton VL, BSD-3-Clause,
# github.com/princeton-vl/infinigen, infinigen/assets/objects). The ranges below are ours.
# Every variant keeps the [0, 1] relative extents (bbox_m scaling is unchanged) and keeps its
# parts touching: each one connects to the ground through a chain of touching parts.

_FRONT = -0.46  # front face of a variant carcass (body y in [-0.46, 0.5])


def _leg(pid: str, w: float, h: float, x: float, y: float, material: str, rnd: bool) -> dict[str, Any]:
    return (_cyl if rnd else _box)(pid, [w, w, h], [x, y, 0], material)


def _cabinet_v(rng: Any, doors: tuple[int, int], drawers: tuple[int, int], legs: bool,
               wood: str) -> list[dict[str, Any]]:
    n_doors, n_drawers = rng.randint(*doors), rng.randint(*drawers)
    if n_doors + n_drawers == 0:
        n_drawers = 1
    z0 = rng.uniform(0.05, 0.15) if legs else 0.0
    leg_w, rnd = rng.uniform(0.03, 0.07), rng.random() < 0.4
    parts = [_box("body", [1.0, 0.96, 1.0 - z0], [0, 0.02, z0], wood)]
    if legs:
        lx, ly = 0.5 - leg_w / 2 - 0.03, 0.46 - leg_w / 2 - 0.03
        parts += [_leg(f"leg_{i}", leg_w, z0, sx * lx, 0.02 + sy * ly, _DARK, rnd)
                  for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))]
    h = 1.0 - z0
    # drawers stacked at the top, doors below them (a dresser has no doors, a wardrobe few drawers)
    drawer_h = h * (n_drawers / (n_drawers + 2 * n_doors))
    door_top = z0 + h - drawer_h
    line, pull_y = _FRONT - 0.005, _FRONT - 0.015
    for k in range(n_drawers):
        top = z0 + h - drawer_h * k / n_drawers
        bottom = z0 + h - drawer_h * (k + 1) / n_drawers
        if k < n_drawers - 1 or n_doors:
            parts.append(_box(f"drawer_line_{k}", [0.94, 0.01, 0.006], [0, line, bottom], _DARK))
        parts.append(_box(f"drawer_pull_{k}", [0.16, 0.03, 0.02], [0, pull_y, (top + bottom) / 2 - 0.01], _STEEL))
    door_h = door_top - z0
    for k in range(n_doors):
        x = -0.5 + (k + 0.5) / n_doors
        if k < n_doors - 1:
            parts.append(_box(f"door_line_{k}", [0.006, 0.01, door_h * 0.9],
                              [x + 0.5 / n_doors, line, z0 + door_h * 0.05], _DARK))
        hx = x + 0.4 / n_doors * (1 if k % 2 else -1) if n_doors > 1 else 0.35
        parts.append(_box(f"handle_{k}", [0.02, 0.03, min(0.14, door_h * 0.5)],
                          [hx, pull_y, z0 + door_h * 0.45], _STEEL))
    return parts


def _stool_v(rng: Any) -> list[dict[str, Any]]:
    w, inset, rnd = rng.uniform(0.05, 0.1), rng.uniform(0.25, 0.35), rng.random() < 0.5
    seat = (_cyl if rng.random() < 0.6 else _box)("seat", [1.0, 1.0, rng.uniform(0.05, 0.09)], [0, 0, 0.0], _WOOD)
    seat["at"] = [0, 0, 1.0 - seat["size"][2]]
    top = seat["at"][2]
    parts = [seat, *[_leg(f"leg_{i}", w, top, sx * inset, sy * inset, _WOOD, rnd)
                     for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))]]
    if rng.random() < 0.7:
        parts.append(_box("rung", [2 * inset, 0.05, 0.04], [0, -inset, rng.uniform(0.2, 0.4)], _WOOD))
    return parts


def _office_chair_v(rng: Any) -> list[dict[str, Any]]:
    back_h, back_w = rng.uniform(0.3, 0.5), rng.uniform(0.6, 0.85)
    n = rng.choice((4, 5))
    import math

    casters = [_sphere(f"caster_{i}", [0.08, 0.08, 0.08],
                       [round(0.4 * math.cos(2 * math.pi * i / n), 4), round(0.4 * math.sin(2 * math.pi * i / n), 4), 0],
                       _DARK) for i in range(n)]
    return [_cyl("base", [0.9, 0.9, 0.05], [0, 0, 0.04], _DARK), *casters,
            _cyl("gas_lift", [0.08, 0.08, 0.3], [0, 0, 0.09], "metal:chrome"),
            _box("seat", [0.85, 0.8, 0.09], [0, -0.02, 0.39], "fabric:weave"),
            _box("back_post", [0.08, 0.06, 1.0 - back_h - 0.45], [0, 0.4, 0.45], _DARK),
            _box("back", [back_w, 0.08, back_h], [0, 0.42, 1.0 - back_h], "fabric:weave")]


def _shelving_v(rng: Any) -> list[dict[str, Any]]:
    n, w = rng.randint(3, 6), rng.uniform(0.03, 0.06)
    zs = [round(0.05 + (0.93 * k / (n - 1)), 4) for k in range(n)]
    parts = [_box(f"post_{i}", [w, w, 1.0], [sx * (0.5 - w / 2), sy * (0.5 - w / 2), 0], _STEEL)
             for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))]
    parts += [_box(f"shelf_{k}", [1.0, 1.0, 0.02], [0, 0, z], _STEEL) for k, z in enumerate(zs)]
    for k, z in enumerate(zs[:-1]):
        gap = zs[k + 1] - z - 0.02
        if rng.random() < 0.6:
            parts.append(_box(f"box_{k}", [rng.uniform(0.2, 0.35), 0.6, min(0.15, gap * 0.8)],
                              [rng.uniform(-0.3, 0.3), 0, z + 0.02], "color:#b08a5a"))
    return parts


_VARIANTS: dict[str, Any] = {
    "dresser": lambda r: _cabinet_v(r, (0, 0), (3, 6), True, _WOOD),
    "nightstand": lambda r: _cabinet_v(r, (0, 1), (1, 3), True, _WALNUT),
    "cabinet": lambda r: _cabinet_v(r, (1, 3), (0, 2), r.random() < 0.3, _WOOD),
    "wardrobe": lambda r: _cabinet_v(r, (2, 3), (0, 2), False, _WOOD),
    "filing_cabinet": lambda r: _cabinet_v(r, (0, 0), (2, 5), False, _STEEL),
    "stool": _stool_v,
    "office_chair": _office_chair_v,
    "shelving_unit": _shelving_v,
}


def _user_parts(category: str) -> list[dict[str, Any]] | None:
    """Parts of the user recipe for ``category`` when it applies: always for a category
    without a built-in, for one with a built-in only if it was saved with overwrite=true."""
    from ai_db_scene.user_recipes import load_user_recipe

    r = load_user_recipe(category)
    if r is None or (category in _RECIPES and not r.get("overrides_builtin")):
        return None
    return [dict(p) for p in r["parts"]]


def user_overrides(category: str) -> bool:
    """True if a user recipe for ``category`` was saved with overwrite=true: it then wins over
    the built-in recipe and over the addon's furniture generator."""
    from ai_db_scene.user_recipes import load_user_recipe

    r = load_user_recipe(category)
    return bool(r and (r.get("overwrite") or r.get("overrides_builtin")))


def has_recipe(category: str) -> bool:
    return category in _RECIPES or _user_parts(category) is not None


def variant_categories() -> list[str]:
    """Categories whose recipe varies with ``recipe_spec(..., seed=...)``."""
    return sorted(_VARIANTS)


def recipe_spec(category: str, name: str, dims: tuple[float, float, float] | list[float],
                seed: int | str | None = None) -> dict[str, Any]:
    """The build_asset spec of ``category``, named ``name``, scaled to exactly ``dims``.

    ``seed`` (None: the fixed recipe) draws a variant for the categories in
    ``variant_categories()``: leg style/thickness, drawer/door/shelf counts, back height.
    The same seed always gives the same parts; other categories ignore it. A user recipe
    (``create_recipe``) is used for categories without a built-in, or instead of the built-in
    when it was saved with overwrite=true."""
    user = _user_parts(category)
    if user is not None:
        parts, style = user, "ai-db user recipe"
    elif seed is not None and category in _VARIANTS:
        import random

        parts, style = _VARIANTS[category](random.Random(f"{category}:{seed}")), "ai-db recipe"
    else:
        parts, style = _RECIPES[category](), "ai-db recipe"
    return {"asset": name, "category": category, "style": style,
            "bbox_m": [float(dims[0]), float(dims[1]), float(dims[2])], "parts": _fine_grain(soften_edges(parts, dims))}


def _fine_grain(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Furniture-scale wood figure: the library default is sized for floors and reads as big
    contour-map swirls on a drawer front."""
    return [{**p, "material": f"{p['material']}:scale={RECIPE_WOOD_GRAIN_SCALE}"}
            if str(p.get("material", "")).startswith("wood") and "scale=" not in p["material"] else p
            for p in parts]


def soften_edges(parts: list[dict[str, Any]], dims: tuple[float, float, float] | list[float],
                 radius_m: float = RECIPE_EDGE_BEVEL_M) -> list[dict[str, Any]]:
    """Round the edges of every box part by about ``radius_m`` on the built object.

    Parts are authored in unit space and the asset is then scaled to ``dims`` per axis, so a
    unit-space radius r becomes r * dims[i] along axis i: r = radius / max(dims) keeps every
    axis at or under ``radius_m``. Thin parts get at most RECIPE_EDGE_BEVEL_MAX_SHARE of their
    thinnest real side (a 2 cm shelf is not turned into a tube). Parts that set ``bevel``
    themselves are left alone."""
    if radius_m <= 0:
        return parts
    scale = max(float(d) for d in dims)
    out = []
    for p in parts:
        if p.get("shape") == "box" and "bevel" not in p and isinstance(p.get("size"), list):
            thinnest_m = min(float(s) * float(d) for s, d in zip(p["size"], dims))
            r_m = min(radius_m, RECIPE_EDGE_BEVEL_MAX_SHARE * thinnest_m)
            if r_m >= 0.001:
                p = {**p, "bevel": round(r_m / scale, 5), "bevel_segments": RECIPE_EDGE_BEVEL_SEGMENTS}
        out.append(p)
    return out


def recipe_categories() -> list[str]:
    return sorted(_RECIPES)
