"""What still makes a valid scene look fake, as warnings for the agent (realistic-modeling skill).

The validator checks that a scene is physically sound; these check that it will read as real:
flat colour on big surfaces, an indoor room with no light source of its own, and bare walls in
a living space. Warnings only, each with the fix.
"""

from __future__ import annotations

from typing import Any

#: categories that light a room besides the daylight rig
LIGHTS = ("floor_lamp", "table_lamp", "pendant_lamp", "ceiling_light", "wall_sconce", "street_lamp", "lamp_post_double")
#: wall pieces that keep a living space from looking unfinished
WALL_DECOR = ("picture_frame", "mirror", "wall_clock", "whiteboard", "tv", "bookshelf", "shelving_unit", "wall_sconce",
              "fireplace", "window")
#: settings where bare walls are expected (work floors, bathrooms)
WORKSPACE = ("pallet_rack", "conveyor", "forklift", "loading_dock", "toilet", "shower_stall", "bathtub")
BIG_M2 = 1.5  # footprint above which a flat colour is very visible


def _warn(message: str, objects: list[str] | None = None) -> dict[str, Any]:
    return {"check": "realism", "severity": "warning", "objects": objects or [], "message": message}


def check(spec: Any, placements: list[Any]) -> list[dict[str, Any]]:
    from ai_db_scene.materials import FLOOR_MATERIALS, WALL_MATERIAL, style_material

    issues: list[dict[str, Any]] = []
    st = spec.setting
    cats = {p.category for p in placements}
    if st.kind == "indoor":
        floor = st.floor_material or style_material(st.style, "floor") or FLOOR_MATERIALS.get("indoor", "")
        wall = st.wall_material or style_material(st.style, "wall") or WALL_MATERIAL
        for what, mat in (("floor", floor), ("walls", wall)):
            if str(mat).startswith("color"):
                issues.append(_warn(f"the {what} are a flat colour ({mat}); use a material with variation, e.g. "
                                    f"{'wood:oak or concrete' if what == 'floor' else 'plaster:#rrggbb'}"))
        if not cats & set(LIGHTS) and not any(o.category == "window" for o in spec.objects):
            issues.append(_warn("no lamp and no window: the room is lit only by the fill light and will look "
                                "staged; add a floor or table lamp, a pendant or a window"))
        if cats and not cats & set(WALL_DECOR) and not cats & set(WORKSPACE) and len(placements) >= 3:
            issues.append(_warn("bare walls: add a picture_frame, mirror, wall_clock or shelving (above / "
                                "against_wall) so the room does not look unfinished"))
    by_id = {o.id: o for o in spec.objects}
    for p in placements:
        o = by_id.get(p.id) or next((x for x in spec.objects if p.id.startswith(x.id + "_")), None)
        mat = getattr(o, "material", None) if o else None
        if mat and str(mat).startswith("color") and p.dims[0] * p.dims[1] >= BIG_M2:
            issues.append(_warn(f"{p.id} is a large object in a flat colour ({mat}); use paint:, plastic:, "
                                f"fabric: or wood: with the same tint", [p.id]))
    return issues
