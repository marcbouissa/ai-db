"""What still makes a valid scene look fake, as warnings for the agent (realistic-modeling skill).

The validator checks that a scene is physically sound; these check that it will read as real:
flat colour on big surfaces, an indoor room with no light source of its own, and bare walls in
a living space. Warnings only, each with the fix.
"""

from __future__ import annotations

import re
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
    issues += off_axis(spec, placements)
    by_id = {o.id: o for o in spec.objects}
    for p in placements:
        o = by_id.get(p.id) or next((x for x in spec.objects if p.id.startswith(x.id + "_")), None)
        mat = getattr(o, "material", None) if o else None
        if mat and str(mat).startswith("color") and p.dims[0] * p.dims[1] >= BIG_M2:
            issues.append(_warn(f"{p.id} is a large object in a flat colour ({mat}); use paint:, plastic:, "
                                f"fabric: or wood: with the same tint", [p.id]))
    return issues


OFF_AXIS_DEG = (5.0, 40.0)  # furniture this far off the room's axes reads as a mistake, not a choice


def off_axis(spec: Any, placements: list[Any]) -> list[dict[str, Any]]:
    """Floor furniture of a rect room turned 5-40 deg off the walls without a facing/around relation
    asking for it (judges: "racks scattered at odd angles", "desks on a diagonal")."""
    from ai_db_scene.priors import load_priors

    st = spec.setting
    if st.kind != "indoor" or st.shape != "rect":
        return []
    priors = load_priors()
    turned = {r.a for r in spec.relations if r.rel in ("facing", "around")}
    ids = {o.id for o in spec.objects}
    out: list[str] = []
    for p in placements:
        base = p.id if p.id in ids else re.sub(r"(_seat)?_\d+$", "", p.id)
        prior = priors.get(p.category)
        if base in turned or "_seat_" in p.id or prior is None or prior.support != "floor" or p.dims[2] <= 0.05:
            continue
        off = abs((p.rotation_deg_z + 45.0) % 90.0 - 45.0)
        if OFF_AXIS_DEG[0] <= off <= OFF_AXIS_DEG[1]:
            out.append(p.id)
    if not out:
        return []
    return [{"check": "off_axis", "severity": "warning", "objects": out[:8],
             "message": (f"{len(out)} object(s) stand {OFF_AXIS_DEG[0]:g}-{OFF_AXIS_DEG[1]:g} deg off the room's axes "
                         f"({', '.join(out[:5])}); turn them square to the walls, or give them a facing relation")}]


_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "pair": 2, "couple": 2}


#: quantity words with no exact count ("a few rocks", "some plants")
_VAGUE = ("few", "several", "some", "many", "lots", "various", "assorted", "more", "dense", "multiple")


def brief_counts(brief: str, categories: set[str]) -> dict[str, int | None]:
    """Categories a brief names, with the count it asks for when one is stated ("two armchairs").

    Numbered mentions of one category add up ("a teacher desk ... twelve student desks" is 13
    desks); a count followed by "each" ("two chairs each") is per something else: no count."""
    from ai_db_scene.feedback import ALIASES

    low = " " + re.sub(r"[^a-z0-9 ,]+", " ", brief.lower()).replace(",", " , ") + " "
    mentions: dict[str, list[int | None]] = {}
    names = {c.replace("_", " "): c for c in categories}
    names.update({w: c for w, c in ALIASES.items() if c in categories})
    for name, cat in sorted(names.items(), key=lambda kv: -len(kv[0])):
        for m in re.finditer(rf"\b{re.escape(name)}(e?s)?\b", low):
            before = low[:m.start()].split()[-2:]
            before = before[before.index(",") + 1:] if "," in before else before  # "a workbench, safety barriers"
            n = next((int(w) if w.isdigit() else _NUMBERS[w] for w in reversed(before)
                      if w.isdigit() or w in _NUMBERS), None)
            if any(w in _VAGUE for w in before):
                n = None  # "a few rocks" is not one rock
            each = low[m.end():].split()[:1] == ["each"]
            mentions.setdefault(cat, []).append(-1 if each else n)
            low = low[:m.start()] + " " * (m.end() - m.start()) + low[m.end():]  # "coffee table" not also "table"
    out: dict[str, int | None] = {}
    for cat, ns in mentions.items():
        nums = [n for n in ns if n is not None]
        out[cat] = None if not nums or -1 in nums else sum(nums)  # type: ignore[arg-type]
    return out


#: a brief's word for a category also counts its more specific kinds (bushes on a tropical beach
#: are built as tropical_bush, its trees as palm_tree)
_KIN = {"bush": ("tropical_bush",), "tree": ("palm_tree",), "chair": ("sun_lounger",)}


def check_brief(brief: str, placements: list[Any]) -> list[dict[str, Any]]:
    """Objects the brief asks for that the scene lacks, and counts that differ from the brief."""
    from ai_db_scene.priors import load_priors

    have: dict[str, int] = {}
    for p in placements:
        have[p.category] = have.get(p.category, 0) + 1
    issues = []
    for cat, n in brief_counts(brief, set(load_priors())).items():
        got = have.get(cat, 0) + sum(have.get(k, 0) for k in _KIN.get(cat, ()))
        if got == 0:
            issues.append(_warn(f"the brief asks for {cat.replace('_', ' ')} but the scene has none"))
        elif n is not None and got != n:
            issues.append(_warn(f"the brief asks for {n} {cat.replace('_', ' ')}, the scene has {got}"))
    return issues
