"""Design families for generated furniture: sofas, armchairs, tables and dining chairs that differ
from scene to scene and match within one.

Before this, every build passed the addon's furniture generator a style preset (``modern`` ->
``mid_century`` sofa, ``modern_lounge`` armchair) and a seed. A preset pins the generator's design
choices (arm, back and leg style), so every modern scene got the same sofa; without a style the
seed drew them with weights that make most pieces track-arm, loose-back boxes; and each object
drew its own, so a sofa and its armchairs did not belong together. Only the colour changed.

Here a scene picks one design family (mid-century, chesterfield, lounge, ...) for its seating and
tables: from an object's ``variant``, else from ``setting.style`` (each style names a few families),
else from the whole list. The pick is a hash of the build's variation key (the setting, the object
list and the build seed; edit_scene passes the stored key back, so an edit keeps the furniture)
and the object's room. Each piece then draws its proportions (seat count, seat/arm height, leg
height, softness) inside the family's ranges from the same key and its category, so the copies of
one category in a room are an identical set. Every design choice is passed to the generator
explicitly; ``FIT_CODE`` then scales the built piece to the placement's exact bbox.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

#: marks a variation key built with design families; keys stored before it keep the old path
VARIANTS_VERSION = 2
_MARK = "variants"

# generator param ranges (the addon refuses values outside them)
_UPH_RANGES = {"arm_width": (0.10, 0.30), "arm_height": (0.52, 0.80), "seat_height": (0.38, 0.50),
               "back_height": (0.70, 1.05), "leg_height": (0.03, 0.22), "softness": (0.0, 1.0)}
_UPH_SIZE = {"sofa": ((1.3, 3.4), (0.80, 1.15)), "armchair": ((0.62, 1.10), (0.72, 1.00))}
_SEAT_W = (0.5, 0.78)  # one seat cushion's width range (m)

#: name -> design. Numbers are (lo, hi) ranges a piece draws from, lists are choices, the rest fixed.
#: ``sofa`` also applies to armchairs unless ``armchair`` overrides a key.
FAMILIES: dict[str, dict[str, Any]] = {
    "mid_century": {
        "doc": "mid-century modern: thin track arms, tight bench back, splayed walnut legs",
        "sofa": {"arm_style": "track", "arm_width": (0.10, 0.14), "back_style": "tight", "tufted": [True, False],
                 "bench_seat": True, "leg_style": "splayed", "leg_height": (0.15, 0.19), "softness": (0.25, 0.4),
                 "piping": True, "seat_height": (0.42, 0.45), "arm_drop": (0.18, 0.24), "pillows": 2,
                 "leg_material": ["walnut", "oak"]},
        "fabrics": ["olive", "mustard_velvet", "slate_blue", "rust_velvet", "charcoal"],
        "pillow_fabrics": ["rust_velvet", "mustard_velvet", "ivory"],
        "table": {"top_shape": "oval", "leg_style": "splayed", "material": ["walnut", "oak"], "apron": False},
        "chair": {"back_style": "panel", "seat_style": "upholstered", "leg_style": "tapered",
                  "material": ["walnut", "oak"]},
    },
    "chesterfield": {
        "doc": "chesterfield: rolled arms as high as the button-tufted back, leather, bun feet",
        "sofa": {"arm_style": "rolled", "arm_width": (0.20, 0.26), "back_style": "tight", "tufted": True,
                 "bench_seat": [True, False], "leg_style": "bun", "leg_height": (0.06, 0.09),
                 "softness": (0.25, 0.35), "piping": False, "seat_height": (0.44, 0.47), "arm_drop": (0.0, 0.0),
                 "pillows": 0, "leg_material": ["dark_wood", "mahogany"]},
        "armchair": {"arm_drop": (0.0, 0.04)},
        "fabrics": ["cognac_leather", "oxblood_leather", "black_leather", "emerald_velvet"],
        "pillow_fabrics": ["ivory"],
        "table": {"top_shape": "rect", "leg_style": "turned", "material": ["mahogany", "dark_wood"], "apron": True},
        "chair": {"back_style": "upholstered", "seat_style": "upholstered", "leg_style": "turned",
                  "material": ["mahogany", "dark_wood"]},
    },
    "lawson": {
        "doc": "traditional: rolled arms, loose back cushions, turned legs, piping",
        "sofa": {"arm_style": "rolled", "arm_width": (0.17, 0.22), "back_style": "loose", "tufted": False,
                 "bench_seat": False, "leg_style": "turned", "leg_height": (0.10, 0.14), "softness": (0.55, 0.75),
                 "piping": True, "seat_height": (0.44, 0.47), "arm_drop": (0.18, 0.26), "pillows": (2, 3),
                 "leg_material": ["cherry", "dark_wood", "walnut"]},
        "fabrics": ["ivory", "stone", "navy_velvet", "sage_velvet", "oatmeal"],
        "pillow_fabrics": ["navy_velvet", "blush_velvet", "mustard_velvet", "terracotta"],
        "table": {"top_shape": "rect", "leg_style": "tapered", "material": ["cherry", "walnut"], "apron": True},
        "chair": {"back_style": "spindles", "seat_style": "wood", "leg_style": "round", "stretchers": True,
                  "material": ["cherry", "walnut"]},
    },
    "modern_track": {
        "doc": "contemporary: square track arms, loose cushions, short metal legs",
        "sofa": {"arm_style": "track", "arm_width": (0.15, 0.20), "back_style": "loose", "tufted": False,
                 "bench_seat": False, "leg_style": "metal", "leg_height": (0.08, 0.12), "softness": (0.45, 0.65),
                 "piping": False, "seat_height": (0.42, 0.45), "arm_drop": (0.18, 0.24), "pillows": 2,
                 "leg_material": ["black_metal", "chrome"]},
        "fabrics": ["charcoal", "slate_blue", "ivory", "stone"],
        "pillow_fabrics": ["mustard_velvet", "terracotta", "ivory", "sage_velvet"],
        "table": {"top_shape": "rect", "leg_style": "metal", "material": ["oak", "black_lacquer", "white_lacquer"],
                  "leg_material": "black_metal", "shelf": [True, False], "apron": False},
        "chair": {"back_style": "panel", "seat_style": "upholstered", "leg_style": "square",
                  "material": ["black_lacquer", "walnut"]},
    },
    "lounge": {
        "doc": "deep lounge: wide plush arms, low seat, very soft cushions on a hidden plinth",
        "sofa": {"arm_style": "track", "arm_width": (0.24, 0.30), "back_style": "loose", "tufted": False,
                 "bench_seat": False, "leg_style": "hidden", "leg_height": (0.03, 0.05), "softness": (0.85, 1.0),
                 "piping": False, "seat_height": (0.40, 0.42), "arm_drop": (0.22, 0.30), "pillows": (3, 4),
                 "leg_material": ["dark_wood"], "max_seats": 3},
        "fabrics": ["oatmeal", "stone", "terracotta", "ivory"],
        "pillow_fabrics": ["terracotta", "sage_velvet", "rust_velvet", "charcoal"],
        "table": {"top_shape": ["round", "rect"], "leg_style": "pedestal", "material": ["oak", "white_lacquer"]},
        "chair": {"back_style": "upholstered", "seat_style": "upholstered", "leg_style": "tapered",
                  "material": ["oak", "walnut"]},
    },
    "scandinavian": {
        "doc": "scandinavian: slim sloped arms, loose back, light tapered oak legs",
        "sofa": {"arm_style": "slope", "arm_width": (0.13, 0.17), "back_style": "loose", "tufted": False,
                 "bench_seat": False, "leg_style": "tapered", "leg_height": (0.12, 0.16), "softness": (0.45, 0.6),
                 "piping": False, "seat_height": (0.42, 0.45), "arm_drop": (0.20, 0.26), "pillows": 2,
                 "leg_material": ["oak", "pine"]},
        "fabrics": ["ivory", "oatmeal", "sage_velvet", "charcoal", "slate_blue"],
        "pillow_fabrics": ["mustard_velvet", "sage_velvet", "terracotta"],
        "table": {"top_shape": ["round", "rect"], "leg_style": "tapered", "material": ["oak", "white_lacquer"],
                  "leg_material": "oak", "apron": False},
        "chair": {"back_style": "slats", "seat_style": "wood", "leg_style": "round", "material": ["oak", "pine"]},
    },
    "bauhaus": {
        "doc": "bauhaus / industrial: square metal tube frame arms, tight leather back",
        "sofa": {"arm_style": "metal_square", "arm_width": (0.16, 0.22), "back_style": "tight", "tufted": False,
                 "bench_seat": [True, False], "leg_style": "metal", "leg_height": (0.06, 0.10),
                 "softness": (0.3, 0.45), "piping": False, "seat_height": (0.42, 0.45), "arm_drop": (0.10, 0.16),
                 "pillows": 0, "leg_material": ["chrome", "black_metal"]},
        "fabrics": ["black_leather", "cognac_leather", "charcoal"],
        "pillow_fabrics": ["ivory"],
        "table": {"top_shape": "rect", "leg_style": "metal", "material": ["black_lacquer", "oak"],
                  "leg_material": ["chrome", "black_metal"], "apron": False},
        "chair": {"back_style": "panel", "seat_style": "upholstered", "leg_style": "square",
                  "material": ["black_lacquer"]},
    },
    "danish_frame": {
        "doc": "danish: open wooden frame arms with a padded rest, loose back cushions",
        "sofa": {"arm_style": "skeletal", "arm_width": (0.10, 0.14), "back_style": "loose", "tufted": False,
                 "bench_seat": False, "leg_style": "tapered", "leg_height": (0.12, 0.16), "softness": (0.4, 0.55),
                 "piping": [True, False], "seat_height": (0.42, 0.45), "arm_drop": (0.16, 0.22), "pillows": (1, 2),
                 "leg_material": ["oak", "walnut", "cherry"]},
        "fabrics": ["olive", "sage_velvet", "ivory", "slate_blue", "terracotta"],
        "pillow_fabrics": ["mustard_velvet", "ivory", "rust_velvet"],
        "table": {"top_shape": "rect", "leg_style": "tapered", "material": ["oak", "walnut"], "apron": True},
        "chair": {"back_style": "spindles", "seat_style": "wood", "leg_style": "round",
                  "material": ["oak", "walnut"]},
    },
    "glam": {
        "doc": "glam: velvet, tight bench back, slim rolled arms (armless slipper chairs), brass legs",
        "sofa": {"arm_style": "rolled", "arm_width": (0.16, 0.20), "back_style": "tight", "tufted": [True, False],
                 "bench_seat": True, "leg_style": "tapered", "leg_height": (0.14, 0.17), "softness": (0.45, 0.6),
                 "piping": True, "seat_height": (0.44, 0.46), "arm_drop": (0.12, 0.2), "pillows": 2,
                 "leg_material": ["brass"]},
        "armchair": {"arm_style": "none", "back_style": "tight", "pillows": 0},
        "fabrics": ["emerald_velvet", "blush_velvet", "navy_velvet", "mustard_velvet"],
        "pillow_fabrics": ["blush_velvet", "ivory", "emerald_velvet"],
        "table": {"top_shape": ["round", "oval"], "leg_style": "pedestal", "material": ["white_lacquer", "black_lacquer"],
                  "leg_material": "brass"},
        "chair": {"back_style": "upholstered", "seat_style": "upholstered", "leg_style": "tapered",
                  "material": ["black_lacquer", "walnut"]},
    },
    "farmhouse": {
        "doc": "rustic: wide track arms, loose cushions, block legs, canvas and pine",
        "sofa": {"arm_style": "track", "arm_width": (0.20, 0.26), "back_style": "loose", "tufted": False,
                 "bench_seat": False, "leg_style": "block", "leg_height": (0.06, 0.10), "softness": (0.6, 0.8),
                 "piping": False, "seat_height": (0.43, 0.46), "arm_drop": (0.16, 0.22), "pillows": (2, 3),
                 "leg_material": ["pine", "dark_wood", "oak"]},
        "fabrics": ["oatmeal", "terracotta", "olive", "cognac_leather"],
        "pillow_fabrics": ["ivory", "rust_velvet", "slate_blue"],
        "table": {"top_shape": "rect", "leg_style": ["turned", "square"], "material": ["pine", "oak"],
                  "top_thickness": (0.045, 0.06), "shelf": [True, False]},
        "chair": {"back_style": "ladder", "seat_style": "wood", "leg_style": "turned", "stretchers": True,
                  "material": ["pine", "oak"]},
    },
}

#: words of a style or variant text -> the families it names (first: the style's signature look)
_STYLE_FAMILIES: dict[str, tuple[str, ...]] = {
    "modern": ("modern_track", "mid_century", "bauhaus", "glam"),
    "contemporary": ("modern_track", "mid_century", "scandinavian"),
    "minimal": ("modern_track", "scandinavian", "bauhaus"),
    "minimalist": ("modern_track", "scandinavian", "bauhaus"),
    "scandinavian": ("scandinavian", "danish_frame", "mid_century"),
    "scandi": ("scandinavian", "danish_frame"),
    "nordic": ("scandinavian", "danish_frame"),
    "danish": ("danish_frame", "mid_century"),
    "japandi": ("danish_frame", "scandinavian"),
    "industrial": ("bauhaus", "modern_track", "farmhouse"),
    "loft": ("bauhaus", "modern_track"),
    "rustic": ("farmhouse", "lawson", "lounge"),
    "farmhouse": ("farmhouse", "lawson"),
    "country": ("farmhouse", "lawson"),
    "cottage": ("lawson", "farmhouse"),
    "boho": ("lounge", "farmhouse", "danish_frame"),
    "bohemian": ("lounge", "farmhouse", "danish_frame"),
    "classic": ("chesterfield", "lawson", "glam"),
    "traditional": ("lawson", "chesterfield"),
    "victorian": ("chesterfield", "lawson"),
    "english": ("chesterfield", "lawson"),
    "vintage": ("chesterfield", "mid_century", "lawson"),
    "retro": ("mid_century",),
    "midcentury": ("mid_century",),
    "mcm": ("mid_century",),
    "cozy": ("lounge", "lawson", "scandinavian", "farmhouse"),
    "cosy": ("lounge", "lawson", "scandinavian", "farmhouse"),
    "comfortable": ("lounge", "lawson"),
    "luxury": ("glam", "chesterfield"),
    "luxurious": ("glam", "chesterfield"),
    "elegant": ("glam", "chesterfield", "lawson"),
    "glamorous": ("glam",),
    "hollywood": ("glam",),
    "art": ("glam",),  # art deco
    "deco": ("glam",),
    "bauhaus": ("bauhaus",),
    "corbusier": ("bauhaus",),
    "lounge": ("lounge",),
    "chesterfield": ("chesterfield",),
    "lawson": ("lawson",),
    "office": ("modern_track", "bauhaus"),
    "corporate": ("modern_track", "bauhaus"),
    "lobby": ("modern_track", "bauhaus", "glam"),
    "hotel": ("glam", "modern_track", "chesterfield"),
}

#: categories a family shapes -> the generator family (the addon's) that builds them
CATEGORY_FAMILY = {"sofa": "sofa", "armchair": "armchair", "coffee_table": "table", "side_table": "table",
                   "dining_table": "table", "conference_table": "table", "dining_chair": "dining_chair",
                   "chair": "dining_chair"}
_TABLE_KIND = {"coffee_table": "coffee", "side_table": "side", "dining_table": "dining",
               "conference_table": "dining"}


def families() -> list[str]:
    return sorted(FAMILIES)


def coerce_variant(value: Any) -> tuple[str | None, str | None]:
    """An object's ``variant`` text -> (family name, note). A family name, a known alias or a
    style word ("chesterfield sofa", "Mid-Century", "industrial") is accepted; what names no
    family is dropped with a note (never a refusal: weak models write free text here)."""
    if value is None:
        return None, None
    text = str(value).strip().lower()
    key = re.sub(r"[^a-z]+", "_", text).strip("_")
    if key in FAMILIES:
        return key, None
    squashed = key.replace("_", "")
    for name in FAMILIES:
        if name.replace("_", "") == squashed or name.replace("_", "") in squashed:
            return name, f"variant {value!r} read as {name!r}"
    for w in re.findall(r"[a-z]+", text):
        if w in _STYLE_FAMILIES:
            return _STYLE_FAMILIES[w][0], f"variant {value!r} read as {_STYLE_FAMILIES[w][0]!r}"
    return None, f"variant {value!r} names no furniture family (allowed: {families()}); ignored"


def style_families(style: str | None) -> tuple[str, ...]:
    """The families a ``setting.style`` text names, in order; all of them when it names none."""
    words = re.findall(r"[a-z]+", (style or "").lower())
    picked: list[str] = []
    for w in words:
        for f in _STYLE_FAMILIES.get(w, ()):
            if f not in picked:
                picked.append(f)
    return tuple(picked) or tuple(families())


def variation_marker(seed: int) -> dict[str, Any]:
    """The element build_scene appends to the variation key: design families on, with the build
    seed (a rebuild with another seed gets other furniture; an edit passes the key back)."""
    return {_MARK: VARIANTS_VERSION, "seed": int(seed)}


def uses_families(key: list[Any] | None) -> bool:
    """True if a variation key was made with design families (keys stored earlier were not)."""
    return bool(key) and isinstance(key[-1], dict) and key[-1].get(_MARK) == VARIANTS_VERSION  # type: ignore[index]


def _hash(*parts: Any) -> int:
    return int(hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:8], 16)


def family_for(key: list[Any], style: str | None, variant: str | None = None, room: str | None = None) -> str:
    """The design family of a piece: its own ``variant``, else one of the style's families picked
    by the scene (variation key) and room, so one room's pieces match and scenes differ."""
    if variant in FAMILIES:
        return variant  # type: ignore[return-value]
    options = style_families(style)
    return options[_hash(key, room or "", "family") % len(options)]


def _draw(rng: random.Random, spec: Any) -> Any:
    if isinstance(spec, tuple):
        lo, hi = spec
        if isinstance(lo, int) and isinstance(hi, int):
            return rng.randint(lo, hi)
        return round(rng.uniform(lo, hi), 3)
    if isinstance(spec, list):
        return rng.choice(spec)
    return spec


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _upholstered(fam: dict[str, Any], category: str, dims: tuple[float, float, float],
                 rng: random.Random, fabric: str, pillow_fabric: str) -> dict[str, Any]:
    d = {**fam["sofa"], **(fam.get(category) or {})} if category == "armchair" else dict(fam["sofa"])
    w, dp, h = dims
    (wlo, whi), (dlo, dhi) = _UPH_SIZE[category]
    p: dict[str, Any] = {k: _draw(rng, d[k]) for k in ("arm_style", "back_style", "tufted", "leg_style",
                                                       "piping", "leg_material")}
    p["softness"] = _draw(rng, d["softness"])
    p["width"], p["depth"] = round(_clamp(w, wlo, whi), 3), round(_clamp(dp, dlo, dhi), 3)
    bh = round(_clamp(h, *_UPH_RANGES["back_height"]), 3)
    sh = _clamp(_draw(rng, d["seat_height"]), 0.38, bh - 0.32)
    p["back_height"], p["seat_height"] = bh, round(sh, 3)
    top = bh if p["back_style"] == "tight" else bh - 0.06
    drop = _draw(rng, d["arm_drop"])  # arm top this far under the back top
    p["arm_height"] = round(_clamp(top - drop, max(sh + 0.13, 0.52), min(top, 0.80)), 3)
    p["leg_height"] = round(_clamp(_draw(rng, d["leg_height"]), *_UPH_RANGES["leg_height"]), 3)
    aw = _draw(rng, d["arm_width"])
    if p["arm_style"] == "rolled":
        aw = max(aw, 0.15)
    aw_eff = 0.0 if p["arm_style"] == "none" else aw
    if category == "sofa":
        inner = p["width"] - 2 * aw_eff
        lo_n = max(1, int(inner / _SEAT_W[1]) + (inner % _SEAT_W[1] > 1e-6))
        hi_n = max(lo_n, min(int(fam["sofa"].get("max_seats", 4)), int(inner / _SEAT_W[0])))
        n = rng.randint(lo_n, hi_n)
        if n * 0.5 + 2 * aw_eff > p["width"]:  # the generator's own minimum: narrower arms
            aw = aw_eff = max(0.10, (p["width"] - n * 0.5) / 2)
        p["seats"] = n
        p["bench_seat"] = bool(_draw(rng, d["bench_seat"]))
        pillows = int(_draw(rng, d["pillows"]))
        p["pillows"] = min(pillows, n + 1)
    else:
        if 0.5 + 2 * aw_eff > p["width"]:
            aw = max(0.10, (p["width"] - 0.5) / 2)
        p["pillows"] = min(1, int(_draw(rng, d["pillows"])))
    if p["arm_style"] == "rolled" and aw < 0.15:  # a scroll needs 15 cm: too narrow a piece gets track arms
        p["arm_style"] = "track"
    p["arm_width"] = round(_clamp(aw, *_UPH_RANGES["arm_width"]), 3)
    p["fabric"], p["pillow_fabric"] = fabric, pillow_fabric
    if p["back_style"] == "loose":
        p["tufted"] = False
    return p


def _table(fam: dict[str, Any], category: str, dims: tuple[float, float, float], rng: random.Random) -> dict[str, Any]:
    d = fam["table"]
    w, dp, h = dims
    shape = _draw(rng, d["top_shape"])
    if shape == "round" and max(w, dp) / max(min(w, dp), 1e-6) > 1.15:
        shape = "oval"
    if shape == "oval" and max(w, dp) / max(min(w, dp), 1e-6) < 1.05:
        shape = "round"
    p: dict[str, Any] = {"kind": _TABLE_KIND[category], "top_shape": shape,
                         "width": round(_clamp(w, 0.35, 2.8), 3), "depth": round(_clamp(dp, 0.35, 1.3), 3),
                         "height": round(_clamp(h, 0.35, 0.8), 3), "leg_style": _draw(rng, d["leg_style"]),
                         "material": _draw(rng, d["material"])}
    for k in ("leg_material", "apron", "shelf", "top_thickness"):
        if k in d:
            p[k] = _draw(rng, d[k])
    if p["leg_style"] == "pedestal":
        p.pop("shelf", None)
    return p


def _chair(fam: dict[str, Any], dims: tuple[float, float, float], rng: random.Random, fabric: str) -> dict[str, Any]:
    d = fam["chair"]
    w, dp, h = dims
    p: dict[str, Any] = {k: _draw(rng, d[k]) for k in ("back_style", "seat_style", "leg_style", "material")}
    p["seat_width"] = round(_clamp(w - 0.01, 0.40, 0.54), 3)  # the frame adds ~1 cm to the seat
    p["seat_depth"] = round(_clamp(dp - 0.01, 0.40, 0.52), 3)
    p["seat_height"] = round(rng.uniform(0.44, 0.47), 3)
    p["back_height"] = round(_clamp(h, 0.77, 1.05), 3)
    p["stretchers"] = bool(d.get("stretchers", False))
    p["slats"] = {"spindles": rng.randint(5, 7), "ladder": rng.randint(3, 4)}.get(p["back_style"], rng.randint(2, 4))
    p["fabric"] = fabric
    return p


def piece(key: list[Any], category: str, dims: tuple[float, float, float] | list[float],
          style: str | None = None, variant: str | None = None, room: str | None = None) -> dict[str, Any]:
    """The generate_asset arguments of one piece: ``{family, params, seed}`` plus ``design`` (the
    family name). Deterministic in (key, category, room, variant); the copies of one category in
    one room get the same piece."""
    name = family_for(key, style, variant, room)
    fam = FAMILIES[name]
    gen = CATEGORY_FAMILY[category]
    dims3 = (float(dims[0]), float(dims[1]), float(dims[2]))
    # scene-level draws: the room's fabrics, shared by its sofa and its armchairs
    scene_rng = random.Random(_hash(key, room or "", name, "fabric"))
    fabric = scene_rng.choice(fam["fabrics"])
    pillow = scene_rng.choice([f for f in fam["pillow_fabrics"] if f != fabric] or fam["pillow_fabrics"])
    rng = random.Random(_hash(key, room or "", name, category))
    if gen in ("sofa", "armchair"):
        params = _upholstered(fam, gen, dims3, rng, fabric, pillow)
    elif gen == "table":
        params = _table(fam, category, dims3, rng)
    else:
        params = _chair(fam, dims3, rng, fabric)
    return {"family": gen, "params": params, "seed": _hash(key, room or "", name, category, "seed") % (2**31),
            "design": name}


def generated_categories() -> list[str]:
    return sorted(CATEGORY_FAMILY)


#: run_code after generate_asset: scale each listed root so its meshes fill ``dims`` exactly,
#: base centred on its placement point (the generator sizes by width/depth/back height and a
#: cushion or a rolled arm can add a few cm)
FIT_CODE = """
import bpy, json
from mathutils import Vector
fitted = {{}}
bpy.context.view_layer.update()
for ent in json.loads({payload!r}):
    root = bpy.data.objects.get(ent["name"])
    if root is None:
        continue
    inv = root.matrix_world.inverted()
    pts = [inv @ (o.matrix_world @ Vector(c)) for o in [root, *root.children_recursive] if o.type == "MESH"
           for c in o.bound_box]
    if not pts:
        continue
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    size = hi - lo
    s = [ent["dims"][i] / size[i] if size[i] > 1e-6 else 1.0 for i in range(3)]
    shift = Vector((-(lo.x + hi.x) / 2 * s[0], -(lo.y + hi.y) / 2 * s[1], -lo.z * s[2]))
    shift = Vector((shift.x * root.scale.x, shift.y * root.scale.y, shift.z * root.scale.z))
    root.location = root.location + root.rotation_euler.to_matrix() @ shift
    root.scale = (root.scale.x * s[0], root.scale.y * s[1], root.scale.z * s[2])
    fitted[ent["name"]] = [round(v, 3) for v in s]
result = {{"fitted": fitted}}
"""


def fit_command(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """The run_code command fitting ``[{name, dims}]`` roots to their exact bbox."""
    return {"type": "run_code", "params": {"code": FIT_CODE.format(payload=json.dumps(entries))}}
