"""Outdoor terrain: a real ground (sand, grass, soil, gravel, rock), a water body with a sloping
shoreline, and where objects may stand relative to the water.

The weak-model bench (beach_cove, 2026-10-17) showed what a model writes for a beach: an object
of category ``sandy_beach`` / ``ground`` / ``water``, ``floor_material: sand`` (built as glossy
paint), relations ``rocks at_edge_of water`` and ``boardwalk leads_to water``. This module reads
those forms (``lift_terrain``), keeps them in ``setting.ground`` / ``setting.water`` and
relations whose ``b`` is ``"water"``, and provides:

* ``Terrain``: the shoreline geometry in the layout frame (x in [0, W], y in [0, D]): signed
  distance to the waterline, the height of the ground anywhere (flat land where objects stand,
  a beach sloping into the water, deepening sea floor, dunes / hills beyond the setting);
* the layout hooks (``violation``, ``cost``, ``propose``): land objects stay on dry flat land,
  ``at_edge_of`` / ``in`` / ``near`` / ``back_of`` / ``facing`` / ``leads_to`` water;
* ``settle``: objects on the slope stand on the ground (rocks bedded in), a boardwalk runs from
  the land to over the water;
* the Blender commands: one terrain mesh (fine grid where the scene is, coarse to the horizon)
  with a procedural ground shader (wet band at the waterline, ripples on sand) and a water
  body (refraction, depth absorption turquoise -> deep blue, small waves).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_db_scene.layout import Placement
    from ai_db_scene.spec import SceneSpec

GROUND_KINDS = ("sand", "grass", "soil", "gravel", "rock")
WATER_KINDS = ("sea", "lagoon", "lake", "pond")
SIDES = ("north", "south", "east", "west")
WATER = "water"  # relation target: {a: rocks, rel: at_edge_of, b: water}
#: relations to the water (b = "water"); facing = the front toward the water
TERRAIN_RELS = ("at_edge_of", "near", "in", "leads_to", "back_of", "facing")
#: categories that face the water unless a relation orients them (loungers, a parasol's tilt)
FACES_WATER = ("sun_lounger", "parasol", "bench", "chair", "armchair", "picnic_table", "gazebo")
#: categories that belong on / over the water when asked to lead there
OVER_WATER = ("boardwalk",)
#: ``{a: boardwalk, rel: starts_at, b: bushes}``: the boardwalk leads to the water from ``b`` (its
#: land end at b). qwen aidb7 beach_cove wrote ``boardwalk1 connects bush1``; read as ``beside``
#: the solver dropped it (a jetty cannot both reach the water and stand beside the back bushes).
LINK_REL = "starts_at"
#: what models write for that link (a over the water, b an object on the land)
LINK_WORDS = ("connects", "connects_to", "connected_to", "connect", "from", "joins", "join", "links", "link",
              "linking", "links_to", "starts_at", "starts_from", "start_at", "runs_from", "leads_from", "between",
              "comes_from", "begins_at")
#: trees frame the scene: at the sides or the back, not on the beach strip in front of the group
FRAMING = ("palm_tree", "tree")
_FRAME_COST = 0.6
_FRAME_SIDES = 0.65  # beyond this share of the half-length from the shore's middle a tree frames it

#: what models write for a ground -> its kind (category, floor_material or an object's material)
GROUND_ALIASES: dict[str, str] = {
    **{w: "sand" for w in ("sand", "sandy", "sandy_beach", "beach", "beach_sand", "sand_beach", "dune", "dunes",
                           "sand_dunes", "desert", "desert_sand", "shore", "seashore", "beach_ground", "sand_ground",
                           "sandy_ground", "beach_floor", "sand_floor", "white_sand", "golden_sand")},
    **{w: "grass" for w in ("grass", "lawn", "meadow", "turf", "grassland", "field", "grass_ground", "green_lawn",
                            "grassy_ground", "lawn_grass", "park_lawn", "garden_lawn")},
    **{w: "soil" for w in ("soil", "dirt", "earth", "mud", "mud_ground", "dirt_ground", "forest_floor", "clay_ground")},
    **{w: "gravel" for w in ("gravel", "pebbles", "pebble", "shingle", "pebble_beach", "gravel_path", "scree")},
    **{w: "rock" for w in ("rock_ground", "rocky_ground", "bedrock", "rocky", "stone_ground", "cliff", "rock_floor")},
}
#: generic ground words (no kind of their own): the kind comes from the material or the brief
_GROUND_WORDS = ("ground", "terrain", "landscape", "land", "floor", "beach_area", "island")
#: what models write for a water body -> its kind
WATER_ALIASES: dict[str, str] = {
    **{w: "sea" for w in ("water", "sea", "ocean", "seawater", "sea_water", "ocean_water", "bay", "surf", "waves",
                          "water_body", "waterbody", "shoreline_water", "turquoise_water", "calm_water",
                          "clear_water", "blue_water", "beach_water", "cove_water", "coast", "seaside")},
    **{w: "lagoon" for w in ("lagoon", "lagoon_water", "tropical_water", "shallow_water", "reef")},
    **{w: "lake" for w in ("lake", "lake_water", "loch", "reservoir", "river", "stream")},
    **{w: "pond" for w in ("pond", "pond_water", "pool", "oasis_pool", "oasis_water", "koi_pond", "puddle", "spring")},
}
#: category names models use for the new vegetation / beach categories
CATEGORY_ALIASES: dict[str, str] = {
    **{w: "palm_tree" for w in ("palm", "palms", "palm_trees", "coconut_palm", "coconut_tree", "date_palm",
                                "tropical_tree", "palmtree")},
    **{w: "fern" for w in ("ferns", "tree_fern", "bracken")},
    **{w: "tropical_bush" for w in ("tropical_bushes", "tropical_plant", "tropical_shrub", "dense_bush",
                                    "dense_green_bushes", "jungle_plant", "banana_plant", "elephant_ear", "monstera",
                                    "philodendron", "undergrowth", "tropical_foliage")},
    **{w: "sun_lounger" for w in ("lounger", "loungers", "sun_loungers", "sunlounger", "deckchair", "deck_chair",
                                  "sunbed", "sun_bed", "beach_chair", "beach_lounger", "chaise_longue",
                                  "chaise_lounge", "pool_lounger", "lounge_chair")},
    **{w: "parasol" for w in ("umbrella", "beach_umbrella", "sun_umbrella", "patio_umbrella", "parasols",
                              "sunshade", "sun_shade")},
    **{w: "boardwalk" for w in ("wooden_boardwalk", "walkway", "wooden_walkway", "jetty", "pier", "dock",
                                "wooden_deck", "deck", "boardwalks", "footbridge", "wooden_path", "plank_path",
                                "beach_path", "pathway", "path")},
    **{w: "rock" for w in ("rocks", "beach_rock", "beach_rocks", "boulder", "boulders", "stone", "stones")},
}
#: relation names models use toward the water -> a terrain relation
REL_ALIASES: dict[str, str] = {
    **{w: "at_edge_of" for w in ("at_edge_of", "at_edge", "edge_of", "at", "at_the", "at_the_edge_of", "along", "along_edge_of",
                                 "on_shore", "at_shore", "shore", "on_edge_of", "bordering", "by_edge_of",
                                 "at_waters_edge", "at_water_edge", "on_the_edge_of", "lining", "against_wall",
                                 "against")},
    **{w: "near" for w in ("near", "close_to", "next_to", "beside", "by", "nearby", "near_to", "in_front_of",
                           "around", "overlooking", "around_edge", "adjacent_to", "close")},
    **{w: "in" for w in ("in", "inside", "within", "in_the", "partly_in", "in_water", "on", "on_top_of", "floating_on",
                         "centered_in")},
    **{w: "leads_to" for w in ("leads_to", "to", "towards", "toward", "into", "connects", "connects_to", "runs_to",
                               "reaches", "extends_to", "extends_into", "goes_to", "over", "across", "from",
                               "leading_to", "ending_at", "ends_at", "joins")},
    **{w: "back_of" for w in ("back_of", "behind", "at_back_of", "along_back_of", "far_from", "away_from",
                              "at_the_back_of", "rear_of", "backing")},
    **{w: "facing" for w in ("facing", "faces", "looking_at", "facing_toward", "facing_towards", "oriented_to",
                             "view_of", "overlooks")},
}

#: per ground kind: (land band sloping to the waterline m, drop over that band m)
_SHORE = {"sand": (3.5, 0.3), "grass": (1.4, 0.25), "soil": (1.4, 0.25), "gravel": (2.5, 0.3), "rock": (1.0, 0.35)}
#: per water kind: (max depth m, sea-floor slope at the waterline, absorption colour, density per m,
#: surface tint)
#: per water kind, what the eye sees in it from the beach: (body colour by the shore, body colour
#: offshore, body share by the shore, offshore, specular level, shallow band m). The transmissive
#: surface alone showed mostly the sky at eye level (qwen aidb7 beach: grey-blue, no turquoise):
#: an opaque water-body layer gives the shallows their colour and a lower specular level cuts the
#: grazing-angle sky reflection; the sea stays a deeper blue.
WATER_LOOK = {
    "sea": ((0.03, 0.34, 0.4), (0.008, 0.07, 0.19), 0.45, 0.3, 0.4, 10.0),
    "lagoon": ((0.08, 0.62, 0.55), (0.03, 0.36, 0.45), 0.75, 0.55, 0.2, 24.0),
    "lake": ((0.06, 0.2, 0.16), (0.01, 0.05, 0.07), 0.4, 0.3, 0.4, 6.0),
    "pond": ((0.08, 0.15, 0.07), (0.02, 0.05, 0.03), 0.45, 0.35, 0.4, 3.0),
}
_WATER = {
    "sea": (5.0, 0.11, (0.3, 0.8, 0.85), 0.75, (0.75, 0.97, 0.95)),
    "lagoon": (2.5, 0.07, (0.4, 0.9, 0.88), 0.55, (0.8, 1.0, 0.97)),
    "lake": (3.5, 0.18, (0.32, 0.6, 0.55), 0.9, (0.9, 0.97, 0.95)),
    "pond": (1.2, 0.3, (0.42, 0.58, 0.38), 1.6, (0.9, 0.96, 0.92)),
}
#: land objects stand where the shore is still dry: above the wet band, on the gentle top of the
#: slope (they are settled onto it), this share of the slope band up from the waterline
_DRY_SHARE = 0.55
_EDGE_BAND_M = (-1.6, 0.8)  # at_edge_of: object centre this far from the waterline (- = land side)
_NEAR_M = 4.0  # near: within this of the top of the beach slope
_JETTY_OVER_M = (0.8, 4.5)  # leads_to: the water end this far out over the water
_HORIZON_M = 400.0  # terrain and water reach this far from the scene (to the horizon)
_FINE_MARGIN_M = 8.0  # fine grid this far beyond the setting
_BEDDED = {"rock": 0.15}  # share of an object's height sunk into the ground (rocks bed in)


@dataclass(frozen=True)
class Water:
    """A water body: on one ``side`` of the setting (a sea, a lagoon, a lake shore; the shoreline
    bows inland into a cove) or, with ``side`` None, an inland ellipse ``region`` (cx, cy, rx, ry)
    in the setting frame (a pond, an oasis pool)."""

    kind: str = "sea"
    side: str | None = "north"
    fraction: float = 0.3  # share of the setting's depth (along the side's axis) under water
    region: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.side is not None:
            out.update(side=self.side, fraction=self.fraction)
        if self.region is not None:
            out["region"] = list(self.region)
        return out


def water_schema() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "description": ("outdoor only: a water body with a sloping shore (beach into the sea, lake shore, "
                            "pond). Relations then take b: 'water' with rel at_edge_of (rocks at the water's "
                            "edge), near, in, leads_to (a boardwalk / jetty), back_of (bushes along the back), "
                            "facing. Example: {kind: 'sea', side: 'north'}; a pond: {kind: 'pond'}"),
            "properties": {
                "kind": {"enum": list(WATER_KINDS), "description": "sea (default), lagoon (turquoise, shallow), "
                                                                    "lake, pond (inland: centred unless region)"},
                "side": {"enum": list(SIDES), "description": "the setting side the water is on (sea/lagoon/lake)"},
                "fraction": {"type": "number", "minimum": 0.1, "maximum": 0.7,
                             "description": "share of the setting under water (default 0.3)"},
                "region": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                           "description": "pond/lake: [cx, cy, rx, ry] ellipse in meters inside bounds_m"}}}


def _word(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(v).strip().lower()).strip("_") if isinstance(v, str) else ""


def ground_kind(value: Any) -> str | None:
    """The ground kind a word names (``sandy_beach`` -> sand, ``lawn`` -> grass), else None."""
    w = _word(value)
    if not w:
        return None
    if w in GROUND_KINDS:
        return w
    if w in GROUND_ALIASES:
        return GROUND_ALIASES[w]
    base = w.split(":")[0]
    return GROUND_ALIASES.get(base)


def water_kind(value: Any) -> str | None:
    """The water kind a word names (``turquoise_water`` -> sea, ``pond`` -> pond), else None."""
    w = _word(value)
    if not w:
        return None
    if w in WATER_KINDS:
        return w
    if w in WATER_ALIASES:
        return WATER_ALIASES[w]
    toks = w.split("_")
    for t in reversed(toks):  # crystal_clear_lagoon, calm_turquoise_sea
        if t in WATER_ALIASES and t != "water":
            return WATER_ALIASES[t]
    return "sea" if "water" in toks else None


def parse_ground(raw: Any, errors: list[str]) -> str | None:
    if raw is None:
        return None
    kind = ground_kind(raw)
    if kind is None:
        errors.append(f"setting.ground: {raw!r} is not allowed; allowed values: {list(GROUND_KINDS)}")
    return kind


def parse_water(raw: Any, bounds: tuple[float, float, float] | None, errors: list[str],
                notes: list[str] | None = None) -> Water | None:
    """``setting.water`` -> Water. Takes a kind or side word, or {kind, side, fraction, region};
    a ``level``/``depth`` the model adds is ignored (the shore decides it), with a note."""
    if raw is None or raw is False:
        return None
    if raw is True:
        raw = {}
    if isinstance(raw, str):
        w = _word(raw)
        raw = {"side": w} if w in SIDES else {"kind": raw}
    if not isinstance(raw, dict):
        errors.append(f"setting.water: expected {{kind, side}}, got {raw!r}")
        return None
    extra = [k for k in raw if k not in ("kind", "side", "fraction", "region")]
    if extra and notes is not None:
        notes.append(f"setting.water: {extra} ignored (the shore sets level and depth)")
    kind = water_kind(raw.get("kind", "sea"))
    if kind is None:
        errors.append(f"setting.water.kind: {raw.get('kind')!r} is not allowed; allowed values: {list(WATER_KINDS)}")
        return None
    region = raw.get("region")
    side = raw.get("side")
    side = _word(side) if side is not None else None
    if side is not None and side not in SIDES:
        errors.append(f"setting.water.side: {raw.get('side')!r} is not allowed; allowed values: {list(SIDES)}")
        return None
    frac = raw.get("fraction", 0.3)
    if not isinstance(frac, (int, float)) or isinstance(frac, bool) or not 0.1 <= frac <= 0.7:
        errors.append(f"setting.water.fraction: expected a number in [0.1, 0.7], got {frac!r}")
        return None
    reg: tuple[float, float, float, float] | None = None
    if region is not None:
        if not (isinstance(region, list) and len(region) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in region)
                and region[2] > 0 and region[3] > 0):
            errors.append(f"setting.water.region: expected [cx, cy, rx, ry] with rx, ry > 0, got {region!r}")
            return None
        reg = (float(region[0]), float(region[1]), float(region[2]), float(region[3]))
        side = None
    elif side is None and kind == "pond":
        if bounds is not None:
            bw, bd = bounds[0], bounds[1]
            reg = (bw / 2, bd / 2, round(0.22 * bw, 3), round(0.22 * bd, 3))
    elif side is None:
        side = "north"
    return Water(kind, side, float(frac), reg)


# ------------------------------------------------------------------------------------------------
# lifting the forms models write


def lift_terrain(data: Any) -> tuple[Any, list[str]]:
    """Rewrite ground / water objects, ground materials and relations to them into
    ``setting.ground`` / ``setting.water`` and ``b: "water"`` relations; map the beach and
    vegetation category names models use. Returns (data, notes); never raises."""
    if not isinstance(data, dict):
        return data, []
    try:
        return _lift(json.loads(json.dumps(data)))
    except Exception:  # noqa: BLE001 - malformed input: validation reports it properly
        return data, []


def _lift(d: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    setting: dict[str, Any] | None = d["setting"] if isinstance(d.get("setting"), dict) else None
    objs: list[Any] = d["objects"] if isinstance(d.get("objects"), list) else []
    from ai_db_scene.priors import load_priors

    priors = load_priors()
    for i, o in enumerate(objs):
        if isinstance(o, dict) and isinstance(o.get("category"), str) and o["category"] not in priors:
            alias = CATEGORY_ALIASES.get(_word(o["category"]))
            if alias and alias in priors:
                notes.append(f"objects[{i}].category: '{o['category']}' read as '{alias}'")
                o["category"] = alias
    if setting is None or setting.get("kind") != "outdoor":
        return d, notes
    style = _word(setting.get("style"))
    tropical = any(w in style for w in ("tropical", "jungle", "island", "caribbean", "beach"))
    for i, o in enumerate(objs):  # a tree on a palm beach is a palm, a fern is a fern
        if not isinstance(o, dict):
            continue
        # the id names the plant; the material only colours it (qwen aidb7: bush1 with material
        # green_fern became a fern, then the brief check asked for the bushes it had)
        cat, hint, mat = o.get("category"), _word(o.get("id")), _word(o.get("material"))
        new = None
        if cat == "tree" and ("palm" in f"{hint} {mat}" or "coconut" in hint or tropical):
            new = "palm_tree"
        elif cat in ("bush", "potted_plant") and "fern" in hint:
            new = "fern"
        elif cat == "bush" and ("tropical" in hint or "jungle" in hint or tropical):
            new = "tropical_bush"
        if new:
            notes.append(f"objects[{i}].category: '{cat}' read as '{new}' ({o.get('id')}"
                         f"{', a tropical setting' if tropical else ''})")
            o["category"] = new
    ground_ids: set[str] = set()
    water_ids: set[str] = set()
    keep = []
    for i, o in enumerate(objs):
        cat = o.get("category") if isinstance(o, dict) else None
        if not isinstance(cat, str) or cat in priors:
            keep.append(o)
            continue
        wk, gk = water_kind(cat), ground_kind(cat)
        if gk is None and _word(cat) in _GROUND_WORDS:
            gk = ground_kind(o.get("material")) or ground_kind(setting.get("floor_material")) or "_any"
        if wk is not None and (gk is None or "water" in _word(cat)):
            water_ids.add(str(o.get("id")))
            if setting.get("water") is None:
                setting["water"] = {"kind": wk}
            notes.append(f"objects[{i}]: '{o.get('id')}' ({cat}) is the setting's water: setting.water "
                         f"{{kind: {setting['water'].get('kind', wk) if isinstance(setting['water'], dict) else wk}}}"
                         "; relations to it use b: 'water'")
            continue
        if gk is not None:
            ground_ids.add(str(o.get("id")))
            if gk != "_any" and setting.get("ground") is None:
                setting["ground"] = gk
            notes.append(f"objects[{i}]: '{o.get('id')}' ({cat}) is the ground: setting.ground "
                         f"{setting.get('ground') or 'by default'}")
            continue
        keep.append(o)
    if len(keep) != len(objs):
        d["objects"] = keep
    fm = setting.get("floor_material")
    if fm is not None and setting.get("ground") is None and ground_kind(fm):
        setting["ground"] = ground_kind(fm)
        setting.pop("floor_material")
        notes.append(f"setting.floor_material: {fm!r} read as setting.ground '{setting['ground']}'")
    elif fm is not None and setting.get("ground") is not None and ground_kind(fm):
        setting.pop("floor_material")
    if isinstance(setting.get("water"), str) and ground_kind(setting["water"]) is None:
        pass  # parse_water reads a kind / side word
    shore_words = (*WATER_ALIASES, "shore", "shoreline", "waterline", "water_edge", "waters_edge", "sea_shore",
                   "water_line", "edge_of_water", "beach_edge", "surf_line")
    rels_raw: list[Any] = d["relations"] if isinstance(d.get("relations"), list) else []
    if setting.get("water") in (None, False) and any(
            isinstance(r, dict) and _word(r.get("b")) in shore_words for r in rels_raw):
        # a relation to the water edge names the water the spec forgot (qwen beach_cove: 'rocks at
        # water_edge' with only floor_material sand): add it rather than refuse the relation
        setting["water"] = {"kind": "sea"}
        notes.append("setting.water: {kind: sea} added (relations name the water's edge)")
    has_water = setting.get("water") not in (None, False)
    cats = {o.get("id"): o.get("category") for o in d.get("objects", []) if isinstance(o, dict)}
    rels = d.get("relations")
    if isinstance(rels, list):
        out = []
        for i, r in enumerate(rels):
            if not isinstance(r, dict):
                out.append(r)
                continue
            a, b, rel = r.get("a"), r.get("b"), _word(r.get("rel"))
            if a in ground_ids or a in water_ids:
                notes.append(f"relations[{i}]: dropped (its subject '{a}' is the ground / water itself)")
                continue
            if has_water and cats.get(a) in OVER_WATER and b in cats and b != a and rel in LINK_WORDS:
                # 'boardwalk connects bushes': it runs from the bushes to the water
                out.append({"a": a, "rel": LINK_REL, "b": b})
                if r.get("rel") == LINK_REL:
                    continue
                lead = any(isinstance(x, dict) and x.get("a") == a and (x.get("b") == WATER or _word(x.get("b")) in
                                                                        shore_words) for x in rels)
                if not lead:
                    out.append({"a": a, "rel": "leads_to", "b": WATER})
                notes.append(f"relations[{i}]: '{a} {r.get('rel')} {b}' read as '{a} {LINK_REL} {b}'"
                             + ("" if lead else f" + '{a} leads_to water'") + f": it runs from {b} to the water")
                continue
            side_rel = _wall_word_rel(b, setting.get("water")) if has_water and b not in cats and rel != "facing" else None
            if side_rel is not None:
                notes.append(f"relations[{i}]: '{a} {r.get('rel')} {b}' read as '{a} {side_rel} water' (outdoors the "
                             f"{'back' if side_rel == 'back_of' else 'front'} is "
                             f"{'away from' if side_rel == 'back_of' else 'toward'} the water)")
                out.append({"a": a, "rel": side_rel, "b": WATER})
                continue
            to_water = b in water_ids or (has_water and _word(b) in shore_words)
            to_ground = b in ground_ids or (_word(b) in GROUND_ALIASES or _word(b) in _GROUND_WORDS) and not to_water
            if to_ground and has_water and (REL_ALIASES.get(rel) == "back_of" or cats.get(a) in OVER_WATER):
                to_water = True  # behind the beach = away from the water; a boardwalk across it leads to it
            if to_water:
                new = "leads_to" if cats.get(a) in OVER_WATER else REL_ALIASES.get(rel)
                if new is None:
                    notes.append(f"relations[{i}]: dropped '{a} {r.get('rel')} {b}' (toward the water allowed: "
                                 f"{list(TERRAIN_RELS)})")
                    continue
                if new != r.get("rel") or b != WATER:
                    notes.append(f"relations[{i}]: '{a} {r.get('rel')} {b}' read as '{a} {new} water'")
                out.append({"a": a, "rel": new, "b": WATER})
                continue
            if to_ground and b not in cats:
                notes.append(f"relations[{i}]: dropped '{a} {r.get('rel')} {b}' (everything stands on the ground)")
                continue
            out.append(r)
        d["relations"] = out
    if has_water:
        _back_plants(d, notes)
    return d, notes


#: dense low plants of a beach / lakeside: along the back, away from the water, unless placed
BACK_PLANTS = ("tropical_bush", "bush", "fern", "hedge")


def _back_plants(d: dict[str, Any], notes: list[str]) -> None:
    """Bushes and ferns with no place of their own (or only ``beside`` something) go along the back
    of the shore (``back_of water``). Bench beach_cove (aidb7): lightning's ten bushes and four ferns
    stood scattered over the sand; qwen's stood beside the boardwalk at the water."""
    rels: list[Any] = d["relations"] if isinstance(d.get("relations"), list) else []
    for o in d.get("objects") or []:
        if not isinstance(o, dict) or o.get("category") not in BACK_PLANTS:
            continue
        oid = o.get("id")
        mine = [r for r in rels if isinstance(r, dict) and r.get("a") == oid]
        if any(r.get("rel") != "beside" for r in mine):
            continue
        rels = [r for r in rels if not any(r is m for m in mine)] + [{"a": oid, "rel": "back_of", "b": WATER}]
        dropped = [f"{m.get('a')} beside {m.get('b')}" for m in mine]
        notes.append(f"'{oid}' ({o['category']}) goes along the back, away from the water: added '{oid} back_of water'"
                     + (f" (dropped {dropped})" if dropped else ""))
    d["relations"] = rels


#: words around a side in a wall name (back_wall, the north side, rear edge of the beach)
_SIDE_FILLER = ("wall", "walls", "side", "the", "edge", "of", "area", "border", "boundary", "perimeter", "line", "end",
                "part", "beach", "scene", "setting", "cove")
_SIDE_WORD = {"back": "back", "rear": "back", "behind": "back", "front": "front",
              "north": "north", "south": "south", "east": "east", "west": "west"}
_OPPOSITE_SIDE = {"north": "south", "south": "north", "east": "west", "west": "east"}


def _wall_word_rel(b: Any, water: Any) -> str | None:
    """The relation to the water a wall word means outdoors, else None: ``back_wall`` / ``rear`` /
    the side opposite the water -> back_of (along the back, away from the water); ``front`` / the
    water's own side -> near. A side across the shore (east with the sea north) means nothing here.
    Bench beach_cove: "ferns along the back" written ``along back_wall`` was dropped (no walls
    outdoors)."""
    words = [w.removesuffix("ern") for w in _word(b).split("_") if w and w not in _SIDE_FILLER]
    sides = {_SIDE_WORD[w] for w in words if w in _SIDE_WORD}
    if len(sides) != 1 or len(words) != 1:
        return None
    side = sides.pop()
    if side in ("back", "front"):
        return "back_of" if side == "back" else "near"
    wside: Any = None
    if isinstance(water, str):
        wside = _word(water) if _word(water) in SIDES else ("north" if water_kind(water) not in (None, "pond") else None)
    elif isinstance(water, dict):
        wside = _word(water.get("side")) or None
        if wside is None and water.get("region") is None and water_kind(water.get("kind", "sea")) != "pond":
            wside = "north"  # parse_water's default side
    elif water is True:
        wside = "north"
    if wside not in SIDES:
        return None
    return "near" if side == wside else "back_of" if side == _OPPOSITE_SIDE[wside] else None


def hold_water_relations(data: Any) -> tuple[Any, list[dict[str, Any]]]:
    """(data without its b: "water" relations and its boardwalk links, those relations, each once):
    the general relation normalisation knows no water and would drop them; validation gets them
    back after it."""
    if not isinstance(data, dict) or not isinstance(data.get("relations"), list):
        return data, []
    held: list[dict[str, Any]] = []
    rest = []
    for r in data["relations"]:
        if isinstance(r, dict) and (r.get("b") == WATER or r.get("rel") == LINK_REL):
            if r not in held:
                held.append(r)
        else:
            rest.append(r)
    return ({**data, "relations": rest} if held else data), held


def release_water_relations(data: Any, held: list[dict[str, Any]]) -> Any:
    if not held or not isinstance(data, dict):
        return data
    return {**data, "relations": [*(data.get("relations") or []), *held]}


def water_relation(rel: Any, category: str | None = None) -> str | None:
    """The terrain relation a relation name toward the water means (after normalisation), else None."""
    if category in OVER_WATER:
        return "leads_to"
    return REL_ALIASES.get(_word(rel))


# ------------------------------------------------------------------------------------------------
# brief vocabulary


#: brief words -> (ground, water dict or None); the first match wins
_BRIEF = (
    (("oasis",), "sand", {"kind": "pond"}),
    (("lagoon", "atoll"), "sand", {"kind": "lagoon", "side": "north"}),
    (("pond", "koi"), "grass", {"kind": "pond"}),
    (("lake", "lakeside", "lakeshore", "loch"), "grass", {"kind": "lake", "side": "north"}),
    # calm, shallow, turquoise: a cove, a tropical beach (beach_cove: "a tropical beach cove ... calm
    # turquoise water" was built as the open sea)
    (("turquoise", "cove", "tropical beach", "tropical island", "tropical sea", "tropical water", "tropical shore"),
     "sand", {"kind": "lagoon", "side": "north"}),
    (("beach", "seaside", "sea shore", "seashore", "coast", "shore", "bay", "ocean", "sea"), "sand",
     {"kind": "sea", "side": "north"}),
    (("desert", "dune", "dunes"), "sand", None),
    (("tropical", "jungle", "rainforest", "garden", "park", "lawn", "meadow", "backyard", "playground"), "grass", None),
    (("quarry", "canyon", "rocky", "mountain"), "rock", None),
    (("gravel",), "gravel", None),
)


def brief_terrain(brief: str) -> tuple[str | None, dict[str, Any] | None]:
    """(ground, water) the brief's words name: 'beach' -> sand + sea, 'pond' -> grass + pond, ..."""
    low = " " + re.sub(r"[^a-z]+", " ", (brief or "").lower()) + " "
    for words, ground, water in _BRIEF:
        if any(f" {w} " in low for w in words):
            return ground, (dict(water) if water else None)
    return None, None


#: water words that name the sea itself (not just "water"): the model chose it, the brief does not overrule
_SEA_WORDS = ("sea", "ocean", "seawater", "bay", "coast", "seaside")


def _generic_water(word: Any) -> bool:
    """True when a word names water but no kind of its own (water, turquoise_water, calm_water)."""
    return water_kind(word) == "sea" and not set(_word(word).split("_")) & set(_SEA_WORDS)


def water_from_brief(data: Any, brief: Any) -> tuple[Any, str | None]:
    """(spec, note): an outdoor spec whose water the model left implicit (no setting.water, or one
    without a kind; a water object or a relation to the water's edge with only a generic word)
    gets the kind of water its brief names: turquoise / cove / tropical beach -> lagoon, pond,
    lake, sea. Done before validation, so a relation to the water's edge or a wall word
    (``ferns along back_wall``) finds the water. A water kind the model wrote stays."""
    if not isinstance(data, dict) or not isinstance(brief, str):
        return data, None
    st = data.get("setting")
    if not isinstance(st, dict) or st.get("kind") != "outdoor":
        return data, None
    _ground, water = brief_terrain(brief)
    if water is None:
        return data, None
    have = st.get("water")
    if have not in (None, True) and not (isinstance(have, dict) and "kind" not in have):
        return data, None
    raw_objs = data.get("objects")
    objs: list[Any] = raw_objs if isinstance(raw_objs, list) else []
    for o in objs:  # a water object with a kind of its own (pond, lake, ocean) decides
        cat = o.get("category") if isinstance(o, dict) else None
        if isinstance(cat, str) and water_kind(cat) is not None and ground_kind(cat) is None and not _generic_water(cat):
            return data, None
    new = {**water, **(have if isinstance(have, dict) else {})}
    return ({**data, "setting": {**st, "water": new}},
            f"setting.water: {{kind: {new['kind']}}} from the brief (the spec named no kind of water)")


#: material words that pave rather than grow: a park's floor_material of this kind is its paving
_PAVING_WORDS = ("concrete", "stone", "paving", "pavement", "asphalt", "brick", "tile", "tiles", "marble", "granite",
                 "cobblestone", "cobble", "flagstone", "slate", "terrazzo", "tarmac", "blacktop", "pavers", "paver")
#: paving too dark for a park's plaza (a road surface, not a square round a fountain)
_DARK_PAVING = ("asphalt", "tarmac", "blacktop", "bitumen", "black", "dark", "charcoal", "anthracite", "basalt",
                "slate")


def paving(material: Any) -> bool:
    """Is a floor material a paving (concrete, stone, brick, ...)?"""
    return isinstance(material, str) and material.split(":")[0].strip().lower() in _PAVING_WORDS


def plaza_material(material: Any) -> tuple[str, str | None]:
    """(paving of a park's plaza, note or None): the spec's floor material when it is a light
    paving; asphalt or a very dark one becomes light stone pavers (the aidb7 park's plaza read as a
    car park) with a note; no paving: light stone."""
    if not paving(material):
        return PLAZA_MATERIAL, None
    words = re.split(r"[^a-z]+", str(material).lower())
    m = re.search(r"#([0-9a-fA-F]{6})", str(material))
    lum = None
    if m:
        r, g, b = (int(m.group(1)[i:i + 2], 16) / 255 for i in (0, 2, 4))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if any(w in _DARK_PAVING for w in words) or (lum is not None and lum < 0.3):
        return PLAZA_MATERIAL, (f"setting.floor_material {material!r} is too dark for a park's plaza: paved in light "
                                f"stone ({PLAZA_MATERIAL}); name a light paving to choose another")
    return str(material), None


def apply_brief_defaults(spec: SceneSpec, brief: str | None) -> tuple[SceneSpec, list[str]]:
    """An outdoor spec without ground / water gets the ones its brief names, with a note."""
    st = spec.setting
    if not brief or st.kind != "outdoor" or (st.ground is not None and st.water is not None):
        return spec, []
    ground, water = brief_terrain(brief)
    notes = []
    new_ground, new_water = st.ground, st.water
    if st.ground is None and ground is not None and st.floor_material is None:
        new_ground = ground
        notes.append(f"setting.ground: '{ground}' from the brief")
    elif st.ground is None and ground in PLAZA_GROUNDS and paving(st.floor_material):
        # a park / garden whose model wrote floor_material concrete: a flat grey slab (bench
        # park_corner, qwen aidb7). The lawn is the ground; the paving goes round the fountain.
        new_ground = ground
        notes.append(f"setting.ground: '{ground}' from the brief; floor_material {st.floor_material!r} paves the "
                     f"plaza round the {' / '.join(PLAZA_CENTRES)} (if any)")
        dark = plaza_material(st.floor_material)[1]
        if dark:
            notes.append(dark)
    if st.water is None and water is not None:
        new_water = parse_water(water, st.bounds_m, [])
        notes.append(f"setting.water: {new_water.to_dict() if new_water else water} from the brief")
    if not notes:
        return spec, []
    return replace(spec, setting=replace(st, ground=new_ground or st.ground, water=new_water)), notes


# ------------------------------------------------------------------------------------------------
# geometry (layout frame: x in [0, W], y in [0, D])


def _hash01(ix: int, iy: int, seed: int) -> float:
    h = (ix * 374761393 + iy * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def _vnoise(x: float, y: float, seed: int) -> float:
    """Smooth value noise in [0, 1] at (x, y) (unit cells)."""
    ix, iy = math.floor(x), math.floor(y)
    fx, fy = x - ix, y - iy
    sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash01(ix, iy, seed), _hash01(ix + 1, iy, seed)
    c, e = _hash01(ix, iy + 1, seed), _hash01(ix + 1, iy + 1, seed)
    return (a + (b - a) * sx) + ((c + (e - c) * sx) - (a + (b - a) * sx)) * sy


class Terrain:
    """The ground of an outdoor setting with ``ground`` and/or ``water``."""

    def __init__(self, spec: SceneSpec) -> None:
        st = spec.setting
        self.W, self.D = st.bounds_m[0], st.bounds_m[1]
        self.ground = st.ground or ("sand" if st.water is not None and st.water.kind in ("sea", "lagoon") else "grass")
        self.water = st.water
        self.band, self.drop = _SHORE[self.ground]
        if self.water is not None and self.water.side is None:
            self.band = min(self.band, 1.6)
        self.level = -self.drop
        kind = self.water.kind if self.water is not None else "sea"
        self.depth, self.floor_slope = _WATER[kind][0], _WATER[kind][1]
        self.seed = 7
        #: per object id: [(rel, label)] toward the water
        self.rels: dict[str, list[str]] = {}
        for r in spec.relations:
            if r.b == WATER:
                self.rels.setdefault(r.a, []).append(r.rel)
        cats = {o.id: o.category for o in spec.objects}
        oriented = {r.a for r in spec.relations if r.b != WATER and r.rel in ("facing", "against_wall", "in_front_of",
                                                                              "around")}
        placed = {r.a for r in spec.relations if r.b != WATER and r.rel not in ("facing", LINK_REL)}
        #: object id -> the boardwalk that starts at it (``starts_at``)
        self.start_of: dict[str, str] = {r.b: r.a for r in spec.relations if r.rel == LINK_REL}
        for oid, cat in cats.items():
            if cat in OVER_WATER and self.water is not None and oid not in placed and not self.rels.get(oid):
                self.rels[oid] = ["leads_to"]  # a boardwalk on a beach runs out to the water
            if cat in FACES_WATER and self.water is not None and oid not in oriented \
                    and "facing" not in self.rels.get(oid, []):
                self.rels.setdefault(oid, []).append("facing")
        self.cats = cats

    # --- the shoreline ----------------------------------------------------------------------
    def _side_frame(self) -> tuple[float, float]:
        """(length along the shore, depth across it) of the setting for the water's side."""
        assert self.water is not None
        return (self.W, self.D) if self.water.side in ("north", "south") else (self.D, self.W)

    def _to_side(self, x: float, y: float) -> tuple[float, float]:
        """(along, v): v grows toward the water."""
        assert self.water is not None
        side = self.water.side
        return {"north": (x, y), "south": (x, self.D - y), "east": (y, x), "west": (y, self.W - x)}[str(side)]

    def _from_side_dir(self, da: float, dv: float) -> tuple[float, float]:
        assert self.water is not None
        side = self.water.side
        return {"north": (da, dv), "south": (da, -dv), "east": (dv, da), "west": (-dv, da)}[str(side)]

    def _shore(self, a: float) -> tuple[float, float]:
        """(v of the waterline, dv/da) at ``a`` along the shore: a cove, bowing inland in the middle,
        headlands beyond the setting."""
        length, depth = self._side_frame()
        assert self.water is not None
        v0 = depth * (1 - self.water.fraction)
        amp = 0.3 * self.water.fraction * depth
        u = (a - length / 2) / (length / 2)
        if abs(u) <= 1.6:
            return v0 - amp * (1 - u * u), amp * 2 * u / (length / 2)
        u0 = math.copysign(1.6, u)
        return v0 - amp * (1 - u0 * u0) - amp * 2 * u0 * (u - u0) * 0.4, amp * 2 * u0 * 0.4 / (length / 2)

    def signed(self, x: float, y: float) -> float:
        """Distance (m) from the waterline: > 0 in the water, < 0 on land."""
        if self.water is None:
            return -1e9
        if self.water.side is None:
            assert self.water.region is not None
            cx, cy, rx, ry = self.water.region
            r = math.hypot((x - cx) / rx, (y - cy) / ry)
            return (1 - r) * min(rx, ry)
        a, v = self._to_side(x, y)
        vs, dvs = self._shore(a)
        return (v - vs) / math.sqrt(1 + dvs * dvs)

    def to_water(self, x: float, y: float) -> tuple[float, float]:
        """Unit vector (layout frame) from (x, y) toward the water, across the shore."""
        if self.water is None:
            return (0.0, 1.0)
        if self.water.side is None:
            assert self.water.region is not None
            cx, cy = self.water.region[0], self.water.region[1]
            dx, dy = cx - x, cy - y
            n = math.hypot(dx, dy) or 1.0
            return dx / n, dy / n
        a, _v = self._to_side(x, y)
        _vs, dvs = self._shore(a)
        n = math.sqrt(1 + dvs * dvs)
        return self._from_side_dir(-dvs / n, 1 / n)

    def height(self, x: float, y: float) -> float:
        """Ground height (m) at (x, y), layout frame; 0 on the flat land the scene stands on."""
        s = self.signed(x, y)
        if s > 0:  # the sea floor deepens from the waterline
            return self.level - self.depth * (1 - math.exp(-s * self.floor_slope / self.depth))
        t = min(max((s + self.band) / self.band, 0.0), 1.0)
        z = -self.drop * t * t
        # beyond the setting: dunes / hills rise away from the scene and the shore
        out = math.hypot(max(-x, 0.0, x - self.W), max(-y, 0.0, y - self.D))
        if out > 0:
            fade = min(max((-s - self.band - 2.0) / 8.0, 0.0), 1.0) if self.water is not None else 1.0
            n1 = _vnoise(x / 9.0, y / 9.0, self.seed)
            n2 = _vnoise(x / 3.5, y / 3.5, self.seed + 1)
            relief = {"sand": 1.6, "grass": 2.2, "soil": 1.8, "gravel": 1.2, "rock": 4.0}[self.ground]
            rise = min(out * 0.06, 3.0) + relief * min(out / 12.0, 1.0) * (0.65 * n1 + 0.35 * n2)
            z += fade * rise
        return z

    # --- the layout hooks -------------------------------------------------------------------
    def _corners(self, x: float, y: float, yaw: float, dims: tuple[float, float, float]) -> list[tuple[float, float]]:
        r, f = (math.cos(yaw), math.sin(yaw)), (math.sin(yaw), -math.cos(yaw))
        hw, hd = dims[0] / 2, dims[1] / 2
        return [(x + sx * hw * r[0] + sy * hd * f[0], y + sx * hw * r[1] + sy * hd * f[1])
                for sx in (-1, 1) for sy in (-1, 1)]

    def _ends(self, x: float, y: float, yaw: float, d: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """(front end, back end) centres of a long object (front = local -Y)."""
        f = (math.sin(yaw), -math.cos(yaw))
        return (x + f[0] * d / 2, y + f[1] * d / 2), (x - f[0] * d / 2, y - f[1] * d / 2)

    def violation(self, oid: str, category: str, x: float, y: float, yaw: float,
                  dims: tuple[float, float, float]) -> str | None:
        """Why an object may not stand here (hard), else None."""
        if self.water is None:
            return None
        rels = self.rels.get(oid, [])
        dry = self.dry()
        if "leads_to" in rels:
            front, back = self._ends(x, y, yaw, dims[1])
            if not _JETTY_OVER_M[0] <= self.signed(*front) <= _JETTY_OVER_M[1] + dims[1] / 2:
                return f"relation '{oid} leads_to water': its end is not over the water"
            if self.signed(*back) > dry:
                return f"relation '{oid} leads_to water': it does not start on the land"
            f = (math.sin(yaw), -math.cos(yaw))
            n = self.to_water(x, y)
            if f[0] * n[0] + f[1] * n[1] < math.cos(math.radians(25)):
                return f"relation '{oid} leads_to water': not pointing at the water"
            return None
        if "at_edge_of" in rels:
            s = self.signed(x, y)
            lo, hi = _EDGE_BAND_M
            return None if lo <= s <= hi else f"relation '{oid} at_edge_of water': {s:+.2f} m from the waterline"
        if "in" in rels:
            s = self.signed(x, y)
            return None if 0.3 <= s <= 6.0 else f"relation '{oid} in water': {s:+.2f} m from the waterline"
        if any(self.signed(cx, cy) > dry for cx, cy in self._corners(x, y, yaw, dims)) or self.signed(x, y) > dry:
            return "in the water or on the wet shore (only at_edge_of / in / leads_to objects go there)"
        return None

    def cost(self, oid: str, x: float, y: float, yaw: float, dims: tuple[float, float, float]) -> float:
        """Soft preferences toward the water."""
        if self.water is None:
            return 0.0
        rels = self.rels.get(oid, [])
        c = 0.0
        s = self.signed(x, y)
        top = self.dry()  # where the dry land ends
        if "at_edge_of" in rels:
            c += 0.6 * abs(s + 0.3)
        if "near" in rels:
            c += 0.3 * abs(s - (top - 1.2)) / _NEAR_M
        if "back_of" in rels:  # along the back: as far from the water as the setting goes
            c += 0.8 * min(max(s - self.back_at(x, y) - max(dims[0], dims[1]) / 2, 0.0) / 4.0, 1.5)
        if "leads_to" in rels:
            front, _back = self._ends(x, y, yaw, dims[1])
            c += 0.3 * abs(self.signed(*front) - 2.0)
        if "facing" in rels and not any(r in rels for r in ("at_edge_of", "near", "in", "back_of")) \
                and self.cats.get(oid) in FACES_WATER:  # loungers, a parasol: toward the water by default
            c += 0.8 * min(abs(s - (top - 2.5)) / 4.0, 1.5)
        if "facing" in rels or "leads_to" in rels:
            f = (math.sin(yaw), -math.cos(yaw))
            n = self.to_water(x, y)
            c += 0.8 * (1 - (f[0] * n[0] + f[1] * n[1])) / 2
        return c + self.frame_cost(oid, x, y)

    def rank(self, oid: str) -> int:
        """Placement order outdoors: 0 the shore groups (a parasol and the loungers under it, what
        else faces the water, a boardwalk, objects at the water's edge or in it), 1 the rest
        (trees, bushes: the solver's usual hero / size order). Bench beach_cove (qwen aidb6 demo):
        five hero palms, placed first as the largest objects, took the beach strip the loungers
        wanted, and the group stood at the back."""
        cat = self.cats.get(oid)
        rels = self.rels.get(oid, [])
        if self.water is not None and (any(r in rels for r in ("at_edge_of", "in", "leads_to"))
                                       or cat in FACES_WATER or cat in OVER_WATER or oid in self.start_of):
            return 0  # what a boardwalk starts at goes down before the palms take its place
        return 1

    def frame_cost(self, oid: str, x: float, y: float) -> float:
        """A tree with no place of its own stands at the sides or along the back, framing the beach,
        not in the middle of the strip in front of the water."""
        if self.water is None or self.water.side is None or self.cats.get(oid) not in FRAMING \
                or any(r in self.rels.get(oid, []) for r in ("at_edge_of", "in", "leads_to", "near", "back_of")):
            return 0.0
        a, _v = self._to_side(x, y)
        length, _depth = self._side_frame()
        u = abs(a - length / 2) / (length / 2)
        back, top, s = self.back_at(x, y), self.dry(), self.signed(x, y)
        front = min(max((s - back) / max(top - back, 1e-6), 0.0), 1.0)  # 0 at the back edge, 1 at the dry line
        return _FRAME_COST * max(0.0, 1 - u / _FRAME_SIDES) * front

    def _link_target(self, link: tuple[float, float, float, float],
                     dims: tuple[float, float, float]) -> tuple[float, float]:
        """Where an object the boardwalk ``link`` (x, y, yaw, length) starts at stands: just behind
        its land end, on its axis (``_extend_back`` then runs the boardwalk up to it)."""
        x, y, yaw, length = link
        f = (math.sin(yaw), -math.cos(yaw))
        back = length / 2 + max(dims[0], dims[1]) / 2 + 0.6
        return x - f[0] * back, y - f[1] * back

    def link_cost(self, link: tuple[float, float, float, float], x: float, y: float,
                  dims: tuple[float, float, float]) -> float:
        tx, ty = self._link_target(link, dims)
        return 0.6 * min(math.hypot(x - tx, y - ty) / 3.0, 2.0)

    def link_spots(self, link: tuple[float, float, float, float], dims: tuple[float, float, float], rng: Any,
                   n: int) -> list[tuple[float, float, float]]:
        """Candidates on the boardwalk's axis behind its land end (up to the setting's back)."""
        yaw = link[2]
        f = (math.sin(yaw), -math.cos(yaw))
        tx, ty = self._link_target(link, dims)
        out = []
        for k in range(max(n, 1)):
            t = 0.0 if k == 0 else rng.uniform(0.0, 6.0)
            px, py = tx - f[0] * t + rng.uniform(-0.3, 0.3), ty - f[1] * t + rng.uniform(-0.3, 0.3)
            if 0.0 <= px <= self.W and 0.0 <= py <= self.D:
                out.append((px, py, rng.uniform(0, 2 * math.pi)))
        return out

    def dry(self) -> float:
        """Signed distance where the dry land ends (land objects stay on the land side of it)."""
        return -self.band * _DRY_SHARE

    def back_at(self, x: float, y: float) -> float:
        """Signed distance of the setting's back edge behind (x, y), away from the water."""
        nx, ny = self.to_water(x, y)
        step = 0.25
        while 0.0 <= x - nx * step <= self.W and 0.0 <= y - ny * step <= self.D:
            x, y = x - nx * step, y - ny * step
        return self.signed(x, y)

    def propose(self, oid: str, dims: tuple[float, float, float], rng: Any, n: int) -> list[tuple[float, float, float]]:
        """Up to ``n`` candidate (x, y, yaw) for an object with a water relation (the solver's
        uniform draws rarely hit a 2 m shore band, never a jetty pointing at the water)."""
        rels = self.rels.get(oid, [])
        if self.water is None or not any(r in rels for r in ("at_edge_of", "in", "leads_to", "near", "back_of")):
            return []
        out: list[tuple[float, float, float]] = []
        for _ in range(n * 6):
            if len(out) >= n:
                break
            x, y = rng.uniform(0, self.W), rng.uniform(0, self.D)
            s = self.signed(x, y)
            nx, ny = self.to_water(x, y)
            top = self.dry()
            if "leads_to" in rels:
                want = rng.uniform(_JETTY_OVER_M[0] + 0.3, _JETTY_OVER_M[1])  # the water end
                tx, ty = x + nx * (want - s), y + ny * (want - s)  # that end
                cx, cy = tx - nx * dims[1] / 2, ty - ny * dims[1] / 2
                yaw = math.atan2(nx, -ny)  # front (local -Y) toward the water
                out.append((cx, cy, yaw))
                continue
            if "at_edge_of" in rels:
                want = rng.uniform(_EDGE_BAND_M[0] * 0.6, _EDGE_BAND_M[1] * 0.6)
            elif "in" in rels:
                want = rng.uniform(0.6, 3.0)
            elif "near" in rels:
                want = rng.uniform(top - _NEAR_M, top - max(dims[0], dims[1]) / 2)
            else:  # back_of
                half = max(dims[0], dims[1]) / 2
                want = self.back_at(x, y) + rng.uniform(half, half + 1.5)
            px, py = x + nx * (want - s), y + ny * (want - s)
            yaw = math.atan2(nx, -ny) if "facing" in rels else rng.uniform(0, 2 * math.pi)
            out.append((px, py, yaw))
        return out

    # --- after the layout -------------------------------------------------------------------
    def ground_under(self, x: float, y: float, yaw: float, dims: tuple[float, float, float]) -> float:
        """Lowest ground under a footprint (centre and corners): the object rests there."""
        return min(self.height(px, py) for px, py in [(x, y), *self._corners(x, y, yaw, dims)])


def terrain_of(spec: SceneSpec) -> Terrain | None:
    st = spec.setting
    if st.kind != "outdoor" or (st.ground is None and st.water is None):
        return None
    return Terrain(spec)


def settle(spec: SceneSpec, placements: list[Placement]) -> list[Placement]:
    """Placements resting on the terrain: objects on the shore slope stand on the ground (rocks
    bedded in), a boardwalk that leads to the water runs back over the land to whatever stands
    behind it (the bushes) or the setting's edge. Loungers under a parasol (any setting) lie side
    by side around its pole."""
    placements = arrange_under_parasols(spec, placements)
    ter = terrain_of(spec)
    if ter is None:
        return placements
    from ai_db_scene.priors import load_priors

    priors = load_priors()
    out = []
    for pl in placements:
        base_id = _base(pl.id, ter.rels)
        x, y, z = pl.location
        h = pl.dims[2]
        yaw = math.radians(pl.rotation_deg_z)
        prior = priors.get(pl.category)
        on_ground = prior is not None and prior.support in ("ground", "floor") and abs(z - h / 2) < 1e-3
        if pl.category in OVER_WATER and "leads_to" in ter.rels.get(base_id, []):
            pl = _extend_back(ter, pl, placements)
            out.append(pl)
            continue
        if not on_ground or pl.category in OVER_WATER:
            out.append(pl)
            continue
        zb = ter.ground_under(x, y, yaw, pl.dims) - _BEDDED.get(pl.category, 0.0) * h
        if abs(zb) > 1e-4:
            pl = replace(pl, location=(x, y, round(zb + h / 2, 6)))
        out.append(pl)
    return out


_POLE_GAP_M = 0.14  # between a parasol's pole and the loungers either side of it


def arrange_under_parasols(spec: SceneSpec, placements: list[Placement]) -> list[Placement]:
    """What stands ``under`` a parasol lies side by side across it, the pole in a gap between
    them, all turned like the first and shifted a little toward their head ends (the shade falls
    on the upper body). The solver only keeps them inside the canopy: a lounger's rail went
    through the pole (bench beach_cove)."""
    cats = {o.id: o.category for o in spec.objects}
    under: dict[str, list[str]] = {}
    for r in spec.relations:
        if r.rel == "under" and cats.get(r.b) == "parasol":
            under.setdefault(r.b, []).append(r.a)
    if not under:
        return placements
    moved: dict[str, Placement] = {}
    for par_id, bases in under.items():
        for par in [p for p in placements if p.category == "parasol" and _base(p.id, {par_id: 1}) == par_id]:
            kids = [p for p in placements if p.id not in moved and _base(p.id, {b: 1 for b in bases}) in bases
                    and abs(p.location[0] - par.location[0]) <= par.dims[0] / 2
                    and abs(p.location[1] - par.location[1]) <= par.dims[1] / 2]
            if not kids:
                continue
            kids.sort(key=lambda p: p.id)
            yaw = math.radians(kids[0].rotation_deg_z)
            rv, fv = (math.cos(yaw), math.sin(yaw)), (math.sin(yaw), -math.cos(yaw))
            step = max(k.dims[0] for k in kids) + _POLE_GAP_M * 2
            n = len(kids)
            offs = [(i - (n - 1) / 2) * step for i in range(n)]
            if n % 2:
                offs = [o + step / 2 for o in offs]  # never one centred on the pole
            back = 0.18 * max(k.dims[1] for k in kids)  # toward the head ends (local +y = -front)
            for k, off in zip(kids, offs, strict=True):
                x = par.location[0] + rv[0] * off - fv[0] * back
                y = par.location[1] + rv[1] * off - fv[1] * back
                moved[k.id] = replace(k, location=(round(x, 6), round(y, 6), k.location[2]),
                                      rotation_deg_z=kids[0].rotation_deg_z)
    return [moved.get(p.id, p) for p in placements]


def _base(pid: str, rels: dict[str, Any]) -> str:
    if pid in rels:
        return pid
    m = re.match(r"^(.+)_\d+$", pid)
    return m.group(1) if m else pid


def _extend_back(ter: Terrain, pl: Placement, placements: list[Placement]) -> Placement:
    """Lengthen a boardwalk at its land end toward the back of the setting, stopping short of the
    first object behind it (it runs from that object to the water) or the setting's edge."""
    x, y, z = pl.location
    yaw = math.radians(pl.rotation_deg_z)
    f = (math.sin(yaw), -math.cos(yaw))
    w, d, h = pl.dims
    front = (x + f[0] * d / 2, y + f[1] * d / 2)
    r = (math.cos(yaw), math.sin(yaw))
    others = [p for p in placements if p.id != pl.id and p.dims[2] > 0.05]
    length = d
    step = 0.25
    while length < 60.0:
        nl = length + step
        bx, by = front[0] - f[0] * nl, front[1] - f[1] * nl
        if not (0.0 <= bx <= ter.W and 0.0 <= by <= ter.D):
            break
        if any(_in_box(bx + r[0] * k * w / 2, by + r[1] * k * w / 2, p, 0.15) for p in others for k in (-1, 0, 1)):
            break
        length = nl
    cx, cy = front[0] - f[0] * length / 2, front[1] - f[1] * length / 2
    return replace(pl, location=(round(cx, 6), round(cy, 6), z), dims=(w, round(length, 4), h))


def _in_box(x: float, y: float, p: Placement, pad: float) -> bool:
    yaw = math.radians(p.rotation_deg_z)
    dx, dy = x - p.location[0], y - p.location[1]
    u = dx * math.cos(yaw) + dy * math.sin(yaw)
    v = dx * math.sin(yaw) - dy * math.cos(yaw)
    return abs(u) <= p.dims[0] / 2 + pad and abs(v) <= p.dims[1] / 2 + pad


# ------------------------------------------------------------------------------------------------
# Blender commands


def _axis(lo: float, hi: float, fine: float) -> list[float]:
    """Grid coordinates: ``fine`` steps over [lo, hi], growing steps out to the horizon."""
    n = max(2, math.ceil((hi - lo) / fine))
    xs = [lo + (hi - lo) * i / n for i in range(n + 1)]
    left, right, step = [], [], fine
    a, b = lo, hi
    while a > lo - _HORIZON_M:
        step *= 1.22
        a -= step
        b += step
        left.append(max(a, lo - _HORIZON_M))
        right.append(min(b, hi + _HORIZON_M))
    return [round(v, 4) for v in [*reversed(left), *xs, *right]]


def terrain_payload(spec: SceneSpec, off: tuple[float, float, float]) -> dict[str, Any]:
    ter = Terrain(spec)
    fine = max(0.25, max(ter.W, ter.D) / 140.0)
    xs = _axis(-_FINE_MARGIN_M, ter.W + _FINE_MARGIN_M, fine)
    ys = _axis(-_FINE_MARGIN_M, ter.D + _FINE_MARGIN_M, fine)
    heights = [round(ter.height(x, y), 4) for y in ys for x in xs]
    water = None
    if ter.water is not None:
        kind = ter.water.kind
        depth, _slope, absorb, density, tint = _WATER[kind]
        bottom = ter.level - depth - 1.0
        if ter.water.side is None:
            assert ter.water.region is not None
            cx, cy, rx, ry = ter.water.region
            shape = {"ellipse": [cx + off[0], cy + off[1], rx + ter.band + 1.0, ry + ter.band + 1.0]}
        else:
            # the water side, from just inland of the waterline out to the horizon
            lo_x, hi_x, lo_y, hi_y = -_HORIZON_M, ter.W + _HORIZON_M, -_HORIZON_M, ter.D + _HORIZON_M
            _length, depth_ = ter._side_frame()
            vmin = depth_ * (1 - ter.water.fraction) - 0.3 * ter.water.fraction * depth_ - ter.band - 2.0
            side = ter.water.side
            if side == "north":
                lo_y = vmin
            elif side == "south":
                hi_y = ter.D - vmin
            elif side == "east":
                lo_x = vmin
            else:
                hi_x = ter.W - vmin
            shape = {"box": [lo_x + off[0], lo_y + off[1], hi_x + off[0], hi_y + off[1]]}
        water = {"name": f"{kind}_water", "kind": kind, "level": ter.level, "bottom": bottom, **shape,
                 "absorb": list(absorb), "density": density, "tint": list(tint), "shore": _shore_line(ter, off),
                 "look": [list(v) if isinstance(v, tuple) else v for v in WATER_LOOK[kind]]}
    return {"name": f"{ter.ground}_ground", "ground": ter.ground, "xs": [x + off[0] for x in xs],
            "ys": [y + off[1] for y in ys], "heights": heights, "level": ter.level,
            "has_water": ter.water is not None, "water": water}


def _shore_line(ter: Terrain, off: tuple[float, float, float]) -> dict[str, Any]:
    """Where the water's shallows start, for its shader: a side water's mean waterline (``axis``
    0 = x / 1 = y in the Blender frame, ``at``, ``dir`` +1/-1 toward the water), or a lake's
    ellipse (``ellipse`` [cx, cy, rx, ry]: shallow at the rim)."""
    assert ter.water is not None
    if ter.water.side is None:
        assert ter.water.region is not None
        cx, cy, rx, ry = ter.water.region
        return {"ellipse": [cx + off[0], cy + off[1], rx, ry]}
    length, _depth = ter._side_frame()
    vs = sum(ter._shore(length * k / 8)[0] for k in range(9)) / 9
    side = ter.water.side
    if side == "north":
        return {"axis": 1, "at": vs + off[1], "dir": 1}
    if side == "south":
        return {"axis": 1, "at": ter.D - vs + off[1], "dir": -1}
    if side == "east":
        return {"axis": 0, "at": vs + off[0], "dir": 1}
    return {"axis": 0, "at": ter.W - vs + off[0], "dir": -1}


def terrain_commands(spec: SceneSpec, off: tuple[float, float, float]) -> list[dict[str, Any]]:
    """run_code: the terrain mesh and its ground shader, the water body and its shader."""
    payload = terrain_payload(spec, off)
    return [{"type": "run_code", "params": {"code": "import json\nCFG = json.loads(" + repr(json.dumps(payload))
                                            + ")\n" + _TERRAIN_CODE}}]


def object_names(spec: SceneSpec) -> list[str]:
    """Names of the terrain objects a build creates (to reserve and to tidy)."""
    ter = terrain_of(spec)
    if ter is None:
        return []
    return [f"{ter.ground}_ground"] + ([f"{ter.water.kind}_water"] if ter.water is not None else [])


_TERRAIN_CODE = r'''
import bpy, math

def sock(sockets, name, kind=None):
    return next(s for s in sockets if s.name == name and (kind is None or s.type == kind) and s.enabled)

def node(nt, kind, **inputs):
    n = nt.nodes.new(kind)
    for k, v in inputs.items():
        sock(n.inputs, k).default_value = v
    return n

def link(nt, a, b):
    nt.links.new(a, b)

def mix(nt, a, b, fac, blend="MIX"):
    m = nt.nodes.new("ShaderNodeMix")
    m.data_type = "RGBA"
    m.blend_type = blend
    for s, v in ((sock(m.inputs, "A", "RGBA"), a), (sock(m.inputs, "B", "RGBA"), b)):
        if isinstance(v, tuple):
            s.default_value = (*v, 1.0) if len(v) == 3 else v
        else:
            link(nt, v, s)
    f = sock(m.inputs, "Factor", "VALUE")
    if isinstance(fac, (int, float)):
        f.default_value = fac
    else:
        link(nt, fac, f)
    return sock(m.outputs, "Result", "RGBA")

def maprange(nt, value, a, b, c, d):
    m = nt.nodes.new("ShaderNodeMapRange")
    m.clamp = True
    link(nt, value, m.inputs["Value"])
    m.inputs["From Min"].default_value, m.inputs["From Max"].default_value = a, b
    m.inputs["To Min"].default_value, m.inputs["To Max"].default_value = c, d
    return m.outputs["Result"]

def math_op(nt, op, a, b):
    m = nt.nodes.new("ShaderNodeMath")
    m.operation = op
    for i, v in enumerate((a, b)):
        if isinstance(v, (int, float)):
            m.inputs[i].default_value = v
        else:
            link(nt, v, m.inputs[i])
    return m.outputs[0]

# ground: colours (dry a, dry b), wet tint, roughness, grain scale, bump strength
# (linear colours: dry a, dry b, wet multiplier, roughness, grain scale, bump strength)
GROUNDS = {
    "sand": ((0.6, 0.45, 0.25), (0.48, 0.35, 0.19), (0.55, 0.47, 0.38), 0.93, 140.0, 0.35),
    "grass": ((0.06, 0.15, 0.025), (0.11, 0.2, 0.035), (0.35, 0.3, 0.25), 0.88, 60.0, 0.45),
    "soil": ((0.12, 0.07, 0.035), (0.18, 0.11, 0.06), (0.55, 0.5, 0.45), 0.92, 50.0, 0.5),
    "gravel": ((0.28, 0.26, 0.22), (0.17, 0.16, 0.13), (0.55, 0.52, 0.48), 0.85, 25.0, 0.8),
    "rock": ((0.19, 0.18, 0.16), (0.11, 0.1, 0.09), (0.55, 0.53, 0.5), 0.8, 8.0, 1.0),
}
# the sea floor seen through the water: shallow -> turquoise over the sand, deep -> dark blue
SHALLOW = {"sea": (0.06, 0.42, 0.42), "lagoon": (0.12, 0.55, 0.5), "lake": (0.08, 0.2, 0.14), "pond": (0.08, 0.14, 0.06)}
DEEP = {"sea": (0.005, 0.05, 0.16), "lagoon": (0.01, 0.12, 0.2), "lake": (0.01, 0.04, 0.05), "pond": (0.02, 0.04, 0.02)}

def ground_material(kind, level, has_water):
    name = "ai_db_ground_" + kind
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    ca, cb, wet_c, rough, grain_scale, bump_s = GROUNDS[kind]
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = node(nt, "ShaderNodeBsdfPrincipled")
    link(nt, bsdf.outputs["BSDF"], out.inputs["Surface"])
    tc = nt.nodes.new("ShaderNodeTexCoord")
    co = tc.outputs["Object"]
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    link(nt, co, sep.inputs[0])
    z = sep.outputs["Z"]
    patches = node(nt, "ShaderNodeTexNoise", Scale=0.18, Detail=4.0, Roughness=0.55)
    link(nt, co, patches.inputs["Vector"])
    grain = node(nt, "ShaderNodeTexNoise", Scale=grain_scale, Detail=2.0)
    link(nt, co, grain.inputs["Vector"])
    base = mix(nt, ca, cb, maprange(nt, patches.outputs["Fac"], 0.35, 0.65, 0.0, 1.0))
    if kind == "gravel":
        vor = node(nt, "ShaderNodeTexVoronoi", Scale=22.0)
        link(nt, co, vor.inputs["Vector"])
        base = mix(nt, base, vor.outputs["Color"], 0.35, "OVERLAY")
        height = vor.outputs["Distance"]
    elif kind == "sand":
        # wind ripples: distorted bands a hand apart, plus the grain
        wave = node(nt, "ShaderNodeTexWave", Scale=1.6, Distortion=7.0, Detail=2.0)
        link(nt, co, wave.inputs["Vector"])
        height = math_op(nt, "ADD", math_op(nt, "MULTIPLY", wave.outputs["Fac"], 0.6),
                         math_op(nt, "MULTIPLY", grain.outputs["Fac"], 0.4))
    elif kind == "grass":
        blades = node(nt, "ShaderNodeTexNoise", Scale=180.0, Detail=1.0)
        link(nt, co, blades.inputs["Vector"])
        base = mix(nt, base, (0.45, 0.5, 0.2), maprange(nt, blades.outputs["Fac"], 0.55, 0.75, 0.0, 0.35))
        height = blades.outputs["Fac"]
    elif kind == "rock":
        big = node(nt, "ShaderNodeTexNoise", Scale=1.2, Detail=8.0, Roughness=0.7)
        link(nt, co, big.inputs["Vector"])
        height = big.outputs["Fac"]
    else:
        height = grain.outputs["Fac"]
    base = mix(nt, base, grain.outputs["Color"], 0.12, "OVERLAY")
    rough_s = rough
    if has_water:
        # wet band above the waterline, darker and glossier; a thin foam line at the waterline
        wet = maprange(nt, z, level + 0.03, level + 0.2, 1.0, 0.0)
        wet_col = mix(nt, base, wet_c, 0.7, "MULTIPLY")
        base = mix(nt, base, wet_col, wet)
        foam_mask = maprange(nt, z, level + 0.012, level + 0.05, 1.0, 0.0)
        foam_low = maprange(nt, z, level - 0.02, level + 0.004, 0.0, 1.0)
        fn = node(nt, "ShaderNodeTexNoise", Scale=6.0, Detail=6.0)
        link(nt, co, fn.inputs["Vector"])
        foam = math_op(nt, "MULTIPLY", math_op(nt, "MULTIPLY", foam_mask, foam_low),
                       maprange(nt, fn.outputs["Fac"], 0.45, 0.6, 0.0, 0.85))
        base = mix(nt, base, (0.92, 0.93, 0.92), foam)
        rough_s = maprange(nt, wet, 0.0, 1.0, rough, 0.35)
        shallow = maprange(nt, z, level - 0.03, level - 0.9, 0.0, 0.9)
        base = mix(nt, base, SHALLOW[CFG["water"]["kind"]], shallow)
        deep = maprange(nt, z, level - 1.5, level - 4.5, 0.0, 1.0)
        base = mix(nt, base, DEEP[CFG["water"]["kind"]], deep)
    link(nt, base, bsdf.inputs["Base Color"])
    if isinstance(rough_s, float):
        bsdf.inputs["Roughness"].default_value = rough_s
    else:
        link(nt, rough_s, bsdf.inputs["Roughness"])
    bump = node(nt, "ShaderNodeBump", Strength=bump_s, Distance=0.02)
    link(nt, height, bump.inputs["Height"])
    link(nt, bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat

def water_material(w):
    mat = bpy.data.materials.new("ai_db_water_" + w["kind"])
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    # a touch rough and a lower IOR than real water: at the low eye-level angle the sky's
    # reflection otherwise hides the turquoise of the shallows completely
    shallow_c, deep_c, share_near, share_far, spec_level, band = w["look"]
    shallow_c, deep_c = tuple(shallow_c), tuple(deep_c)
    bsdf = node(nt, "ShaderNodeBsdfPrincipled", Roughness=0.06, IOR=1.12)
    sock(bsdf.inputs, "Transmission Weight").default_value = 1.0
    sock(bsdf.inputs, "Specular IOR Level").default_value = spec_level
    bsdf.inputs["Base Color"].default_value = (*w["tint"], 1.0)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    co = tc.outputs["Object"]
    # the water body: shallow (by the shore) -> its turquoise, offshore -> deeper blue
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    link(nt, co, sep.inputs[0])
    sh = w["shore"]
    if "ellipse" in sh:
        cx, cy, rx, ry = sh["ellipse"]
        dx = math_op(nt, "DIVIDE", math_op(nt, "SUBTRACT", sep.outputs["X"], cx), rx)
        dy = math_op(nt, "DIVIDE", math_op(nt, "SUBTRACT", sep.outputs["Y"], cy), ry)
        r = math_op(nt, "SQRT", math_op(nt, "ADD", math_op(nt, "MULTIPLY", dx, dx), math_op(nt, "MULTIPLY", dy, dy)), 0.0)
        near = maprange(nt, r, 1.0, max(0.0, 1.0 - band / max(min(rx, ry), 0.1)), 1.0, 0.0)
    else:
        coord = sep.outputs["Y" if sh["axis"] == 1 else "X"]
        near = maprange(nt, coord, sh["at"], sh["at"] + sh["dir"] * band, 1.0, 0.0)
    body = node(nt, "ShaderNodeBsdfPrincipled", Roughness=0.12, IOR=1.33)
    sock(body.inputs, "Specular IOR Level").default_value = spec_level * 0.6
    link(nt, mix(nt, deep_c, shallow_c, near), body.inputs["Base Color"])
    share = maprange(nt, near, 0.0, 1.0, share_far, share_near)
    swell = node(nt, "ShaderNodeTexNoise", Scale=0.35, Detail=3.0)
    link(nt, co, swell.inputs["Vector"])
    ripple = node(nt, "ShaderNodeTexWave", Scale=2.2, Distortion=4.0, Detail=3.0)
    link(nt, co, ripple.inputs["Vector"])
    h = math_op(nt, "ADD", swell.outputs["Fac"], math_op(nt, "MULTIPLY", ripple.outputs["Fac"], 0.25))
    bump = node(nt, "ShaderNodeBump", Strength=0.22, Distance=0.05)
    link(nt, h, bump.inputs["Height"])
    link(nt, bump.outputs["Normal"], bsdf.inputs["Normal"])
    link(nt, bump.outputs["Normal"], body.inputs["Normal"])
    both = nt.nodes.new("ShaderNodeMixShader")
    link(nt, share, both.inputs["Fac"])
    link(nt, bsdf.outputs["BSDF"], both.inputs[1])
    link(nt, body.outputs["BSDF"], both.inputs[2])
    link(nt, both.outputs["Shader"], out.inputs["Surface"])
    vol = node(nt, "ShaderNodeVolumeAbsorption", Density=w["density"])
    vol.inputs["Color"].default_value = (*w["absorb"], 1.0)
    link(nt, vol.outputs["Volume"], out.inputs["Volume"])
    return mat

def link_obj(ob):
    bpy.context.scene.collection.objects.link(ob)
    ob["category"] = "ground"
    ob["ai_db_terrain"] = 1
    ob["support"] = "ground"

for nm in [CFG["name"]] + ([CFG["water"]["name"]] if CFG["water"] else []):
    old = bpy.data.objects.get(nm)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)

xs, ys, hs = CFG["xs"], CFG["ys"], CFG["heights"]
nx, ny = len(xs), len(ys)
verts = [(xs[i], ys[j], hs[j * nx + i]) for j in range(ny) for i in range(nx)]
faces = [(j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i) for j in range(ny - 1) for i in range(nx - 1)]
me = bpy.data.meshes.new(CFG["name"])
me.from_pydata(verts, [], faces)
for p in me.polygons:
    p.use_smooth = True
ground = bpy.data.objects.new(CFG["name"], me)
link_obj(ground)
# a closed slab (Solidify, applied, 1 m thick under the surface): validate_scene's open-mesh fix
# would otherwise fill the terrain's outline with one huge face
sol = ground.modifiers.new("ai_db_solid", "SOLIDIFY")
sol.thickness, sol.offset, sol.use_rim = 1.0, -1.0, True
dg = bpy.context.evaluated_depsgraph_get()
closed_me = bpy.data.meshes.new_from_object(ground.evaluated_get(dg))
ground.modifiers.remove(sol)
ground.data = closed_me
bpy.data.meshes.remove(me)
closed_me.name = CFG["name"]
me = closed_me
me.materials.append(ground_material(CFG["ground"], CFG["level"], CFG["has_water"]))
made = [ground.name]
w = CFG["water"]
if w:
    import bmesh
    bm = bmesh.new()
    top, bot = w["level"], w["bottom"]
    if "box" in w:
        x0, y0, x1, y1 = w["box"]
        vs = [bm.verts.new(c) for c in ((x0, y0, bot), (x1, y0, bot), (x1, y1, bot), (x0, y1, bot),
                                         (x0, y0, top), (x1, y0, top), (x1, y1, top), (x0, y1, top))]
        for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            bm.faces.new([vs[i] for i in f])
    else:
        cx, cy, rx, ry = w["ellipse"]
        n = 64
        lo = [bm.verts.new((cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n), bot)) for k in range(n)]
        hi = [bm.verts.new((cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n), top)) for k in range(n)]
        bm.faces.new(list(reversed(lo)))
        bm.faces.new(hi)
        for k in range(n):
            bm.faces.new((lo[k], lo[(k + 1) % n], hi[(k + 1) % n], hi[k]))
    bm.normal_update()
    top_faces = [f for f in bm.faces if f.normal.z > 0.9 and all(abs(v.co.z - top) < 1e-6 for v in f.verts)]
    bmesh.ops.subdivide_edges(bm, edges=list({e for f in top_faces for e in f.edges}), cuts=15, use_grid_fill=True)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    wm = bpy.data.meshes.new(w["name"])
    bm.to_mesh(wm)
    bm.free()
    water = bpy.data.objects.new(w["name"], wm)
    link_obj(water)
    wm.materials.append(water_material(w))
    made.append(water.name)
result = {"terrain": made, "verts": len(verts)}
'''


# ------------------------------------------------------------------------------------------------
# boardwalk / jetty

BOARDWALK_PARTS = {"deck": "wood:pine:#b8a07a", "frame": "wood:dark", "posts": "wood:dark"}
_POST_SPACING_M = 1.8
_POST_EMBED_M = 0.3  # posts reach this far into the sand / sea floor


#: a park's paved plaza: round a fountain / gazebo standing on grass or soil, with a path from it
#: to the nearest open edge of the setting (bench park_corner, aidb7: the fountain and its benches
#: stood on a flat grey slab, or anywhere on the lawn)
PLAZA_CENTRES = ("fountain", "gazebo")
PLAZA_GROUNDS = ("grass", "soil")
PLAZA_MATERIAL = "stone:#a59e92"  # light paving; a spec's floor_material (concrete, stone) wins
_PLAZA_RING_CATS = ("bench", "chair", "armchair", "dining_chair", "trash_can")
_PLAZA_MARGIN_M = 0.9  # paving beyond the ring of benches: the walkway behind them
_PLAZA_REACH_M = 4.5  # objects this near the centre belong to the plaza
_PLAZA_MAX_R_M = 7.0
_PATH_W_M = 1.8
_PAVE_Z_M = 0.02  # above the ground: never coplanar with it


def plaza_layout(spec: SceneSpec, placements: list[Placement]) -> list[dict[str, Any]]:
    """Per plaza centre: {centre (layout frame), radius, path: (dx, dy) unit direction, length}."""
    st = spec.setting
    if st.kind != "outdoor" or terrain_of(spec) is None or Terrain(spec).ground not in PLAZA_GROUNDS:
        return []
    out = []
    for c in placements:
        if c.category not in PLAZA_CENTRES:
            continue
        cx, cy = c.location[0], c.location[1]
        r = max(c.dims[0], c.dims[1]) / 2 + 1.5
        for p in placements:
            d = math.hypot(p.location[0] - cx, p.location[1] - cy)
            if p is not c and p.category in _PLAZA_RING_CATS and d <= _PLAZA_REACH_M:
                r = max(r, d + max(p.dims[0], p.dims[1]) / 2 + _PLAZA_MARGIN_M)
        r = min(r, _PLAZA_MAX_R_M)
        best = None
        for k in range(16):  # the path: the clearest short way out to an edge of the setting
            a = 2 * math.pi * k / 16
            u = (math.cos(a), math.sin(a))
            tx = ((st.bounds_m[0] - cx) / u[0] if u[0] > 1e-6 else -cx / u[0] if u[0] < -1e-6 else 1e9)
            ty = ((st.bounds_m[1] - cy) / u[1] if u[1] > 1e-6 else -cy / u[1] if u[1] < -1e-6 else 1e9)
            length = min(tx, ty)
            if length <= r:
                continue
            hits = 0
            for p in placements:
                if p is c:
                    continue
                ox, oy = p.location[0] - cx, p.location[1] - cy
                along = ox * u[0] + oy * u[1]
                if r - 0.5 < along < length and abs(-ox * u[1] + oy * u[0]) < _PATH_W_M / 2 + max(p.dims[0], p.dims[1]) / 2:
                    hits += 1
            key = (hits, length)
            if best is None or key < best[0]:
                best = (key, u, length)
        out.append({"id": c.id, "centre": (cx, cy), "radius": round(r, 3),
                    "path": None if best is None else {"dir": best[1], "length": round(best[2], 3)}})
    return out


def plaza_commands(spec: SceneSpec, placements: list[Placement], off: tuple[float, float, float],
                   material: str) -> list[dict[str, Any]]:
    """run_code + apply_material: a paved disc round each plaza centre and its path (flat land)."""
    cmds: list[dict[str, Any]] = []
    for pz in plaza_layout(spec, placements):
        payload = {"name": f"plaza_{pz['id']}", "centre": [pz["centre"][0] + off[0], pz["centre"][1] + off[1]],
                   "z": off[2] + _PAVE_Z_M, "radius": pz["radius"], "path": pz["path"], "path_w": _PATH_W_M}
        cmds.append({"type": "run_code", "params": {
            "code": "import json\nCFG = json.loads(" + repr(json.dumps(payload)) + ")\n" + _PLAZA_CODE}})
        cmds.append({"type": "apply_material", "params": {"name": payload["name"], "material": material}})
    return cmds


_PLAZA_CODE = r'''
import bpy, bmesh, math
name, (cx, cy), z, R = CFG["name"], CFG["centre"], CFG["z"], CFG["radius"]
old = bpy.data.objects.get(name)
if old is not None:
    bpy.data.objects.remove(old, do_unlink=True)
bm = bmesh.new()
n = 64
mid = bm.verts.new((cx, cy, z))
rim = [bm.verts.new((cx + R * math.cos(2 * math.pi * k / n), cy + R * math.sin(2 * math.pi * k / n), z)) for k in range(n)]
for k in range(n):
    bm.faces.new((mid, rim[k], rim[(k + 1) % n]))
p = CFG["path"]
if p:
    ux, uy = p["dir"]
    w = CFG["path_w"] / 2
    al = math.asin(min(w / R, 1.0))
    a0 = R * math.cos(al)
    # from the plaza's rim (the arc between the two corners), no overlap with the disc
    arc = [(R * math.cos(t), R * math.sin(t)) for t in (al - 2 * al * k / 8 for k in range(1, 8))]
    q = [(a0, -w), (p["length"], -w), (p["length"], w), (a0, w), *arc]
    vs = [bm.verts.new((cx + a * ux - b * uy, cy + a * uy + b * ux, z)) for a, b in q]
    bm.faces.new(vs)
bm.normal_update()
for f in bm.faces:  # every face up (a face seen from below rendered black)
    if f.normal.z < 0:
        f.normal_flip()
me = bpy.data.meshes.new(name)
bm.to_mesh(me)
bm.free()
ob = bpy.data.objects.new(name, me)
bpy.context.scene.collection.objects.link(ob)
ob["category"] = "ground"  # flat paving: part of the ground for the checks (bounds, open mesh)
result = {"plaza": name, "radius": R}
'''


def boardwalk_commands(spec: SceneSpec, pl: Placement, off: tuple[float, float, float], seed: int,
                       deck_material: str | None = None) -> list[dict[str, Any]]:
    """A boardwalk on posts: planks across its width, two stringers, posts every ~1.8 m down to
    the ground under them (into the sea floor where it runs out over the water). Its deck is
    ``dims[2]`` above its base; on flat ground the build fills the placement box exactly."""
    w, length, h = pl.dims
    x, y, z = pl.location
    zb = z - h / 2
    yaw = math.radians(pl.rotation_deg_z)
    f = (math.sin(yaw), -math.cos(yaw))
    ter = terrain_of(spec)
    n = max(2, math.ceil(length / _POST_SPACING_M) + 1)
    posts = []
    for i in range(n):
        ly = -length / 2 + 0.15 + (length - 0.3) * i / (n - 1)  # local y (front = -y)
        px, py = x - f[0] * ly, y - f[1] * ly
        ground = ter.height(px, py) - _POST_EMBED_M if ter is not None else zb
        posts.append([round(ly, 4), round(min(ground, zb) - zb, 4)])
    payload = {"name": pl.id, "dims": [w, length, h], "location": [x + off[0], y + off[1], zb], "rot_z": pl.rotation_deg_z,
               "posts": posts, "seed": seed}
    cmds: list[dict[str, Any]] = [{"type": "run_code", "params": {
        "code": "import json\nCFG = json.loads(" + repr(json.dumps(payload)) + ")\n" + _BOARDWALK_CODE}}]
    for part, mat in BOARDWALK_PARTS.items():
        chosen = deck_material if part == "deck" and deck_material else mat
        cmds.append({"type": "apply_material", "params": {
            "name": f"{pl.id}_{part}", "material": chosen if ":proj=" in chosen else f"{chosen}:scale=3.0:proj=box"}})
    return cmds


_BOARDWALK_CODE = r'''
import bpy, bmesh, math, random
from mathutils import Matrix, Vector
rng = random.Random(CFG["seed"])
W, L, H = CFG["dims"]
name = CFG["name"]
for suffix in ("", "_deck", "_frame", "_posts"):
    old = bpy.data.objects.get(name + suffix)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)

def box(bm, cx, cy, cz0, sx, sy, sz):
    """A box by its bottom centre (local frame of the boardwalk: base at z=0)."""
    m = Matrix.Translation((cx, cy, cz0 + sz / 2)) @ Matrix.Diagonal((sx, sy, sz, 1.0))
    bmesh.ops.create_cube(bm, size=1.0, matrix=m)

plank, gap, thick = 0.14, 0.012, min(0.04, H * 0.25)
deck_z = H - thick
deck = bmesh.new()
y = -L / 2
while y < L / 2 - 1e-6:
    pw = min(plank, L / 2 - y)
    box(deck, rng.uniform(-0.004, 0.004), y + pw / 2, deck_z + rng.uniform(-0.003, 0.0), W, max(pw - gap, 0.005), thick)
    y += plank
frame = bmesh.new()
beam_h = min(0.16, deck_z * 0.6) if deck_z > 0.02 else 0.0
if beam_h > 0:
    for sx in (-1, 1):
        box(frame, sx * (W / 2 - 0.12), 0.0, deck_z - beam_h, 0.08, L, beam_h)
posts = bmesh.new()
post_top = deck_z - beam_h
for ly, bottom in CFG["posts"]:
    for sx in (-1, 1):
        if post_top - bottom > 0.02:
            box(posts, sx * (W / 2 - 0.12), ly, bottom, 0.11, 0.11, post_top - bottom)
root = bpy.data.objects.new(name, None)
bpy.context.scene.collection.objects.link(root)
root["asset"] = name
made = []
for suffix, bm in (("_deck", deck), ("_frame", frame), ("_posts", posts)):
    me = bpy.data.meshes.new(name + suffix)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name + suffix, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = root
    made.append(ob.name)
root.location = CFG["location"]
root.rotation_euler[2] = math.radians(CFG["rot_z"])
result = {"boardwalk": name, "parts": made}
'''
