"""Which inventory objects count for a brief's checklist item (bench.checklist).

An object counts for an item when its ``category`` (ai-db tags every object it builds; the
category of a recipe kind, too) is one the item names. A name only counts for objects with no
category, or with a generic one (``ground``: sand and water are both terrain), and then only
when

* a token starts a word of the name (``tree`` is not in ``street_lamp``, ``board`` not in
  ``baseboard``),
* the item is the name's head: with several items matching, the one whose token sits latest in
  the name wins (``desk_chair`` is a chair, ``bar_stool`` a stool, ``pallet_rack`` a rack, ``tv_stand``
  a stand), and
* for a broad category (``table``, ``chair``, ``lamp``...) the name decides first: a ``table`` named
  ``tv_stand`` is a TV stand, one named ``table_3`` a table;
* the object's size is roughly that of the item (its first category's prior, x SIZE_GATE either
  way), when both are known.

The old name match counted a ``fence_section`` named "parasol" as a parasol, street lamps as trees
and the room's baseboard as a whiteboard.
"""

from __future__ import annotations

import re
from typing import Any

#: categories that say what an object is, per checklist item label (briefs.BRIEFS); the first
#: one's size prior gates name matches
ITEM_CATEGORIES: dict[str, tuple[str, ...]] = {
    "sofa": ("sofa", "sectional", "couch", "loveseat"),
    "sofas": ("sofa", "sectional", "couch", "loveseat"),
    "tv stand": ("tv_stand", "tvstand", "media_console", "media", "console", "console_table", "sideboard",
                 "cabinet"),
    "tv": ("tv", "television", "electronics", "screen", "monitor"),
    "coffee table": ("coffee_table", "table", "low_table"),
    "rug": ("rug", "carpet", "textile", "mat"),
    "armchair": ("armchair", "lounge_chair", "club_chair", "accent_chair", "wingback_chair"),
    "armchairs": ("armchair", "lounge_chair", "club_chair", "accent_chair", "wingback_chair"),
    "floor lamp": ("floor_lamp", "lamp", "standing_lamp"),
    "desks": ("desk", "table", "school_desk", "student_desk", "teacher_desk", "writing_desk"),
    "chairs": ("chair", "dining_chair", "office_chair", "school_chair", "desk_chair", "side_chair", "cafe_chair",
               "armchair"),
    "whiteboard": ("whiteboard", "blackboard", "chalkboard", "board"),
    "tables": ("table", "dining_table", "cafe_table", "bistro_table", "round_table", "coffee_table", "side_table"),
    "counter": ("bar_counter", "counter", "kitchen_counter", "bar", "kitchen_island"),
    "stools": ("bar_stool", "stool", "counter_stool"),
    "plants": ("potted_plant", "plant", "houseplant", "planter"),
    "reception desk": ("reception_desk", "desk", "front_desk", "counter"),
    "low table": ("coffee_table", "table", "low_table"),
    "pendant lamp": ("pendant_lamp", "pendant", "ceiling_light", "chandelier", "lamp"),
    "benches": ("bench", "park_bench"),
    "fountain": ("fountain",),
    "trees": ("tree", "broadleaf_tree", "oak", "conifer", "palm_tree"),
    "trash bin": ("trash_can", "bin", "litter_bin", "waste_bin"),
    "street lamps": ("street_lamp", "lamp_post", "lamp_post_double", "lamp", "light_pole"),
    "pallet racks": ("pallet_rack", "rack", "shelving_unit"),
    "pallets": ("pallet",),
    "crates": ("crate", "box", "cardboard_box", "carton"),
    "forklift": ("forklift",),
    "workbench": ("workbench", "work_table"),
    "safety barriers": ("safety_barrier", "barrier", "railing", "guard_rail", "bollard"),
    "ceiling lights": ("ceiling_light", "pendant_lamp", "high_bay_light", "light", "lamp"),
    "sand": ("sand", "beach"),
    "water": ("water", "sea", "ocean", "lagoon", "lake", "pond"),
    "palm trees": ("palm_tree", "palm", "tree"),
    "bushes": ("bush", "tropical_bush", "fern", "shrub", "hedge"),
    "boardwalk": ("boardwalk", "walkway", "deck", "path", "pier", "jetty"),
    "sun loungers": ("sun_lounger", "lounger", "deckchair", "deck_chair", "sunbed"),
    "parasol": ("parasol", "umbrella", "beach_umbrella"),
    "rocks": ("rock", "boulder", "stone"),
    "desk": ("desk", "writing_desk", "executive_desk"),
    "fireplace": ("fireplace", "mantel"),
    "windows": ("window",),
    "paintings": ("picture_frame", "painting", "picture", "artwork"),
}

#: name words that also name an item (the brief's tokens are kept as they are): a desk's
#: "student_desk_seat_3" is a chair
EXTRA_TOKENS: dict[str, tuple[str, ...]] = {"chairs": ("seat",), "armchairs": ("seat",), "armchair": ("seat",),
                                            "stools": ("barstool",)}
#: categories that do not say what an object is: the name decides (terrain is tagged "ground")
GENERIC_CATEGORIES = frozenset({"", "ground", "terrain", "object", "prop", "props", "furniture", "decor",
                                "decoration", "misc", "other", "generic", "mesh", "asset"})
#: categories too broad to say which item an object is (a "table" may be a coffee table, a desk or
#: a TV stand): the name decides first, the category only when the name matches no item
BROAD_CATEGORIES = frozenset({"table", "chair", "lamp", "light", "desk", "box", "cabinet", "counter", "board",
                              "bench", "seat", "seating", "stand", "shelf", "plant", "tree"})
#: a name match must be within this factor of the item's prior (min / SIZE_GATE .. max x SIZE_GATE):
#: loose, it only rejects the wrong kind of thing (a 3 m x 6 cm fence section as a parasol)
SIZE_GATE = 2.0
THIN_M = 0.1
#: never a checklist object, whatever its name (lights, cameras: "ai_db_daylight_Sun" is no street lamp)
_NOT_OBJECTS = frozenset({"LIGHT", "CAMERA", "SPEAKER", "LIGHT_PROBE", "LIGHTPROBE"})
_COPY = re.compile(r"(\.\d{3})+$")


def _category(o: dict[str, Any]) -> str:
    return re.sub(r"_\d+$", "", str(o.get("category") or "").strip().lower().replace(" ", "_").replace("-", "_"))


def _words(name: str) -> str:
    """Lower-case name with camelCase and separators turned into '_' (word starts are findable)."""
    name = _COPY.sub("", name)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[^a-z0-9]+", "_", name.lower())


def name_match_end(name: str, tokens: tuple[str, ...]) -> int:
    """Where the latest token that starts a word of ``name`` ends, -1 if none does."""
    w = "_" + _words(name)
    best = -1
    for t in tokens:
        t = _words(t).strip("_")
        if not t:
            continue
        for m in re.finditer(r"(?<=[_\d])" + re.escape(t), w):
            best = max(best, m.end())
    return best


def size_gate_ok(size: Any, prior: Any) -> bool:
    """Roughly the item's size: height and both footprint sides (either way round) within
    SIZE_GATE of the prior's range. Unknown size or prior: no gate."""
    if prior is None or not size or len(size) < 3:
        return True
    lo, hi = prior.min_m, prior.max_m

    def within(v: float, a: float, b: float) -> bool:
        # thin sides (a board, a rug) may be modelled as a plane: no lower bound below THIN_M
        return (a / SIZE_GATE if a >= THIN_M else 0.0) <= v <= b * SIZE_GATE

    if not within(size[2], lo[2], hi[2]):
        return False
    w, d = size[0], size[1]
    return (within(w, lo[0], hi[0]) and within(d, lo[1], hi[1])) or \
        (within(d, lo[0], hi[0]) and within(w, lo[1], hi[1]))


def match(items: list[tuple[str, int, tuple[str, ...]]], roots: list[dict[str, Any]],
          priors: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Per checklist item, the roots that count for it (see the module doc)."""
    hits: list[list[dict[str, Any]]] = [[] for _ in items]
    for o in roots:
        if o.get("type") in _NOT_OBJECTS:
            continue
        cat = _category(o)
        if cat not in GENERIC_CATEGORIES and cat not in BROAD_CATEGORIES:  # a tagged object: its category decides
            for k in _by_category(items, cat):
                hits[k].append(o)
            continue
        # no category, a generic or a broad one: the name's head item, if the size agrees
        ends = [name_match_end(o["name"], (*tokens, *EXTRA_TOKENS.get(label, ()))) for label, _need, tokens in items]
        best = max(ends, default=-1)
        named = [k for k, (label, _need, _tokens) in enumerate(items) if best >= 0 and ends[k] == best
                 and size_gate_ok(o.get("size_m"), _prior(priors, label))]
        if not named and cat in BROAD_CATEGORIES:
            # "table_3" tagged table in a cafe, "student_desks_seat_9" tagged chair (a desk's seat):
            # no item by name, the category still says it - unless the name is another kind
            # ("Side_Table_1" tagged table is no coffee table)
            named = [k for k in _by_category(items, cat) if not _names_other_kind(o["name"], items[k][0], priors)]
        for k in named:
            hits[k].append(o)
    return hits


def _names_other_kind(name: str, label: str, priors: dict[str, Any]) -> bool:
    """The name holds a two-word category (side_table, bar_stool) the item does not list."""
    w = "_" + _words(name) + "_"
    own = set(ITEM_CATEGORIES.get(label, ()))
    return any("_" in c and c not in own and re.search(r"(?<=[_\d])" + re.escape(c) + r"(?=[_\d])", w)
               for c in priors)


def _prior(priors: dict[str, Any], label: str) -> Any:
    cats = ITEM_CATEGORIES.get(label, ())
    return priors.get(cats[0]) if cats else None


def _by_category(items: list[tuple[str, int, tuple[str, ...]]], cat: str) -> list[int]:
    """Items whose category list has ``cat`` (an item without one: its tokens as categories)."""
    out = []
    for k, (label, _need, tokens) in enumerate(items):
        cats = ITEM_CATEGORIES.get(label)
        if cats is None:
            cats = tuple(_words(t).strip("_") for t in tokens)
        if cat in cats:
            out.append(k)
    return out
