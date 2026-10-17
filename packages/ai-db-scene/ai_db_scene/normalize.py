"""Rewrite the SceneSpec forms weak models write whose intent is unambiguous, before validation.

Benchmark mining (docs/WEAK_MODEL_FAILURES_2026-10-16.md, item 5) found the same refusals in
scene after scene: ``wall_north`` in a round room, ``gap_m: [0.3, 0.5]``, ``on floor``, a
coffee table ``on`` the rug, a picture frame with no wall, ``count`` next to ``grid``. Each one
cost a retry or an abandoned build. ``normalize_spec`` rewrites them and says what it did; the
notes travel with the spec (``SceneSpec.notes``) into the tool results.

Only shapes with one reading are rewritten; anything else is left for validation to report.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any

from ai_db_scene.constants import LAYOUT_FLAT_MAX_H_M
from ai_db_scene.priors import load_priors

#: the compass walls of a rect room -> the at_deg of the same side of an ellipse/polygon room
#: (x east, y north; at_deg counter-clockwise from east)
_WALL_DEG = {"wall_east": 0.0, "wall_north": 90.0, "wall_west": 180.0, "wall_south": 270.0}
_RECT_WALLS = ("wall_north", "wall_east", "wall_west", "wall_south")  # tie order for a free wall
_FLOOR_WORDS = ("room", "floor", "ground", "the_floor", "the_room")
_COPY = re.compile(r"^(.+?)_?(\d+)$")
#: category names weak models wrote -> the category (bench: a cafe counter as kitchen_counter or
#: "counter", a reception desk as desk or "front_desk")
CATEGORY_ALIASES = {
    "counter": "bar_counter", "bar": "bar_counter", "cafe_counter": "bar_counter", "service_counter": "bar_counter",
    "coffee_counter": "bar_counter", "serving_counter": "bar_counter", "checkout_counter": "bar_counter",
    "front_desk": "reception_desk", "reception": "reception_desk", "reception_counter": "reception_desk",
    "receptionist_desk": "reception_desk", "help_desk": "reception_desk", "information_desk": "reception_desk",
    "barstool": "bar_stool", "counter_stool": "bar_stool", "high_stool": "bar_stool",
    "couch": "sofa", "settee": "sofa", "loveseat": "sofa", "television": "tv", "fridge": "refrigerator",
    "bookcase": "bookshelf", "plant": "potted_plant", "carpet": "rug", "cupboard": "cabinet",
    "table": "dining_table", "lamp": "floor_lamp", "light": "ceiling_light", "painting": "picture_frame",
    "picture": "picture_frame", "artwork": "picture_frame", "art": "picture_frame", "clock": "wall_clock",
}


#: relation names weak models wrote (aidb5/aidb6 invalid results) -> the relation that means it
REL_ALIASES = {
    "near": "beside", "next_to": "beside", "nearby": "beside", "near_to": "beside", "close_to": "beside",
    "adjacent": "beside", "adjacent_to": "beside", "by": "beside", "alongside": "beside",
    "in_row": "beside", "in_a_row": "beside", "row": "beside", "aligned_with": "beside", "in_line_with": "beside",
    "aligned": "beside", "connects": "beside", "connected_to": "beside", "at_edge_of": "beside",
    "at_the_edge_of": "beside", "edge_of": "beside", "along": "beside",
    "opposite": "facing", "opposite_to": "facing", "faces": "facing", "facing_towards": "facing",
    "facing_toward": "facing", "looks_at": "facing", "towards": "facing",
    "in_front": "in_front_of", "front_of": "in_front_of", "before": "in_front_of",
    "on_top_of": "on", "atop": "on", "on_top": "on", "resting_on": "on", "stacked_on": "on", "placed_on": "on",
    "inside": "on", "in": "on", "onto": "on",
    "against": "against_wall", "along_wall": "against_wall", "on_wall": "against_wall", "wall": "against_wall",
    "mounted_on": "against_wall", "hung_on": "against_wall",
    "centered": "centered_in", "center": "centered_in", "centre": "centered_in", "in_center_of": "centered_in",
    "in_centre_of": "centered_in", "middle_of": "centered_in", "in_middle_of": "centered_in",
    "centered_on": "centered_in",
    "below": "under", "beneath": "under", "underneath": "under",
    "over": "above", "hanging_above": "above", "hangs_above": "above",
    "surrounding": "around", "surround": "around", "surrounds": "around", "circle": "around", "encircle": "around",
}
#: corner relations: against both walls of the corner (b names it: north_west, ne, corner_se, ...)
_CORNER_RELS = ("corner", "at_corner", "in_corner", "in_the_corner", "corner_of")
#: relations that say only "spread the copies" or "somewhere": nothing for the solver to do
_EMPTY_RELS = ("evenly_spaced", "spaced", "distributed", "spread", "scattered", "position", "positioned",
               "placed", "located", "random", "anywhere")
#: compass words and room sides -> the rect wall (front: the wall a room is entered toward)
_SIDE = {"north": "wall_north", "n": "wall_north", "front": "wall_north", "top": "wall_north",
         "south": "wall_south", "s": "wall_south", "back": "wall_south", "rear": "wall_south", "bottom": "wall_south",
         "east": "wall_east", "e": "wall_east", "right": "wall_east",
         "west": "wall_west", "w": "wall_west", "left": "wall_west"}
_WALL_WORDS = ("wall", "side", "the", "corner", "of", "room", "ern")
#: shape words -> setting.shape
SHAPE_ALIASES = {"round": "ellipse", "circle": "ellipse", "circular": "ellipse", "oval": "ellipse",
                 "elliptical": "ellipse", "ellipse": "ellipse", "rotunda": "ellipse", "square": "rect",
                 "rectangle": "rect", "rectangular": "rect", "box": "rect", "rect": "rect"}
#: a brief that asks for a round room (the model left setting.shape out: qwen aidb5 round_lobby)
_ROUND_BRIEF = re.compile(r"\b(round|circular|oval|elliptical|rotunda)\b(?!\s+(?:table|tables|rug|rugs|mirror|"
                          r"mirrors|window|windows|pot|pots|lamp|lamps|light|lights|stool|stools|ottoman|ottomans|"
                          r"plate|plates|coffee|side|dining|planter|planters|fountain|bed|beds))", re.IGNORECASE)


def _key(v: Any) -> str:
    return re.sub(r"[\s\-]+", "_", v.strip().lower()) if isinstance(v, str) else ""


def wall_side(b: Any) -> str | None:
    """``north_wall`` / ``north`` / ``the north side`` / ``front_wall`` -> ``wall_north``; else None."""
    k = _key(b)
    if k in _RECT_WALLS:
        return k
    words = [w for w in re.split(r"_+", k) if w and w not in _WALL_WORDS]
    words = [w.removesuffix("ern") if w.endswith("ern") else w for w in words]
    sides = {_SIDE[w] for w in words if w in _SIDE}
    return next(iter(sides)) if len(sides) == 1 and len(words) == 1 else None


def corner_walls(b: Any) -> list[str] | None:
    """``north_west`` / ``nw`` / ``corner_se`` / ``northeast`` -> its two walls; one side -> that wall."""
    k = _key(b)
    for a, c in (("north", "n"), ("south", "s")):
        for e, f in (("east", "e"), ("west", "w")):
            for form in (f"{a}_{e}", f"{a}{e}", f"{c}{f}", f"{e}_{a}", f"{f}{c}"):
                if re.search(rf"(^|_){form}($|_)", k):
                    return [_SIDE[a], _SIDE[e]]
    side = wall_side(b)
    return [side] if side else None


def room_shape_from_brief(data: Any, brief: Any) -> tuple[Any, str | None]:
    """(spec, note): a brief asking for a round room gives an indoor spec with no ``setting.shape``
    an ellipse. qwen3-14b-local-aidb5 round_lobby left shape out and built a rectangular lobby."""
    if not isinstance(data, dict) or not isinstance(brief, str) or not _ROUND_BRIEF.search(brief):
        return data, None
    st = data.get("setting")
    if not isinstance(st, dict) or st.get("kind", "indoor") != "indoor" or "shape" in st or st.get("rooms"):
        return data, None
    out = {**data, "setting": {**st, "shape": "ellipse"}}
    word = _ROUND_BRIEF.search(brief).group(1)  # type: ignore[union-attr]
    return out, f"setting.shape: the brief asks for a {word.lower()} room: shape 'ellipse' (one curved wall)"


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _gap(v: Any) -> tuple[Any, bool]:
    """(gap_m, changed): a range [lo, hi] -> its midpoint, "0.5" -> 0.5."""
    if isinstance(v, (list, tuple)) and v and all(_num(x) for x in v):
        return round(sum(v) / len(v), 3), True
    if isinstance(v, str):
        try:
            return float(v.strip().removesuffix("m").strip()), True
        except ValueError:
            return v, False
    return v, False


def _base_id(v: Any, ids: set[str]) -> str | None:
    """``plant1_1`` / ``chair_3`` (a built copy) -> its declared id, else None."""
    if not isinstance(v, str) or v in ids:
        return None
    m = _COPY.match(v)
    if m and m.group(1) in ids:
        return m.group(1)
    return None


def normalize_spec(data: Any) -> tuple[Any, list[str]]:
    """(rewritten copy of ``data``, one note per rewrite). Never raises: on input it cannot read
    it returns ``data`` unchanged and lets validation explain."""
    if not isinstance(data, dict):
        return data, []
    try:
        return _normalize(copy.deepcopy(data))
    except Exception:  # noqa: BLE001 - malformed input: validation reports it properly
        return data, []


def _normalize(d: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    raw_setting = d.get("setting")
    setting: dict[str, Any] = raw_setting if isinstance(raw_setting, dict) else {}
    alias = SHAPE_ALIASES.get(_key(setting.get("shape")))
    if "shape" in setting and alias and alias != setting["shape"]:
        notes.append(f"setting.shape: {setting['shape']!r} read as '{alias}'")
        setting["shape"] = alias
    shape = setting.get("shape", "rect")
    indoor = setting.get("kind") == "indoor"
    objs = [o for o in d.get("objects") or [] if isinstance(o, dict)] if isinstance(d.get("objects"), list) else []
    ids = {o["id"] for o in objs if isinstance(o.get("id"), str)}
    priors = load_priors()
    cat = {o["id"]: o.get("category") for o in objs if isinstance(o.get("id"), str)}

    def prior(oid: Any) -> Any:
        return priors.get(cat.get(oid)) if isinstance(oid, str) else None  # type: ignore[arg-type]

    for i, o in enumerate(objs):  # names models use for a category: the category
        c = o.get("category")
        alias = CATEGORY_ALIASES.get(c.strip().lower().replace(" ", "_")) if isinstance(c, str) else None
        if alias and c not in priors and alias in priors:
            o["category"] = alias
            cat[o.get("id")] = alias
            notes.append(f"objects[{i}].category: '{c}' read as '{alias}'")
    for i, o in enumerate(objs):  # count next to grid: the grid decides
        g = o.get("grid")
        if isinstance(g, dict) and isinstance(g.get("rows"), int) and isinstance(g.get("cols"), int) \
                and "count" in o and o["count"] != g["rows"] * g["cols"]:
            notes.append(f"objects[{i}]: count {o.pop('count')!r} ignored; the grid "
                         f"{g['rows']}x{g['cols']} decides ({g['rows'] * g['cols']})")
    rels = d.get("relations")
    if not isinstance(rels, list):
        return d, notes
    out: list[Any] = []
    for i, raw in enumerate(rels):
        if not isinstance(raw, dict):
            out.append(raw)
            continue
        p = f"relations[{i}]"
        got = _vocab(dict(raw), p, ids, shape, setting.get("kind"), notes, objs)
        if not got:
            continue
        if len(got) > 1:
            out += got[:-1]
        r = got[-1]
        for k in ("a", "b") if r.get("rel") != "against_wall" else ("a",):
            base = _base_id(r.get(k), ids)
            if base:
                notes.append(f"{p}.{k}: '{r[k]}' read as '{base}' (relations name the declared object id)")
                r[k] = base
        if "gap_m" in r:
            gap, changed = _gap(r["gap_m"])
            if changed:
                notes.append(f"{p}.gap_m: {r['gap_m']!r} read as {gap!r}")
                r["gap_m"] = gap
        rel, b = r.get("rel"), r.get("b")
        if rel == "on" and isinstance(b, str) and b.lower() in _FLOOR_WORDS:
            notes.append(f"{p}: dropped '{r.get('a')} on {b}' (floor objects stand on the floor already)")
            continue
        if rel == "centered_in" and isinstance(b, str) and b.lower() in _FLOOR_WORDS and b != "room":
            r["b"] = "room"
        if rel == "against_wall" and shape in ("ellipse", "polygon") and b in _WALL_DEG:
            r["b"] = "wall"
            if not _num(r.get("at_deg")):
                r["at_deg"] = _WALL_DEG[b]
            notes.append(f"{p}: '{b}' read as wall at_deg {r['at_deg']:g} (a {shape} room has one wall)")
        if "at_deg" in r and (rel != "against_wall" or shape == "rect"):
            notes.append(f"{p}.at_deg: ignored ({'rect rooms use wall_north..' if shape == 'rect' else 'only on against_wall'})")
            r.pop("at_deg")
        out.append(r)
    seen: set[str] = set()
    for r in list(out):  # 'next_to' read as 'beside' next to its own 'beside'
        k = repr(sorted(r.items())) if isinstance(r, dict) else ""
        if k and k in seen:
            out.remove(r)
            notes.append(f"dropped a repeated relation '{r.get('a')} {r.get('rel')} {r.get('b')}'")
        seen.add(k)
    out = _rugs(out, ids, prior, notes)
    if indoor:
        out = _desk_rows(d, out, setting, prior, notes)
        objs = [o for o in d.get("objects") or [] if isinstance(o, dict)]
        out = _seating_group(out, objs, setting, notes)
    out = _seats(d, out, objs, notes)
    objs = [o for o in d.get("objects") or [] if isinstance(o, dict)]
    _spread_centered(out, objs, setting, prior, notes)
    if indoor:
        out = _ceiling_centres(out, objs, prior, notes)
    if indoor:
        out = _wall_mounts(out, objs, shape, prior, notes)
    if indoor and shape == "ellipse":
        out = _curved_wall_pieces(out, objs, notes)
    d["relations"] = out
    return d, notes


#: "the copies of A in a row": A's count becomes a one-row grid
_ROW_RELS = ("in_row", "in_a_row", "row", "aligned", "aligned_with", "in_line_with")


def _vocab(r: dict[str, Any], p: str, ids: set[str], shape: Any, kind: Any, notes: list[str],
           objs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The relation(s) a relation written with another word means, or [] when it says nothing the
    solver can use (each rewrite noted). qwen aidb6 wrote near, in_row, opposite, corner,
    at_corner, aligned_with, connects, at_edge_of, evenly_spaced and walls as north_wall / south /
    front_wall; each was refused and cost a retry."""
    rel, a, b = r.get("rel"), r.get("a"), r.get("b")
    if not isinstance(rel, str):
        return [r]
    k = _key(rel)
    if k != rel and k in (*_REL_NAMES, *REL_ALIASES, *_CORNER_RELS, *_EMPTY_RELS):
        r["rel"] = rel = k
    label = f"'{a} {k} {b}'"
    obj = next((o for o in objs if o.get("id") == a), None)
    if a == b and k in _ROW_RELS and obj is not None and obj.get("grid") is None and _n(obj) > 1:
        obj["grid"] = {"rows": 1, "cols": _n(obj)}
        obj.pop("count", None)
        notes.append(f"{p}: {label} read as a one-row grid of the {_n(obj)} copies")
        return []
    if k in _EMPTY_RELS or (k not in _REL_NAMES and a == b):
        notes.append(f"{p}: dropped {label} (copies of an object are spread by the solver; use count or a grid "
                     "for a row)")
        return []
    on_obj = isinstance(b, str) and b in ids
    if k in _CORNER_RELS or (k not in _REL_NAMES and not on_obj and corner_walls(b) and len(corner_walls(b) or []) == 2):
        if on_obj:
            r["rel"] = "beside"
            notes.append(f"{p}: {label} read as beside '{b}'")
            return [r]
        walls = corner_walls(b) if kind == "indoor" else None
        if not walls:
            notes.append(f"{p}: dropped {label} ({'no walls outdoors' if kind != 'indoor' else 'name the corner, e.g. north_west'})")
            return []
        if shape in ("ellipse", "polygon"):
            ang = [_WALL_DEG[w] for w in walls]
            deg = ang[0] if len(ang) == 1 else (sum(ang) / 2 if abs(ang[0] - ang[1]) <= 180 else (sum(ang) / 2 + 180) % 360)
            notes.append(f"{p}: {label} read as against_wall wall at_deg {deg:g}")
            return [{"a": a, "rel": "against_wall", "b": "wall", "at_deg": deg}]
        notes.append(f"{p}: {label} read as against_wall " + " + ".join(walls) + " (in the corner)")
        return [{"a": a, "rel": "against_wall", "b": w} for w in walls]
    if k in REL_ALIASES:
        r["rel"] = REL_ALIASES[k]
        if on_obj or r["rel"] == "against_wall" or b == "room" or wall_side(b):
            notes.append(f"{p}: rel '{k}' read as '{r['rel']}'")
    rel = r["rel"]
    side = None if on_obj else wall_side(b)
    if rel == "on" and _key(b) in ("ceiling", "the_ceiling", "roof"):
        notes.append(f"{p}: dropped '{a} on {b}' (ceiling objects hang from the ceiling already)")
        return []
    if side is not None and rel in ("against_wall", "on", "beside", "in_front_of", "centered_in"):
        if kind != "indoor":
            notes.append(f"{p}: dropped '{a} {rel} {b}' (no walls outdoors)")
            return []
        if rel != "against_wall" or b != side:
            notes.append(f"{p}: '{rel} {b}' read as against_wall {side}")
        r["rel"], r["b"] = "against_wall", side
        return [r]
    if rel == "against_wall" and kind not in (None, "indoor"):
        notes.append(f"{p}: dropped '{a} against_wall {b}' (no walls outdoors)")
        return []
    if k in REL_ALIASES and rel != "against_wall" and not on_obj and isinstance(b, str) and b not in ("room",):
        notes.append(f"{p}: dropped '{a} {k} {b}' ('{b}' is not an object of the spec)")
        return []
    return [r]


#: spec.RELATIONS (kept local: spec imports this module)
_REL_NAMES = ("on", "against_wall", "in_front_of", "beside", "facing", "centered_in", "under", "around", "above")


def _positional(rels: list[Any], oid: str, skip: Any = None) -> bool:
    return any(isinstance(r, dict) and r is not skip and r.get("a") == oid and r.get("rel") != "facing" for r in rels)


def _rugs(rels: list[Any], ids: set[str], prior: Any, notes: list[str]) -> list[Any]:
    """``coffee_table on rug`` / ``rug under coffee_table``: a rug carries nothing and is wider
    than what stands on it. The rug is centred under the object (``rug centered_in X``), or, when
    the rug has a place of its own, the object stands within the rug (``X centered_in rug``)."""
    out = list(rels)
    for r in list(out):
        if not isinstance(r, dict) or r.get("a") not in ids or r.get("b") not in ids:
            continue
        rel = r.get("rel")
        if rel == "on":
            x, rug = r["a"], r["b"]
        elif rel == "under":
            rug, x = r["a"], r["b"]
        else:
            continue
        pr, px = prior(rug), prior(x)
        if pr is None or px is None or pr.can_support_others or pr.typical_m[2] > LAYOUT_FLAT_MAX_H_M \
                or px.typical_m[2] <= LAYOUT_FLAT_MAX_H_M:
            continue
        i = out.index(r)
        if _positional(out, rug, skip=r):
            out[i] = {"a": x, "rel": "centered_in", "b": rug}
            notes.append(f"'{r['a']} {rel} {r['b']}' read as '{x} centered_in {rug}' (stands within the rug)")
        else:
            out[i] = {"a": rug, "rel": "centered_in", "b": x}
            notes.append(f"'{r['a']} {rel} {r['b']}' read as '{rug} centered_in {x}' (the rug lies under it)")
    return out


#: what the students look at: desks in rows face it
_BOARDS = ("whiteboard", "blackboard", "chalkboard", "projector_screen")
#: desks that make rows, and how many copies make a block of rows
_ROW_DESKS = ("desk",)
_ROW_MIN = 4
#: the board's end of the room kept free of student rows (board, teacher desk, aisle), m
_FRONT_ZONE_M = 2.5
#: a chair's depth behind its desk, added to the desk's depth for one row
_SEAT_DEPTH_M = 0.6


def _desk_rows(d: dict[str, Any], rels: list[Any], setting: dict[str, Any], prior: Any,
               notes: list[str]) -> list[Any]:
    """Student desks as a grid in rows facing the board.

    Bench classroom: qwen aidb4/aidb5 declared student_desk_1 .. student_desk_12 chained by
    ``beside`` (each with its own chair ``beside`` it) and the solver could not place the chain
    (aidb5: 976x out of bounds); qwen aidb6 wrote one desk object with count 12 ``in_front_of``
    the whiteboard and the copies stood scattered over the room. Four or more desks of one name
    chained by beside / in_row, or one desk object with four or more copies that only faces the
    board, become one grid of rows facing the board; per-desk chairs become one chair object
    beside the block (``_seats`` then gives every desk its chair)."""
    objs = [o for o in d.get("objects") or [] if isinstance(o, dict) and isinstance(o.get("id"), str)]
    boards = [o["id"] for o in objs if o.get("category") in _BOARDS]
    board_wall = next((r.get("b") for r in rels if isinstance(r, dict) and r.get("a") in boards
                       and r.get("rel") == "against_wall"), None)
    out = list(rels)
    # 1. separate desks chained by beside: one block
    single = [o["id"] for o in objs if o.get("category") in _ROW_DESKS and o.get("grid") is None and _n(o) == 1]
    root = {i: i for i in single}

    def find(i: str) -> str:
        while root[i] != i:
            i = root[i]
        return i

    for r in out:
        if isinstance(r, dict) and r.get("rel") == "beside" and r.get("a") in root and r.get("b") in root:
            root[find(r["a"])] = find(r["b"])
    comps: dict[str, list[str]] = {}
    for i in single:
        comps.setdefault(find(i), []).append(i)
    for comp in comps.values():
        if len(comp) < _ROW_MIN:
            continue
        # the copies share a name (student_desk_1 .. 12); a teacher desk chained to them stays out
        names: dict[str, list[str]] = {}
        for i in comp:
            m = _COPY.match(i)
            names.setdefault(m.group(1) if m else i, []).append(i)
        base, members = max(names.items(), key=lambda kv: len(kv[1]))
        if len(members) < _ROW_MIN:
            continue
        out = _merge_block(d, out, base, members, boards, board_wall, setting, prior, notes)
    # 2. one desk object with copies that only faces the board (or has no place at all)
    for o in [o for o in d.get("objects") or [] if isinstance(o, dict)]:
        oid = o.get("id")
        if not boards or o.get("category") not in _ROW_DESKS or o.get("grid") is not None or _n(o) < _ROW_MIN:
            continue
        mine = [r for r in out if isinstance(r, dict) and r.get("a") == oid]
        if any(r.get("rel") not in ("in_front_of", "facing") or r.get("b") not in boards for r in mine):
            continue
        out = _merge_block(d, out, str(oid), [str(oid)], boards, board_wall, setting, prior, notes)
    return out


def _merge_block(d: dict[str, Any], rels: list[Any], base: str, members: list[str], boards: list[str],
                 board_wall: Any, setting: dict[str, Any], prior: Any, notes: list[str]) -> list[Any]:
    objs = [o for o in d.get("objects") or [] if isinstance(o, dict)]
    by_id = {o.get("id"): o for o in objs}
    first = by_id[members[0]]
    n = sum(_n(by_id[m]) for m in members)
    block = base if base not in by_id or base in members else members[0]
    p = prior(members[0])
    dw, dd = (p.typical_m[0], p.typical_m[1]) if p is not None else (1.2, 0.6)
    rows, cols = _rows_cols(n, setting, board_wall, dw, dd + _SEAT_DEPTH_M)
    new = {k: v for k, v in first.items() if k != "count"}
    new.update(id=block, grid={"rows": rows, "cols": cols})
    keep = set(members)
    d["objects"] = [new if o is first else o for o in d["objects"]
                    if not (isinstance(o, dict) and o.get("id") in keep and o is not first)]
    out: list[Any] = []
    for r in rels:
        if not isinstance(r, dict):
            out.append(r)
            continue
        a, b = r.get("a"), r.get("b")
        if a in keep and b in keep:
            continue  # the chain itself: the grid's rows
        r = {**r, "a": block if a in keep else a, "b": block if b in keep else b}
        if r["a"] == block and r.get("rel") == "in_front_of" and r["b"] in boards:
            r = {"a": block, "rel": "facing", "b": r["b"]}  # the board's front is the teacher's place
        if r not in out:
            out.append(r)
    if boards and not any(r.get("a") == block and r.get("rel") == "facing" for r in out if isinstance(r, dict)):
        out.append({"a": block, "rel": "facing", "b": boards[0]})
    what = f"{len(members)} desks chained by beside" if len(members) > 1 else f"'{block}' ({n} copies)"
    notes.append(f"{what} laid out as '{block}', a {rows}x{cols} grid in rows"
                 + (f" facing '{boards[0]}'" if boards else ""))
    return _block_chairs(d, out, block, n, boards, notes)


def _block_chairs(d: dict[str, Any], rels: list[Any], block: str, n: int, boards: list[str],
                  notes: list[str]) -> list[Any]:
    """Per-desk chairs (student_chair_1 beside student_desk_1, ...) -> one chair object beside the
    block, so ``_seats`` puts a chair at every desk of the grid."""
    objs = [o for o in d.get("objects") or [] if isinstance(o, dict)]
    groups: dict[str, list[dict[str, Any]]] = {}
    loose: dict[str, int] = {}  # chairs with no relation at all (qwen aidb5): one per desk only
    for o in objs:
        oid = o.get("id")
        if o.get("category") not in _SEAT_CATS or o.get("grid") is not None or _n(o) != 1:
            continue
        mine = [r for r in rels if isinstance(r, dict) and oid in (r.get("a"), r.get("b"))]
        if not mine or (all(r.get("a") == oid and (r.get("b") == block or r.get("b") in boards) for r in mine)
                        and any(r.get("b") == block for r in mine)):
            groups.setdefault(o["category"], []).append(o)
            loose[o["category"]] = loose.get(o["category"], 0) + (not mine)
    for cat_, chairs in groups.items():
        if len(chairs) < 2 or n % len(chairs) or (loose[cat_] and len(chairs) != n):
            continue
        ids = {c["id"] for c in chairs}
        m = _COPY.match(chairs[0]["id"])
        cid = m.group(1) if m and m.group(1) not in {o.get("id") for o in objs} else chairs[0]["id"]
        merged = {**{k: v for k, v in chairs[0].items() if k != "count"}, "id": cid, "count": len(chairs)}
        d["objects"] = [merged if o is chairs[0] else o for o in d["objects"]
                        if not (isinstance(o, dict) and o.get("id") in ids and o is not chairs[0])]
        rels = [r for r in rels if not (isinstance(r, dict) and r.get("a") in ids)]
        rels.append({"a": cid, "rel": "beside", "b": block})
        notes.append(f"{len(chairs)} {cat_}s for the desks merged into '{cid}' (count {len(chairs)}), beside '{block}'")
    return rels


def _rows_cols(n: int, setting: dict[str, Any], board_wall: Any, cell_w: float, cell_d: float) -> tuple[int, int]:
    """(rows, cols) of ``n`` desks: the block fits the room in front of the board and keeps its
    proportions (cols run along the board's wall)."""
    bounds = setting.get("bounds_m")
    w, dp = (float(bounds[0]), float(bounds[1])) if isinstance(bounds, list) and len(bounds) >= 2 \
        and all(_num(v) for v in bounds[:2]) else (10.0, 8.0)
    across, depth = (dp, w) if board_wall in ("wall_east", "wall_west") else (w, dp)
    depth = max(depth - _FRONT_ZONE_M, 1.0)
    from ai_db_scene.constants import LAYOUT_GRID_GAP_M

    gx, gy = LAYOUT_GRID_GAP_M
    best: tuple[tuple[bool, float], int, int] | None = None
    for rows in range(1, n + 1):
        if n % rows:
            continue
        cols = n // rows
        bw, bd = cols * cell_w + (cols - 1) * gx, rows * cell_d + (rows - 1) * gy
        score = (not (bw <= 0.85 * across and bd <= depth), abs(math.log((bw / bd) / (across / depth))))
        if best is None or score < best[0]:
            best = (score, rows, cols)
    assert best is not None
    return best[1], best[2]


#: categories a ``seats`` table may have, and the seats it may get
_SEAT_CATS = ("chair", "dining_chair", "office_chair", "stool", "bar_stool")


def _n(o: dict[str, Any]) -> int:
    g = o.get("grid")
    if isinstance(g, dict) and isinstance(g.get("rows"), int) and isinstance(g.get("cols"), int):
        return int(g["rows"] * g["cols"])
    c = o.get("count", 1)
    return c if isinstance(c, int) and not isinstance(c, bool) else 1


def _seats(d: dict[str, Any], rels: list[Any], objs: list[dict[str, Any]], notes: list[str]) -> list[Any]:
    """Chairs as a block of their own next to a block of desks/tables (``chairs in_front_of desks``,
    both grids; bench classroom: "chairs clustered apart from the desks", 2300x out of bounds) become
    ``seats`` on the desks: the solver then puts a chair at every desk."""
    from ai_db_scene.layout import SEAT_CATEGORY
    from ai_db_scene.spec import MAX_SEATS

    by_id = {o.get("id"): o for o in objs}
    for r in list(rels):
        if not isinstance(r, dict) or r.get("rel") not in ("in_front_of", "around", "beside"):
            continue
        chairs, table = by_id.get(r.get("a")), by_id.get(r.get("b"))
        if chairs is None or table is None or chairs.get("category") not in _SEAT_CATS \
                or table.get("category") not in SEAT_CATEGORY or table.get("seats"):
            continue
        nt, nc = _n(table), _n(chairs)
        if not (table.get("grid") or nt > 1 or chairs.get("grid")) or nc % nt or not 1 <= nc // nt <= MAX_SEATS:
            continue
        cid, tid = chairs["id"], table["id"]
        table["seats"] = nc // nt
        if chairs["category"] != SEAT_CATEGORY[table["category"]]:
            table["seat_category"] = chairs["category"]
        d["objects"] = [o for o in d["objects"] if not (isinstance(o, dict) and o.get("id") == cid)]
        objs.remove(chairs)
        gone = [x for x in rels if isinstance(x, dict) and cid in (x.get("a"), x.get("b"))]
        rels = [x for x in rels if x not in gone]
        if isinstance(d.get("camera"), dict) and d["camera"].get("focus") == cid:
            d["camera"]["focus"] = tid
        labels = [" ".join(str(x.get(k)) for k in ("a", "rel", "b")) for x in gone]
        notes.append(f"'{cid}' ({nc} {chairs['category']}) became seats: {nc // nt} on each '{tid}', so every "
                     f"{table['category']} has its chair (dropped: {labels})")
    return rels


def _spread_centered(rels: list[Any], objs: list[dict[str, Any]], setting: dict[str, Any], prior: Any,
                     notes: list[str]) -> None:
    """``ceiling_lights (count 6) centered_in room``: six copies cannot share the centre (bench:
    the repair loop grew a warehouse to 69 x 104 m). The copies become a grid spread over the
    middle of the room, still centred."""
    bounds = setting.get("bounds_m")
    if not (isinstance(bounds, list) and len(bounds) == 3 and all(_num(v) for v in bounds)):
        return
    w, d = float(bounds[0]), float(bounds[1])
    for o in objs:
        n, oid = o.get("count"), o.get("id")
        if not isinstance(n, int) or n < 2 or o.get("grid") is not None or prior(oid) is None:
            continue
        mine = [r for r in rels if isinstance(r, dict) and r.get("a") == oid]
        if not any(r.get("rel") == "centered_in" and r.get("b") == "room" for r in mine) \
                or any(r.get("rel") not in ("centered_in", "facing") for r in mine):
            continue
        # rows x cols == n, the block's aspect close to the room's
        rows, cols = min(((r, n // r) for r in range(1, n + 1) if n % r == 0),
                         key=lambda rc: abs(math.log((rc[1] / rc[0]) / (w / d))))
        cw, cd = prior(oid).typical_m[:2]
        gx = max(0.3, (0.6 * w - cols * cw) / (cols - 1)) if cols > 1 else 0.5
        gy = max(0.3, (0.6 * d - rows * cd) / (rows - 1)) if rows > 1 else 0.8
        o["grid"] = {"rows": rows, "cols": cols, "gap_m": [round(gx, 2), round(gy, 2)]}
        o.pop("count")
        notes.append(f"'{oid}': {n} copies centered_in room spread as a {rows}x{cols} grid in the middle of the room")


#: the opposite wall of a rect room
_OPPOSITE = {"wall_north": "wall_south", "wall_south": "wall_north", "wall_east": "wall_west", "wall_west": "wall_east"}
#: coffee table in front of the sofa (m): knee room, still in reach
_SOFA_TABLE_GAP_M = 0.45
#: armchair beside the coffee table's short end (m)
_CHAIR_TABLE_GAP_M = 0.35
#: armchair relations a model writes for "two armchairs" that the group reads naturally instead
_LOOSE_CHAIR_RELS = ("beside", "facing", "in_front_of", "centered_in", "around")


def _seating_group(rels: list[Any], objs: list[dict[str, Any]], setting: dict[str, Any], notes: list[str]) -> list[Any]:
    """A living-room seating group: the sofa anchors it against a wall, the coffee table stands
    centred in front of it, the rug lies centred under the table, armchairs sit at the table's
    short ends facing it and the TV stand stands against the opposite wall facing the sofa.

    Bench (Marc's review of the aidb5 living rooms): armchairs ``beside sofa`` sat at the sofa's
    ends facing the same way, the table was off the sofa's axis, the rug ran half under one end
    of the sofa or floated mid-room, the TV stand was off-axis. Only what the model left open or
    wrote loosely is rewritten (each change in a note); explicit walls and other anchors stay."""
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for o in objs:
        if isinstance(o.get("id"), str) and isinstance(o.get("category"), str):
            by_cat.setdefault(o["category"], []).append(o)
    sofas, tables = by_cat.get("sofa", []), by_cat.get("coffee_table", [])
    if len(sofas) != 1 or len(tables) != 1 or _n(sofas[0]) != 1 or _n(tables[0]) != 1:
        return rels
    sofa, table = sofas[0]["id"], tables[0]["id"]
    out = [r for r in rels]

    def mine(oid: str) -> list[dict[str, Any]]:
        return [r for r in out if isinstance(r, dict) and r.get("a") == oid]

    def wall_of(oid: str) -> str | None:
        return next((r.get("b") for r in mine(oid) if r.get("rel") == "against_wall"), None)

    def replace(oid: str, new: list[dict[str, Any]], why: str) -> None:
        old = [" ".join(str(r.get(k)) for k in ("a", "rel", "b")) for r in mine(oid)]
        now = [" ".join(str(r.get(k)) for k in ("a", "rel", "b")) for r in new]
        if old == now:
            return
        at = next((i for i, r in enumerate(out) if isinstance(r, dict) and r.get("a") == oid), len(out))
        rest = [r for r in out if not (isinstance(r, dict) and r.get("a") == oid)]
        at -= sum(1 for r in out[:at] if isinstance(r, dict) and r.get("a") == oid)
        out[:] = rest[:at] + new + rest[at:]  # where the object's first relation was: error indices hold
        notes.append(f"seating group: '{oid}' {now}" + (f" instead of {old}" if old else "") + f" ({why})")

    stands = [o["id"] for o in by_cat.get("tv_stand", []) + by_cat.get("media_console", []) if _n(o) == 1]
    stand = stands[0] if len(stands) == 1 else None
    rect = setting.get("shape", "rect") == "rect"
    bounds = setting.get("bounds_m")
    # 1. the sofa against a wall: opposite the TV stand's, else a long wall
    if rect and wall_of(sofa) is None and not any(r.get("rel") in ("on", "centered_in", "in_front_of", "beside", "around")
                                                  for r in mine(sofa)):
        tv_wall = wall_of(stand) if stand else None
        long_x = not (isinstance(bounds, list) and len(bounds) >= 2 and _num(bounds[0]) and _num(bounds[1])
                      and float(bounds[0]) < float(bounds[1]))
        w = _OPPOSITE.get(tv_wall or "") or ("wall_south" if long_x else "wall_west")
        replace(sofa, [{"a": sofa, "rel": "against_wall", "b": w}], "the sofa anchors the group; it faces into the room")
    # 2. the TV stand against the opposite wall, facing the sofa (on the sofa's axis)
    sw = wall_of(sofa)
    if stand and rect and sw in _OPPOSITE and not any(r.get("rel") in ("on", "beside", "in_front_of", "centered_in", "around")
                                                      for r in mine(stand)):
        replace(stand, [{"a": stand, "rel": "against_wall", "b": wall_of(stand) or _OPPOSITE[sw]},
                        {"a": stand, "rel": "facing", "b": sofa}], "on the sofa's axis, facing it")
    # 2b. the group centred between the sofa's wall and the TV: a sofa flush with the wall of a
    # deep room left half the room empty behind the coffee table (qwen aidb6 living room)
    if stand and rect and sw in _OPPOSITE and wall_of(stand) == _OPPOSITE[sw]:
        _float_sofa(out, sofa, stand, sw, bounds, by_cat, notes)
    # 3. the coffee table centred in front of the sofa
    trels = mine(table)
    if not any(r.get("rel") == "in_front_of" for r in trels) and \
            all((r.get("rel") == "facing" and r.get("b") == sofa) or (r.get("rel") == "centered_in" and r.get("b") == "room")
                for r in trels):
        replace(table, [{"a": table, "rel": "in_front_of", "b": sofa, "gap_m": _SOFA_TABLE_GAP_M}],
                "centred in front of the sofa")
    # 4. the rug centred under the coffee table (symmetric about the sofa's axis)
    for rug in by_cat.get("rug", []):
        rid = rug["id"]
        if _n(rug) == 1 and all(r.get("rel") == "centered_in" for r in mine(rid)) and \
                not any(isinstance(r, dict) and r.get("b") == rid for r in out):
            replace(rid, [{"a": rid, "rel": "centered_in", "b": table}], "the rug lies centred under the coffee table")
            break
    # 5. armchairs at the coffee table's short ends, facing it
    chairs = [o["id"] for o in by_cat.get("armchair", []) if _n(o) == 1]
    if len(chairs) in (1, 2) and all(all(r.get("rel") in _LOOSE_CHAIR_RELS and r.get("b") in (sofa, table, stand, "room", *chairs)
                                         for r in mine(c)) for c in chairs):
        for c in chairs:
            replace(c, [{"a": c, "rel": "beside", "b": table, "gap_m": _CHAIR_TABLE_GAP_M},
                        {"a": c, "rel": "facing", "b": table}], "at the coffee table's short end, facing it")
    return out


#: never closer than this from the sofa's front to the TV stand's (m), nor a narrower walkway behind it
_VIEW_MIN_M = 2.0
_WALKWAY_M = 0.6
_FLOAT_MAX_M = 1.5


def _float_sofa(rels: list[Any], sofa: str, stand: str, wall: str, bounds: Any, by_cat: dict[str, list[dict[str, Any]]],
                notes: list[str]) -> None:
    """The sofa's ``against_wall`` gets a ``gap_m`` that centres the seating group (sofa, coffee
    table, armchairs at its ends) between the sofa's wall and the TV stand: half the free depth
    left over on each side, none when that would leave less than a walkway behind the sofa or put
    the sofa closer than _VIEW_MIN_M to the TV. A gap the model wrote is kept."""
    if not (isinstance(bounds, list) and len(bounds) >= 2 and all(_num(v) for v in bounds[:2])):
        return
    rel = next((r for r in rels if isinstance(r, dict) and r.get("a") == sofa and r.get("rel") == "against_wall"), None)
    if rel is None or rel.get("gap_m") is not None:
        return
    priors = load_priors()

    def depth(cat: str, axis: int = 1) -> float:
        p = priors.get(cat)
        return float(p.typical_m[axis]) if p is not None else 0.0

    room = float(bounds[1] if wall in ("wall_north", "wall_south") else bounds[0])
    stand_cat = next(o["category"] for k in ("tv_stand", "media_console") for o in by_cat.get(k, []) if o["id"] == stand)
    free = room - depth(stand_cat)
    table = depth("coffee_table")
    reach = table / 2 + (max(table / 2, depth("armchair", 0) / 2) if by_cat.get("armchair") else table / 2)
    group = depth("sofa") + _SOFA_TABLE_GAP_M + reach
    gap = min((free - group) / 2, _FLOAT_MAX_M, free - depth("sofa") - _VIEW_MIN_M)
    if gap < _WALKWAY_M:
        return
    rel["gap_m"] = round(gap, 2)
    notes.append(f"seating group: '{sofa}' stands {gap:.2f} m off {wall} so the group is centred between that wall "
                 f"and '{stand}' (gap_m on its against_wall)")


def _ceiling_centres(rels: list[Any], objs: list[dict[str, Any]], prior: Any, notes: list[str]) -> list[Any]:
    """Several ceiling objects each ``centered_in room`` (light1..light4, lightning aidb5's
    warehouse) cannot all hang at the centre: the repair loop grew the room until it gave up.
    Their centre relation goes; the solver spreads free ceiling objects over an even grid."""
    def only_centre(oid: Any) -> bool:
        mine = [r for r in rels if isinstance(r, dict) and r.get("a") == oid]
        return bool(mine) and all(r.get("rel") == "centered_in" and r.get("b") == "room" for r in mine)

    ids = [o.get("id") for o in objs if getattr(prior(o.get("id")), "support", None) == "ceiling"
           and only_centre(o.get("id"))]
    if len(ids) < 2:
        return rels
    notes.append(f"{ids}: {len(ids)} ceiling objects centered_in room cannot share the centre; "
                 "spread over the ceiling in an even grid")
    return [r for r in rels if not (isinstance(r, dict) and r.get("a") in ids)]


def _wall_mounts(rels: list[Any], objs: list[dict[str, Any]], shape: Any, prior: Any, notes: list[str]) -> list[Any]:
    """A wall-mounted object (picture, whiteboard, window) with no against_wall gets one: the wall
    of what it hangs ``above``, else the wall with the fewest objects (curved rooms: the free-est
    angle). Without it the solver refused the spec, and repair enlarged the room instead."""
    out = list(rels)
    walls = [r for r in out if isinstance(r, dict) and r.get("rel") == "against_wall"]
    for o in objs:
        oid = o.get("id")
        pr = prior(oid)
        if pr is None or pr.support != "wall" or any(r.get("a") == oid for r in walls):
            continue
        above = next((r.get("b") for r in out if isinstance(r, dict) and r.get("a") == oid and r.get("rel") == "above"), None)
        host = next((r for r in walls if above is not None and r.get("a") == above), None)
        if host is not None:
            new = {"a": oid, "rel": "against_wall", "b": host["b"],
                   **({"at_deg": host["at_deg"]} if _num(host.get("at_deg")) else {})}
        elif shape == "rect":
            used = {w: sum(1 for r in walls if r.get("b") == w) for w in _RECT_WALLS}
            new = {"a": oid, "rel": "against_wall", "b": min(_RECT_WALLS, key=lambda w: used[w])}
        else:
            taken = [float(r["at_deg"]) for r in walls if _num(r.get("at_deg"))]

            def spare(deg: float, taken: list[float] = taken) -> float:
                return min((abs((deg - t + 180) % 360 - 180) for t in taken), default=math.inf)

            new = {"a": oid, "rel": "against_wall", "b": "wall", "at_deg": max(range(0, 360, 30), key=spare)}
        out.append(new)
        walls.append(new)
        where = new["b"] + (f" at_deg {new['at_deg']:g}" if "at_deg" in new else "")
        notes.append(f"'{oid}' ({o.get('category')}) is wall-mounted: added '{oid} against_wall {where}'")
    return out


#: furniture that stands with its back to a wall: in a round room, left free, it floated mid-floor
#: at an odd angle (lightning aidb5 round_lobby: the reception desk skewed in the middle)
_WALL_PIECES = ("reception_desk", "bar_counter", "kitchen_counter", "bookshelf", "cabinet", "wardrobe", "dresser",
                "tv_stand", "shelving_unit", "filing_cabinet", "sofa", "fireplace")


def _curved_wall_pieces(rels: list[Any], objs: list[dict[str, Any]], notes: list[str]) -> list[Any]:
    """A desk, counter, shelf or sofa of an ellipse room with no relation placing it gets
    ``against_wall wall``: its back to the curved wall, tangent to it, facing into the room."""
    out = list(rels)
    for o in objs:
        oid = o.get("id")
        if o.get("category") not in _WALL_PIECES or o.get("grid") is not None or _n(o) != 1:
            continue
        if any(isinstance(r, dict) and r.get("a") == oid and r.get("rel") != "facing" for r in out):
            continue
        out.append({"a": oid, "rel": "against_wall", "b": "wall"})
        notes.append(f"'{oid}' ({o.get('category')}) has its back to the curved wall (added '{oid} against_wall wall'), "
                     "tangent to it and facing into the room")
    return out
