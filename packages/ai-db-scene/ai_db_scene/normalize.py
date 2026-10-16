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
    shape = setting.get("shape", "rect")
    indoor = setting.get("kind") == "indoor"
    objs = [o for o in d.get("objects") or [] if isinstance(o, dict)] if isinstance(d.get("objects"), list) else []
    ids = {o["id"] for o in objs if isinstance(o.get("id"), str)}
    priors = load_priors()
    cat = {o["id"]: o.get("category") for o in objs if isinstance(o.get("id"), str)}

    def prior(oid: Any) -> Any:
        return priors.get(cat.get(oid)) if isinstance(oid, str) else None  # type: ignore[arg-type]

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
        r = dict(raw)
        p = f"relations[{i}]"
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
    out = _rugs(out, ids, prior, notes)
    out = _seats(d, out, objs, notes)
    objs = [o for o in d.get("objects") or [] if isinstance(o, dict)]
    _spread_centered(out, objs, setting, prior, notes)
    if indoor:
        out = _wall_mounts(out, objs, shape, prior, notes)
    d["relations"] = out
    return d, notes


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
