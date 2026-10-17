"""Layout sanity: deterministic, explainable checks of where things stand and what the camera sees.

The quality score saturated (~99 for aidb7) while the visual review found real problems:
armchairs beside the sofa facing the same way, park benches 6-15 m from the fountain, rooms far
too big for their content, a trunk filling the picture, chairs stranded by a wall. Each check
below is pass / fail / n.a. (None: nothing to judge) with a short reason; ``score`` is the share
of applicable checks passed, and the bench score (v3) takes up to a quarter off for the failed ones.

Input: the inventory's ``layout`` (blender.py _INVENTORY): per root object its world box, origin
and yaw; the scene camera's position, view direction and what a 24 x 14 grid of rays through its
frame hits first (per root: rays, median distance). The front of an object is its local -Y (the
ai-db convention; a raw arm's own model may differ, which then counts against it).

Checks:

* ``facing_focus``: seats face what they belong to (within 60 deg): sofas / armchairs the coffee
  table, TV or fireplace (within 5 m), benches a fountain / gazebo / picnic table (within 8 m),
  chairs and stools the table, desk or counter they stand at (within 1.5 m); >= 75 % of the
  seats that have such a focus.
* ``seat_groups``: sofas / armchairs stand within 2 m (box to box) of the coffee table / TV /
  fireplace they belong to (looked for within 8 m), benches within 3 m of their fountain /
  gazebo / picnic table (looked for within 20 m): the aidb7 park benches stood 6-15 m away.
* ``chairs_at_tables``: every chair / stool stands within 0.9 m of a table, desk or counter.
* ``fill_ratio`` (rooms only): furniture footprint / floor area within 0.10-0.60 (rugs and the
  shell aside): an empty hall or a crammed box fails.
* ``in_bounds`` (rooms only): every object inside the room's walls (0.15 m slack).
* ``grounded``: at most 5 % of the objects float: their bottom more than 5 cm above the floor
  (z = 0 outdoors) and above every object under them; wall- and ceiling-mounted kinds aside.
* ``no_occluder``: nothing in the camera's foreground: no object (ground, floor, walls aside)
  that takes >= 6 % of the frame (or >= 3 % of it in the middle half of its width) stands nearer
  than 75 % of the median distance of everything else the camera sees (a trunk in the middle of
  a park view, an armchair or plant at the lens); pieces lower than 0.85 m aside.
* ``seating_fronts``: the camera sees the fronts of the main seating: of the seats of the main
  kind (sofa > bench > armchair > chairs), >= 1/3 turn their front toward the camera.
"""

from __future__ import annotations

import math
import re
from typing import Any

#: categories (or name words) of the shell and the ground: never furniture, never occluders
STRUCTURE = ("floor", "ground", "wall", "ceiling", "room", "water", "terrain", "paving", "trim", "plane", "window",
             "door")
SEATS = {"sofa": 0, "bench": 1, "armchair": 2, "dining_chair": 3, "office_chair": 3, "chair": 3, "stool": 4,
         "bar_stool": 4}
CHAIRS = ("dining_chair", "office_chair", "chair", "stool", "bar_stool")
TABLES = ("dining_table", "table", "desk", "conference_table", "bar_counter", "kitchen_island", "kitchen_counter",
          "counter", "workbench", "picnic_table", "reception_desk", "coffee_table", "side_table")
LOUNGE_FOCUS = ("coffee_table", "tv", "tv_stand", "fireplace")
BENCH_FOCUS = ("fountain", "gazebo", "picnic_table")
#: kinds that hang on a wall or from the ceiling (or sit on top of something by design)
MOUNTED = ("picture_frame", "wall_clock", "mirror", "wall_sconce", "ceiling_light", "pendant_lamp", "whiteboard",
           "window", "door", "shelf", "wall_shelf", "tv", "painting", "poster", "light", "lamp_shade", "curtain",
           "blackboard", "chalkboard", "projector_screen", "clock", "sign")
FACING_TOL_DEG = 60.0
FOCUS_REACH_M = {"lounge": 5.0, "bench": 8.0, "chair": 1.5}
CHAIR_REACH_M = 0.9
#: seat groups: a seat whose table / fountain is within GROUP_LOOK_M must stand within GROUP_NEAR_M of it
GROUP_LOOK_M = {"lounge": 8.0, "bench": 20.0}
GROUP_NEAR_M = {"lounge": 2.0, "bench": 3.0}
FILL_BAND = (0.10, 0.60)
BOUNDS_SLACK_M = 0.15
FLOAT_TOL_M = 0.05
FLOAT_SHARE_MAX = 0.05
OCCLUDER_SHARE = 0.06  # of all rays through the frame
OCCLUDER_MID_SHARE = 0.03  # of all rays, in the middle half of the frame's width
OCCLUDER_NEAR = 0.75  # nearer than this share of the subject's distance
OCCLUDER_MIN_TOP_M = 0.85  # low pieces (a coffee table before the sofa) frame a view, they do not block it
FRONTS_SHARE = 1 / 3
FACING_SHARE = 0.75

_WORDS: tuple[str, ...] = ()


def _known() -> tuple[str, ...]:
    """Category words, longest first (``dining_chair`` before ``chair``)."""
    global _WORDS
    if not _WORDS:
        from ai_db_scene.priors import load_priors

        words = set(load_priors()) | set(SEATS) | set(TABLES) | set(MOUNTED) | {"fountain", "gazebo", "rug", "carpet"}
        _WORDS = tuple(sorted(words, key=len, reverse=True))
    return _WORDS


def category(name: str, given: str) -> str:
    """The object's category: its own, else the longest known category word in its name."""
    if given:
        return given
    low = re.sub(r"[^a-z]+", "_", name.lower())
    for w in _known():
        if re.search(rf"(^|_){re.escape(w)}(s|es)?(_|$)", low):
            return w
    return ""


def _structural(name: str, cat: str, box: list[list[float]], span: float) -> bool:
    low = name.lower()
    if cat in STRUCTURE or low.startswith((*STRUCTURE, "room_")):
        return True
    w, d = box[1][0] - box[0][0], box[1][1] - box[0][1]
    return max(w, d) > 0.8 * span and box[1][2] - box[0][2] < 0.3  # a slab across the scene: a floor


def _front(yaw_deg: float) -> tuple[float, float]:
    y = math.radians(yaw_deg)
    return math.sin(y), -math.cos(y)


def _centre(b: list[list[float]]) -> tuple[float, float]:
    return (b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2


def _gap(a: list[list[float]], b: list[list[float]]) -> float:
    """xy distance between two boxes (0 when they overlap)."""
    dx = max(a[0][0] - b[1][0], b[0][0] - a[1][0], 0.0)
    dy = max(a[0][1] - b[1][1], b[0][1] - a[1][1], 0.0)
    return math.hypot(dx, dy)


def _faces(o: dict[str, Any], target: tuple[float, float]) -> bool:
    cx, cy = _centre(o["box"])
    dx, dy = target[0] - cx, target[1] - cy
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return True
    f = _front(o["yaw_deg"])
    return (f[0] * dx + f[1] * dy) / n >= math.cos(math.radians(FACING_TOL_DEG))


def check(inv: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    """{check: {"pass": True | False | None, "why": str}}, or None for an inventory without layout."""
    lay = inv.get("layout")
    if not lay or not lay.get("roots"):
        return None
    cats = {o["name"]: o.get("category") or "" for o in inv.get("objects", []) if o.get("parent") is None}
    allbox = list(lay["roots"].values())
    span = max(max(r["box"][1][0] for r in allbox) - min(r["box"][0][0] for r in allbox),
               max(r["box"][1][1] for r in allbox) - min(r["box"][0][1] for r in allbox), 1.0)
    objs: list[dict[str, Any]] = []
    shell: list[dict[str, Any]] = []
    for name, r in lay["roots"].items():
        cat = category(name, cats.get(name, ""))
        o = {"name": name, "cat": cat, **r}
        (shell if r.get("terrain") or _structural(name, cat, r["box"], span) else objs).append(o)
    out: dict[str, dict[str, Any]] = {}
    out["facing_focus"] = _facing(objs)
    out["seat_groups"] = _groups(objs)
    out["chairs_at_tables"] = _chairs(objs)
    room = _room(shell)
    out["fill_ratio"] = _fill(objs, room)
    out["in_bounds"] = _bounds(objs, room)
    out["grounded"] = _grounded(objs, room)
    cam = lay.get("camera")
    low = {o["name"] for o in objs if o["box"][1][2] < OCCLUDER_MIN_TOP_M}
    out["no_occluder"] = _occluder(cam, {o["name"] for o in shell}, low)
    out["seating_fronts"] = _fronts(objs, cam)
    return out


def _focus(o: dict[str, Any], objs: list[dict[str, Any]]) -> dict[str, Any] | None:
    kinds: tuple[str, ...]
    if o["cat"] in ("sofa", "armchair"):
        kinds, reach = LOUNGE_FOCUS, FOCUS_REACH_M["lounge"]
    elif o["cat"] == "bench":
        kinds, reach = BENCH_FOCUS, FOCUS_REACH_M["bench"]
    elif o["cat"] in CHAIRS:
        kinds, reach = TABLES, FOCUS_REACH_M["chair"]
    else:
        return None
    near = [(_gap(o["box"], t["box"]), t) for t in objs if t is not o and t["cat"] in kinds]
    near = [(g, t) for g, t in near if g <= reach]
    return min(near, key=lambda gt: gt[0])[1] if near else None


def _facing(objs: list[dict[str, Any]]) -> dict[str, Any]:
    judged, bad = 0, []
    for o in objs:
        if o["cat"] not in SEATS:
            continue
        f = _focus(o, objs)
        if f is None:
            continue
        judged += 1
        if not _faces(o, _centre(f["box"])):
            bad.append(f"{o['name']}->{f['name']}")
    if not judged:
        return {"pass": None, "why": "no seat with a table, TV or fountain near it"}
    ok = (judged - len(bad)) / judged >= FACING_SHARE
    return {"pass": ok, "why": f"{judged - len(bad)}/{judged} seats face their focus" + (f"; not: {bad[:6]}" if bad else "")}


def _groups(objs: list[dict[str, Any]]) -> dict[str, Any]:
    """Sofas / armchairs near their coffee table or TV, benches near their fountain."""
    judged, far = 0, []
    kinds: tuple[str, ...]
    for o in objs:
        if o["cat"] in ("sofa", "armchair"):
            kinds, look, near = LOUNGE_FOCUS, GROUP_LOOK_M["lounge"], GROUP_NEAR_M["lounge"]
        elif o["cat"] == "bench":
            kinds, look, near = BENCH_FOCUS, GROUP_LOOK_M["bench"], GROUP_NEAR_M["bench"]
        else:
            continue
        cand = [(_gap(o["box"], t["box"]), t["name"]) for t in objs if t["cat"] in kinds]
        cand = [c for c in cand if c[0] <= look]
        if not cand:
            continue
        judged += 1
        g, name = min(cand)
        if g > near:
            far.append(f"{o['name']} {g:.1f} m from {name}")
    if not judged:
        return {"pass": None, "why": "no sofa, armchair or bench with a table, TV or fountain"}
    return {"pass": not far, "why": f"{judged - len(far)}/{judged} seats by their group" + (f"; far: {far[:6]}" if far else "")}


def _chairs(objs: list[dict[str, Any]]) -> dict[str, Any]:
    chairs = [o for o in objs if o["cat"] in CHAIRS]
    if not chairs:
        return {"pass": None, "why": "no chairs"}
    tables = [t for t in objs if t["cat"] in TABLES]
    loose = [c["name"] for c in chairs if not any(_gap(c["box"], t["box"]) <= CHAIR_REACH_M for t in tables)]
    return {"pass": not loose, "why": f"{len(chairs) - len(loose)}/{len(chairs)} chairs at a table/desk"
            + (f"; loose: {loose[:6]}" if loose else "")}


def _room(shell: list[dict[str, Any]]) -> list[list[float]] | None:
    """The room's box from its floor and walls (None outdoors: no walls)."""
    walls = [s for s in shell if "wall" in s["name"].lower() or s["cat"] == "wall"]
    if not walls:
        return None
    parts = walls + [s for s in shell if "floor" in s["name"].lower() or s["cat"] == "floor"]
    return [[min(p["box"][0][i] for p in parts) for i in range(3)], [max(p["box"][1][i] for p in parts) for i in range(3)]]


def _area(b: list[list[float]]) -> float:
    return max(b[1][0] - b[0][0], 0.0) * max(b[1][1] - b[0][1], 0.0)


def _fill(objs: list[dict[str, Any]], room: list[list[float]] | None) -> dict[str, Any]:
    if room is None:
        return {"pass": None, "why": "outdoors: no floor area"}
    floor = _area(room)
    furn = sum(_area(o["box"]) for o in objs if o["cat"] not in ("rug", "carpet")
               and o["box"][1][2] - o["box"][0][2] > 0.05 and o["cat"] not in MOUNTED)
    r = furn / floor if floor > 0 else 0.0
    lo, hi = FILL_BAND
    return {"pass": lo <= r <= hi, "why": f"furniture {furn:.1f} m2 / floor {floor:.1f} m2 = {r:.2f} (band {lo}-{hi})"}


def _bounds(objs: list[dict[str, Any]], room: list[list[float]] | None) -> dict[str, Any]:
    if room is None:
        return {"pass": None, "why": "outdoors"}
    s = BOUNDS_SLACK_M
    out = [o["name"] for o in objs if o["box"][0][0] < room[0][0] - s or o["box"][1][0] > room[1][0] + s
           or o["box"][0][1] < room[0][1] - s or o["box"][1][1] > room[1][1] + s]
    return {"pass": not out, "why": f"{len(objs) - len(out)}/{len(objs)} inside the walls" + (f"; out: {out[:6]}" if out else "")}


def _grounded(objs: list[dict[str, Any]], room: list[list[float]] | None) -> dict[str, Any]:
    floor = room[0][2] if room is not None else 0.0
    judged = [o for o in objs if o["cat"] not in MOUNTED and not any(w in o["name"].lower() for w in MOUNTED)]
    if not judged:
        return {"pass": None, "why": "nothing standing"}
    floating = []
    for o in judged:
        bottom = o["box"][0][2]
        if bottom <= max(floor, 0.0) + FLOAT_TOL_M:
            continue
        cx, cy = _centre(o["box"])
        under = [u for u in objs if u is not o and u["box"][0][0] <= cx <= u["box"][1][0]
                 and u["box"][0][1] <= cy <= u["box"][1][1] and u["box"][1][2] <= bottom + FLOAT_TOL_M]
        if under and bottom - max(u["box"][1][2] for u in under) <= FLOAT_TOL_M + 0.05:
            continue
        floating.append(o["name"])
    ok = len(floating) / len(judged) <= FLOAT_SHARE_MAX
    return {"pass": ok, "why": f"{len(floating)}/{len(judged)} float" + (f": {floating[:6]}" if floating else "")}


def _occluder(cam: dict[str, Any] | None, shell: set[str], low: set[str] | None = None) -> dict[str, Any]:
    if not cam or not cam.get("median_m"):
        return {"pass": None, "why": "no scene camera (or it sees nothing)"}
    rays = cam["rays"]
    hits = {k: (h[0], h[1], h[2] if len(h) > 2 else 0) for k, h in cam["hits"].items() if k not in shell}
    bad = []
    for k, (n, d, mid) in hits.items():
        if k in (low or set()) or n / rays < OCCLUDER_SHARE and mid / rays < OCCLUDER_MID_SHARE:
            continue
        others = sorted(dd for kk, (nn, dd, _m) in hits.items() if kk != k for _ in range(nn))
        if not others:
            continue
        subject = others[len(others) // 2]  # the median distance of what else the camera sees
        if d < OCCLUDER_NEAR * subject:
            bad.append(f"{k} ({n / rays:.0%} of the frame, {mid / rays:.0%} in its middle, at {d:.1f} m; "
                       f"the rest at {subject:.1f} m)")
    return {"pass": not bad, "why": f"foreground: {bad}" if bad else "nothing in front of the subject fills the frame"}


def _fronts(objs: list[dict[str, Any]], cam: dict[str, Any] | None) -> dict[str, Any]:
    seats = [o for o in objs if o["cat"] in SEATS]
    if not seats or not cam:
        return {"pass": None, "why": "no seating" if not seats else "no camera"}
    rank = min(SEATS[o["cat"]] for o in seats)
    main = [o for o in seats if SEATS[o["cat"]] == rank]
    cx, cy = cam["loc"][0], cam["loc"][1]
    seen = [o["name"] for o in main if _faces(o, (cx, cy))]
    ok = len(seen) / len(main) >= FRONTS_SHARE
    return {"pass": ok, "why": f"{len(seen)}/{len(main)} {main[0]['cat']} fronts toward the camera"}


def score(checks: dict[str, dict[str, Any]] | None) -> float | None:
    """100 x passed / applicable, None when nothing applies."""
    if not checks:
        return None
    judged = [c["pass"] for c in checks.values() if c["pass"] is not None]
    return round(100.0 * sum(judged) / len(judged), 1) if judged else None
