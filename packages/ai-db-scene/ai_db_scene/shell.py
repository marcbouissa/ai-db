"""The architectural shell of an indoor room: a door, windows and trim, part of the room.

Rooms built from a model's spec were empty boxes: four bare walls, no door, no window, no
skirting (judges and Marc's reviews: "reads as a box", "flat grey light"). Unless the spec lists
its own windows or doors, a single rect or round room gets

* windows on the walls the camera looks at (the room's "outside" walls), sized to the room, on
  free stretches of wall: never behind furniture that stands taller than the sill, never over a
  picture or a TV; low furniture (a sofa, a desk) may stand under one;
* one door on a free stretch with a clear walkway in front of it, near a corner, preferably on a
  side wall;
* a skirting board, and a small cove at the ceiling where the style suits (not in industrial
  rooms; classic rooms keep their crown and baseboard).

"Windowless" briefs (a basement, a server room, a cinema...) get the door but no window. The
shell is planned from the solved placements, so a rebuild or an edit plans it again the same way;
it is not part of the stored spec. build_scene reports it as a note.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from ai_db_scene.layout import Placement
from ai_db_scene.spec import NONRECT_WALL, ObjectSpec, Relation, SceneSpec

SHELL_PREFIX = "shell_"
SILL_M = 0.9
WINDOW_HEAD_GAP_M = 0.3  # window head this far under the ceiling at least
WINDOW_MAX_H_M = 1.6
TALL_WINDOW_MAX_H_M = 2.6  # rooms over 3.2 m high (a lobby): taller windows
WORK_WINDOW_H_M = 1.2  # a work hall's high windows
WINDOW_MIN_H_M = 0.8
WINDOW_DEPTH_M = 0.08
WINDOW_SPACING_M = 2.9  # about one window per this much wall
WINDOW_GAP_M = 0.5  # between two windows, and from a corner
WINDOW_CLEAR_DEPTH_M = 0.7  # furniture this close to the wall, taller than the sill, blocks a window
DOOR_W_M, DOOR_H_M, DOOR_DEPTH_M = 0.9, 2.05, 0.05
DOOR_CLEAR_DEPTH_M = 1.0  # a walkway in front of the door
DOOR_MIN_CLEAR_M = 0.6  # in a crowded room, at least this
DOOR_CORNER_M = 0.35  # the door's casing this far from the corner at least
SIDE_MARGIN_M = 0.12
CAMERA_CLEAR_M = 1.5  # no window right beside the build camera (a frame filling the picture's edge)  # clear wall beside an opening's frame
STEP_M = 0.05
#: briefs that ask for a room without windows
WINDOWLESS = re.compile(r"\b(windowless|no windows?|without (any )?windows?|basement|cellar|bunker|vault|"
                        r"server room|data cent(er|re)|darkroom|dark room|cinema|home theat(er|re)|screening room|"
                        r"recording studio|sound ?proof)\b", re.IGNORECASE)
#: styles without a ceiling cove (bare industrial rooms); classic has its own crown
NO_COVE_STYLES = ("industrial", "warehouse", "factory", "logistics", "storage", "loft", "brutalist", "minimal",
                  "minimalist", "classic")
#: rooms with these get no skirting either (a warehouse floor runs into steel cladding)
WORK_CATEGORIES = ("pallet_rack", "forklift", "conveyor", "loading_dock")


@dataclass(frozen=True)
class Opening:
    id: str
    kind: str  # window | door
    wall: str
    placement: Placement


@dataclass(frozen=True)
class Shell:
    openings: tuple[Opening, ...]
    trim: tuple[str, ...]  # moulding kinds (scene_pipeline._MOULDING_PROFILES)
    note: str

    @property
    def placements(self) -> list[Placement]:
        return [o.placement for o in self.openings]


@dataclass(frozen=True)
class _Wall:
    name: str
    p0: tuple[float, float]  # start of the inner face (layout frame)
    t: tuple[float, float]  # along the wall
    n: tuple[float, float]  # inward normal
    length: float
    curved: bool = False


def _yaw_deg(n: tuple[float, float]) -> float:
    """layout.wall's yaw: the object's front (-Y) faces along the inward normal."""
    return round(math.degrees(math.atan2(n[0], -n[1])) % 360.0, 3)


def _rect_walls(w: float, d: float) -> list[_Wall]:
    return [_Wall("wall_north", (0.0, d), (1.0, 0.0), (0.0, -1.0), w),
            _Wall("wall_south", (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), w),
            _Wall("wall_east", (w, 0.0), (0.0, 1.0), (-1.0, 0.0), d),
            _Wall("wall_west", (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), d)]


def _axes(yaw_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    a = math.radians(yaw_deg)
    return (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))


def _overlap_2d(c1: tuple[float, float], ax1: tuple[tuple[float, float], tuple[float, float]], h1: tuple[float, float],
                c2: tuple[float, float], ax2: tuple[tuple[float, float], tuple[float, float]], h2: tuple[float, float]) -> bool:
    """Separating-axis test of two oriented rectangles (centre, unit axes, half sizes)."""
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    for u in (*ax1, *ax2):
        r1 = sum(h * abs(a[0] * u[0] + a[1] * u[1]) for a, h in zip(ax1, h1, strict=True))
        r2 = sum(h * abs(a[0] * u[0] + a[1] * u[1]) for a, h in zip(ax2, h2, strict=True))
        if abs(dx * u[0] + dy * u[1]) > r1 + r2:
            return False
    return True


def _blocked(placements: list[Placement], centre: tuple[float, float], t: tuple[float, float], n: tuple[float, float],
             half_w: float, depth: float, z0: float, z1: float) -> bool:
    """Does any object stand in the box ``half_w`` either side of ``centre`` along the wall, ``depth``
    into the room, between heights z0 and z1?"""
    c = (centre[0] + n[0] * depth / 2, centre[1] + n[1] * depth / 2)
    for p in placements:
        lo, hi = p.location[2] - p.dims[2] / 2, p.location[2] + p.dims[2] / 2
        if hi <= z0 or lo >= z1:
            continue
        if _overlap_2d(c, (t, n), (half_w, depth / 2), (p.location[0], p.location[1]), _axes(p.rotation_deg_z),
                       (p.dims[0] / 2, p.dims[1] / 2)):
            return True
    return False


def windowless(brief: str | None) -> bool:
    return bool(brief and WINDOWLESS.search(brief))


def _view(spec: SceneSpec, placements: list[Placement]) -> tuple[float, float, float, float] | None:
    """(x, y, dx, dy): where the build camera stands and looks (layout frame), if it can tell."""
    try:
        from ai_db_scene.tools.scene_pipeline import camera_poses, frame_offset

        off = frame_offset(spec)
        pose = camera_poses(spec, placements, off)[0]
    except Exception:  # noqa: BLE001 - no camera guess: walls ranked by free length alone
        return None
    yaw = math.radians(pose["rotation"][2])  # rotation [90, 0, yaw] looks along +Y turned by yaw
    return pose["location"][0] - off[0], pose["location"][1] - off[1], -math.sin(yaw), math.cos(yaw)


def _work_room(spec: SceneSpec, placements: list[Placement]) -> bool:
    """A warehouse / factory hall (its style or its racks and forklifts)."""
    from ai_db_scene.tools.scene_pipeline import WAREHOUSE_STYLES

    return (spec.setting.style or "").lower() in WAREHOUSE_STYLES or any(p.category in WORK_CATEGORIES
                                                                          for p in placements)


def _trim(spec: SceneSpec, placements: list[Placement]) -> tuple[str, ...]:
    st = spec.setting
    if st.mouldings is not None or st.style == "classic":  # the spec's own, or the classic crown + baseboard
        return ()
    style = (st.style or "").lower()
    if _work_room(spec, placements):
        return ()
    return ("skirting",) if style in NO_COVE_STYLES else ("skirting", "cove")


def _window_size(length: float, h: float, work: bool = False) -> tuple[float, float, float] | None:
    """(width, height, sill) of the windows of a wall ``length`` long in a room ``h`` high: taller in a
    tall room (a lobby), high and wide in a work hall (clerestory windows over the racks)."""
    if work:
        sill = max(SILL_M, h - WORK_WINDOW_H_M - 0.6)
        return 2.0, WORK_WINDOW_H_M, round(sill, 3)
    room = h - SILL_M - WINDOW_HEAD_GAP_M
    wh = min(WINDOW_MAX_H_M, room) if h <= 3.2 else min(room - 0.2, TALL_WINDOW_MAX_H_M)
    if wh < WINDOW_MIN_H_M:
        return None
    ww = 1.5 if length >= 4.5 else 1.2 if length >= 3.0 else 0.9
    return ww, round(wh, 3), SILL_M


def _place(wall: _Wall, s: float, dims: tuple[float, float, float], z: float) -> Placement:
    x = wall.p0[0] + wall.t[0] * s + wall.n[0] * dims[1] / 2
    y = wall.p0[1] + wall.t[1] * s + wall.n[1] * dims[1] / 2
    return Placement(id="", location=(round(x, 4), round(y, 4), round(z, 4)), rotation_deg_z=_yaw_deg(wall.n),
                     dims=dims)


def _free_spots(wall: _Wall, placements: list[Placement], half: float, depth: float, z0: float, z1: float,
                corner: float, taken: list[tuple[float, float]]) -> list[float]:
    """Positions along the wall (STEP_M grid) where an opening of half width ``half`` fits."""
    out = []
    k = 0
    while True:
        s = corner + half + k * STEP_M
        if s > wall.length - corner - half + 1e-9:
            break
        k += 1
        if any(abs(s - a) < half + b for a, b in taken):
            continue
        pt = (wall.p0[0] + wall.t[0] * s, wall.p0[1] + wall.t[1] * s)
        if not _blocked(placements, pt, wall.t, wall.n, half, depth, z0, z1):
            out.append(s)
    return out


def _rect_plan(spec: SceneSpec, placements: list[Placement], with_windows: bool,
               view: tuple[float, float, float, float] | None, work: bool = False) -> list[Opening]:
    w, d, h = spec.setting.bounds_m
    walls = _rect_walls(w, d)
    # how much a wall faces the camera: 1 = the far wall, 0 = a side wall, -1 = behind the camera
    facing = {wl.name: (-(wl.n[0] * view[2] + wl.n[1] * view[3]) if view else 0.0) for wl in walls}
    openings: list[Opening] = []
    taken: dict[str, list[tuple[float, float]]] = {wl.name: [] for wl in walls}
    # the door first: a side wall (it shows in the picture without stealing it), then the far wall,
    # then behind the camera; near a corner, with a clear walkway in front
    order = sorted(walls, key=lambda wl: (abs(facing[wl.name]) >= 0.75, -facing[wl.name]))
    dh = min(DOOR_H_M, h - 0.15)
    half = DOOR_W_M / 2 + 0.08
    placed = False
    for clear, corner in ((DOOR_CLEAR_DEPTH_M, DOOR_CORNER_M), (DOOR_MIN_CLEAR_M, 0.1)):  # crowded: a tighter fit
        for wl in order:
            spots = _free_spots(wl, placements, half, clear, 0.0, dh, corner, taken[wl.name])
            if not spots:
                continue
            s = min(spots, key=lambda v: min(v, wl.length - v))  # nearest a corner
            pl = _place(wl, s, (DOOR_W_M, DOOR_DEPTH_M, dh), dh / 2)
            openings.append(Opening(f"{SHELL_PREFIX}door", "door", wl.name, replace(pl, id=f"{SHELL_PREFIX}door",
                                                                                     category="door")))
            taken[wl.name].append((s, half + WINDOW_GAP_M))
            placed = True
            break
        if placed:
            break
    if not with_windows:
        return openings
    # windows on every wall but the one behind the camera: the far wall first. The build's
    # coverage check may still swing the camera to another spot, and furniture often fills
    # one wall (a bar counter, a TV), so more than one outside wall keeps windows in the picture.
    k = 0
    for wl in sorted(walls, key=lambda wl: (-facing[wl.name], -wl.length)):
        if facing[wl.name] < -0.5 and view is not None:
            continue
        size = _window_size(wl.length, h, work)
        if size is None:
            break
        ww, wh, sill = size
        n = max(1, min(4, round(wl.length / WINDOW_SPACING_M)))
        half = ww / 2 + SIDE_MARGIN_M
        for i in range(n):
            spots = [v for v in _free_spots(wl, placements, half, WINDOW_CLEAR_DEPTH_M, sill - 0.03, sill + wh,
                                            WINDOW_GAP_M, taken[wl.name])
                     if view is None or math.hypot(wl.p0[0] + wl.t[0] * v - view[0],
                                                   wl.p0[1] + wl.t[1] * v - view[1]) > CAMERA_CLEAR_M + half]
            if not spots:
                break
            target = wl.length * (i + 0.5) / n
            s = min(spots, key=lambda v: abs(v - target))
            if abs(s - target) > wl.length / (2 * n) + 0.6:  # pushed far off its share: crowded
                continue
            taken[wl.name].append((s, half + WINDOW_GAP_M / 2))
            k += 1
            pl = _place(wl, s, (ww, WINDOW_DEPTH_M, wh), sill + wh / 2)
            openings.append(Opening(f"{SHELL_PREFIX}window_{k}", "window", wl.name,
                                    replace(pl, id=f"{SHELL_PREFIX}window_{k}", category="window")))
    return openings


def _ellipse_plan(spec: SceneSpec, placements: list[Placement], with_windows: bool,
                  view: tuple[float, float, float, float] | None) -> list[Opening]:
    """A round room: windows spread all round from the arc the camera faces, the door to one side."""
    w, d, h = spec.setting.bounds_m
    a, b, cx, cy = w / 2, d / 2, w / 2, d / 2
    far = math.atan2(view[3], view[2]) if view else math.pi / 2  # the angle the camera looks toward

    def at(theta: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        p = (cx + a * math.cos(theta), cy + b * math.sin(theta))
        nx, ny = -math.cos(theta) / a, -math.sin(theta) / b
        k = math.hypot(nx, ny)
        n = (nx / k, ny / k)
        return p, (-n[1], n[0]), n

    def fits(theta: float, half: float, depth: float, z0: float, z1: float) -> bool:
        p, t, n = at(theta)
        # the flat opening is a chord: set it in by the sagitta so its ends stay in the wall
        return not _blocked(placements, p, t, n, half, depth, z0, z1)

    perim = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    openings: list[Opening] = []
    used: list[float] = []

    def add(kind: str, oid: str, theta: float, dims: tuple[float, float, float], z: float) -> None:
        p, _t, n = at(theta)
        sag = dims[0] ** 2 / (8 * max(min(a, b), 0.5))
        loc = (p[0] + n[0] * (dims[1] / 2 + sag), p[1] + n[1] * (dims[1] / 2 + sag), z)
        openings.append(Opening(oid, kind, NONRECT_WALL, Placement(
            id=oid, location=(round(loc[0], 4), round(loc[1], 4), round(z, 4)), rotation_deg_z=_yaw_deg(n),
            dims=dims, category=kind)))
        used.append(theta)

    def clear_of_used(theta: float, half_m: float) -> bool:
        return all(abs(math.remainder(theta - u, 2 * math.pi)) * min(a, b) > half_m + 0.9 for u in used)

    dh = min(DOOR_H_M, h - 0.15)
    half = DOOR_W_M / 2 + 0.08
    for off_deg in range(90, 271, 6):
        th = far + math.radians(off_deg)
        if clear_of_used(th, half) and fits(th, half, DOOR_CLEAR_DEPTH_M, 0.0, dh):
            add("door", f"{SHELL_PREFIX}door", th, (DOOR_W_M, DOOR_DEPTH_M, dh), dh / 2)
            break
    size = _window_size(perim / 4, h) if with_windows else None
    if size is not None:
        ww, wh, sill = size
        # all round the room: the build's coverage check may swing the camera to another spot
        # (qwen aidb7 lobby: pose 10 of 38), and a round lobby is glazed all round anyway
        n = max(3, min(8, round(perim / (WINDOW_SPACING_M * 1.6))))
        half = ww / 2 + SIDE_MARGIN_M
        k = 0
        for i in range(n):
            target = far + 2 * math.pi * i / n
            for off_deg in range(0, 31, 2):
                hit = None
                for sgn in (1, -1):
                    th = target + sgn * math.radians(off_deg)
                    if clear_of_used(th, half) and fits(th, half, WINDOW_CLEAR_DEPTH_M, sill - 0.03, sill + wh):
                        hit = th
                        break
                if hit is not None:
                    k += 1
                    add("window", f"{SHELL_PREFIX}window_{k}", hit, (ww, WINDOW_DEPTH_M, wh),
                        sill + wh / 2)
                    break
    return openings


def plan(spec: SceneSpec, placements: list[Placement], brief: str | None = None) -> Shell | None:
    """The shell for a single indoor rect or round room (None: none applies)."""
    st = spec.setting
    if st.kind != "indoor" or st.rooms or st.shape not in ("rect", "ellipse"):
        return None
    own = {o.category for o in spec.objects} & {"window", "door", "niche"}
    trim = _trim(spec, placements)
    openings: list[Opening] = []
    if not own:
        with_windows = not windowless(brief)
        view = _view(spec, placements) if placements else None
        if st.shape == "rect":
            openings = _rect_plan(spec, placements, with_windows, view, _work_room(spec, placements))
        else:
            openings = _ellipse_plan(spec, placements, with_windows, view)
    if not openings and not trim:
        return None
    wins = sum(o.kind == "window" for o in openings)
    door = any(o.kind == "door" for o in openings)
    parts = []
    if wins:
        walls = sorted({o.wall for o in openings if o.kind == "window"})
        parts.append(f"{wins} window{'s' if wins > 1 else ''} ({', '.join(walls)})")
    if door:
        parts.append(f"a door ({next(o.wall for o in openings if o.kind == 'door')})")
    if trim:
        parts.append(" and ".join({"skirting": "a skirting board", "cove": "a ceiling cove"}[t] for t in trim))
    why = ("the spec has its own windows/doors" if own else
           "windowless brief: no windows" if windowless(brief) and not wins else "")
    note = ("shell: the room got " + ", ".join(parts) + " (part of the room, kept clear of the furniture)"
            + (f"; {why}" if why else ""))
    return Shell(tuple(openings), trim, note)


def apply(spec: SceneSpec, placements: list[Placement], shell: Shell | None) -> tuple[SceneSpec, list[Placement]]:
    """The spec and placements with the shell's openings as window/door objects against their walls
    (what room_shell cuts openings for, the daylight rig puts portals in, and the recipes build)."""
    if shell is None or not shell.openings:
        return spec, placements
    ids = {o.id for o in spec.objects}
    ops = [o for o in shell.openings if o.id not in ids]
    objs = tuple(ObjectSpec(id=o.id, category=o.kind) for o in ops)
    rels = tuple(Relation(a=o.id, rel="against_wall", b=o.wall) for o in ops)
    return (replace(spec, objects=(*spec.objects, *objs), relations=(*spec.relations, *rels)),
            [*placements, *(o.placement for o in ops)])
