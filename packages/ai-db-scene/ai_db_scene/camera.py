"""Where the build camera stands: on the open side, looking at the fronts of the seating.

Marc's review of the aidb5 round lobby: "the camera in the lobby is flipped so sofas are facing
the wall, preventing the open feel". The old camera took the room corner with the most furniture
clearance, farthest from the heroes, and aimed at them: often behind the seating, looking past
its backs at a blank wall. Here every candidate spot (along the walls, or round a curved room)
is scored by what a photographer looks for:

* the fronts of the seating and of the feature objects (TV, fireplace, reception desk, board):
  each one in frame adds ``w * -dot(view ray, its front)`` - a sofa seen from the front counts,
  one seen from behind counts against (seating backs double);
* the heroes and features in frame (the lens widens up to ``MAX_HFOV_DEG`` to take them in);
* room to step back (distance to what it looks at), no furniture at the camera, nothing tall
  right in front of the lens.

Plain geometry in the layout frame (x east, y north, front of an object = local -Y); no Blender.
``view_issue`` judges a camera a model set itself with the same score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ai_db_scene.layout import Placement
from ai_db_scene.spec import SceneSpec

#: objects whose front a view should show, and how much it matters (seating first)
SEATING = {"sofa": 1.5, "armchair": 1.0, "bench": 0.8, "chair": 0.4, "dining_chair": 0.3, "office_chair": 0.3,
           "bed": 0.6}
FEATURES = {"fireplace": 1.2, "reception_desk": 1.2, "bar_counter": 0.8, "whiteboard": 0.8, "tv": 0.6,
            "tv_stand": 0.5, "kitchen_counter": 0.5, "kitchen_island": 0.5, "bookshelf": 0.4, "desk": 0.4,
            "stove": 0.3, "dresser": 0.3, "wardrobe": 0.3, "fountain": 0.8}
MIN_HFOV_DEG = 55.0  # never narrower: a room photo, not a close-up
MAX_HFOV_DEG = 96.0  # ~16 mm on a 36 mm sensor: wider distorts the corners
SENSOR_MM = 36.0
_INSET_M = 0.45  # spots stand this far in from the walls
_CLEAR_M = 0.35  # and at least this far from any furniture footprint


@dataclass(frozen=True)
class View:
    """A camera in the layout frame: eye at (x, y, z), looking toward ``aim`` with ``hfov_deg``."""

    x: float
    y: float
    z: float
    aim: tuple[float, float, float]
    hfov_deg: float
    score: float
    seen: tuple[str, ...]  # focus objects in frame
    fronts: float  # the front term alone (< 0: it mostly sees backs)

    @property
    def yaw_deg(self) -> float:
        """Blender Euler Z with rotation [90, 0, yaw] (yaw 0 looks along +Y)."""
        return math.degrees(math.atan2(-(self.aim[0] - self.x), self.aim[1] - self.y))

    @property
    def lens_mm(self) -> float:
        return SENSOR_MM / 2 / math.tan(math.radians(self.hfov_deg) / 2)


def _front(p: Placement) -> tuple[float, float]:
    yaw = math.radians(p.rotation_deg_z)
    return math.sin(yaw), -math.cos(yaw)


def _footprint_dist(x: float, y: float, p: Placement) -> float:
    """Distance from (x, y) to p's oriented footprint (0 inside)."""
    yaw = math.radians(p.rotation_deg_z)
    dx, dy = x - p.location[0], y - p.location[1]
    u = dx * math.cos(yaw) + dy * math.sin(yaw)
    v = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return math.hypot(max(abs(u) - p.dims[0] / 2, 0.0), max(abs(v) - p.dims[1] / 2, 0.0))


def _base(pid: str, ids: set[str]) -> str:
    head, _, tail = pid.rpartition("_")
    return head if tail.isdigit() and head in ids else pid


def focus_weights(spec: SceneSpec, placements: list[Placement]) -> dict[str, float]:
    """Objects a view should show and how much each counts: heroes, seating, features."""
    heroes = {o.id for o in spec.objects if o.hero}
    ids = {o.id for o in spec.objects}
    focus = spec.camera.get("focus")
    out: dict[str, float] = {}
    for p in placements:
        w = SEATING.get(p.category, 0.0) + FEATURES.get(p.category, 0.0)
        if _base(p.id, ids) in heroes:
            w += 0.5
        if _base(p.id, ids) == focus:
            w += 2.0  # camera.focus: what the view is about
        if w > 0:
            out[p.id] = w
    if not out:  # nothing marked: the largest objects
        solid = sorted(placements, key=lambda p: -(p.dims[0] * p.dims[1] * p.dims[2]))[:3]
        out = {p.id: 1.0 for p in solid}
    return out


def spots(spec: SceneSpec, placements: list[Placement]) -> list[tuple[float, float]]:
    """Candidate camera spots inside the setting, clear of furniture: along each wall of a
    rectangle (corners, quarters, middle), round the wall of an ellipse every 30 deg, along the
    edges of a polygon; outdoors round the bounds."""
    from ai_db_scene.shapes import RoomShape

    st = spec.setting
    w, d, _h = st.bounds_m
    i = min(_INSET_M, w / 4, d / 4)
    out: list[tuple[float, float]] = []
    if st.shape == "ellipse":
        a, b = w / 2 - i, d / 2 - i
        out = [(w / 2 + a * math.cos(math.radians(t)), d / 2 + b * math.sin(math.radians(t))) for t in range(0, 360, 30)]
    elif st.shape == "polygon" and len(st.points) >= 3:
        shape = RoomShape("polygon", w, d, tuple((float(p[0]), float(p[1])) for p in st.points))
        pts = list(st.points)
        for k in range(len(pts)):
            (x0, y0), (x1, y1) = pts[k - 1], pts[k]
            for f in (0.15, 0.5, 0.85):
                x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
                cx, cy = shape.cx - x, shape.cy - y
                n = math.hypot(cx, cy) or 1.0
                out.append((x + cx / n * i * 1.5, y + cy / n * i * 1.5))
        out = [p for p in out if shape.contains(p[0], p[1], 0.0)]
    else:
        fr = (0.0, 0.25, 0.5, 0.75, 1.0)
        out = [(i + (w - 2 * i) * f, y) for f in fr for y in (i, d - i)]
        out += [(x, i + (d - 2 * i) * f) for f in fr[1:-1] for x in (i, w - i)]
    solid = [p for p in placements if p.dims[2] > 0.05]
    clear = [s for s in out if all(_footprint_dist(s[0], s[1], p) >= _CLEAR_M for p in solid)]
    return clear or out


def _aim(placements: dict[str, Placement], weights: dict[str, float]) -> tuple[float, float, float]:
    tot = sum(weights.values())
    x = sum(placements[k].location[0] * w for k, w in weights.items()) / tot
    y = sum(placements[k].location[1] * w for k, w in weights.items()) / tot
    z = sum(min(placements[k].location[2], 1.2) * w for k, w in weights.items()) / tot
    return x, y, max(0.6, min(z, 1.0))


def score_view(spec: SceneSpec, placements: list[Placement], x: float, y: float, z: float,
               aim: tuple[float, float, float] | None = None, hfov_deg: float | None = None) -> View:
    """Score a camera at (x, y, z) looking at ``aim`` (default: the focus centroid). With no
    ``hfov_deg`` the lens is chosen to take in the focus objects (MIN..MAX_HFOV_DEG)."""
    by_id = {p.id: p for p in placements}
    weights = focus_weights(spec, placements)
    aim = aim or _aim(by_id, weights)
    vx, vy = aim[0] - x, aim[1] - y
    dist = math.hypot(vx, vy) or 1e-6
    vx, vy = vx / dist, vy / dist
    ang: dict[str, float] = {}
    for k in weights:
        p = by_id[k]
        ox, oy = p.location[0] - x, p.location[1] - y
        r = math.hypot(ox, oy) or 1e-6
        off = math.atan2(vx * oy - vy * ox, vx * ox + vy * oy)  # signed angle from the view axis
        ang[k] = abs(off) + math.atan2(max(p.dims[0], p.dims[1]) / 2, r) * 0.6
    if hfov_deg is None:
        need = max((a for k, a in ang.items() if math.degrees(a) * 2 <= MAX_HFOV_DEG), default=0.0)
        hfov_deg = max(MIN_HFOV_DEG, min(MAX_HFOV_DEG, math.degrees(need) * 2 + 8.0))
    half = math.radians(hfov_deg) / 2
    seen = tuple(k for k, a in ang.items() if a <= half)
    fronts = 0.0
    for k in seen:
        p = by_id[k]
        wf = SEATING.get(p.category, 0.0) + FEATURES.get(p.category, 0.0)
        if wf <= 0:
            continue
        ox, oy = p.location[0] - x, p.location[1] - y
        r = math.hypot(ox, oy) or 1e-6
        f = _front(p)
        c = -(ox * f[0] + oy * f[1]) / r
        # the back of a sofa or armchair in frame is what reads as "facing the wall": it counts
        # double, and more the nearer it stands to the middle of the picture (a chair's back
        # dead centre in the foreground, lightning aidb5's lobby ring seen along its axis)
        centre = max(0.0, 1.0 - ang[k] / half)
        fronts += wf * c * ((2.0 + 2.0 * centre) if c < 0 and p.category in SEATING else 1.0)
    tot = sum(weights.values())
    framed = sum(weights[k] for k in seen) / tot
    solid = [p for p in placements if p.dims[2] > 0.05]
    clearance = min((_footprint_dist(x, y, p) for p in solid), default=9.0)
    blocked = 0.0
    for p in solid:  # something tall right in front of the lens hides the room
        if p.dims[2] < z - 0.2 or p.id in seen and SEATING.get(p.category):
            continue
        for t in (0.4, 0.8, 1.2):
            if t < dist and _footprint_dist(x + vx * t, y + vy * t, p) < 0.05:
                blocked += 1.0
                break
    score = (2.0 * framed + fronts / max(1.0, len(seen)) ** 0.5 + 0.4 * min(dist, 6.0) / 6.0
             - (3.0 if clearance < _CLEAR_M else 0.0) - 1.0 * blocked)
    return View(x, y, z, aim, hfov_deg, score, seen, fronts)


def best_views(spec: SceneSpec, placements: list[Placement], eye_m: float) -> list[View]:
    """Every candidate spot at eye height, best first."""
    views = [score_view(spec, placements, x, y, eye_m) for x, y in spots(spec, placements)]
    return sorted(views, key=lambda v: -v.score)


def view_issue(spec: SceneSpec, placements: list[Placement], location: Any, rotation_deg: Any,
               lens_mm: float = 50.0, eye_m: float = 1.4,
               off: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str | None:
    """Why a camera a model placed itself is a poor view, with the pose ai-db would use, or None.
    ``location`` and the pose in the message are in the frame ``off`` shifts the layout to (the
    Blender frame). Poor: no focus object in frame (it looks at a blank wall), or it sees the
    backs of the seating."""
    try:
        x, y, z = (float(v) - o for v, o in zip(location, off, strict=True))
        rx, _ry, rz = (math.radians(float(v)) for v in rotation_deg)
    except (TypeError, ValueError):
        return None
    # Blender camera looks along its local -Z: Euler XYZ (rx, 0, rz) -> direction
    dx, dy = -math.sin(rz) * math.sin(rx), math.cos(rz) * math.sin(rx)
    if math.hypot(dx, dy) < 0.2:
        return None  # looking straight down: a plan view, not judged here
    hfov = math.degrees(2 * math.atan(SENSOR_MM / 2 / max(lens_mm, 1.0)))
    mine = score_view(spec, placements, x, y, z, aim=(x + dx * 5, y + dy * 5, z), hfov_deg=hfov)
    best = best_views(spec, placements, eye_m)
    if not best:
        return None
    if not mine.seen:
        why = "it looks at a blank wall: none of the main objects is in frame"
    elif mine.fronts < 0 and best[0].fronts > mine.fronts + 0.5:
        why = "it looks at the backs of the seating (from behind the furniture)"
    else:
        return None
    b = best[0]
    return (f"camera at [{x + off[0]:.1f}, {y + off[1]:.1f}, {z + off[2]:.1f}]: {why}; from the open side: "
            f"location [{b.x + off[0]:.2f}, {b.y + off[1]:.2f}, {b.z + off[2]:.2f}], rotation [90, 0, {b.yaw_deg:.1f}], "
            f"lens {b.lens_mm:.0f}")
