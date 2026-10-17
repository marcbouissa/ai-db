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
#: outdoor subjects a view should frame, front or back alike (a beach's shore group: loungers face
#: the water, the camera looks across them at it)
SUBJECTS = {"sun_lounger": 0.8, "parasol": 0.6, "boardwalk": 0.4, "picnic_table": 0.5}
#: what an audience faces: desks and chairs turned to one are seen from behind it (a classroom
#: photo looks over the class at the board, not from the board at the class)
BOARDS = ("whiteboard", "blackboard", "chalkboard", "projector_screen")
_AUDIENCE = ("desk", "chair", "dining_chair", "office_chair")
#: foreground occluders: anything nearer than this along the view and covering more of the
#: frame than _COVER_MAX (aidb7: a plant, an armchair, a palm trunk filling a corner of the picture)
NEAR_M = 1.5
COVER_MAX = 0.08
OCCLUDER_MIN_TOP_M = 0.85  # plants, armchairs, chair backs, trunks: tall enough to block the view
OUTDOOR_OCCLUDER_MIN_TOP_M = 0.3  # outdoors the eye is higher: ferns at the lens fill the bottom
OUTDOOR_NEAR_M = 4.0
MIN_HFOV_DEG = 55.0  # never narrower: a room photo, not a close-up
MAX_HFOV_DEG = 96.0  # ~16 mm on a 36 mm sensor: wider distorts the corners
SENSOR_MM = 36.0
#: outdoors: tree trunks and lamp posts between the lens and the subject (qwen aidb7 park: a trunk
#: filled the middle of the picture) and tops cut by the top of the frame count against a view
TRUNKS = ("tree", "palm_tree", "street_lamp", "lamp_post_double", "signpost")
CANOPIES = ("tree", "palm_tree")
TRUNK_W = 1.5
NEAR_TREE_W = 0.5  # per tree nearer than the aim (by its crown's angular size)
TRUNK_HIDE_W = 1.5  # per main object (weight up to 1) with a trunk in front of it
TRUNK_R_M = 0.3  # a trunk's half-width as seen (with a margin)
CANOPY_CUT_W = 0.7
OUTDOOR_TREE_W = 0.3
OUTDOOR_BACK_M = 14.0
#: a view toward the water (cosine with the way to it): lightning's aidb7 beach looked inland
WATER_VIEW_W = 1.2
#: by the water a view stands at least this far from the group (the eye is a dune top, 4 m up)
WATER_MIN_DIST_M = 9.0
#: by the water the frame reaches this far above the horizon (tan): the sea under a band of sky
WATER_SKY_TAN = 0.1
#: outdoors the lens starts from this (the group fills the frame; MIN_HFOV_DEG is a room photo)
OUTDOOR_MIN_HFOV_DEG = 40.0
#: outdoors the camera may step back this far beyond the setting's bounds (the ground goes on)
OUTDOOR_REACH_M = 8.0
_ASPECT = 16 / 9
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
        if spec.setting.kind == "outdoor":
            w += SUBJECTS.get(p.category, 0.0)
        if spec.setting.kind == "outdoor" and p.category in CANOPIES and spec.setting.water is None:
            w += OUTDOOR_TREE_W  # by the water the shore group is the subject: palms frame it  # a park view shows its trees whole, not a trunk at the edge
        if _base(p.id, ids) in heroes:
            w += 0.5
        if _base(p.id, ids) == focus:
            w += 2.0  # camera.focus: what the view is about
        if w > 0:
            out[p.id] = w
    if len(out) < 3:  # little marked (a warehouse's forklift): the largest objects too
        solid = sorted((p for p in placements if p.id not in out and p.dims[2] > 0.05),
                       key=lambda p: -(p.dims[0] * p.dims[1] * p.dims[2]))
        out.update({p.id: 0.5 for p in solid[:3 - len(out)]})
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
    if st.kind == "outdoor" and st.water is not None:  # never a camera standing in the water
        from ai_db_scene.terrain import terrain_of

        ter = terrain_of(spec)
        if ter is not None:
            out = [p for p in out if ter.signed(p[0], p[1]) < ter.dry() - 0.5]
    solid = [p for p in placements if p.dims[2] > 0.05]
    clear = [s for s in out if all(_footprint_dist(s[0], s[1], p) >= _CLEAR_M for p in solid)]
    return clear or out


def _aim(placements: dict[str, Placement], weights: dict[str, float]) -> tuple[float, float, float]:
    tot = sum(weights.values())
    x = sum(placements[k].location[0] * w for k, w in weights.items()) / tot
    y = sum(placements[k].location[1] * w for k, w in weights.items()) / tot
    z = sum(min(placements[k].location[2], 1.2) * w for k, w in weights.items()) / tot
    return x, y, max(0.6, min(z, 1.0))


def foreground_cover(placements: list[Placement], x: float, y: float, z: float, aim: tuple[float, float, float],
                     hfov_deg: float, aspect: float = 16 / 9,
                     min_top: float = OCCLUDER_MIN_TOP_M, near_m: float = NEAR_M) -> list[tuple[str, float]]:
    """(id, fraction of the frame) of the objects within NEAR_M of the camera in its view. In the
    image plane (tangent space): an object's box seen at the depth of its nearest face, clipped to
    the frame; the frame shifted down/up as the build's level camera does with its lens shift."""
    from ai_db_scene.constants import SCENE_CAMERA_MAX_SHIFT

    vx, vy = aim[0] - x, aim[1] - y
    n = math.hypot(vx, vy) or 1e-6
    vx, vy = vx / n, vy / n
    th = math.tan(math.radians(hfov_deg) / 2)
    tv = th / aspect
    # lens shift (scene_pipeline.camera_poses): the frame centre at tan = dz / dist, clamped
    t0 = max(-2 * SCENE_CAMERA_MAX_SHIFT * th, min(2 * SCENE_CAMERA_MAX_SHIFT * th, (aim[2] - z) / max(n, 0.5)))
    out = []
    for p in placements:
        near = _footprint_dist(x, y, p)
        if p.location[2] + p.dims[2] / 2 < min_top or near > near_m:
            continue  # low pieces (a desk or coffee-table top at the bottom edge) frame a room photo
        ox, oy = p.location[0] - x, p.location[1] - y
        ahead = ox * vx + oy * vy
        if ahead <= 0:
            continue  # behind the lens
        side = ox * -vy + oy * vx
        span = max(p.dims[0], p.dims[1]) / 2
        depth = max(ahead - span, near, 0.25)  # its nearest face (a desk seen corner-on: the corner)
        h0, h1 = max((side - span) / depth, -th), min((side + span) / depth, th)
        bottom, top = p.location[2] - p.dims[2] / 2, p.location[2] + p.dims[2] / 2
        far = max(ahead + span, depth)  # a top below the eye shows its far edge highest (a desk top)
        v0, v1 = max((bottom - z) / depth, t0 - tv), min((top - z) / (far if top < z else depth), t0 + tv)
        if h1 <= h0 or v1 <= v0:
            continue
        out.append((p.id, (h1 - h0) / (2 * th) * (v1 - v0) / (2 * tv)))
    return out


def _audience(placements: list[Placement]) -> set[str]:
    """Desks and chairs that face a board (within 45 deg)."""
    boards = [p for p in placements if p.category in BOARDS]
    out = set()
    for p in placements:
        if p.category not in _AUDIENCE:
            continue
        f = _front(p)
        for b in boards:
            ox, oy = b.location[0] - p.location[0], b.location[1] - p.location[1]
            r = math.hypot(ox, oy) or 1e-6
            if (ox * f[0] + oy * f[1]) / r > 0.7:
                out.add(p.id)
                break
    return out


def score_view(spec: SceneSpec, placements: list[Placement], x: float, y: float, z: float,
               aim: tuple[float, float, float] | None = None, hfov_deg: float | None = None) -> View:
    """Score a camera at (x, y, z) looking at ``aim`` (default: the focus centroid). With no
    ``hfov_deg`` the lens is chosen to take in the focus objects (MIN..MAX_HFOV_DEG)."""
    by_id = {p.id: p for p in placements}
    weights = focus_weights(spec, placements)
    auto_aim = aim is None
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
    outdoor = spec.setting.kind == "outdoor"
    water_scene = False
    if outdoor:
        from ai_db_scene.terrain import terrain_of

        ter0 = terrain_of(spec)
        water_scene = ter0 is not None and ter0.water is not None
    if hfov_deg is None:
        need = max((a for k, a in ang.items() if math.degrees(a) * 2 <= MAX_HFOV_DEG), default=0.0)
        hfov_deg = max(MIN_HFOV_DEG if not outdoor else OUTDOOR_MIN_HFOV_DEG,
                       min(MAX_HFOV_DEG, math.degrees(need) * 2 + 8.0))
        if outdoor and auto_aim:
            # frame the group from the ground at its near edge up to the tree tops (by the water:
            # up past the horizon), the lens shift centring that band: the old frame centred on
            # the group's middle height needed a 93 deg lens for the canopies, and the group
            # filled 40 % of the picture over a foreground of empty grass (qwen aidb7 park)
            hfov_deg, aim = _outdoor_frame(spec, placements, weights, x, y, z, aim, dist, hfov_deg, water_scene)
        elif outdoor:  # wide enough (up to the limit) for the canopies of the trees in frame
            t0 = _shift_tan(aim, z, dist, hfov_deg)
            up = max((_top_tan(p, x, y, z) - t0 for p in placements if p.category in CANOPIES
                      and abs(_off_angle(p, x, y, vx, vy)) <= math.radians(hfov_deg) / 2), default=0.0)
            if up > 0:
                hfov_deg = max(hfov_deg, min(MAX_HFOV_DEG, math.degrees(2 * math.atan(up * 1.08 * _ASPECT))))
    half = math.radians(hfov_deg) / 2
    seen = tuple(k for k, a in ang.items() if a <= half)
    fronts = 0.0
    facing_board = 0.0
    audience = _audience(placements)
    for k in seen:
        p = by_id[k]
        wf = SEATING.get(p.category, 0.0) + FEATURES.get(p.category, 0.0)
        if audience and p.category in BOARDS:
            wf *= 3.0  # the board is what a classroom view is about
        if wf <= 0:
            continue
        ox, oy = p.location[0] - x, p.location[1] - y
        r = math.hypot(ox, oy) or 1e-6
        f = _front(p)
        c = -(ox * f[0] + oy * f[1]) / r
        if k in audience:
            c = -c  # seen from behind the class, looking where it looks: right; from the board: wrong
        if audience and p.category in BOARDS:
            facing_board += 1.5 * c  # the board face-on from the back, not edge-on from a corner
        # the back of a sofa or armchair in frame is what reads as "facing the wall": it counts
        # double, and more the nearer it stands to the middle of the picture (a chair's back
        # dead centre in the foreground, lightning aidb5's lobby ring seen along its axis)
        centre = max(0.0, 1.0 - ang[k] / half)
        fronts += wf * c * ((2.0 + 2.0 * centre) if c < 0 and p.category in SEATING and k not in audience else 1.0)
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
    # how much of the picture the subject fills: a few racks in one corner of a 50 m hall seen
    # from the far wall were specks (lightning aidb5 warehouse)
    fill = min(1.0, max((ang[k] for k in seen), default=0.0) / half)
    cover = sum(f for _k, f in foreground_cover(placements, x, y, z, aim, hfov_deg,
                                                min_top=OUTDOOR_OCCLUDER_MIN_TOP_M if outdoor else OCCLUDER_MIN_TOP_M,
                                                near_m=OUTDOOR_NEAR_M if outdoor else NEAR_M)
                if f > COVER_MAX)
    trunks = cut = water = hides = 0.0
    if outdoor:
        from ai_db_scene.terrain import terrain_of

        ter = terrain_of(spec)
        if ter is not None and ter.water is not None:  # a beach photo looks across the shore to the water
            n = ter.to_water(x, y)
            water = vx * n[0] + vy * n[1]
            if water < 0.3:
                water -= 2.0  # along the shore or inland: the water is not the picture's background
            if dist < WATER_MIN_DIST_M:  # from the dune top close up is a drone shot of the loungers
                water -= (WATER_MIN_DIST_M - dist) / WATER_MIN_DIST_M * 2.0
        t_top = _shift_tan(aim, z, dist, hfov_deg) + math.tan(half) / _ASPECT
        for p in placements:
            if p.category not in TRUNKS:
                continue
            off = _off_angle(p, x, y, vx, vy)
            r = math.hypot(p.location[0] - x, p.location[1] - y)
            crown = math.atan2(max(p.dims[0], p.dims[1]) / 2, max(r, 0.5)) if p.category in CANOPIES else 0.0
            if abs(off) > half + crown:
                continue  # (a palm whose crown reaches into the frame shows its trunk at the edge)
            aoff = min(abs(off), half)
            if r < 0.75 * dist:  # a trunk nearer than the subject, the more the nearer the middle
                # (one near the lens at the edge of the frame still shows: a palm trunk down the side)
                edge = 0.35 if r < 0.5 * dist else 0.0
                trunks += max(1.0 - aoff / half, edge) * min(1.0, 4.0 / max(r, 1.0))
            if p.category in CANOPIES and r < dist - 0.5:
                # a tree on the near side of the group: its crown fills a side of the picture
                trunks += NEAR_TREE_W * min(1.0, max(p.dims[0], p.dims[1]) / max(r, 1.0))
            hides += _trunk_hides(p, x, y, r, off, by_id, weights, seen, vx, vy)
            if p.category in CANOPIES and _top_tan(p, x, y, z) > t_top:
                cut += 1.0
    # room to step back: outdoors a park or beach is seen whole from further off (the qwen aidb7
    # park view stood 4 m from the fountain with a 96 deg lens, the trees cut at the top)
    back = 0.8 * min(dist, OUTDOOR_BACK_M) / OUTDOOR_BACK_M if outdoor else 0.4 * min(dist, 6.0) / 6.0
    score = (2.0 * framed + fronts / max(1.0, len(seen)) ** 0.5 + back + 0.8 * fill
             - (3.0 if clearance < _CLEAR_M else 0.0) - 1.0 * blocked - 6.0 * cover + facing_board
             - TRUNK_W * trunks - CANOPY_CUT_W * cut + WATER_VIEW_W * water - TRUNK_HIDE_W * hides)
    return View(x, y, z, aim, hfov_deg, score, seen, fronts)


def _trunk_hides(t: Placement, x: float, y: float, r: float, off: float, by_id: dict[str, Placement],
                 weights: dict[str, float], seen: tuple[str, ...], vx: float, vy: float) -> float:
    """How much a trunk (or lamp post) stands in front of the main objects in frame: a ring tree
    between the lens and the fountain (qwen aidb7 park) cuts the picture in two."""
    tw = math.atan2(TRUNK_R_M, max(r, 0.5))
    out = 0.0
    for k in seen:
        p = by_id[k]
        if p.category in TRUNKS or p.category in CANOPIES:
            continue
        rp = math.hypot(p.location[0] - x, p.location[1] - y)
        if rp < r + 0.5:
            continue  # the subject stands in front of the trunk
        pw = math.atan2(max(p.dims[0], p.dims[1]) / 2, max(rp, 0.5))
        if abs(_off_angle(p, x, y, vx, vy) - off) < pw + tw:
            out += min(1.0, weights.get(k, 0.0))
    return out


def _outdoor_frame(spec: SceneSpec, placements: list[Placement], weights: dict[str, float], x: float, y: float,
                   z: float, aim: tuple[float, float, float], dist: float, hfov_deg: float,
                   water: bool) -> tuple[float, tuple[float, float, float]]:
    """(hfov, aim) of an outdoor view: the vertical band from the ground at the near edge of the
    focus group to the tops in frame (tree canopies, the parasol; past the horizon by the water),
    centred by the lens shift, the lens just wide enough for the band and the group's width."""
    vx, vy = (aim[0] - x) / dist, (aim[1] - y) / dist
    lo = hi = None
    for _ in range(2):  # the band depends on what is in frame, the frame on the band
        half = math.radians(hfov_deg) / 2
        inside = [p for p in placements if abs(_off_angle(p, x, y, vx, vy)) <= half and p.dims[2] > 0.05
                  and (p.id in weights or p.category in CANOPIES)]
        if not inside:
            break
        near = min(math.hypot(p.location[0] - x, p.location[1] - y) - max(p.dims[0], p.dims[1]) / 2
                   for p in inside if p.id in weights) if any(p.id in weights for p in inside) else dist
        lo = -z / max(near, 1.0)
        hi = max(_top_tan(p, x, y, z) for p in inside)
        if water:
            hi = max(hi, WATER_SKY_TAN)  # the horizon and a band of sky over the water
        span = (hi - lo) / 2 * 1.06
        need = math.degrees(2 * math.atan(span * _ASPECT))
        hfov_deg = max(hfov_deg, min(MAX_HFOV_DEG, need))
    if lo is None or hi is None:
        return hfov_deg, aim
    # the frame's bottom edge at the near edge of the group: what the band leaves over goes to the
    # sky and the trees, not to empty ground in front (lightning aidb7 beach: half the picture sand)
    half_v = math.tan(math.radians(hfov_deg) / 2) / _ASPECT
    centre = max((hi + lo) / 2, lo - 0.04 + half_v)
    if water:  # the horizon no higher than a third from the top: the sea, not an empty sky
        centre = min(centre, max((hi + lo) / 2, -half_v / 3))
    return hfov_deg, (aim[0], aim[1], z + centre * dist)


def _off_angle(p: Placement, x: float, y: float, vx: float, vy: float) -> float:
    """Signed angle of p from the view axis (vx, vy unit)."""
    ox, oy = p.location[0] - x, p.location[1] - y
    return math.atan2(vx * oy - vy * ox, vx * ox + vy * oy)


def _top_tan(p: Placement, x: float, y: float, z: float) -> float:
    """tan of the elevation of p's top seen from the eye (at its near side)."""
    r = math.hypot(p.location[0] - x, p.location[1] - y) - max(p.dims[0], p.dims[1]) / 2
    return (p.location[2] + p.dims[2] / 2 - z) / max(r, 0.5)


def _shift_tan(aim: tuple[float, float, float], z: float, dist: float, hfov_deg: float) -> float:
    """tan of the frame centre's elevation: the build's level camera shifts its lens toward the aim."""
    from ai_db_scene.constants import SCENE_CAMERA_MAX_SHIFT

    th = math.tan(math.radians(hfov_deg) / 2)
    return max(-2 * SCENE_CAMERA_MAX_SHIFT * th, min(2 * SCENE_CAMERA_MAX_SHIFT * th, (aim[2] - z) / max(dist, 0.5)))


def near_spots(spec: SceneSpec, placements: list[Placement]) -> list[tuple[float, float]]:
    """Spots round the focus objects, far enough to take them in at a medium lens: in a big room
    the walls are too far from a small group."""
    from ai_db_scene.shapes import RoomShape

    by_id = {p.id: p for p in placements}
    weights = focus_weights(spec, placements)
    ax, ay, _az = _aim(by_id, weights)
    reach = max((math.hypot(by_id[k].location[0] - ax, by_id[k].location[1] - ay)
                 + max(by_id[k].dims[0], by_id[k].dims[1]) / 2 for k in weights), default=1.0)
    w, d, _h = spec.setting.bounds_m
    pts = tuple((float(p[0]), float(p[1])) for p in spec.setting.points)
    shape = RoomShape(spec.setting.shape, w, d, pts) if spec.setting.shape != "polygon" or len(pts) >= 3 else None
    solid = [p for p in placements if p.dims[2] > 0.05]
    outdoor = spec.setting.kind == "outdoor"
    ter = None
    if outdoor:
        from ai_db_scene.terrain import terrain_of

        ter = terrain_of(spec)
    radii = [reach / math.tan(math.radians(35)) + 1.0, reach / math.tan(math.radians(25)) + 1.0]
    if outdoor:  # step back further: the whole group and the tree tops in one picture
        radii.append(reach / math.tan(math.radians(18)) + 1.0)
    out = []
    cands = [(ax + r * math.cos(math.radians(t)), ay + r * math.sin(math.radians(t))) for r in radii
             for t in range(0, 360, 30)]
    if ter is not None and ter.water is not None:
        # inland of the group, looking back across it at the water (the rings are too far apart
        # for a wide beach: lightning's 50 m cove had one spot left, looking along the shore)
        n = ter.to_water(ax, ay)
        for dist in (6.0, 9.0, 11.0, 13.0, 15.0, 18.0, 24.0):
            for t in (-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0):  # (diagonals: along the beach)
                c, sn = math.cos(math.radians(t)), math.sin(math.radians(t))
                ux, uy = n[0] * c - n[1] * sn, n[0] * sn + n[1] * c
                cands.append((ax - ux * dist, ay - uy * dist))
    for x, y in cands:
        if outdoor:  # anywhere on the land round the setting, never in the water
            m = OUTDOOR_REACH_M
            inside = -m <= x <= w + m and -m <= y <= d + m and (
                ter is None or ter.water is None or ter.signed(x, y) < ter.dry() - 0.5)
        else:
            inside = shape.contains(x, y, -_INSET_M) if shape is not None and spec.setting.shape != "rect" else (
                _INSET_M <= x <= w - _INSET_M and _INSET_M <= y <= d - _INSET_M)
        if inside and all(_footprint_dist(x, y, p) >= _CLEAR_M for p in solid):
            out.append((x, y))
    return out


def best_views(spec: SceneSpec, placements: list[Placement], eye_m: float) -> list[View]:
    """Every candidate spot at eye height (by the walls, and round the focus in a big room), best first."""
    views = [score_view(spec, placements, x, y, eye_m)
             for x, y in dict.fromkeys(spots(spec, placements) + near_spots(spec, placements))]
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
