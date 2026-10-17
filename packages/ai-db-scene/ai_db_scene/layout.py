"""Deterministic layout solver: SceneSpec + priors -> oriented-box placements.

Conventions
-----------
* Room: x in [0, W] (east-west), y in [0, D] (north-south), z up. wall_south is y=0,
  wall_north y=D, wall_west x=0, wall_east x=W. Only ``indoor`` settings have walls.
* Object dims are [w, d, h]: w along the local x axis, d along local y. The *front* of an
  object is its local -Y side (shared asset convention). ``rotation_deg_z`` is the yaw
  (counter-clockwise, 0 = front faces -y; front vector = (sin yaw, -cos yaw)).
* Room shape ``rect`` (default) | ``ellipse`` | ``polygon`` inside the box [0,W]x[0,D]. Rect walls
  are the four named walls; ellipse has ``wall`` (+ ``at_deg`` = ellipse parameter angle; copies
  of one object at one at_deg stand side by side along the wall);
  polygons have ``wall_<n>`` (edge n, 1-based) and ``wall`` with ``at_deg`` (ray from the center). ``location`` is the center of the bounding box (z included).
* ``A against_wall W`` with ``gap_m``: A's back stands that far off the wall (a sofa floated
  into a deep living room); wall-mounted objects always hang flush.
* ``A above B``: A is wall-mounted (support ``wall``, with its against_wall relation) straight
  above B: the same position along the wall, A's base ``gap_m`` (default 0.25 m) over B's top.
  A ceiling-hung A (pendant lamp) hangs from the ceiling straight over B's centre.
* ``count > 1`` expands an object into instances ``<id>_1 .. <id>_n``. Relations of the
  object apply to every instance; a target with several instances is paired by index.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

from ai_db_scene.constants import (
    LAYOUT_BREATHING_M,
    LAYOUT_CANDIDATES,
    LAYOUT_DEFAULT_ABOVE_GAP_M,
    LAYOUT_DEFAULT_AROUND_GAP_M,
    LAYOUT_DEFAULT_BESIDE_GAP_M,
    LAYOUT_DEFAULT_IN_FRONT_GAP_M,
    LAYOUT_DOOR_CLEAR_M,
    LAYOUT_DOOR_HEIGHT_M,
    LAYOUT_EPS_M,
    LAYOUT_FACING_ATTEMPTS,
    LAYOUT_FACING_MISS_COST,
    LAYOUT_FACING_TOL_DEG,
    LAYOUT_FLAT_MAX_H_M,
    LAYOUT_LOCAL_ITERS,
    LAYOUT_LOCAL_STEP_M,
    LAYOUT_LOCAL_YAW_DEG,
    LAYOUT_REL_TOL_M,
    LAYOUT_RESTARTS,
    LAYOUT_RETRY_CANDIDATES,
    LAYOUT_SEED,
)
from ai_db_scene.priors import CategoryPrior, load_priors
from ai_db_scene.shapes import RoomShape
from ai_db_scene.spec import ROOM, SceneSpec

Dims = tuple[float, float, float]
_HALF_PI = math.pi / 2
_GAP_RELS = ("in_front_of", "beside", "around")
_PRIMARY_ORDER = ("on", "under", "above", "centered_in", "around", "in_front_of", "beside")
_DEP_RELS = ("on", "under", "above", "centered_in", "around", "in_front_of", "beside")
_ORIENT_RELS = ("against_wall", "facing", "in_front_of", "beside", "on", "under", "centered_in", "around", "above")
_CURVED_WALL_GAP_M = 0.25  # max back-face gap to a curved wall (sagitta of a flat back on a curve)
_LATERAL_TOL_M = 0.05
#: shelf surfaces of storage furniture, as fractions of its height (the built-in recipes' deck
#: and shelf tops): ``A on B`` puts A on one of B's levels, bottom first, side by side along B's
#: width - pallets on a pallet rack were refused ("cannot support other objects") and repair
#: dropped the relation (lightning aidb5 warehouse)
SHELF_LEVELS: dict[str, tuple[float, ...]] = {"pallet_rack": (0.0, 0.381, 0.711),
                                              "shelving_unit": (0.07, 0.35, 0.64, 0.97)}
_SHELF_SPACING_M = 0.08  # between neighbours on one level
#: categories that stack on copies of themselves when their support's top is full (crates on a pallet)
STACKABLE = ("crate", "pallet", "book_stack", "barrel")
#: clearance (m, box to box) between kinds with no relation of their own: a street lamp stood
#: 0.3 m from a park bench's end (lightning aidb7 park), crowding it
PAIR_CLEARANCE_M: dict[frozenset[str], float] = {
    **{frozenset((lamp, seat)): 1.5 for lamp in ("street_lamp", "lamp_post_double")
       for seat in ("bench", "picnic_table", "trash_can")},
    frozenset(("street_lamp", "street_lamp")): 3.0}
_RING_YAW_TOL_DEG = 3.0  # a ring member faces the centre of what it stands round


@dataclass(frozen=True)
class Placement:
    id: str
    location: tuple[float, float, float]  # bbox center, meters
    rotation_deg_z: float
    dims: Dims  # [w, d, h] meters
    category: str = ""
    facing_target: str | None = None  # instance id this object was oriented toward, if any

    @property
    def yaw_deg(self) -> float:
        return self.rotation_deg_z


class LayoutError(Exception):
    """The spec cannot be laid out; ``conflicts`` names the relations/objects involved."""

    def __init__(self, message: str, conflicts: list[str]) -> None:
        super().__init__(message)
        self.conflicts = conflicts


_SPREAD_GAP_M = 0.6  # between copies of one object placed along a curved wall at one at_deg

#: ObjectSpec.seats: the chair that goes with a table when the spec names none
SEAT_CATEGORY = {"desk": "chair", "dining_table": "dining_chair", "conference_table": "office_chair",
                 "kitchen_island": "stool", "workbench": "stool", "picnic_table": "stool",
                 "bar_counter": "bar_stool", "reception_desk": "office_chair"}
_SQUARE_FACING_TOL_DEG = 15.0  # a facing object is squared to the room when it then faces within this
_SEAT_GAP_M = 0.05  # table edge -> chair (pulled up to it, not tucked: no interpenetration)
_SEAT_SPACING_M = 0.15  # between chairs along one side


@dataclass(frozen=True)
class _SeatPlan:
    """A table with its chairs, solved as one box. Offsets are in the box's (right, front) frame
    from the box centre; front chairs come first, then back, left, right."""

    table: Dims
    category: str
    chair: Dims
    table_off: tuple[float, float]
    seats: tuple[tuple[float, float], ...]
    box: Dims


#: what rows of desks face, and the walkway kept behind their last row (m)
_BOARDS = ("whiteboard", "blackboard", "chalkboard", "projector_screen")
_BACK_AISLE_M = 0.6
#: round table diameter (m) by the chairs round it: a cafe two-top, a four-top, a family table
_ROUND_DIAMETER_M = ((2, 0.8), (4, 1.0), (6, 1.2))


def round_table_dims(category: str, typical: Dims, seats: int) -> Dims:
    """A round table's footprint: a square of its diameter (dining by its seats, else the
    category's typical depth: a round coffee table as deep as a rect one)."""
    if category in ("dining_table", "conference_table"):
        dia = next((d for n, d in _ROUND_DIAMETER_M if seats <= n), 1.4)
    else:
        dia = min(typical[0], max(typical[1], 0.5) * 1.3)
    return (dia, dia, typical[2])


def _seat_plan(table: Dims, n: int, chair: Dims, category: str, back_first: bool = False,
               front_only: bool = False) -> _SeatPlan:
    """``back_first``: a table that *faces* something (desks facing the board) seats its first
    chair behind it, so the person looks the same way; otherwise the front (a desk against a
    wall faces into the room, its chair in front of it). ``front_only``: every seat along the
    front (a counter against a wall: stools never between it and the wall)."""
    w, d, h = table
    first, second = ("back", "front") if back_first else ("front", "back")
    cw, cd = chair[0], chair[1]
    pitch = cw + _SEAT_SPACING_M
    cap_long = max(1, int((w + _SEAT_SPACING_M) // pitch))
    cap_end = max(1, int((d + _SEAT_SPACING_M) // pitch))
    sides = {"front": 0, "back": 0, "left": 0, "right": 0}
    for _ in range(n):  # front/back alternately, then the ends, then overflow on the long sides
        if front_only:
            sides[first if not back_first else "front"] += 1
            continue
        side = next((sd for sd, cap in ((first, cap_long), (second, cap_long)) if sides[sd] < cap
                     and sides[sd] <= min(sides["front"], sides["back"])), None) \
            or next((sd for sd, cap in (("left", cap_end), ("right", cap_end)) if sides[sd] < cap
                     and sides[sd] <= min(sides["left"], sides["right"])), None) \
            or (first if sides[first] <= sides[second] else second)
        sides[side] += 1
    reach = _SEAT_GAP_M + cd / 2
    if front_only:  # spread along the whole front (two stools at a 2.4 m counter, not huddled mid-way)
        pitch = max(pitch, w / max(n, 1))
    pts: list[tuple[float, float]] = []
    for side in ("front", "back", "left", "right"):
        k = sides[side]
        for i in range(k):
            along = (i - (k - 1) / 2) * pitch
            if side == "front":
                pts.append((along, d / 2 + reach))
            elif side == "back":
                pts.append((along, -d / 2 - reach))
            elif side == "left":
                pts.append((-w / 2 - reach, along))
            else:
                pts.append((w / 2 + reach, along))
    xs = [-w / 2, w / 2] + [x + sgn * (cw / 2 if abs(y) > d / 2 else cd / 2) for x, y in pts for sgn in (-1, 1)]
    ys = [-d / 2, d / 2] + [y + sgn * (cd / 2 if abs(y) > d / 2 else cw / 2) for x, y in pts for sgn in (-1, 1)]
    rc, fc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return _SeatPlan(table, category, chair, (-rc, -fc), tuple((x - rc, y - fc) for x, y in pts),
                     (max(xs) - min(xs), max(ys) - min(ys), h))


def _local(p: _Pose, off: tuple[float, float]) -> tuple[float, float]:
    r, f = _axes(p.yaw)
    return p.x + off[0] * r[0] + off[1] * f[0], p.y + off[0] * r[1] + off[1] * f[1]


@dataclass(frozen=True)
class _Inst:
    id: str
    base: str
    category: str
    prior: CategoryPrior
    dims: Dims
    hero: bool
    idx: int
    n: int
    #: ObjectSpec.turn_deg: the object stands this many degrees turned from its solved yaw; the
    #: solver works with its footprint as turned (``dims`` swapped for 90/270)
    turn: int = 0

    @property
    def flat(self) -> bool:
        return self.dims[2] <= LAYOUT_FLAT_MAX_H_M


@dataclass(frozen=True)
class _Rel:
    a: str
    rel: str
    b: str
    gap: float
    ring_i: int
    ring_n: int
    label: str
    at_deg: float | None = None


@dataclass
class _Pose:
    x: float
    y: float
    yaw: float
    zb: float  # base z


def _axes(yaw: float) -> tuple[tuple[float, float], tuple[float, float]]:
    return (math.cos(yaw), math.sin(yaw)), (math.sin(yaw), -math.cos(yaw))  # (right, front); front = -Y at yaw 0


def _ext(dims: Dims, yaw: float, u: tuple[float, float]) -> float:
    r, f = _axes(yaw)
    return dims[0] / 2 * abs(u[0] * r[0] + u[1] * r[1]) + dims[1] / 2 * abs(u[0] * f[0] + u[1] * f[1])


def _sep(pa: _Pose, da: Dims, pb: _Pose, db: Dims) -> float:
    """Separating-axis distance between two oriented rectangles (<0 when overlapping)."""
    dx, dy = pa.x - pb.x, pa.y - pb.y
    best = -1e9
    for yaw in (pa.yaw, pb.yaw):
        for u in _axes(yaw):
            s = abs(dx * u[0] + dy * u[1]) - _ext(da, pa.yaw, u) - _ext(db, pb.yaw, u)
            best = max(best, s)
    return best


def _front_hits(pa: _Pose, pb: _Pose, db: Dims) -> bool:
    """Does the ray from A's centre along A's front cross B's oriented footprint?"""
    rb, fb = _axes(pb.yaw)
    fa = _axes(pa.yaw)[1]
    # A's centre and front direction in B's frame (x along B's right, y along B's front)
    ox, oy = (pa.x - pb.x) * rb[0] + (pa.y - pb.y) * rb[1], (pa.x - pb.x) * fb[0] + (pa.y - pb.y) * fb[1]
    dx, dy = fa[0] * rb[0] + fa[1] * rb[1], fa[0] * fb[0] + fa[1] * fb[1]
    t0, t1 = 0.0, math.inf
    for o, dv, half in ((ox, dx, db[0] / 2), (oy, dy, db[1] / 2)):
        if abs(dv) < 1e-9:
            if abs(o) > half:
                return False
            continue
        ta, tb = (-half - o) / dv, (half - o) / dv
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
    return t0 <= t1


def _corners(p: _Pose, d: Dims) -> list[tuple[float, float]]:
    r, f = _axes(p.yaw)
    out = []
    for sw in (-1, 1):
        for sd in (-1, 1):
            out.append((p.x + sw * d[0] / 2 * r[0] + sd * d[1] / 2 * f[0], p.y + sw * d[0] / 2 * r[1] + sd * d[1] / 2 * f[1]))
    return out


def _inside(p: _Pose, d: Dims, outer: _Pose, od: Dims, eps: float) -> bool:
    r, f = _axes(outer.yaw)
    for cx, cy in _corners(p, d):
        dx, dy = cx - outer.x, cy - outer.y
        if abs(dx * r[0] + dy * r[1]) > od[0] / 2 + eps or abs(dx * f[0] + dy * f[1]) > od[1] / 2 + eps:
            return False
    return True


def _face_yaw(ax: float, ay: float, bx: float, by: float) -> float:
    return math.atan2(bx - ax, -(by - ay))


#: outdoor sizes of categories whose typical size is an indoor one (a park bin, not a wastebasket)
OUTDOOR_DIMS: dict[str, Dims] = {"trash_can": (0.5, 0.5, 0.9)}


def _angdiff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class _Problem:
    """Instances, expanded relations and the geometric predicates shared by solver and checker."""

    def __init__(self, spec: SceneSpec, priors: dict[str, CategoryPrior], dims_override: dict[str, Dims]) -> None:
        self.spec = spec
        self.kind = spec.setting.kind
        self.W, self.D, self.H = spec.setting.bounds_m
        st = spec.setting
        self.shape = RoomShape(st.shape, self.W, self.D, st.points)
        #: (free ceiling ids, grid cells) of ceiling_grid
        self.ceiling_cache: tuple[tuple[str, ...], list[tuple[float, float]]] = ((), [])
        self.insts: dict[str, _Inst] = {}
        #: grid objects: id -> (rows, cols, gap x, gap y, cell dims). Solved as one block
        #: instance (relations apply to the block), expanded into cells by _to_placements.
        self.grids: dict[str, tuple[int, int, float, float, Dims]] = {}
        #: objects with seats: their instances (grid cells) are solved as table + chairs boxes
        self.seated: dict[str, _SeatPlan] = {}
        by_base: dict[str, list[str]] = {}
        cats = {o.id: o.category for o in spec.objects}
        for o in spec.objects:
            prior = priors.get(o.category)
            if prior is None:
                raise LayoutError(f"object '{o.id}': category '{o.category}' has no prior", [o.id])
            if self.kind == "outdoor" and o.category in OUTDOOR_DIMS and not (
                    dims_override.get(o.id + "_1") or dims_override.get(o.id)):
                dims_override = {**dims_override, o.id: OUTDOOR_DIMS[o.category]}
            if o.shape == "round" and not (dims_override.get(o.id + "_1") or dims_override.get(o.id)):
                dims_override = {**dims_override, o.id: round_table_dims(o.category, prior.typical_m, o.seats)}
            if o.seats:
                sc = o.seat_category or SEAT_CATEGORY.get(o.category, "chair")
                if sc not in priors:
                    raise LayoutError(f"object '{o.id}': seat category '{sc}' has no prior", [o.id])
                table = (dims_override.get(o.id + "_1") or dims_override.get(o.id) or prior.typical_m)
                faces = any(r.a == o.id and r.rel == "facing" for r in spec.relations)
                # against a wall (a bar counter, a desk): every seat on the room side, none behind
                walled = any(r.a == o.id and r.rel == "against_wall" for r in spec.relations)
                plan = _seat_plan(table, o.seats, priors[sc].typical_m, sc, back_first=faces, front_only=walled)
                self.seated[o.id] = plan
                dims_override = {**dims_override, o.id: plan.box, o.id + "_1": plan.box,
                                 **{f"{o.id}_{i}": plan.box for i in range(2, o.count + 1)}}
            if o.grid is not None:
                rows, cols, gx, gy = o.grid
                cell = dims_override.get(o.id + "_1") or dims_override.get(o.id) or prior.typical_m
                self.grids[o.id] = (rows, cols, gx, gy, cell)
                block = (cols * cell[0] + (cols - 1) * gx, rows * cell[1] + (rows - 1) * gy, cell[2])
                if o.seats and any(r.a == o.id and r.rel == "facing" and r.b in cats and cats[r.b] in _BOARDS
                                   for r in spec.relations):
                    # rows of seated desks facing a board keep a walkway behind the last row (aidb7
                    # classrooms: the back row's chairs stood against the wall, every camera spot
                    # at the back had a desk in its face); the rows stay at the block's front
                    block = (block[0], block[1] + _BACK_AISLE_M, block[2])
                self.insts[o.id] = _Inst(o.id, o.id, o.category, prior, block, o.hero, 0, 1)
                by_base[o.id] = [o.id]
                continue
            for i in range(1, o.count + 1):
                iid = o.id if o.count == 1 else f"{o.id}_{i}"
                dims = dims_override.get(iid) or dims_override.get(o.id) or prior.typical_m
                turn = 0 if o.id in self.seated else o.turn_deg
                if turn % 180:
                    dims = (dims[1], dims[0], dims[2])
                self.insts[iid] = _Inst(iid, o.id, o.category, prior, dims, o.hero, i - 1, o.count, turn)
                by_base.setdefault(o.id, []).append(iid)
        self.rels: list[_Rel] = []
        from ai_db_scene.terrain import LINK_REL, WATER, terrain_of

        self.terrain = terrain_of(spec)  # outdoor ground/water: relations to the water live there
        self.by_base = by_base
        for r in spec.relations:
            if r.b == WATER or r.rel == LINK_REL:  # the terrain's (a boardwalk's land end: terrain.link_cost)
                continue
            for ai in by_base[r.a]:
                a = self.insts[ai]
                if r.b in by_base:
                    bl = by_base[r.b]
                    bi = bl[0] if len(bl) == 1 else bl[a.idx % len(bl)]
                else:
                    bi = r.b
                default = {"in_front_of": LAYOUT_DEFAULT_IN_FRONT_GAP_M, "beside": LAYOUT_DEFAULT_BESIDE_GAP_M,
                           "around": LAYOUT_DEFAULT_AROUND_GAP_M, "above": LAYOUT_DEFAULT_ABOVE_GAP_M}.get(r.rel, 0.0)
                gap = default if r.gap_m is None else r.gap_m
                if r.gap_m is None and r.rel == "beside" and r.b in by_base:  # a lamp beside a bench: not at its end
                    gap = max(gap, PAIR_CLEARANCE_M.get(frozenset((a.category, self.insts[bi].category)), 0.0))
                if r.rel == "against_wall" and a.prior.support in ("wall", "ceiling"):
                    gap = 0.0  # a picture or window hangs on the wall itself
                label = f"{ai} {r.rel} {bi}" + (f" (gap {gap:g} m)" if r.rel in (*_GAP_RELS, "above")
                                                or (r.rel == "against_wall" and gap) else "")
                at = self._spread_at(a, r.b, r.at_deg) if r.rel == "against_wall" else r.at_deg
                self.rels.append(_Rel(ai, r.rel, bi, gap, a.idx, a.n, label, at))
        # every object `around` one target shares one ring, evenly spaced: four armchairs written
        # as four objects each `around low_table` were rings of one, each at a random angle and
        # yaw (qwen / lightning aidb5-6 round lobbies)
        ring: dict[str, list[int]] = {}
        for j, rl in enumerate(self.rels):
            if rl.rel == "around" and rl.b in self.insts:
                ring.setdefault(rl.b, []).append(j)
        for js in ring.values():
            for i, j in enumerate(js):
                self.rels[j] = replace(self.rels[j], ring_i=i, ring_n=len(js))
        self.by_a: dict[str, list[_Rel]] = {}
        self.by_b: dict[str, list[_Rel]] = {}
        for rl in self.rels:
            self.by_a.setdefault(rl.a, []).append(rl)
            if rl.b in self.insts:
                self.by_b.setdefault(rl.b, []).append(rl)
        self.pair_gap: dict[frozenset[str], float] = {}
        for rl in self.rels:
            if rl.rel in _GAP_RELS and rl.b in self.insts:
                k = frozenset((rl.a, rl.b))
                self.pair_gap[k] = min(self.pair_gap.get(k, 1e9), rl.gap)
        #: pairs placed against each other on purpose (A beside B, A beside B in_front_of C): no
        #: breathing room between them, or the local search drifts armchairs off the coffee table
        self.grouped: set[frozenset[str]] = set()
        near = {(rl.a, rl.b) for rl in self.rels if rl.rel in ("beside", "in_front_of") and rl.b in self.insts}
        for ga, gb in near:
            self.grouped.add(frozenset((ga, gb)))
            self.grouped |= {frozenset((ga, gc)) for gb2, gc in near if gb2 == gb}
        focus = spec.camera.get("focus")
        self.focus = by_base[focus][0] if focus in by_base else None
        self.under_pairs = {frozenset((r.a, r.b)) for r in self.rels if r.rel == "under"}
        #: A on a shelf holder: (level height fraction, offset along B's width, turned a quarter) per
        #: instance; a pallet deeper than the rack turns its long side along the beams
        self.shelf_slot: dict[str, tuple[float, float, bool]] = {}
        for bid in dict.fromkeys(r.b for r in self.rels if r.rel == "on" and self.shelf(r.b)):
            self._shelf_slots(bid)
        # what stands on a shelf (and what stands on that: crates on a racked pallet) is inside
        # the holder's box on purpose
        for rl in self.rels:
            hold, seen = rl.b, {rl.a}
            while rl.rel == "on" and hold in self.insts and hold not in seen and not self.shelf(hold):
                seen.add(hold)
                hold = next((q.b for q in self.rels if q.a == hold and q.rel == "on"), "")
            if rl.rel == "on" and self.shelf(hold):
                self.under_pairs.add(frozenset((rl.a, hold)))
        #: footprints nothing standing below LAYOUT_DOOR_HEIGHT_M may overlap: (label, pose, dims)
        self.blocked: list[tuple[str, _Pose, Dims]] = []
        #: ring members at their even places (a crowded room falls back to any place round B)
        self.strict_rings = True
        #: a stacked copy needs its copy below among the placed poses (check_layout: a final pass)
        self.verify_stacks = True
        self._static_checks()

    # ---- static (pose-independent) validation -------------------------------------------
    def _static_checks(self) -> None:
        indoor = self.kind == "indoor"
        has_on = {r.a for r in self.rels if r.rel == "on"}
        has_wall = {r.a for r in self.rels if r.rel == "against_wall"}
        for i in self.insts.values():
            sup = i.prior.support
            if sup == "surface" and i.id not in has_on:
                raise LayoutError(
                    f"'{i.id}' ({i.category}) must rest on a surface: add a relation {{a: '{i.base}', rel: 'on', b: <object that can support others>}}",
                    [i.id],
                )
            if sup == "wall" and i.id not in has_wall:
                raise LayoutError(f"'{i.id}' ({i.category}) is wall-mounted: add a relation {{a: '{i.base}', rel: 'against_wall', b: 'wall_north'|...}}", [i.id])
            if sup == "ceiling" and not indoor:
                raise LayoutError(f"'{i.id}' ({i.category}) is ceiling-mounted but the setting is '{self.kind}', not indoor", [i.id])
        for r in self.rels:
            if r.rel == "on":
                b = self.insts[r.b]
                if not b.prior.can_support_others and not self.shelf(r.b):
                    raise LayoutError(f"relation '{r.label}': category '{b.category}' cannot support other objects", [r.label])
                if self.insts[r.a].prior.support in ("wall", "ceiling"):
                    raise LayoutError(f"relation '{r.label}': '{r.a}' is {self.insts[r.a].prior.support}-mounted and cannot rest on an object", [r.label])
            if r.rel == "above":
                a = self.insts[r.a]
                if a.prior.support == "ceiling" and indoor:
                    continue  # hangs over B's centre (a pendant over the dining table)
                if a.prior.support != "wall" or a.id not in has_wall:
                    raise LayoutError(
                        f"relation '{r.label}': 'above' places a wall-mounted object over another; '{r.a}' "
                        f"({a.category}) is {a.prior.support}-supported" + ("" if a.id in has_wall else " and has no against_wall relation")
                        + " (e.g. a mirror against_wall + above a sink)", [r.label])
            if r.rel == "under" and self.insts[r.a].dims[2] >= self.insts[r.b].dims[2]:
                raise LayoutError(f"relation '{r.label}': '{r.a}' is as tall as or taller than '{r.b}'", [r.label])

    # ---- walls ---------------------------------------------------------------------------
    def _spread_at(self, inst: _Inst, wall: str, at_deg: float | None) -> float | None:
        """at_deg of copy idx of n against a curved/polygon ``wall``: the copies side by side
        along the wall (widths + _SPREAD_GAP_M apart), centred on at_deg. One at_deg put every
        copy on the same spot (three windows of an oval room could never be placed)."""
        if at_deg is None or inst.n <= 1 or self.shape.kind == "rect" or wall != "wall":
            return at_deg
        a, _n = self.shape.wall_ref(wall, at_deg - 0.5, 0.0, 0.0)
        b, _n = self.shape.wall_ref(wall, at_deg + 0.5, 0.0, 0.0)
        step = (inst.dims[0] + _SPREAD_GAP_M) / max(math.dist(a, b), 1e-6)
        return round(at_deg + (inst.idx - (inst.n - 1) / 2) * step, 3)

    def wall(self, name: str, at_deg: float | None = None, x: float = 0.0, y: float = 0.0) -> tuple[tuple[float, float], tuple[float, float], float]:
        """(inward normal, point on wall, yaw whose front faces inward). (x, y) = object center."""
        if self.shape.kind != "rect":
            p, n = self.shape.wall_ref(name, at_deg, x, y)
            return n, p, math.atan2(n[0], -n[1])
        return {
            "wall_north": ((0.0, -1.0), (0.0, self.D), 0.0),
            "wall_south": ((0.0, 1.0), (0.0, 0.0), math.pi),
            "wall_east": ((-1.0, 0.0), (self.W, 0.0), 3 * _HALF_PI),
            "wall_west": ((1.0, 0.0), (0.0, 0.0), _HALF_PI),
        }[name]

    def in_room(self, inst: _Inst, pose: _Pose, eps: float) -> bool:
        return all(self.shape.contains(cx, cy, eps) if self.shape.kind != "rect" else (-eps <= cx <= self.W + eps and -eps <= cy <= self.D + eps)
                   for cx, cy in _corners(pose, inst.dims))

    def snap_walls(self, inst: _Inst, pose: _Pose, set_yaw: bool) -> None:
        walls = [r for r in self.by_a.get(inst.id, []) if r.rel == "against_wall"]
        for k, r in enumerate(walls):
            n, p, wyaw = self.wall(r.b, r.at_deg, pose.x, pose.y)
            if k == 0 and set_yaw:
                pose.yaw = wyaw
            if r.at_deg is not None and self.shape.kind != "rect":
                pose.x, pose.y = p
            s = (pose.x - p[0]) * n[0] + (pose.y - p[1]) * n[1]
            delta = _ext(inst.dims, pose.yaw, n) + r.gap - s  # gap_m: stands that far off the wall
            pose.x += delta * n[0]
            pose.y += delta * n[1]
            if k == 0 and self.shape.curved:  # a flat back on a curve: slide in until the corners fit
                for _ in range(int(_CURVED_WALL_GAP_M / 0.005)):
                    if self.in_room(inst, pose, LAYOUT_EPS_M):
                        break
                    pose.x += 0.005 * n[0]
                    pose.y += 0.005 * n[1]

    def _shelf_slots(self, bid: str) -> None:
        """Fill ``shelf_slot`` for what stands on shelf holder ``bid``: level by level from the
        bottom, side by side along its width, each row centred; what does not fit gets no slot."""
        b = self.insts[bid]
        on = [q.a for q in self.rels if q.rel == "on" and q.b == bid]
        turned = {a: self.insts[a].dims[1] > b.dims[1] >= self.insts[a].dims[0] for a in on}
        width = {a: self.insts[a].dims[1 if turned[a] else 0] for a in on}
        shelves: list[list[str]] = [[]]
        used = 0.0
        for a in on:
            if shelves[-1] and used + _SHELF_SPACING_M + width[a] > b.dims[0] + 1e-9:
                if len(shelves) == len(SHELF_LEVELS[b.category]):
                    break  # the rack is full
                shelves.append([])
                used = 0.0
            used += (_SHELF_SPACING_M if shelves[-1] else 0.0) + width[a]
            shelves[-1].append(a)
        for lvl, row in zip(SHELF_LEVELS[b.category], shelves, strict=False):
            x = -(sum(width[a] for a in row) + _SHELF_SPACING_M * (len(row) - 1)) / 2
            for a in row:
                self.shelf_slot[a] = (lvl, x + width[a] / 2, turned[a])
                x += width[a] + _SHELF_SPACING_M

    def shelf(self, iid: str) -> bool:
        """Is ``iid`` storage furniture whose shelf levels carry what is ``on`` it?"""
        inst = self.insts.get(iid)
        return inst is not None and inst.category in SHELF_LEVELS

    def shelf_pose(self, iid: str, pb: _Pose) -> _Pose | None:
        """Where ``iid`` stands on its shelf holder at ``pb`` (None: not on a shelf, or no room left)."""
        slot = self.shelf_slot.get(iid)
        hold = next((r.b for r in self.by_a.get(iid, []) if r.rel == "on" and self.shelf(r.b)), None)
        if slot is None or hold is None:
            return None
        x, y = _local(pb, (slot[1], 0.0))
        return _Pose(x, y, pb.yaw + (_HALF_PI if slot[2] else 0.0), pb.zb + slot[0] * self.insts[hold].dims[2])

    def stack_below(self, inst: _Inst, pose: _Pose, poses: dict[str, _Pose]) -> str | None:
        """The placed copy (same category, same support) ``inst`` stands on top of, if any."""
        if inst.category not in STACKABLE:
            return None
        sup = next((r.b for r in self.by_a.get(inst.id, []) if r.rel == "on"), None)
        for oid, op in poses.items():
            o = self.insts[oid]
            if oid == inst.id or o.category != inst.category or abs(op.zb + o.dims[2] - pose.zb) > 1e-3:
                continue
            if sup is not None and not any(r.rel == "on" and r.b == sup for r in self.by_a.get(oid, [])):
                continue
            if _inside(pose, inst.dims, op, o.dims, LAYOUT_EPS_M):
                return oid
        return None

    def carries_stack(self, iid: str, poses: dict[str, _Pose]) -> bool:
        """Does a stacked copy stand on ``iid``? Then it keeps its pose (the copy would float)."""
        inst = self.insts[iid]
        return inst.category in STACKABLE and any(
            self.stack_below(self.insts[o], q, {iid: poses[iid]}) == iid for o, q in poses.items() if o != iid)

    def ring_slot(self, r: _Rel, pb: _Pose) -> _Pose:
        """Where ring member ``r.a`` of ``r.b`` sits: slot ``ring_i`` of ``ring_n`` evenly spaced
        round B, counted from B's right axis (two chairs at a table's ends, four at its sides),
        ``gap`` from B's edge, facing B's centre."""
        a, b = self.insts[r.a], self.insts[r.b]
        phi = pb.yaw + 2 * math.pi * r.ring_i / max(r.ring_n, 1)
        u = (math.cos(phi), math.sin(phi))
        yaw = _face_yaw(0, 0, -u[0], -u[1])
        dist = _ext(b.dims, pb.yaw, u) + r.gap + _ext(a.dims, yaw, u)
        return _Pose(pb.x + u[0] * dist, pb.y + u[1] * dist, yaw, 0.0)

    def ring_anchor(self, iid: str) -> bool:
        """Does a ring stand round this object (and nothing else place it)? It keeps to the open middle."""
        return any(r.rel == "around" for r in self.by_b.get(iid, [])) and not any(
            r.rel != "facing" for r in self.by_a.get(iid, []))

    def default_face(self, inst: _Inst, poses: dict[str, _Pose]) -> tuple[float, float] | None:
        """Point an object with no orienting relation faces: the camera focus, else the centre of
        the setting - indoors and outdoors alike (park benches facing away from the fountain read
        as a mistake)."""
        if any(r.rel in _ORIENT_RELS for r in self.by_a.get(inst.id, [])):
            return None
        if self.focus and self.focus != inst.id and self.focus in poses:
            return poses[self.focus].x, poses[self.focus].y
        if self.ring_anchor(inst.id):
            return None  # a table in the middle of its ring: square to the room, not turned to a point
        return (self.shape.cx, self.shape.cy) if self.kind in ("indoor", "outdoor") else None

    def base_z(self, inst: _Inst, poses: dict[str, _Pose]) -> float:
        sup = inst.prior.support
        if sup == "wall":
            for r in self.by_a.get(inst.id, []):
                if r.rel == "above" and r.b in poses:
                    return poses[r.b].zb + self.insts[r.b].dims[2] + r.gap
            return inst.prior.mount_height_m
        if sup == "ceiling":
            return self.H - inst.dims[2]
        for r in self.by_a.get(inst.id, []):
            if r.rel == "on" and r.b in poses:
                sp = self.shelf_pose(inst.id, poses[r.b])
                return sp.zb if sp is not None else poses[r.b].zb + self.insts[r.b].dims[2]
        return 0.0

    # ---- axis-aligned yaw (rect rooms) ----------------------------------------------------
    def yaw_free(self, iid: str) -> bool:
        """May the solver choose this object's yaw freely? Then, in a rect room, it is kept on the
        room's axes (or its target's): judges called racks, desks and sofas at odd angles mistakes.
        ``facing`` / ``around`` ask for a yaw of their own; walls fix it anyway."""
        if self.shape.kind != "rect" or self.kind != "indoor":
            return False
        return not any(r.rel in ("facing", "around", "against_wall") for r in self.by_a.get(iid, []))

    def square_facing(self, iid: str) -> bool:
        """A ``facing`` object of a rect room whose yaw may be squared when it still faces."""
        rels = self.by_a.get(iid, [])
        return (self.shape.kind == "rect" and self.kind == "indoor" and any(r.rel == "facing" for r in rels)
                and not any(r.rel in ("around", "against_wall") for r in rels))

    def axis_ref(self, iid: str, poses: dict[str, _Pose]) -> float:
        """The yaw an object's quarter turns count from: its target's (in_front_of, beside, on, ...)
        when it has a placed one, else the room's axes."""
        for r in self.by_a.get(iid, []):
            if r.rel in ("in_front_of", "beside", "on", "under", "centered_in", "above") and r.b in poses:
                return poses[r.b].yaw
        return 0.0

    def snap_yaw(self, iid: str, yaw: float, poses: dict[str, _Pose]) -> float:
        if not self.yaw_free(iid):
            return yaw
        ref = self.axis_ref(iid, poses)
        return ref + round((yaw - ref) / _HALF_PI) * _HALF_PI

    def table_of(self, iid: str, pose: _Pose) -> tuple[_Pose, Dims]:
        """The table itself inside a table + chairs box (what ``on``/``under``/``above`` mean)."""
        inst = self.insts.get(iid)
        plan = self.seated.get(inst.base) if inst is not None and inst.base not in self.grids else None
        if plan is None:
            return pose, inst.dims if inst is not None else (0.0, 0.0, 0.0)
        x, y = _local(pose, plan.table_off)
        return _Pose(x, y, pose.yaw, pose.zb), plan.table

    # ---- predicates ----------------------------------------------------------------------
    def rel_violation(self, r: _Rel, pa: _Pose, pb: _Pose | None) -> str | None:
        """Reason a hard relation is violated, else None. facing is soft: handled elsewhere."""
        a = self.insts[r.a]
        tol = LAYOUT_REL_TOL_M
        if r.rel == "against_wall":
            n, p, _ = self.wall(r.b, r.at_deg, pa.x, pa.y)
            gap = (pa.x - p[0]) * n[0] + (pa.y - p[1]) * n[1] - _ext(a.dims, pa.yaw, n) - r.gap
            f = _axes(pa.yaw)[1]
            lo, hi = (-1e-3, _CURVED_WALL_GAP_M) if self.shape.curved else (-1e-3, 1e-3)
            # the first wall turns the object to face the room; a second one (a corner) only
            # has it flush with its side or back
            first = next(x for x in self.by_a[r.a] if x.rel == "against_wall") is r
            if not lo <= gap <= hi or (first and f[0] * n[0] + f[1] * n[1] < 0.99):
                return f"'{r.label}': not flush against the wall (gap {gap:.3f} m)"
            if r.at_deg is not None and self.shape.kind != "rect":
                off = gap + r.gap + _ext(a.dims, pa.yaw, n)
                lat = math.hypot(pa.x - p[0] - off * n[0], pa.y - p[1] - off * n[1])
                if lat > _LATERAL_TOL_M:
                    return f"'{r.label}': {lat:.2f} m away from the requested at_deg position"
            return None
        if r.rel == "centered_in":
            if pb is None:
                tx, ty = self.shape.cx, self.shape.cy
            else:
                if self.insts[r.b].flat:
                    # "centered in the rug" means standing on it: inside its footprint. Read
                    # literally it fights the usual in_front_of the sofa (benchmark: the small
                    # model wrote both in most living rooms, and no layout existed).
                    if not _inside(pa, a.dims, pb, self.insts[r.b].dims, LAYOUT_EPS_M):
                        return f"'{r.label}': not within the footprint of '{r.b}'"
                    return None
                tx, ty = pb.x, pb.y
            if math.hypot(pa.x - tx, pa.y - ty) > tol:
                return f"'{r.label}': center is {math.hypot(pa.x - tx, pa.y - ty):.3f} m off"
            return None
        if pb is None:
            return None  # target not placed yet
        b = self.insts[r.b]
        if r.rel in ("on", "under", "above") and b.base in self.seated:
            pb, bdims = self.table_of(r.b, pb)  # on the table, not on its chairs
        else:
            bdims = b.dims
        if r.rel == "on":
            if self.shelf(r.b):
                sp = self.shelf_pose(r.a, pb)
                if sp is None:
                    return f"'{r.label}': no room left on the shelves of '{r.b}'"
                if abs(pa.zb - sp.zb) > 1e-3 or math.hypot(pa.x - sp.x, pa.y - sp.y) > tol \
                        or min(_angdiff(pa.yaw, sp.yaw), _angdiff(pa.yaw, sp.yaw + math.pi)) > 1e-3:
                    return f"'{r.label}': not at its place on the shelves of '{r.b}'"
                return None
            top = pb.zb + b.dims[2]
            layer = (pa.zb - top) / a.dims[2] if a.category in STACKABLE else 0.0
            if abs(layer - round(layer)) * a.dims[2] > 1e-3 or not 0 <= round(layer) <= 3 \
                    or not _inside(pa, a.dims, pb, bdims, LAYOUT_EPS_M):
                return f"'{r.label}': does not fit on top of '{r.b}'"
            return None
        if r.rel == "under":
            if not _inside(pa, a.dims, pb, bdims, LAYOUT_EPS_M):
                return f"'{r.label}': not within the footprint of '{r.b}'"
            return None
        if r.rel == "above" and a.prior.support == "ceiling":
            off = math.hypot(pa.x - pb.x, pa.y - pb.y)
            return f"'{r.label}': {off:.3f} m off the centre of '{r.b}'" if off > tol else None
        if r.rel == "above":
            ra, fa = _axes(pa.yaw)
            d = (pb.x - pa.x, pb.y - pa.y)
            lat = abs(d[0] * ra[0] + d[1] * ra[1])  # offset along the wall
            depth = d[0] * fa[0] + d[1] * fa[1]  # B stands in front of the wall A hangs on
            rise = pa.zb - (pb.zb + b.dims[2])
            if lat > tol or depth < -tol or abs(rise - r.gap) > tol:
                return (f"'{r.label}': offset {lat:.3f} m along the wall, {rise:.3f} m above the top of '{r.b}'"
                        + (" (behind the wall)" if depth < -tol else ""))
            return None
        if r.rel == "in_front_of":
            fb, rb = _axes(pb.yaw)[1], _axes(pb.yaw)[0]
            d = (pa.x - pb.x, pa.y - pb.y)
            along = d[0] * fb[0] + d[1] * fb[1] - _ext(b.dims, pb.yaw, fb) - _ext(a.dims, pa.yaw, fb)
            lat = abs(d[0] * rb[0] + d[1] * rb[1])
            if abs(along - r.gap) > tol or lat > _ext(b.dims, pb.yaw, rb) + tol:
                return f"'{r.label}': measured gap {along:.3f} m, lateral offset {lat:.3f} m"
            if not _front_hits(pa, pb, b.dims):
                # validate_scene reads in_front_of as "A faces B" (its front ray reaches B);
                # local search used to turn A away and leave a blocking facing issue.
                return f"'{r.label}': does not face '{r.b}'"
            return None
        if r.rel == "beside":
            s = _sep(pa, a.dims, pb, b.dims)
            if abs(s - r.gap) > tol:
                return f"'{r.label}': measured gap {s:.3f} m"
            return None
        if r.rel == "around":
            dx, dy = pa.x - pb.x, pa.y - pb.y
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                return f"'{r.label}': coincident centers"
            u = (dx / dist, dy / dist)
            gap = dist - _ext(b.dims, pb.yaw, u) - _ext(a.dims, pa.yaw, u)
            if abs(gap - r.gap) > tol:
                return f"'{r.label}': measured ring gap {gap:.3f} m"
            if r.ring_n > 1 and self.strict_rings:  # evenly spaced round B, each facing its centre
                slot = self.ring_slot(r, pb)
                if math.hypot(pa.x - slot.x, pa.y - slot.y) > tol or _angdiff(pa.yaw, slot.yaw) > math.radians(_RING_YAW_TOL_DEG):
                    return f"'{r.label}': not at its place {r.ring_i + 1} of {r.ring_n} round '{r.b}'"
            return None
        return None

    def facing_error_deg(self, r: _Rel, pa: _Pose, pb: _Pose) -> float:
        return math.degrees(_angdiff(pa.yaw, _face_yaw(pa.x, pa.y, pb.x, pb.y)))

    def margin(self, a: _Inst, b: _Inst) -> float:
        m = max(a.prior.default_clearance_m, b.prior.default_clearance_m)
        g = self.pair_gap.get(frozenset((a.id, b.id)))
        if g is None:  # no relation between them: kinds that must not crowd each other keep apart
            m = max(m, PAIR_CLEARANCE_M.get(frozenset((a.category, b.category)), 0.0))
        return m if g is None else min(m, g)

    def pair_conflict(self, a: _Inst, pa: _Pose, b: _Inst, pb: _Pose) -> bool:
        if a.flat != b.flat:
            return False
        if frozenset((a.id, b.id)) in self.under_pairs:
            return False
        if pa.zb >= pb.zb + b.dims[2] - LAYOUT_EPS_M or pb.zb >= pa.zb + a.dims[2] - LAYOUT_EPS_M:
            return False
        return _sep(pa, a.dims, pb, b.dims) < self.margin(a, b) - LAYOUT_EPS_M

    def link_end(self, inst: _Inst, poses: dict[str, _Pose]) -> tuple[float, float, float, float] | None:
        """(x, y, yaw, length) of the placed boardwalk that starts at ``inst`` (the first copy of
        its object), else None."""
        a = None if self.terrain is None or inst.idx else self.terrain.start_of.get(inst.base)
        aid = self.by_base.get(a, [None])[0] if a is not None else None
        pa = poses.get(aid) if aid is not None else None
        if pa is None or aid is None:
            return None
        return pa.x, pa.y, pa.yaw, self.insts[aid].dims[1]

    def hard_violation(self, inst: _Inst, pose: _Pose, poses: dict[str, _Pose]) -> str | None:
        """First hard-constraint violation of `inst` at `pose` against already-placed poses."""
        e = LAYOUT_EPS_M
        if not self.in_room(inst, pose, e):
            return "out of bounds"
        if self.terrain is not None and pose.zb < e:  # the shore: terrain.py
            tv = self.terrain.violation(inst.base, inst.category, pose.x, pose.y, pose.yaw, inst.dims)
            if tv:
                return tv
        if self.kind == "indoor" and (pose.zb < -e or pose.zb + inst.dims[2] > self.H + e):
            return "out of bounds (height)"
        for r in self.by_a.get(inst.id, []):
            if r.rel == "facing":
                continue
            pb = poses.get(r.b) if r.b in self.insts else None
            if r.b in self.insts and pb is None:
                continue
            v = self.rel_violation(r, pose, pb)
            if v:
                return f"relation {v}"
        for r in self.by_b.get(inst.id, []):
            if r.rel == "facing" or r.a not in poses:
                continue
            v = self.rel_violation(r, poses[r.a], pose)
            if v:
                return f"relation {v}"
        if self.verify_stacks and inst.category in STACKABLE and any(
                r.rel == "on" and r.b in poses and pose.zb > poses[r.b].zb + self.insts[r.b].dims[2] + 1e-3
                for r in self.by_a.get(inst.id, []) if not self.shelf(r.b)) and self.stack_below(inst, pose, poses) is None:
            return "stacked on nothing"
        for r in self.by_b.get(inst.id, []):
            # the ring's places must be in the room before the table is put down
            if r.rel == "around" and r.ring_n > 1 and self.strict_rings and r.a not in poses \
                    and not self.in_room(self.insts[r.a], self.ring_slot(r, pose), e):
                return f"relation '{r.label}': no room for its place round '{r.b}'"
        if not inst.flat and pose.zb < LAYOUT_DOOR_HEIGHT_M:
            for label, bp, bd in self.blocked:
                if _sep(pose, inst.dims, bp, bd) < -LAYOUT_EPS_M:
                    return f"blocks {label}"
        for oid, op in poses.items():
            if oid != inst.id and self.pair_conflict(inst, pose, self.insts[oid], op):
                return f"overlaps '{oid}'"
        return None

    def soft_cost(self, inst: _Inst, pose: _Pose, poses: dict[str, _Pose]) -> float:
        cost = 0.0 if self.terrain is None else self.terrain.cost(inst.base, pose.x, pose.y, pose.yaw, inst.dims)
        link = self.link_end(inst, poses)
        if link is not None:
            cost += self.terrain.link_cost(link, pose.x, pose.y, inst.dims)  # type: ignore[union-attr]
        cells, at = ceiling_grid(self, inst)
        if cells:  # a free ceiling light keeps to its grid cell
            cost += math.hypot(pose.x - cells[at][0], pose.y - cells[at][1]) / max(self.W, self.D, 1.0)
        for r in self.by_a.get(inst.id, []):
            pb = poses.get(r.b)
            if pb is None:
                continue
            if r.rel == "facing":
                err = self.facing_error_deg(r, pose, pb)
                cost += err / 180.0 + (LAYOUT_FACING_MISS_COST if err > LAYOUT_FACING_TOL_DEG else 0.0)
            elif r.rel in ("in_front_of", "beside"):
                rb, fb = _axes(pb.yaw)
                d = (pose.x - pb.x, pose.y - pb.y)
                along, across = d[0] * rb[0] + d[1] * rb[1], d[0] * fb[0] + d[1] * fb[1]
                b = self.insts[r.b]
                if r.rel == "in_front_of":
                    # centred on B's axis: a coffee table off the sofa's axis read as a mistake
                    cost += 0.5 * abs(along) / max(b.dims[0] / 2, 0.1)
                    if not any(q.rel == "facing" for q in self.by_a[inst.id]):
                        cost += 0.3 * _angdiff(pose.yaw, pb.yaw + math.pi) / math.pi  # A faces B
                else:
                    # at B's side, level with it (armchairs at the coffee table's short ends,
                    # not in front of it); front or back of B only when the sides are taken
                    side = abs(along) - b.dims[0] / 2 > -LAYOUT_EPS_M
                    cost += (0.0 if side else 0.5) + 0.3 * min(abs(across) / max(b.dims[1] / 2, 0.1), 1.0)
                    if not any(q.rel == "facing" for q in self.by_a[inst.id]):
                        cost += 0.3 * _angdiff(pose.yaw, pb.yaw) / math.pi
            elif r.rel == "on":
                # faces the way its support does (a TV on its stand turned to the wall was "fixed"
                # by the validator rotating it in Blender, qwen aidb6 living room)
                if not any(q.rel == "facing" for q in self.by_a[inst.id]):
                    cost += 0.3 * _angdiff(pose.yaw, pb.yaw) / math.pi
                if pose.zb > pb.zb + self.insts[r.b].dims[2] + 1e-3 and not self.shelf(r.b):
                    cost += 0.3  # a stacked layer: only when the top below is full
                if sum(1 for q in self.by_b.get(r.b, []) if q.rel == "on") == 1:
                    # the only thing on its support stands in the middle of it
                    tp, td = self.table_of(r.b, pb)
                    cost += 0.2 * math.hypot(pose.x - tp.x, pose.y - tp.y) / max(td[0] / 2, 0.1)
            elif r.rel == "centered_in" and self.insts[r.a].flat:
                # a rug under a table lies parallel to it
                cost += 0.3 * min(_angdiff(pose.yaw, pb.yaw), _angdiff(pose.yaw, pb.yaw + math.pi)) / _HALF_PI
        dt = self.default_face(inst, poses)
        if dt is not None:
            cost += 0.3 * _angdiff(pose.yaw, _face_yaw(pose.x, pose.y, dt[0], dt[1])) / math.pi
        if self.ring_anchor(inst.id) and self.kind == "indoor":
            # a table with its ring of chairs: in the open middle, square to the room
            cost += 0.5 * math.hypot(pose.x - self.shape.cx, pose.y - self.shape.cy) / max(self.W, self.D)
            cost += 0.2 * abs((pose.yaw + _HALF_PI / 2) % _HALF_PI - _HALF_PI / 2) / (_HALF_PI / 2)
        if self.shape.kind == "rect" and any(r.rel == "facing" for r in self.by_b.get(inst.id, [])):
            # what others face (the board, the TV) sits mid-wall, so they can face it square
            wall = next((r for r in self.by_a.get(inst.id, []) if r.rel == "against_wall"), None)
            if wall is not None:
                n = self.wall(wall.b)[0]
                lateral = abs((pose.x - self.W / 2) * n[1]) + abs((pose.y - self.D / 2) * n[0])
                cost += 0.5 * lateral / max(self.W, self.D)
        if self.square_facing(inst.id):
            # prefer spots from which the facing yaw is square to the room (snapped at the end)
            cost += 0.2 * abs((pose.yaw + _HALF_PI / 2) % _HALF_PI - _HALF_PI / 2) / (_HALF_PI / 2)
        for oid, op in poses.items():
            if oid == inst.id:
                continue
            o = self.insts[oid]
            if o.flat or inst.flat or frozenset((inst.id, oid)) in self.pair_gap or frozenset((inst.id, oid)) in self.grouped:
                continue
            s = _sep(pose, inst.dims, op, o.dims)
            if s < LAYOUT_BREATHING_M:
                cost += 0.1 * (LAYOUT_BREATHING_M - max(s, 0.0)) / LAYOUT_BREATHING_M
        return cost

    # ---- ordering ------------------------------------------------------------------------
    @staticmethod
    def _cyclic(deps: dict[str, set[str]]) -> bool:
        done: set[str] = set()
        left = set(deps)
        while left:
            ready = {i for i in left if deps[i] <= done}
            if not ready:
                return True
            done |= ready
            left -= ready
        return False

    def order(self) -> list[str]:
        def deps_of(rels: tuple[str, ...]) -> dict[str, set[str]]:
            return {i: {r.b for r in self.by_a.get(i, []) if r.rel in rels and r.b in self.insts} for i in self.insts}

        # `facing` is soft: its target is placed first when possible (so the yaw is exact),
        # and the facing edges are dropped only if they would make the order circular.
        deps = deps_of((*_DEP_RELS, "facing"))
        if self._cyclic(deps):
            deps = deps_of(_DEP_RELS)
        done: list[str] = []
        left = set(self.insts)
        def anchored(i: str) -> bool:
            return any(r.rel in ("against_wall", "centered_in") for r in self.by_a.get(i, []))

        def shore_rank(i: str) -> int:  # outdoors: the shore groups before decoration (terrain.rank)
            return 1 if self.terrain is None else self.terrain.rank(self.insts[i].base)

        # wall/center-anchored objects first (they have the fewest legal poses), then hero, then big
        key = lambda i: (not anchored(i), shore_rank(i), not self.insts[i].hero,
                         -self.insts[i].dims[0] * self.insts[i].dims[1], i)
        while left:
            ready = sorted((i for i in left if deps[i] <= set(done)), key=key)
            if not ready:
                cyc = sorted(left)
                labels = [r.label for r in self.rels if r.a in left and r.b in left and r.rel in _DEP_RELS]
                raise LayoutError(f"circular relations between {cyc}: {labels}", labels)
            # hero (or largest) first, but only one at a time so dependents see updated order
            done.append(ready[0])
            left.discard(ready[0])
        return done


def ceiling_grid(pr: _Problem, inst: _Inst) -> tuple[list[tuple[float, float]], int]:
    """(cell centres, this instance's cell) of an even grid over the room for the ceiling objects
    with no relation: four ceiling lights all ``centered_in room`` could never share the centre,
    and the repair loop grew a 50 x 30 m warehouse to 72 x 43 m before giving up (lightning
    aidb5). Indoor ceiling-mounted objects only; ([], 0) otherwise."""
    if inst.prior.support != "ceiling" or pr.kind != "indoor":
        return [], 0
    free = sorted(i.id for i in pr.insts.values() if i.prior.support == "ceiling" and not pr.by_a.get(i.id))
    if inst.id not in free:
        return [], 0
    key = tuple(free)
    if pr.ceiling_cache[0] != key:
        n = len(free)
        aspect = pr.W / max(pr.D, 1e-6)
        # fewest empty cells first, then the cell shape closest to the room's
        cols = min(range(1, n + 1), key=lambda c: (c * math.ceil(n / c) - n,
                                                   abs(math.log((c / math.ceil(n / c)) / aspect)), c))
        rows = math.ceil(n / cols)
        cells = [((c + 0.5) * pr.W / cols, (r + 0.5) * pr.D / rows) for r in range(rows) for c in range(cols)]
        pr.ceiling_cache = (key, [xy for xy in cells if pr.shape.contains(xy[0], xy[1], 0.0)] or cells)
    return pr.ceiling_cache[1], free.index(inst.id)


def _candidates(pr: _Problem, inst: _Inst, poses: dict[str, _Pose], rng: random.Random, n: int) -> list[_Pose]:
    rels = pr.by_a.get(inst.id, [])
    walls = [r for r in rels if r.rel == "against_wall"]
    placed = [r for k in _PRIMARY_ORDER for r in rels if r.rel == k and (r.b in poses or r.b == ROOM)]
    # centered_in a flat target (rug) only asks for containment: let a positional relation
    # (in_front_of, beside, ...) place the object when there is one.
    prim = next((r for r in placed if not (r.rel == "centered_in" and r.b in pr.insts and pr.insts[r.b].flat)),
                placed[0] if placed else None)
    pb = poses.get(prim.b) if prim is not None else None
    b = pr.insts.get(prim.b) if prim is not None else None
    bdims = b.dims if b is not None else (0.0, 0.0, 0.0)
    if prim is not None and pb is not None and b is not None and prim.rel in ("on", "under", "above"):
        pb, bdims = pr.table_of(prim.b, pb)  # a seated table: its own top, not the chairs' box
    cards = [k * _HALF_PI for k in range(4)]
    yaws = list(cards)
    for r in rels:
        t = poses.get(r.b)
        if t is None:
            continue
        if r.rel in ("on", "under", "in_front_of", "beside", "centered_in"):
            yaws = [(t.yaw + k * _HALF_PI) % (2 * math.pi) for k in range(4)] + yaws
    facing_targets = [poses[r.b] for r in rels if r.rel == "facing" and r.b in poses]
    out: list[_Pose] = []
    dface = pr.default_face(inst, poses)
    stack = [oid for oid in poses if inst.category in STACKABLE and pr.insts[oid].category == inst.category
             and prim is not None and prim.rel == "on" and not pr.shelf(prim.b)
             and any(r.rel == "on" and r.b == prim.b for r in pr.by_a.get(oid, []))]
    ceiling_cells, ceiling_i = ceiling_grid(pr, inst) if prim is None and not rels else ([], 0)
    for k in range(n):
        yaw = rng.choice(yaws[: 4 if k % 2 == 0 else len(yaws)]) if yaws else 0.0
        if prim is not None and pb is not None and prim.rel in ("in_front_of", "beside") and k % 5 != 4:
            yaw = pb.yaw + (math.pi if prim.rel == "in_front_of" else 0.0)  # A faces B / matches its neighbour
        if walls and facing_targets and k % 2 == 0:
            # against a wall, facing something: the wall fixes the yaw, so the position along
            # the wall must line up with the target (snap_walls below moves it onto the wall)
            ft = facing_targets[(k // 2) % len(facing_targets)]
            x = ft.x + rng.uniform(-0.3, 0.3) * (k > 0)
            y = ft.y + rng.uniform(-0.3, 0.3) * (k > 0)
        elif prim is None and facing_targets and k % 3 == 1 and pr.shape.kind == "rect":
            # straight out from what it faces (desks in front of the board, not off to one side
            # at an angle): the facing yaw is then square to the room
            ft = facing_targets[(k // 3) % len(facing_targets)]
            f = _axes(ft.yaw)[1]
            # only distances that keep A inside the room (a 7 x 5 m block of desks drew 2000
            # out-of-bounds candidates in a 10 x 8 m classroom and the repair grew the room)
            room = [t for t in ((pr.W - ft.x) / f[0] if f[0] > 1e-6 else (-ft.x / f[0] if f[0] < -1e-6 else math.inf),
                                (pr.D - ft.y) / f[1] if f[1] > 1e-6 else (-ft.y / f[1] if f[1] < -1e-6 else math.inf))]
            far = min(room) - _ext(inst.dims, ft.yaw, f) if pr.shape.kind == "rect" else max(pr.W, pr.D)
            dist = rng.uniform(0.5, far) if far > 0.5 else rng.uniform(0.5, max(pr.W, pr.D))
            x, y = ft.x + f[0] * dist, ft.y + f[1] * dist
        elif prim is None and ceiling_cells and k < 2 * len(ceiling_cells):
            # a free ceiling light: its own cell of an even grid first, then the other cells
            x, y = ceiling_cells[(ceiling_i + k) % len(ceiling_cells)]
            if k >= len(ceiling_cells):
                x, y = x + rng.uniform(-0.25, 0.25), y + rng.uniform(-0.25, 0.25)
        elif prim is None and k < 2 and pr.ring_anchor(inst.id):
            x, y = pr.shape.cx, pr.shape.cy  # a table with its ring: the middle of the room first
            yaw = cards[k]
        elif prim is None or pb is None and prim.rel != "centered_in":
            x = rng.uniform(0, pr.W)
            y = rng.uniform(0, pr.D)
        elif prim.rel == "centered_in":
            x, y = (pr.shape.cx, pr.shape.cy) if pb is None else (pb.x, pb.y)
        else:
            assert b is not None and pb is not None
            rb, fb = _axes(pb.yaw)
            jit = 0.0 if k == 0 else rng.uniform(-1, 1)
            sp = pr.shelf_pose(inst.id, pb) if prim.rel == "on" else None
            if sp is not None:
                x, y, yaw = sp.x, sp.y, sp.yaw  # its place on the rack's levels
            elif prim.rel == "above":
                x, y = pb.x, pb.y  # snap_walls then moves A onto its wall: same position along it
            elif prim.rel == "on" and prim.b in pr.grids:
                # on a grid block (crates on a staging grid of pallets): one cell each, not the gaps
                cells = _grid_cells(pr, prim.b, pb)
                _, x, y = cells[inst.idx % len(cells)]
                if k:
                    cw, cd = pr.grids[prim.b][4][:2]
                    x += rng.uniform(-1, 1) * max(cw / 2 - inst.dims[0] / 2, 0) * rb[0]
                    y += rng.uniform(-1, 1) * max(cd / 2 - inst.dims[1] / 2, 0) * fb[1]
                    x += rng.uniform(-1, 1) * max(cd / 2 - inst.dims[1] / 2, 0) * fb[0]
            elif prim.rel in ("on", "under"):
                x = pb.x + jit * max(bdims[0] / 2 - inst.dims[0] / 2, 0) * rb[0] + rng.uniform(-1, 1) * (0 if k == 0 else max(bdims[1] / 2 - inst.dims[1] / 2, 0)) * fb[0]
                y = pb.y + jit * max(bdims[0] / 2 - inst.dims[0] / 2, 0) * rb[1] + rng.uniform(-1, 1) * (0 if k == 0 else max(bdims[1] / 2 - inst.dims[1] / 2, 0)) * fb[1]
            elif prim.rel == "around" and prim.ring_n > 1 and (pr.strict_rings or k % 3 == 0):
                slot = pr.ring_slot(prim, pb)  # its place in the ring: nothing to choose
                x, y, yaw = slot.x, slot.y, slot.yaw
            elif prim.rel == "around":
                phi = 2 * math.pi * prim.ring_i / max(prim.ring_n, 1) + (0.0 if k % 3 == 0 else rng.uniform(0, 2 * math.pi))
                u = (math.cos(phi), math.sin(phi))
                yaw = _face_yaw(0, 0, -u[0], -u[1])
                dist = _ext(b.dims, pb.yaw, u) + prim.gap + _ext(inst.dims, yaw, u)
                x, y = pb.x + u[0] * dist, pb.y + u[1] * dist
            else:  # in_front_of / beside
                dirs = [fb] if prim.rel == "in_front_of" else [rb, (-rb[0], -rb[1]), fb, (-fb[0], -fb[1])]
                u = dirs[k % len(dirs)]
                tg = (-u[1], u[0])
                dist = _ext(b.dims, pb.yaw, u) + prim.gap + _ext(inst.dims, yaw, u)
                span = _ext(b.dims, pb.yaw, tg)
                off = 0.0 if k < len(dirs) else rng.uniform(-span, span)
                x, y = pb.x + u[0] * dist + tg[0] * off, pb.y + u[1] * dist + tg[1] * off
        if facing_targets and not walls and (prim is None or prim.rel in ("on", "under", "centered_in", "beside")) and k % 4 != 3:
            ft = facing_targets[k % len(facing_targets)]
            yaw = _face_yaw(x, y, ft.x, ft.y)
        elif dface is not None and not walls and k % 4 != 3:
            yaw = _face_yaw(x, y, dface[0], dface[1])
        p = _Pose(x, y, pr.snap_yaw(inst.id, yaw, poses), 0.0)
        pr.snap_walls(inst, p, set_yaw=True)
        p.zb = pr.base_z(inst, poses)
        if stack and k % 3 == 2:  # on top of a copy already standing on the same support
            below = poses[stack[(k // 3) % len(stack)]]
            p = _Pose(below.x, below.y, below.yaw, below.zb + inst.dims[2])
        out.append(p)
    if pr.terrain is not None and prim is None:  # shore bands and jetties: terrain.py proposes
        out += [_Pose(x, y, yaw, 0.0) for x, y, yaw in pr.terrain.propose(inst.base, inst.dims, rng, n // 2)]
        link = pr.link_end(inst, poses)
        if link is not None:  # where the boardwalk starts: behind its land end
            out += [_Pose(x, y, yaw, 0.0) for x, y, yaw in pr.terrain.link_spots(link, inst.dims, rng, n // 3)]
    return out


def _fail(pr: _Problem, inst: _Inst, tally: Counter[str], tried: int) -> LayoutError:
    rels = [r.label for r in pr.by_a.get(inst.id, [])]
    top = tally.most_common(3)
    conflicts: list[str] = []
    for reason, _ in top:
        for r in pr.rels:
            if f"'{r.label}'" in reason and r.label not in conflicts:
                conflicts.append(r.label)
        if reason.startswith("overlaps '"):
            conflicts.append(reason.split("'")[1])
    if not conflicts:
        conflicts = rels or [inst.id]
    w, d, h = inst.dims
    why = "; ".join(f"{c}x {r}" for r, c in top) or "no candidate pose could be generated"
    return LayoutError(
        f"cannot place '{inst.id}' ({inst.category}, {w:g}x{d:g}x{h:g} m) in {pr.W:g}x{pr.D:g}x{pr.H:g} m {pr.kind} setting "
        f"after {tried} candidates; main blockers: {why}. relations of '{inst.id}': {rels or 'none'}",
        conflicts,
    )


def _place_greedy(pr: _Problem, rng: random.Random) -> dict[str, _Pose]:
    poses: dict[str, _Pose] = {}
    for iid in pr.order():
        inst = pr.insts[iid]
        best: tuple[float, int, _Pose] | None = None
        tally: Counter[str] = Counter()
        tried = 0
        for n in (LAYOUT_CANDIDATES, LAYOUT_RETRY_CANDIDATES):
            for k, cand in enumerate(_candidates(pr, inst, poses, rng, n)):
                tried += 1
                v = pr.hard_violation(inst, cand, poses)
                if v:
                    tally[v] += 1
                    continue
                c = pr.soft_cost(inst, cand, poses)
                if best is None or c < best[0] - 1e-12:
                    best = (c, k, cand)
            if best is not None:
                break
        if best is None:
            raise _fail(pr, inst, tally, tried)
        poses[iid] = best[2]
    return poses


def _total_cost(pr: _Problem, poses: dict[str, _Pose]) -> float:
    return sum(pr.soft_cost(pr.insts[i], p, poses) for i, p in poses.items())


def _local_search(pr: _Problem, poses: dict[str, _Pose], rng: random.Random) -> None:
    ids = sorted(poses)
    yaw_locked = {i for i in ids if any(r.rel == "against_wall" for r in pr.by_a.get(i, []))}

    def try_move(iid: str, new: _Pose) -> bool:
        inst = pr.insts[iid]
        old = poses[iid]
        before = _total_cost(pr, poses)
        poses[iid] = new
        ok = pr.hard_violation(inst, new, poses) is None
        # dependents' `on` height follows the base of their support
        if ok and _total_cost(pr, poses) < before - 1e-12:
            return True
        poses[iid] = old
        return False

    for iid in ids:  # deterministic facing pass
        inst = pr.insts[iid]
        if iid in yaw_locked:
            continue
        for r in pr.by_a.get(iid, []):
            if r.rel == "facing" and r.b in poses:
                p = poses[iid]
                try_move(iid, _Pose(p.x, p.y, _face_yaw(p.x, p.y, poses[r.b].x, poses[r.b].y), p.zb))
        dt = pr.default_face(inst, poses)
        if dt is not None:
            p = poses[iid]
            try_move(iid, _Pose(p.x, p.y, pr.snap_yaw(iid, _face_yaw(p.x, p.y, dt[0], dt[1]), poses), p.zb))
    step, dyaw = LAYOUT_LOCAL_STEP_M, math.radians(LAYOUT_LOCAL_YAW_DEG)
    for _ in range(LAYOUT_LOCAL_ITERS):
        iid = rng.choice(ids)
        inst = pr.insts[iid]
        p = poses[iid]
        if pr.by_b.get(iid) and not any(r.rel == "facing" for r in pr.by_b[iid]):
            continue  # support/anchor objects keep their position: dependents were placed relative to them
        if pr.carries_stack(iid, poses) or iid in pr.shelf_slot:  # a racked pallet keeps its bay
            continue
        new = _Pose(p.x + rng.uniform(-step, step), p.y + rng.uniform(-step, step), p.yaw, p.zb)
        if iid not in yaw_locked:
            new.yaw = p.yaw + rng.uniform(-dyaw, dyaw)
            for r in pr.by_a.get(iid, []):
                if r.rel == "facing" and r.b in poses and rng.random() < 0.5:
                    new.yaw = _face_yaw(new.x, new.y, poses[r.b].x, poses[r.b].y)
            dt = pr.default_face(inst, poses)
            if dt is not None and rng.random() < 0.5:
                new.yaw = _face_yaw(new.x, new.y, dt[0], dt[1])
            new.yaw = pr.snap_yaw(iid, new.yaw, poses)  # no ±15 deg skew on furniture in rect rooms
        pr.snap_walls(inst, new, set_yaw=False)
        new.zb = pr.base_z(inst, poses)
        try_move(iid, new)


def _facing_target(pr: _Problem, iid: str) -> str | None:
    rels = pr.by_a.get(iid, [])
    for rel in ("facing", "in_front_of"):
        for r in rels:
            if r.rel == rel and r.b in pr.insts:
                return r.b
    return None


def _grid_cells(pr: _Problem, iid: str, p: _Pose) -> list[tuple[str, float, float]]:
    """(id, x, y) of every cell of grid block ``iid`` at pose ``p``; row 1 is the front row."""
    rows, cols, gx, gy, cell = pr.grids[iid]
    block = pr.insts[iid].dims
    r_ax, f_ax = _axes(p.yaw)
    out = []
    for row in range(rows):
        for col in range(cols):
            lx = -block[0] / 2 + cell[0] / 2 + col * (cell[0] + gx)
            ly = block[1] / 2 - cell[1] / 2 - row * (cell[1] + gy)  # +front first
            out.append((f"{iid}_{row * cols + col + 1}",
                        p.x + lx * r_ax[0] + ly * f_ax[0], p.y + lx * r_ax[1] + ly * f_ax[1]))
    return out


def _seat_placements(pr: _Problem, base: str, boxes: list[_Pose]) -> list[Placement]:
    """The chairs of seated object ``base``, given its table + chairs boxes in instance order."""
    plan = pr.seated[base]
    out: list[Placement] = []
    k = 0
    ox, oy = plan.table_off
    for i, p in enumerate(boxes):
        owner = base if len(boxes) == 1 else f"{base}_{i + 1}"
        for off in plan.seats:
            k += 1
            x, y = _local(p, off)
            cr, cf = off[0] - ox, off[1] - oy  # the chair relative to the table centre
            # square to its side of the table (not turned to the centre: four chairs read as a skew)
            tx, ty = _local(p, (ox + cr, oy) if abs(cf) > plan.table[1] / 2 else (ox, oy + cf))
            out.append(Placement(f"{base}_seat_{k}", (round(x, 6), round(y, 6), round(p.zb + plan.chair[2] / 2, 6)),
                                 round(math.degrees(_face_yaw(x, y, tx, ty)) % 360.0, 6), plan.chair, plan.category,
                                 owner))
    return out


def _to_placements(pr: _Problem, poses: dict[str, _Pose]) -> list[Placement]:
    out = []
    boxes: dict[str, list[_Pose]] = {}
    for iid, inst in pr.insts.items():
        p = poses[iid]
        plan = pr.seated.get(inst.base)
        if iid in pr.grids:
            cell = pr.grids[iid][4]
            for cid, x, y in _grid_cells(pr, iid, p):
                dims = cell
                if plan is not None:  # the cell is a table + chairs box: the table sits inside it
                    boxes.setdefault(inst.base, []).append(_Pose(x, y, p.yaw, p.zb))
                    x, y = _local(_Pose(x, y, p.yaw, p.zb), plan.table_off)
                    dims = plan.table
                # no per-cell facing target: the block faces it, cells stay parallel (a desk at
                # the end of a row faces the board wall, not the board's centre)
                out.append(Placement(cid, (round(x, 6), round(y, 6), round(p.zb + dims[2] / 2, 6)),
                                     round(math.degrees(p.yaw) % 360.0, 6), dims, inst.category, None))
            continue
        x, y, dims = p.x, p.y, inst.dims
        if plan is not None:
            boxes.setdefault(inst.base, []).append(p)
            x, y = _local(p, plan.table_off)
            dims = plan.table
        if inst.turn % 180:
            dims = (dims[1], dims[0], dims[2])  # the object's own dims: it stands turned
        out.append(
            Placement(
                iid,
                (round(x, 6), round(y, 6), round(p.zb + dims[2] / 2, 6)),
                round((math.degrees(p.yaw) + inst.turn) % 360.0, 6),
                dims,
                inst.category,
                _facing_target(pr, iid),
            )
        )
    for base, bx in boxes.items():
        out += _seat_placements(pr, base, bx)
    return out


def _seat_index(pid: str, base: str) -> int:
    tail = pid[len(base) + 1:] if pid.startswith(base + "_") else ""
    return int(tail) if tail.isdigit() else 0


def _unseat(pr: _Problem, placements: list[Placement], issues: list[str]) -> list[Placement]:
    """check_layout of seated objects: each table placement back to the centre of its table +
    chairs box (what the solver placed), and every chair checked against where its table puts it."""
    if not pr.seated:
        return placements
    out: list[Placement] = []
    boxes: dict[str, list[tuple[int, _Pose]]] = {}
    chairs: dict[str, Placement] = {}
    for pl in placements:
        base = next((b for b in pr.seated if pl.id == b or (pl.id.startswith(b + "_") and pl.id[len(b) + 1:].isdigit())), None)
        if base is None:
            if any(pl.id.startswith(b + "_seat_") and pl.id[len(b) + 6:].isdigit() for b in pr.seated):
                chairs[pl.id] = pl
            else:
                out.append(pl)
            continue
        plan = pr.seated[base]
        yaw = math.radians(pl.rotation_deg_z)
        r, f = _axes(yaw)
        ox, oy = plan.table_off
        bx, by = pl.location[0] - ox * r[0] - oy * f[0], pl.location[1] - ox * r[1] - oy * f[1]
        boxes.setdefault(base, []).append((_seat_index(pl.id, base), _Pose(bx, by, yaw, pl.location[2] - pl.dims[2] / 2)))
        out.append(Placement(pl.id, (bx, by, pl.location[2]), pl.rotation_deg_z, pl.dims, pl.category, pl.facing_target))
    for base, found in boxes.items():
        for want in _seat_placements(pr, base, [p for _i, p in sorted(found, key=lambda t: t[0])]):
            got = chairs.pop(want.id, None)
            if got is None:
                issues.append(f"hard: no placement for seat '{want.id}'")
            elif math.dist(got.location[:2], want.location[:2]) > LAYOUT_REL_TOL_M:
                issues.append(f"hard: seat '{want.id}' is not at its table")
    issues += [f"hard: placement '{cid}' is not an object instance of the spec" for cid in chairs]
    return out


def _settle_facing(pr: _Problem, poses: dict[str, _Pose]) -> None:
    """Turn objects still facing off-target toward it, where that breaks nothing.

    facing is soft for the solver but blocking for validate_scene: an armchair left 48 deg
    off its coffee table failed an otherwise clean scene. Only the yaw changes, and only if
    every hard constraint still holds.
    """
    for r in pr.rels:
        if r.rel != "facing" or r.a not in poses or r.b not in poses:
            continue
        pa, pb = poses[r.a], poses[r.b]
        if pr.facing_error_deg(r, pa, pb) <= LAYOUT_FACING_TOL_DEG:
            continue
        inst = pr.insts[r.a]
        tries = [_Pose(pa.x, pa.y, _face_yaw(pa.x, pa.y, pb.x, pb.y), pa.zb)]
        if any(rl.rel == "against_wall" for rl in pr.by_a.get(r.a, [])):
            # back to a wall: the yaw is fixed, so slide along the wall to the target instead
            slid = _Pose(pb.x, pb.y, pa.yaw, pa.zb)
            pr.snap_walls(inst, slid, set_yaw=True)
            tries.append(slid)
        others = {k: v for k, v in poses.items() if k != r.a}
        for cand in tries:
            if pr.facing_error_deg(r, cand, pb) > LAYOUT_FACING_TOL_DEG:
                continue
            if pr.hard_violation(inst, cand, others) is None and not any(
                    pr.hard_violation(pr.insts[k], v, {**others, r.a: cand}) for k, v in others.items()
                    if any(rl.b == r.a for rl in pr.by_a.get(k, []))):
                poses[r.a] = cand
                break


def solve_layout(
    spec: SceneSpec,
    priors: dict[str, CategoryPrior] | None = None,
    measured_dims: dict[str, Dims] | None = None,
    seed: int = LAYOUT_SEED,
    *,
    blocked: list[tuple[str, _Pose, Dims]] | None = None,
) -> list[Placement]:
    """Place every object instance. Deterministic for a given (spec, priors, dims, seed).

    Raises LayoutError (with ``conflicts``) when the spec cannot be satisfied.
    """
    if spec.setting.rooms:
        return _solve_rooms(spec, priors, measured_dims, seed)
    pr = _Problem(spec, priors or load_priors(), {k: tuple(v) for k, v in (measured_dims or {}).items()})  # type: ignore[misc]
    pr.blocked = list(blocked or [])
    # facing is soft here but blocking in validate_scene, and whether a wall-bound object can
    # face its target depends on where earlier objects landed: try a few more seeds (still
    # deterministic) and keep the layout that misses the fewest facings.
    best: tuple[int, list[Placement]] | None = None
    error: LayoutError | None = None
    for attempt in range(1 + LAYOUT_FACING_ATTEMPTS):
        try:
            poses = _solve_once(pr, seed + 7919 * attempt)
        except LayoutError as e:
            if attempt == 0 and pr.strict_rings and any(r.rel == "around" and r.ring_n > 1 for r in pr.rels):
                pr.strict_rings = False  # no room for even rings: any place round the target
                poses = _solve_once(pr, seed)
            elif attempt == 0:
                raise
            else:
                error = e
                continue
        misses = sum(1 for r in pr.rels if r.rel == "facing" and r.a in poses and r.b in poses
                     and pr.facing_error_deg(r, poses[r.a], poses[r.b]) > LAYOUT_FACING_TOL_DEG)
        if best is None or misses < best[0]:
            best = (misses, _to_placements(pr, poses))
        if misses == 0:
            break
    if best is None:
        assert error is not None
        raise error
    return best[1]


def _solve_rooms(
    spec: SceneSpec, priors: dict[str, CategoryPrior] | None, measured_dims: dict[str, Dims] | None, seed: int,
) -> list[Placement]:
    """A multi-room building, room by room: each room is a one-room spec of its own size (its
    walls are wall_north.. of that rectangle), then shifted to its offset. Relations never cross
    rooms (validate_spec), so the rooms are independent."""
    from dataclasses import replace

    out: list[Placement] = []
    for room in spec.setting.rooms:
        objs = tuple(replace(o, room=None) for o in spec.objects if o.room == room.id)
        if not objs:
            continue
        ids = {o.id for o in objs}
        sub = SceneSpec(replace(spec.setting, bounds_m=(*room.size_m, spec.setting.bounds_m[2]), rooms=()), objs,
                        tuple(r for r in spec.relations if r.a in ids), spec.lighting, spec.camera)
        dx, dy = room.offset_m
        zones = [(lbl, _Pose(p.x - dx, p.y - dy, p.yaw, p.zb), dd) for lbl, p, dd, rids in _door_zones(spec)
                 if room.id in rids]
        try:
            pls = solve_layout(sub, priors, measured_dims, seed, blocked=zones)
        except LayoutError as e:
            raise LayoutError(f"room '{room.id}': {e}", e.conflicts) from e
        out += [replace(p, location=(p.location[0] + dx, p.location[1] + dy, p.location[2])) for p in pls]
    return out


def room_doors(spec: SceneSpec, width_m: float = 0.9, margin_m: float = 0.1) -> list[dict[str, Any]]:
    """One door per pair of rooms sharing a wall long enough for it, at the shared segment's middle.

    ``at`` is in the layout frame; ``axis`` is the wall's direction ("x": a wall running east-west).
    """
    doors: list[dict[str, Any]] = []
    rooms = spec.setting.rooms
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            for k in (0, 1):  # k = 0: a wall at constant x (runs along y); k = 1: at constant y
                o = 1 - k
                if abs(a.offset_m[k] + a.size_m[k] - b.offset_m[k]) < 1e-6:
                    touch = b.offset_m[k]
                elif abs(b.offset_m[k] + b.size_m[k] - a.offset_m[k]) < 1e-6:
                    touch = a.offset_m[k]
                else:
                    continue
                lo = max(a.offset_m[o], b.offset_m[o])
                hi = min(a.offset_m[o] + a.size_m[o], b.offset_m[o] + b.size_m[o])
                if hi - lo < width_m + 2 * margin_m:
                    continue
                at = [0.0, 0.0]
                at[k], at[o] = touch, (lo + hi) / 2
                doors.append({"id": f"door_{a.id}_{b.id}", "rooms": [a.id, b.id], "at": at,
                              "axis": "y" if k == 0 else "x", "width": width_m})
    return doors


def _door_zones(spec: SceneSpec) -> list[tuple[str, _Pose, Dims, list[str]]]:
    """The clear zone of each inner door (layout frame): door width x LAYOUT_DOOR_CLEAR_M deep on
    both sides of the wall, with the ids of the two rooms it joins."""
    out = []
    for dr in room_doors(spec):
        w, c = dr["width"], 2 * LAYOUT_DOOR_CLEAR_M
        dims = (c, w, LAYOUT_DOOR_HEIGHT_M) if dr["axis"] == "y" else (w, c, LAYOUT_DOOR_HEIGHT_M)
        out.append((f"doorway '{dr['id']}'", _Pose(dr["at"][0], dr["at"][1], 0.0, 0.0), dims, dr["rooms"]))
    return out


def _check_rooms(spec: SceneSpec, placements: list[Placement], priors: dict[str, CategoryPrior] | None) -> list[str]:
    from dataclasses import replace

    def base(pid: str) -> str:
        head, _, tail = pid.rpartition("_")
        head = head.removesuffix("_seat") if tail.isdigit() and head.endswith("_seat") else head
        return head if tail.isdigit() and head else pid

    room_of = {o.id: o.room for o in spec.objects}
    issues = [f"hard: placement '{p.id}' is not an object instance of the spec" for p in placements
              if p.id not in room_of and base(p.id) not in room_of]
    zones = _door_zones(spec)
    for room in spec.setting.rooms:
        objs = tuple(replace(o, room=None) for o in spec.objects if o.room == room.id)
        if not objs:
            continue
        ids = {o.id for o in objs}
        dx, dy = room.offset_m
        pls = [replace(p, location=(p.location[0] - dx, p.location[1] - dy, p.location[2])) for p in placements
               if p.id in ids or (p.id not in room_of and base(p.id) in ids)]
        sub = SceneSpec(replace(spec.setting, bounds_m=(*room.size_m, spec.setting.bounds_m[2]), rooms=()), objs,
                        tuple(r for r in spec.relations if r.a in ids), spec.lighting, spec.camera)
        issues += check_layout(sub, pls, priors, blocked=[(lbl, _Pose(p.x - dx, p.y - dy, p.yaw, p.zb), d)
                                                         for lbl, p, d, rids in zones if room.id in rids])
    return issues


def _dependents(pr: _Problem, iid: str) -> list[str]:
    """Objects placed relative to ``iid`` (transitively): they turn with it."""
    out: list[str] = []
    todo = [iid]
    while todo:
        cur = todo.pop()
        for r in pr.by_b.get(cur, []):
            if r.rel != "facing" and r.a not in out and r.a != iid:
                out.append(r.a)
                todo.append(r.a)
    return out


def _snap_axes(pr: _Problem, poses: dict[str, _Pose]) -> None:
    """Last pass: every free yaw of a rect room onto the room's (or its target's) axes. The object
    turns about its centre and whatever stands relative to it turns with it; a snap that would
    break a hard constraint is not made."""
    for iid in pr.order():
        p = poses.get(iid)
        if p is None or pr.carries_stack(iid, poses):
            continue
        target = pr.snap_yaw(iid, p.yaw, poses)
        facing = [r for r in pr.by_a.get(iid, []) if r.rel == "facing" and r.b in poses]
        if target == p.yaw and facing and pr.square_facing(iid):
            # facing: onto the room's axes too when that still faces the target (desks 3 deg off the
            # board read as "rotated haphazardly")
            square = round(p.yaw / _HALF_PI) * _HALF_PI
            sq = _Pose(p.x, p.y, square, p.zb)
            if all(pr.facing_error_deg(r, sq, poses[r.b]) <= _SQUARE_FACING_TOL_DEG for r in facing):
                target = square
        delta = target - p.yaw
        if abs(delta) < 1e-9:
            continue
        moved = {iid: _Pose(p.x, p.y, target, p.zb)}
        c, sn = math.cos(delta), math.sin(delta)
        for dep in _dependents(pr, iid):
            q = poses[dep]
            dx, dy = q.x - p.x, q.y - p.y
            moved[dep] = _Pose(p.x + c * dx - sn * dy, p.y + sn * dx + c * dy, q.yaw + delta, q.zb)
        trial = {**poses, **moved}
        if all(pr.hard_violation(pr.insts[k], trial[k], {o: v for o, v in trial.items() if o != k}) is None
               for k in moved):
            poses.update(moved)


def _solve_once(pr: _Problem, seed: int) -> dict[str, _Pose]:
    rng = random.Random(seed)
    for attempt in range(LAYOUT_RESTARTS):
        try:
            poses = _place_greedy(pr, rng)
            break
        except LayoutError:
            if attempt == LAYOUT_RESTARTS - 1:
                raise  # the last attempt's conflict is reported
    _local_search(pr, poses, rng)
    _settle_facing(pr, poses)
    _snap_axes(pr, poses)
    for iid, p in poses.items():  # final proof that nothing was broken by the search
        v = pr.hard_violation(pr.insts[iid], p, poses)
        if v:
            raise LayoutError(f"internal: final layout violates a hard constraint for '{iid}': {v}", [iid])
    return poses


def _collapse_grids(pr: _Problem, placements: list[Placement], issues: list[str]) -> list[Placement]:
    """Grid cells -> their block placement, after checking every cell is where the grid puts it."""
    if not pr.grids:
        return placements
    out = [pl for pl in placements if not any(pl.id.startswith(g + "_") and pl.id[len(g) + 1:].isdigit()
                                               for g in pr.grids)]
    for gid, (rows, cols, _gx, gy, cell) in pr.grids.items():
        cells = {pl.id: pl for pl in placements if pl.id.startswith(gid + "_") and pl.id[len(gid) + 1:].isdigit()}
        if len(cells) != rows * cols:
            issues.append(f"hard: grid '{gid}': {len(cells)} of {rows * cols} cells placed")
            continue
        xs = [c.location[0] for c in cells.values()]
        ys = [c.location[1] for c in cells.values()]
        first = cells[f"{gid}_1"]
        yaw = math.radians(first.rotation_deg_z)
        back = (pr.insts[gid].dims[1] - (rows * cell[1] + (rows - 1) * gy)) / 2  # a walkway behind the rows
        f = _axes(yaw)[1]
        block = _Pose(sum(xs) / len(xs) - f[0] * back, sum(ys) / len(ys) - f[1] * back, yaw,
                      first.location[2] - cell[2] / 2)
        for cid, x, y in _grid_cells(pr, gid, block):
            c = cells[cid]
            if math.hypot(c.location[0] - x, c.location[1] - y) > LAYOUT_REL_TOL_M:
                issues.append(f"hard: grid '{gid}': {cid} is off its grid position")
        out.append(Placement(gid, (block.x, block.y, first.location[2]), first.rotation_deg_z,
                             pr.insts[gid].dims, first.category))
    return out


def check_layout(
    spec: SceneSpec,
    placements: list[Placement],
    priors: dict[str, CategoryPrior] | None = None,
    *,
    blocked: list[tuple[str, _Pose, Dims]] | None = None,
) -> list[str]:
    """Independent verification. Returns 'hard: ...' / 'soft: ...' issue strings (empty = clean).

    A multi-room spec is checked room by room, like solve_layout, with each room's doorways kept clear."""
    if spec.setting.rooms:
        return _check_rooms(spec, placements, priors)
    pr = _Problem(spec, priors or load_priors(), {p.id: p.dims for p in placements})
    pr.blocked = list(blocked or [])
    turned = {o.id: o.turn_deg for o in spec.objects if o.turn_deg and o.grid is None and not o.seats}
    if turned:  # back to the solver's frame: the yaw the relations give, the footprint as turned
        def base(pid: str) -> str:
            head, _, tail = pid.rpartition("_")
            return head if tail.isdigit() and head in turned else pid

        placements = [replace(pl, rotation_deg_z=(pl.rotation_deg_z - turned[base(pl.id)]) % 360.0,
                              dims=(pl.dims[1], pl.dims[0], pl.dims[2]) if turned[base(pl.id)] % 180 else pl.dims)
                      if base(pl.id) in turned else pl for pl in placements]
    pr.strict_rings = False  # an uneven ring is reported soft, below
    pr.verify_stacks = False  # a stacked copy is checked once every pose is known, below
    poses: dict[str, _Pose] = {}
    issues: list[str] = []
    placements = _unseat(pr, placements, issues)
    placements = _collapse_grids(pr, placements, issues)
    for pl in placements:
        if pl.id not in pr.insts:
            issues.append(f"hard: placement '{pl.id}' is not an object instance of the spec")
            continue
        poses[pl.id] = _Pose(pl.location[0], pl.location[1], math.radians(pl.rotation_deg_z), pl.location[2] - pl.dims[2] / 2)
    for iid in pr.insts:
        if iid not in poses:
            issues.append(f"hard: no placement for '{iid}'")
    placed: dict[str, _Pose] = {}
    for iid, p in poses.items():
        v = pr.hard_violation(pr.insts[iid], p, placed)
        placed[iid] = p
        if v:
            issues.append(f"hard: {iid}: {v}")
        inst = pr.insts[iid]
        if inst.prior.support in ("floor", "ground") and not any(r.rel == "on" for r in pr.by_a.get(iid, [])) and abs(p.zb) > 1e-3:
            issues.append(f"hard: {iid}: floats at z={p.zb:.3f} m")
    for iid, p in poses.items():
        inst = pr.insts[iid]
        sup = next((r.b for r in pr.by_a.get(iid, []) if r.rel == "on" and r.b in poses and not pr.shelf(r.b)), None)
        if inst.category in STACKABLE and sup is not None and p.zb > poses[sup].zb + pr.insts[sup].dims[2] + 1e-3 \
                and pr.stack_below(inst, p, poses) is None:
            issues.append(f"hard: {iid}: stacked on nothing")
    for r in pr.rels:
        if r.rel == "around" and r.ring_n > 1 and r.a in poses and r.b in poses:
            slot = pr.ring_slot(r, poses[r.b])
            if math.hypot(poses[r.a].x - slot.x, poses[r.a].y - slot.y) > LAYOUT_REL_TOL_M:
                issues.append(f"soft: '{r.label}': not at its place {r.ring_i + 1} of {r.ring_n} round '{r.b}'")
        if r.rel == "facing" and r.a in poses and r.b in poses:
            err = pr.facing_error_deg(r, poses[r.a], poses[r.b])
            if err > LAYOUT_FACING_TOL_DEG:
                issues.append(f"soft: '{r.label}': facing off by {err:.0f} deg")
    return issues
