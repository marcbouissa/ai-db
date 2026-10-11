"""Deterministic layout solver: SceneSpec + priors -> oriented-box placements.

Conventions
-----------
* Room: x in [0, W] (east-west), y in [0, D] (north-south), z up. wall_south is y=0,
  wall_north y=D, wall_west x=0, wall_east x=W. Only ``indoor`` settings have walls.
* Object dims are [w, d, h]: w along the local x axis, d along local y. The *front* of an
  object is its local -Y side (shared asset convention). ``rotation_deg_z`` is the yaw
  (counter-clockwise, 0 = front faces -y; front vector = (sin yaw, -cos yaw)).
* Room shape ``rect`` (default) | ``ellipse`` | ``polygon`` inside the box [0,W]x[0,D]. Rect walls
  are the four named walls; ellipse has ``wall`` (+ ``at_deg`` = ellipse parameter angle);
  polygons have ``wall_<n>`` (edge n, 1-based) and ``wall`` with ``at_deg`` (ray from the center). ``location`` is the center of the bounding box (z included).
* ``A above B``: A is wall-mounted (support ``wall``, with its against_wall relation) straight
  above B: the same position along the wall, A's base ``gap_m`` (default 0.25 m) over B's top.
* ``count > 1`` expands an object into instances ``<id>_1 .. <id>_n``. Relations of the
  object apply to every instance; a target with several instances is paired by index.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ai_db_scene.constants import (
    LAYOUT_BREATHING_M,
    LAYOUT_CANDIDATES,
    LAYOUT_DEFAULT_ABOVE_GAP_M,
    LAYOUT_DEFAULT_AROUND_GAP_M,
    LAYOUT_DEFAULT_BESIDE_GAP_M,
    LAYOUT_DEFAULT_IN_FRONT_GAP_M,
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
        self.insts: dict[str, _Inst] = {}
        #: grid objects: id -> (rows, cols, gap x, gap y, cell dims). Solved as one block
        #: instance (relations apply to the block), expanded into cells by _to_placements.
        self.grids: dict[str, tuple[int, int, float, float, Dims]] = {}
        by_base: dict[str, list[str]] = {}
        for o in spec.objects:
            prior = priors.get(o.category)
            if prior is None:
                raise LayoutError(f"object '{o.id}': category '{o.category}' has no prior", [o.id])
            if o.grid is not None:
                rows, cols, gx, gy = o.grid
                cell = dims_override.get(o.id + "_1") or dims_override.get(o.id) or prior.typical_m
                self.grids[o.id] = (rows, cols, gx, gy, cell)
                block = (cols * cell[0] + (cols - 1) * gx, rows * cell[1] + (rows - 1) * gy, cell[2])
                self.insts[o.id] = _Inst(o.id, o.id, o.category, prior, block, o.hero, 0, 1)
                by_base[o.id] = [o.id]
                continue
            for i in range(1, o.count + 1):
                iid = o.id if o.count == 1 else f"{o.id}_{i}"
                dims = dims_override.get(iid) or dims_override.get(o.id) or prior.typical_m
                self.insts[iid] = _Inst(iid, o.id, o.category, prior, dims, o.hero, i - 1, o.count)
                by_base.setdefault(o.id, []).append(iid)
        self.rels: list[_Rel] = []
        for r in spec.relations:
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
                label = f"{ai} {r.rel} {bi}" + (f" (gap {gap:g} m)" if r.rel in (*_GAP_RELS, "above") else "")
                self.rels.append(_Rel(ai, r.rel, bi, gap, a.idx, a.n, label, r.at_deg))
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
        focus = spec.camera.get("focus")
        self.focus = by_base[focus][0] if focus in by_base else None
        self.under_pairs = {frozenset((r.a, r.b)) for r in self.rels if r.rel == "under"}
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
                if not b.prior.can_support_others:
                    raise LayoutError(f"relation '{r.label}': category '{b.category}' cannot support other objects", [r.label])
                if self.insts[r.a].prior.support in ("wall", "ceiling"):
                    raise LayoutError(f"relation '{r.label}': '{r.a}' is {self.insts[r.a].prior.support}-mounted and cannot rest on an object", [r.label])
            if r.rel == "above":
                a = self.insts[r.a]
                if a.prior.support != "wall" or a.id not in has_wall:
                    raise LayoutError(
                        f"relation '{r.label}': 'above' places a wall-mounted object over another; '{r.a}' "
                        f"({a.category}) is {a.prior.support}-supported" + ("" if a.id in has_wall else " and has no against_wall relation")
                        + " (e.g. a mirror against_wall + above a sink)", [r.label])
            if r.rel == "under" and self.insts[r.a].dims[2] >= self.insts[r.b].dims[2]:
                raise LayoutError(f"relation '{r.label}': '{r.a}' is as tall as or taller than '{r.b}'", [r.label])

    # ---- walls ---------------------------------------------------------------------------
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
            delta = _ext(inst.dims, pose.yaw, n) - s
            pose.x += delta * n[0]
            pose.y += delta * n[1]
            if k == 0 and self.shape.curved:  # a flat back on a curve: slide in until the corners fit
                for _ in range(int(_CURVED_WALL_GAP_M / 0.005)):
                    if self.in_room(inst, pose, LAYOUT_EPS_M):
                        break
                    pose.x += 0.005 * n[0]
                    pose.y += 0.005 * n[1]

    def default_face(self, inst: _Inst, poses: dict[str, _Pose]) -> tuple[float, float] | None:
        """Point an object with no orienting relation faces: the camera focus, else the room center (indoor)."""
        if any(r.rel in _ORIENT_RELS for r in self.by_a.get(inst.id, [])):
            return None
        if self.focus and self.focus != inst.id and self.focus in poses:
            return poses[self.focus].x, poses[self.focus].y
        return (self.shape.cx, self.shape.cy) if self.kind == "indoor" else None

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
                return poses[r.b].zb + self.insts[r.b].dims[2]
        return 0.0

    # ---- predicates ----------------------------------------------------------------------
    def rel_violation(self, r: _Rel, pa: _Pose, pb: _Pose | None) -> str | None:
        """Reason a hard relation is violated, else None. facing is soft: handled elsewhere."""
        a = self.insts[r.a]
        tol = LAYOUT_REL_TOL_M
        if r.rel == "against_wall":
            n, p, _ = self.wall(r.b, r.at_deg, pa.x, pa.y)
            gap = (pa.x - p[0]) * n[0] + (pa.y - p[1]) * n[1] - _ext(a.dims, pa.yaw, n)
            f = _axes(pa.yaw)[1]
            lo, hi = (-1e-3, _CURVED_WALL_GAP_M) if self.shape.curved else (-1e-3, 1e-3)
            if not lo <= gap <= hi or f[0] * n[0] + f[1] * n[1] < 0.99:
                return f"'{r.label}': not flush against the wall (gap {gap:.3f} m)"
            if r.at_deg is not None and self.shape.kind != "rect":
                lat = math.hypot(pa.x - p[0] - (gap + _ext(a.dims, pa.yaw, n)) * n[0], pa.y - p[1] - (gap + _ext(a.dims, pa.yaw, n)) * n[1])
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
        if r.rel == "on":
            if abs(pa.zb - (pb.zb + b.dims[2])) > 1e-3 or not _inside(pa, a.dims, pb, b.dims, LAYOUT_EPS_M):
                return f"'{r.label}': does not fit on top of '{r.b}'"
            return None
        if r.rel == "under":
            if not _inside(pa, a.dims, pb, b.dims, LAYOUT_EPS_M):
                return f"'{r.label}': not within the footprint of '{r.b}'"
            return None
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
            return None
        return None

    def facing_error_deg(self, r: _Rel, pa: _Pose, pb: _Pose) -> float:
        return math.degrees(_angdiff(pa.yaw, _face_yaw(pa.x, pa.y, pb.x, pb.y)))

    def margin(self, a: _Inst, b: _Inst) -> float:
        m = max(a.prior.default_clearance_m, b.prior.default_clearance_m)
        g = self.pair_gap.get(frozenset((a.id, b.id)))
        return m if g is None else min(m, g)

    def pair_conflict(self, a: _Inst, pa: _Pose, b: _Inst, pb: _Pose) -> bool:
        if a.flat != b.flat:
            return False
        if frozenset((a.id, b.id)) in self.under_pairs:
            return False
        if pa.zb >= pb.zb + b.dims[2] - LAYOUT_EPS_M or pb.zb >= pa.zb + a.dims[2] - LAYOUT_EPS_M:
            return False
        return _sep(pa, a.dims, pb, b.dims) < self.margin(a, b) - LAYOUT_EPS_M

    def hard_violation(self, inst: _Inst, pose: _Pose, poses: dict[str, _Pose]) -> str | None:
        """First hard-constraint violation of `inst` at `pose` against already-placed poses."""
        e = LAYOUT_EPS_M
        if not self.in_room(inst, pose, e):
            return "out of bounds"
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
        for oid, op in poses.items():
            if oid != inst.id and self.pair_conflict(inst, pose, self.insts[oid], op):
                return f"overlaps '{oid}'"
        return None

    def soft_cost(self, inst: _Inst, pose: _Pose, poses: dict[str, _Pose]) -> float:
        cost = 0.0
        for r in self.by_a.get(inst.id, []):
            pb = poses.get(r.b)
            if pb is None:
                continue
            if r.rel == "facing":
                err = self.facing_error_deg(r, pose, pb)
                cost += err / 180.0 + (LAYOUT_FACING_MISS_COST if err > LAYOUT_FACING_TOL_DEG else 0.0)
            elif r.rel == "in_front_of" and not any(q.rel == "facing" for q in self.by_a[inst.id]):
                cost += 0.3 * _angdiff(pose.yaw, pb.yaw + math.pi) / math.pi  # A faces B
            elif r.rel == "beside":
                cost += 0.3 * _angdiff(pose.yaw, pb.yaw) / math.pi
        dt = self.default_face(inst, poses)
        if dt is not None:
            cost += 0.3 * _angdiff(pose.yaw, _face_yaw(pose.x, pose.y, dt[0], dt[1])) / math.pi
        for oid, op in poses.items():
            if oid == inst.id:
                continue
            o = self.insts[oid]
            if o.flat or inst.flat or frozenset((inst.id, oid)) in self.pair_gap:
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

        # wall/center-anchored objects first (they have the fewest legal poses), then hero, then big
        key = lambda i: (not anchored(i), not self.insts[i].hero, -self.insts[i].dims[0] * self.insts[i].dims[1], i)
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
        elif prim is None or pb is None and prim.rel != "centered_in":
            x = rng.uniform(0, pr.W)
            y = rng.uniform(0, pr.D)
        elif prim.rel == "centered_in":
            x, y = (pr.shape.cx, pr.shape.cy) if pb is None else (pb.x, pb.y)
        else:
            assert b is not None and pb is not None
            rb, fb = _axes(pb.yaw)
            jit = 0.0 if k == 0 else rng.uniform(-1, 1)
            if prim.rel == "above":
                x, y = pb.x, pb.y  # snap_walls then moves A onto its wall: same position along it
            elif prim.rel in ("on", "under"):
                x = pb.x + jit * max(b.dims[0] / 2 - inst.dims[0] / 2, 0) * rb[0] + rng.uniform(-1, 1) * (0 if k == 0 else max(b.dims[1] / 2 - inst.dims[1] / 2, 0)) * fb[0]
                y = pb.y + jit * max(b.dims[0] / 2 - inst.dims[0] / 2, 0) * rb[1] + rng.uniform(-1, 1) * (0 if k == 0 else max(b.dims[1] / 2 - inst.dims[1] / 2, 0)) * fb[1]
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
        if facing_targets and not walls and (prim is None or prim.rel in ("on", "under", "centered_in")) and k % 4 != 3:
            ft = facing_targets[k % len(facing_targets)]
            yaw = _face_yaw(x, y, ft.x, ft.y)
        elif dface is not None and not walls and k % 4 != 3:
            yaw = _face_yaw(x, y, dface[0], dface[1])
        p = _Pose(x, y, yaw, 0.0)
        pr.snap_walls(inst, p, set_yaw=True)
        p.zb = pr.base_z(inst, poses)
        out.append(p)
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
            try_move(iid, _Pose(p.x, p.y, _face_yaw(p.x, p.y, dt[0], dt[1]), p.zb))
    step, dyaw = LAYOUT_LOCAL_STEP_M, math.radians(LAYOUT_LOCAL_YAW_DEG)
    for _ in range(LAYOUT_LOCAL_ITERS):
        iid = rng.choice(ids)
        inst = pr.insts[iid]
        p = poses[iid]
        if pr.by_b.get(iid) and not any(r.rel == "facing" for r in pr.by_b[iid]):
            continue  # support/anchor objects keep their position: dependents were placed relative to them
        new = _Pose(p.x + rng.uniform(-step, step), p.y + rng.uniform(-step, step), p.yaw, p.zb)
        if iid not in yaw_locked:
            new.yaw = p.yaw + rng.uniform(-dyaw, dyaw)
            for r in pr.by_a.get(iid, []):
                if r.rel == "facing" and r.b in poses and rng.random() < 0.5:
                    new.yaw = _face_yaw(new.x, new.y, poses[r.b].x, poses[r.b].y)
            dt = pr.default_face(inst, poses)
            if dt is not None and rng.random() < 0.5:
                new.yaw = _face_yaw(new.x, new.y, dt[0], dt[1])
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


def _to_placements(pr: _Problem, poses: dict[str, _Pose]) -> list[Placement]:
    out = []
    for iid, inst in pr.insts.items():
        p = poses[iid]
        if iid in pr.grids:
            cell = pr.grids[iid][4]
            for cid, x, y in _grid_cells(pr, iid, p):
                # no per-cell facing target: the block faces it, cells stay parallel (a desk at
                # the end of a row faces the board wall, not the board's centre)
                out.append(Placement(cid, (round(x, 6), round(y, 6), round(p.zb + cell[2] / 2, 6)),
                                     round(math.degrees(p.yaw) % 360.0, 6), cell, inst.category, None))
            continue
        out.append(
            Placement(
                iid,
                (round(p.x, 6), round(p.y, 6), round(p.zb + inst.dims[2] / 2, 6)),
                round(math.degrees(p.yaw) % 360.0, 6),
                inst.dims,
                inst.category,
                _facing_target(pr, iid),
            )
        )
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
) -> list[Placement]:
    """Place every object instance. Deterministic for a given (spec, priors, dims, seed).

    Raises LayoutError (with ``conflicts``) when the spec cannot be satisfied.
    """
    if spec.setting.rooms:
        return _solve_rooms(spec, priors, measured_dims, seed)
    pr = _Problem(spec, priors or load_priors(), {k: tuple(v) for k, v in (measured_dims or {}).items()})  # type: ignore[misc]
    # facing is soft here but blocking in validate_scene, and whether a wall-bound object can
    # face its target depends on where earlier objects landed: try a few more seeds (still
    # deterministic) and keep the layout that misses the fewest facings.
    best: tuple[int, list[Placement]] | None = None
    error: LayoutError | None = None
    for attempt in range(1 + LAYOUT_FACING_ATTEMPTS):
        try:
            poses = _solve_once(pr, seed + 7919 * attempt)
        except LayoutError as e:
            if attempt == 0:
                raise
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
        try:
            pls = solve_layout(sub, priors, measured_dims, seed)
        except LayoutError as e:
            raise LayoutError(f"room '{room.id}': {e}", e.conflicts) from e
        dx, dy = room.offset_m
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
    for gid, (rows, cols, _gx, _gy, cell) in pr.grids.items():
        cells = {pl.id: pl for pl in placements if pl.id.startswith(gid + "_") and pl.id[len(gid) + 1:].isdigit()}
        if len(cells) != rows * cols:
            issues.append(f"hard: grid '{gid}': {len(cells)} of {rows * cols} cells placed")
            continue
        xs = [c.location[0] for c in cells.values()]
        ys = [c.location[1] for c in cells.values()]
        first = cells[f"{gid}_1"]
        block = _Pose(sum(xs) / len(xs), sum(ys) / len(ys), math.radians(first.rotation_deg_z),
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
) -> list[str]:
    """Independent verification. Returns 'hard: ...' / 'soft: ...' issue strings (empty = clean)."""
    pr = _Problem(spec, priors or load_priors(), {p.id: p.dims for p in placements})
    poses: dict[str, _Pose] = {}
    issues: list[str] = []
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
    for r in pr.rels:
        if r.rel == "facing" and r.a in poses and r.b in poses:
            err = pr.facing_error_deg(r, poses[r.a], poses[r.b])
            if err > LAYOUT_FACING_TOL_DEG:
                issues.append(f"soft: '{r.label}': facing off by {err:.0f} deg")
    return issues
