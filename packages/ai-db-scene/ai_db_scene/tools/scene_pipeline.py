"""MCP tools for the spec -> layout -> build pipeline (docs/3D_PIPELINE.md stages 1, 3, 4, 7).

Table-driven like scene_tools: (name, description, schema) rows plus handlers. Command
construction is pure (``build_layout_commands``) so it is testable without Blender.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from ai_db.errors import AiDbConfigError, AiDbError
from ai_db_scene.blender_client import BlenderClient, BlenderError
from ai_db_scene.build_report import present_build
from ai_db_scene.constants import (
    FOLIAGE_CATEGORIES,
    LAYOUT_DOOR_HEIGHT_M,
    SCENE_CAMERA_FACED_EYE_M,
    SCENE_CAMERA_HEIGHT_M,
    SCENE_CAMERA_MAX_SHIFT,
    SCENE_CAMERA_WATER_EYE_M,
    SCENE_CEILING,
    SCENE_EXPOSURE_INDOOR,
    SCENE_EXPOSURE_OUTDOOR,
    SCENE_EXPOSURE_PER_DOUBLING,
    SCENE_EXPOSURE_RANGE,
    SCENE_FILL_CEILING_GAP_M,
    SCENE_FILL_SIZE_FRACTION,
    SCENE_FILL_W_PER_M2,
    SCENE_FILL_W_RANGE,
    SCENE_FOLIAGE_DENSITY,
    SCENE_FOLIAGE_LEAF_M,
    SCENE_FOLIAGE_MAX_LEAVES,
    SCENE_GROUND_DROP_M,
    SCENE_GROUND_EXTENT_M,
    SCENE_LAMP_W,
    SCENE_LINE_MATERIAL,
    SCENE_LINE_WIDTH_M,
    SCENE_PLANK_FLOOR,
    SCENE_PLANK_GRAIN_SCALE,
    SCENE_PLANK_TOP_M,
    SCENE_REF_FLOOR_M2,
    SCENE_RUG_PILE_DENSITY,
    SCENE_SKY_STRENGTH_OUTDOOR,
    SCENE_SUN_BEHIND_CAMERA_DEG,
    SCENE_SUN_COLOR_OUTDOOR,
    SCENE_SUN_ELEVATION_OUTDOOR,
    SCENE_SUN_ENERGY,
    SCENE_SUN_ENERGY_OUTDOOR,
    SCENE_VIEW_LOOK_OUTDOOR,
    SCENE_WALKWAY_INSET_M,
    TREE_BARK_MATERIAL,
)
from ai_db_scene.layout import LayoutError, Placement, solve_layout
from ai_db_scene.materials import (
    FLOOR_MATERIALS,
    STYLE_PALETTES,
    interior_wall,
    legacy_fallback,
    material_for,
    pbr_preset,
    style_material,
)
from ai_db_scene.models import RUN_OUTCOMES, RunRecord
from ai_db_scene.priors import load_priors
from ai_db_scene.recipes import has_recipe, recipe_spec, user_overrides
from ai_db_scene.spec import RELATIONS, SETTING_KINDS, SceneSpec, spec_schema, validate_spec

WALL_THICKNESS_M = 0.1
_WALLS = ("wall_north", "wall_south", "wall_east", "wall_west")
_SUFFIX = re.compile(r"_\d+$")
_MAX_EXEMPLARS = 3


def _memory() -> Any:
    from ai_db_scene.tools.scene_tools import get_scene_memory

    return get_scene_memory()


def _err(message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "error", "message": message, **extra}


def _parse_spec(raw: Any) -> tuple[SceneSpec | None, list[str]]:
    return validate_spec(raw)


#: SceneSpec fields a call may pass at its top level instead of inside ``spec``
SPEC_KEYS = ("setting", "objects", "relations", "lighting", "camera")


def spec_arg(args: dict[str, Any]) -> tuple[Any, str | None]:
    """(the spec a call passes, error). Weak models wrote ``build_scene({setting, objects})``
    (26 calls rejected with "spec: Missing key") and ``spec`` as a JSON string: both are
    accepted. A JSON error names its line and column."""
    raw = args.get("spec")
    for _ in range(2):  # a JSON string, or {"spec": {...}} wrapped twice
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as e:
                return None, (f"spec: not valid JSON ({e.msg} at line {e.lineno} column {e.colno}); "
                              "pass the spec as an object")
        if isinstance(raw, dict) and set(raw) == {"spec"}:
            raw = raw["spec"]
    if raw is None and any(k in args for k in SPEC_KEYS):
        raw = {k: args[k] for k in SPEC_KEYS if k in args}
    return raw, None


#: valid specs this server has seen (submit_scene_spec, solve_layout, build_scene), newest last.
#: OpenCode's Code Mode runs every tool call in a fresh JS sandbox, so a model that validated
#: its spec in one call had no ``spec`` variable left for build_scene in the next (13 calls,
#: 9 bench scenes): build_scene({spec_id}) or build_scene() builds the remembered one.
_REMEMBERED: dict[str, dict[str, Any]] = {}
_REMEMBER_MAX = 32


def remember_spec(spec: SceneSpec) -> str:
    """Keep a valid spec for later calls; returns its ``spec_id`` (same spec, same id)."""
    data = spec.to_dict()
    sid = "spec_" + hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:10]
    _REMEMBERED.pop(sid, None)
    _REMEMBERED[sid] = data
    while len(_REMEMBERED) > _REMEMBER_MAX:
        _REMEMBERED.pop(next(iter(_REMEMBERED)))
    return sid


def remembered_spec(spec_id: Any = None) -> tuple[dict[str, Any] | None, str | None]:
    """(spec, error): the spec with ``spec_id``, or the last valid one when it is None."""
    if spec_id is not None:
        if spec_id in _REMEMBERED:
            return _REMEMBERED[spec_id], None
        return None, (f"spec_id: unknown {spec_id!r} (known: {list(_REMEMBERED)[-5:]}); "
                      "pass the spec itself, or submit_scene_spec it first")
    if not _REMEMBERED:
        return None, ("spec: missing, and no valid spec was submitted yet; pass {spec: {setting, objects, "
                      "relations}} (see get_scene_spec_schema); after that build_scene() alone rebuilds it")
    return _REMEMBERED[next(reversed(_REMEMBERED))], None


def _spec_from_args(args: dict[str, Any], remembered: bool = False) -> tuple[SceneSpec | None, list[str]]:
    """The call's spec, parsed. ``remembered``: with no spec in the call, the one ``spec_id``
    names, else the last valid spec. A valid spec is remembered for the next call."""
    raw, err = spec_arg(args)
    if err:
        return None, [err]
    if raw is None and (remembered or args.get("spec_id") is not None):
        raw, err = remembered_spec(args.get("spec_id"))
        if err:
            return None, [err]
    if raw is None:
        return None, ["spec: missing; pass {spec: {setting, objects, relations}} (see get_scene_spec_schema)"]
    spec, errors = _parse_spec(raw)
    if spec is not None:
        remember_spec(spec)
    return spec, errors


#: the invalid reply's hint: fix only what the errors name, then build without resending the spec
_INVALID_HINT = ("fix what the errors name and resubmit; once a spec is valid, build_scene() with no "
                 "arguments (or {spec_id}) builds it")


def parse_placements(raw: Any) -> list[Placement]:
    out = []
    for p in raw or []:
        out.append(
            Placement(
                id=str(p["id"]),
                location=tuple(float(v) for v in p["location"]),  # type: ignore[arg-type]
                rotation_deg_z=float(p.get("rotation_deg_z", 0.0)),
                dims=tuple(float(v) for v in p["dims"]),  # type: ignore[arg-type]
                category=str(p.get("category", "")),
                facing_target=p.get("facing_target"),
            )
        )
    return out


def placement_to_dict(p: Placement) -> dict[str, Any]:
    return {
        "id": p.id,
        "location": list(p.location),
        "rotation_deg_z": p.rotation_deg_z,
        "dims": list(p.dims),
        "category": p.category,
        "yaw_deg": p.rotation_deg_z,
        **({"facing_target": p.facing_target} if p.facing_target else {}),
    }


# --- pure command construction -------------------------------------------------


_OPENING_KINDS = ("window", "door", "niche")


def frame_offset(spec: SceneSpec) -> tuple[float, float, float]:
    """Layout frame -> Blender frame.

    solve_layout places objects in ``[0, w] x [0, d]``; the Blender addon's ``room_shell`` (and
    its validator's ``room_bounds``) centre a sized room on the origin. Without this shift
    every indoor object sat half a room away from its walls. Polygon rooms are drawn at
    their given points, so they keep the layout frame.
    """
    if spec.setting.kind == "single_asset" or spec.setting.shape == "polygon":
        return (0.0, 0.0, 0.0)
    w, d, _h = spec.setting.bounds_m
    return (-w / 2, -d / 2, 0.0)


def _shift(loc: Any, off: tuple[float, float, float]) -> list[float]:
    return [float(loc[0]) + off[0], float(loc[1]) + off[1], float(loc[2]) + off[2]]


def _openings(spec: SceneSpec, placements: list[Placement]) -> list[dict[str, Any]]:
    """Wall openings from window/door/niche objects that have an against_wall relation."""
    base_wall = {r.a: r for r in spec.relations if r.rel == "against_wall"}
    ids = {o.id for o in spec.objects}
    out = []
    for pl in placements:
        if pl.category not in _OPENING_KINDS:
            continue
        base = pl.id if pl.id in ids else _SUFFIX.sub("", pl.id)
        r = base_wall.get(base)
        if r is None:
            continue
        w, d, h = pl.dims
        # room_shell takes the opening's point ("at": snapped to the nearest wall) and its
        # sill; the solver already put the window/door object exactly there.
        out.append({
            "id": pl.id, "kind": pl.category, "wall": r.b,
            "at": _shift(pl.location, frame_offset(spec))[:2],
            "width": w, "height": h, "depth": d, "sill": max(pl.location[2] - h / 2, 0.0),
        })
    return out


def _room_commands(spec: SceneSpec, placements: list[Placement] | None = None) -> list[dict[str, Any]]:
    kind = spec.setting.kind
    if kind == "single_asset":
        return []
    w, d, h = spec.setting.bounds_m
    t = WALL_THICKNESS_M
    if kind == "indoor":
        st = spec.setting
        params: dict[str, Any] = {
            "shape": st.shape, "height": h, "wall_thickness": t,
            "openings": _openings(spec, placements or []),
            "floor": True, "ceiling": SCENE_CEILING, "trim": False,
        }
        if st.shape == "polygon":
            params["points"] = [list(p) for p in st.points]
        else:
            params["size"] = [w, d]
        if st.shape == "ellipse":
            params["segments"] = _ROOM_ELLIPSE_SEGMENTS
        return [{"type": "room_shell", "params": params}]
    from ai_db_scene.terrain import terrain_commands, terrain_of

    if terrain_of(spec) is not None:  # setting.ground / setting.water: terrain, shore and water (terrain.py)
        return terrain_commands(spec, frame_offset(spec))
    shp = spec.setting.shape
    # outdoors the ground runs on to the horizon: a slab ending at the bounds floats in the void
    beyond = [{"type": "create_primitive", "params": {
        "primitive": "cube", "name": "ground_beyond", "location": [0.0, 0.0, -t / 2 - SCENE_GROUND_DROP_M],
        "scale": [SCENE_GROUND_EXTENT_M, SCENE_GROUND_EXTENT_M, t / 2]}}] if kind == "outdoor" else []
    return beyond + [{
        "type": "create_primitive",
        "params": {"primitive": "cube", "name": "floor", "location": [0.0, 0.0, -t / 2], "scale": [w / 2, d / 2, t / 2]},
    }] if shp == "rect" else [{"type": "room_shell", "params": {
        "shape": shp, "height": 0.0, "wall_thickness": t, "openings": [], "floor": True, "ceiling": False, "trim": False,
        **({"segments": _ROOM_ELLIPSE_SEGMENTS} if shp == "ellipse" else {}),
        **({"points": [list(p) for p in spec.setting.points]} if shp == "polygon" else {"size": [w, d]})}}]


#: setting.mouldings -> (addon profile, projection m, height m, bottom z: "floor" | "rail" | "crown")
_MOULDING_PROFILES: dict[str, tuple[str, float, float, str]] = {
    "crown": ("crown_classic", 0.10, 0.12, "crown"),
    "chair_rail": ("chair_rail", 0.03, 0.08, "rail"),
    "baseboard": ("baseboard_classic", 0.02, 0.15, "floor"),
}
_MOULDING_GAP_M = 0.0015  # off the wall and floor, as the addon's trim: never coplanar
_CHAIR_RAIL_Z_M = 0.9
_MOULDING_CLEAR_M = 0.03  # beyond each side of an opening the moulding stops for
#: facets of a round room (room_shell and its moulding alike): 64 showed as vertical bands on a
#: 10 m lobby wall; 128 plus smooth shading (_ROOM_MATERIAL) reads as one curved surface
_ROOM_ELLIPSE_SEGMENTS = 128
#: faces meeting at more than this stay sharp when the shell is smooth-shaded (rect corners, openings)
_ROOM_SMOOTH_ANGLE_DEG = 30.0


def _inset(pts: list[list[float]], d: float) -> list[list[float]]:
    """Mitre offset of a CCW polygon by d towards its inside."""
    import math

    n, out = len(pts), []
    for i in range(n):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[(i + 1) % n]
        l0, l1 = math.hypot(x1 - x0, y1 - y0) or 1.0, math.hypot(x2 - x1, y2 - y1) or 1.0
        n0, n1 = (-(y1 - y0) / l0, (x1 - x0) / l0), (-(y2 - y1) / l1, (x2 - x1) / l1)  # inward (left)
        den = max(0.2, 1 + n0[0] * n1[0] + n0[1] * n1[1])
        out.append([x1 + d * (n0[0] + n1[0]) / den, y1 + d * (n0[1] + n1[1]) / den])
    return out


def _outline_pieces(pts: list[list[float]], gaps: list[tuple[float, float]]) -> list[tuple[list[list[float]], bool]]:
    """The closed outline without the arc-length intervals ``gaps`` (s0, s1; may wrap):
    [(points, closed)]: the whole loop when nothing is cut, else open runs between the gaps."""
    import math

    n = len(pts)
    cum = [0.0]
    for i in range(n):
        cum.append(cum[-1] + math.dist(pts[i], pts[(i + 1) % n]))
    total = cum[-1]
    if not gaps:
        return [(pts, True)]

    def at(s: float) -> list[float]:
        s %= total
        i = max(k for k in range(n) if cum[k] <= s + 1e-12)
        t = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-12)
        a, b = pts[i], pts[(i + 1) % n]
        return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]

    cut = sorted(((g0 % total, (g0 % total) + (g1 - g0)) for g0, g1 in gaps), key=lambda g: g[0])
    merged: list[list[float]] = []
    for g0, g1 in cut:
        if merged and g0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], g1)
        else:
            merged.append([g0, g1])
    if merged[-1][1] - merged[0][0] >= total:
        return []
    pieces = []
    for k, (_g0, g1) in enumerate(merged):
        nxt = merged[(k + 1) % len(merged)][0] + (total if k == len(merged) - 1 else 0.0)
        if nxt - g1 < 0.05:
            continue
        run = [at(g1)] + [pts[j % n] for j in range(n * 2) if g1 < cum[j % n] + total * (j // n) < nxt] + [at(nxt)]
        pieces.append((run, False))
    return pieces


def _arc_position(pts: list[list[float]], q: list[float]) -> float:
    """Arc length along the outline of the point nearest to q."""
    import math

    best, s, pos = math.inf, 0.0, 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        ab = (b[0] - a[0], b[1] - a[1])
        ln = math.hypot(*ab) or 1e-12
        u = max(0.0, min(1.0, ((q[0] - a[0]) * ab[0] + (q[1] - a[1]) * ab[1]) / (ln * ln)))
        dd = math.dist(q, (a[0] + u * ab[0], a[1] + u * ab[1]))
        if dd < best:
            best, pos = dd, s + u * ln
        s += ln
    return pos


#: sweeps each named profile along its paths with the addon's own sweep (shapes._sweep_into, as its
#: "sweep" shape does for a named profile: projection to the left of the path, so a CCW outline puts
#: it on the room side). Objects keep world coordinates; untagged, category trim.
_MOULDING = """
import bpy, bmesh, json, sys
from mathutils import Vector
jobs = json.loads({payload!r})
shp = next(m for n, m in sys.modules.items() if n.endswith(".shapes") and hasattr(m, "_sweep_into") and hasattr(m, "_profile_pts"))
mat = bpy.data.materials.get("ai_db_moulding")
if mat is None:
    mat = bpy.data.materials.new("ai_db_moulding")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.92, 0.9, 0.86, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.45
    mat.diffuse_color = (0.92, 0.9, 0.86, 1.0)
made = []
for job in jobs:
    prof = [(y, -x) for x, y in shp._profile_pts(job["profile"], job["width"], job["height"])]
    for k, (path, closed) in enumerate(job["paths"]):
        old = bpy.data.objects.get(job["name"] + "_" + str(k))
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
        bm = bmesh.new()
        shp._sweep_into(bm, [Vector((x, y, job["z"])) for x, y in path], prof, closed)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        me = bpy.data.meshes.new(job["name"] + "_" + str(k))
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat)
        obj = bpy.data.objects.new(me.name, me)
        bpy.context.scene.collection.objects.link(obj)
        obj["category"] = "trim"
        obj["moulding"] = job["kind"]
        made.append(obj.name)
result = {{"mouldings": made}}
"""


def moulding_commands(spec: SceneSpec, placements: list[Placement]) -> list[dict[str, Any]]:
    """run_code sweeping setting.mouldings (or the style's) along the room's inner outline, 1.5 mm
    inside the walls, stopping at every opening whose height range it crosses (doors cut the
    baseboard and chair rail, windows the chair rail when their sill is low)."""
    kinds = spec.setting.moulding_list()
    if not kinds or spec.setting.kind != "indoor":
        return []
    off = frame_offset(spec)
    h = spec.setting.bounds_m[2]
    outline = _inset(footprint(spec, off, _ROOM_ELLIPSE_SEGMENTS), _MOULDING_GAP_M)
    opens = _openings(spec, placements)
    scale = min(max(h / 2.8, 1.0), 2.0)  # taller rooms, heavier crown
    jobs = []
    for kind in kinds:
        profile, width, height, where = _MOULDING_PROFILES[kind]
        if kind == "crown":
            width, height = width * scale, height * scale
        z = {"floor": _MOULDING_GAP_M, "rail": _CHAIR_RAIL_Z_M - height / 2,
             "crown": h - height - _MOULDING_GAP_M}[where]
        gaps = []
        for op in opens:
            if op["sill"] < z + height and op["sill"] + op["height"] > z:
                s = _arc_position(outline, op["at"])
                half = op["width"] / 2 + _MOULDING_CLEAR_M
                gaps.append((s - half, s + half))
        pieces = _outline_pieces(outline, gaps)
        if pieces:
            jobs.append({"kind": kind, "name": f"Moulding_{kind}", "profile": profile, "width": round(width, 4),
                         "height": round(height, 4), "z": round(z, 4),
                         "paths": [[[[round(x, 4), round(y, 4)] for x, y in pts], closed] for pts, closed in pieces]})
    return [{"type": "run_code", "params": {"code": _MOULDING.format(payload=json.dumps(jobs))}}] if jobs else []


def _interior_wall_commands(spec: SceneSpec) -> list[dict[str, Any]]:
    """Multi-room buildings: a wall slab on every room edge inside the outer shell (shared with
    another room or facing empty space), with a door gap at each door (layout.room_doors) and a
    lintel above it from LAYOUT_DOOR_HEIGHT_M to the ceiling. A shared edge is built once: by the
    room on its west/south side; a room's west/south edges only get the parts no neighbour builds."""
    from ai_db_scene.layout import room_doors

    rooms = spec.setting.rooms
    if not rooms:
        return []
    w, d, h = spec.setting.bounds_m
    t, off = WALL_THICKNESS_M, frame_offset(spec)
    doors = room_doors(spec)
    mat = material_for("", interior_wall(spec.setting.wall_material or style_material(spec.setting.style, "wall")))
    cmds: list[dict[str, Any]] = []

    def slab(name: str, k: int, line: float, a: float, b: float, z0: float, z1: float) -> None:
        o = 1 - k
        centre, half = [0.0, 0.0, (z0 + z1) / 2], [0.0, 0.0, (z1 - z0) / 2]
        centre[k], half[k] = line, t / 2
        centre[o], half[o] = (a + b) / 2, (b - a) / 2
        cmds.append({"type": "create_primitive", "params": {
            "primitive": "cube", "name": name, "location": _shift(centre, off), "scale": half}})
        cmds.append({"type": "apply_material", "params": {"name": name, "material": mat}})

    def minus(lo: float, hi: float, cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        spans = []
        for a, b in sorted(cuts):
            if b <= lo or a >= hi:
                continue
            spans.append((lo, max(lo, a)))
            lo = max(lo, b)
        spans.append((lo, hi))
        return [s for s in spans if s[1] - s[0] > 1e-3]

    for r in rooms:
        for k, side, far in ((0, "east", True), (1, "north", True), (0, "west", False), (1, "south", False)):
            o = 1 - k
            line = r.offset_m[k] + (r.size_m[k] if far else 0.0)
            if abs(line - (w, d)[k]) < 1e-6 or abs(line) < 1e-6:
                continue  # the outer shell already has this wall
            lo, hi = r.offset_m[o], r.offset_m[o] + r.size_m[o]
            if far:
                cuts = [(dr["at"][o] - dr["width"] / 2, dr["at"][o] + dr["width"] / 2) for dr in doors
                        if abs(dr["at"][k] - line) < 1e-6 and lo < dr["at"][o] < hi]
            else:  # the parts a neighbour's east/north edge on this line does not already build
                cuts = [(n.offset_m[o], n.offset_m[o] + n.size_m[o]) for n in rooms
                        if n is not r and abs(n.offset_m[k] + n.size_m[k] - line) < 1e-6]
            for i, (a, b) in enumerate(minus(lo, hi, cuts)):
                slab(f"wall_{r.id}_{side}_{i}", k, line, a, b, 0.0, h)
    top = min(LAYOUT_DOOR_HEIGHT_M, h)
    for dr in doors:
        if h - top > 1e-3:
            k = 0 if dr["axis"] == "y" else 1
            c = dr["at"][1 - k]
            slab(f"lintel_{dr['id']}", k, dr["at"][k], c - dr["width"] / 2, c + dr["width"] / 2, top, h)
    return cmds


def _lookup_asset(asset_map: dict[str, Any], pl: Placement, base_id: str) -> dict[str, Any] | None:
    for key in (pl.id, base_id, pl.category):
        v = asset_map.get(key)
        if isinstance(v, dict) and v.get("filepath") and v.get("asset_name"):
            return v
    return None


#: first command of every build batch. A name the batch is about to create may still be held by an
#: object outside the current scene (in another scene, or a leftover clear_scene could not remove):
#: Blender would then call the new object "sofa.001", and every later command addressing "sofa"
#: (modify_object, apply_material, the tags below) would hit the leftover. Move such holders aside.
_RESERVE = """
import bpy, json
scene = bpy.context.scene
moved = []
for name in json.loads({payload!r}):
    held = bpy.data.objects.get(name)
    if held is not None and scene.objects.get(name) is None:
        held.name = name + "_stale"
        moved.append(held.name)
result = {{"moved": moved}}
"""

#: after clear_scene in build_scene/edit_scene: drop every object and data block nothing uses any
#: more. clear_scene's own purge (an outliner operator) does nothing without a UI context, so
#: orphans piled up over edits and pushed rebuilt names to ".001".
_PURGE = """
import bpy
removed = 0
for _ in range(8):  # an orphan object frees its mesh, the mesh its material: repeat until stable
    n = 0
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.collections, bpy.data.node_groups, bpy.data.images):
        for blk in list(coll):
            if blk.users == 0 and not blk.use_fake_user:
                coll.remove(blk)
                n += 1
    removed += n
    if not n:
        break
result = {"purged": removed}
"""

_FINALIZE = """
import bpy, json, re
scene = bpy.context.scene
suffix = re.compile(r"[.][0-9]{{3,}}$")

def fresh(base):
    # the untagged object this batch made for `base` when Blender had to rename it ("base.001")
    for o in scene.objects:
        if o.get("ai_db_id") is None and o.name.startswith(base + ".") and suffix.search(o.name):
            return o
    return None

def claim(obj, name):
    held = bpy.data.objects.get(name)
    if held is not None and held is not obj and scene.objects.get(name) is None:
        held.name = name + "_stale"  # a leftover outside the scene: the scene object gets the name
    obj.name = name

for ent in json.loads({payload!r}):
    obj = scene.objects.get(ent["name"])
    if obj is not None and obj.get("ai_db_id") not in (None, ent["name"]):
        obj = None  # an older object of another id
    if obj is None:
        obj = fresh(ent.get("asset_name") or ent["name"])
        if obj is None and ent.get("asset_name"):
            obj = scene.objects.get(ent["asset_name"])
            obj = obj if obj is not None and obj.get("ai_db_id") is None else None
        if obj is not None:
            claim(obj, ent["name"])
    if obj is None:
        continue
    if ent.get("asset_name"):
        obj.location = ent["location"]
        obj.rotation_euler[2] = ent["rot_z"]
    for k, v in ent["props"].items():
        obj[k] = v
result = {{"tagged": len(json.loads({payload!r}))}}
"""


#: leaves on foliage: geometry nodes scatter small cupped leaf cards (random turn and size) over
#: every foliage-material mesh under the given roots; the lobes stay as the darker inner mass.
_FOLIAGE = """
import bpy, json, math
from mathutils import Matrix
cfg = json.loads({payload!r})

def sock(sockets, name, kind=None):
    return next(s for s in sockets if s.name == name and s.enabled and (kind is None or s.type == kind))

def leaf_object():
    ob = bpy.data.objects.get("ai_db_leaf")
    if ob is None:
        # a pointed-oval leaf, 1 unit long, slightly cupped, in the XY plane; scaled per instance
        verts = [(0.0, -0.5, 0.0), (0.16, -0.25, 0.03), (0.2, 0.05, 0.04), (0.12, 0.3, 0.02), (0.0, 0.5, 0.0),
                 (-0.12, 0.3, 0.02), (-0.2, 0.05, 0.04), (-0.16, -0.25, 0.03), (0.0, 0.0, 0.0)]
        faces = [(8, i, (i + 1) % 8) for i in range(8)]
        me = bpy.data.meshes.new("ai_db_leaf")
        me.from_pydata(verts, [], faces)
        for p in me.polygons:
            p.use_smooth = True
        ob = bpy.data.objects.new("ai_db_leaf", me)
    # a rebuild's tidy removed the holder but left the leaf: it stood at the origin, in the middle
    # of the room ("low_table_column and ai_db_leaf intersect"): hidden and parked again every time
    coll = bpy.data.collections.get("ai_db_hidden") or bpy.data.collections.new("ai_db_hidden")
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    if ob.name not in coll.objects:
        coll.objects.link(ob)
    holder = bpy.data.objects.get("ai_db_templates")
    if holder is None:
        holder = bpy.data.objects.new("ai_db_templates", None)  # asset-marked: checks skip the template
    holder["asset"] = "templates"
    if holder.name not in coll.objects:
        coll.objects.link(holder)
    ob.parent = holder
    holder.location = (0.0, 0.0, -1000.0)  # far from every scene; instances use its shape only
    # a hidden collection is never evaluated: its matrix_world stayed at the origin, where a
    # table in the middle of the room "intersected" it (round lobby, the ring's table)
    holder.matrix_world = Matrix.Translation((0.0, 0.0, -1000.0))
    ob.matrix_world = Matrix.Translation((0.0, 0.0, -1000.0))
    coll.hide_render = True
    coll.hide_viewport = True
    return ob

def group():
    ng = bpy.data.node_groups.get("ai_db_foliage")
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new("ai_db_foliage", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    s = ng.interface.new_socket(name="Density", in_out="INPUT", socket_type="NodeSocketFloat"); s.default_value = 500.0
    s = ng.interface.new_socket(name="Leaf Size", in_out="INPUT", socket_type="NodeSocketVector")
    ng.interface.new_socket(name="Seed", in_out="INPUT", socket_type="NodeSocketInt")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    dist = N.new("GeometryNodeDistributePointsOnFaces")
    L.new(sock(gi.outputs, "Geometry"), sock(dist.inputs, "Mesh"))
    L.new(sock(gi.outputs, "Density"), sock(dist.inputs, "Density"))
    L.new(sock(gi.outputs, "Seed"), sock(dist.inputs, "Seed"))
    info = N.new("GeometryNodeObjectInfo")
    sock(info.inputs, "Object").default_value = leaf_object()
    rot = N.new("FunctionNodeRandomValue"); rot.data_type = "FLOAT_VECTOR"
    sock(rot.inputs, "Max", "VECTOR").default_value = (math.tau, math.tau, math.tau)
    sc = N.new("FunctionNodeRandomValue"); sc.data_type = "FLOAT"
    sock(sc.inputs, "Min", "VALUE").default_value = 0.6
    sock(sc.inputs, "Max", "VALUE").default_value = 1.35
    mul = N.new("ShaderNodeVectorMath"); mul.operation = "SCALE"
    L.new(sock(gi.outputs, "Leaf Size"), mul.inputs[0])
    L.new(sock(sc.outputs, "Value", "VALUE"), sock(mul.inputs, "Scale"))
    inst = N.new("GeometryNodeInstanceOnPoints")
    L.new(sock(dist.outputs, "Points"), sock(inst.inputs, "Points"))
    L.new(sock(info.outputs, "Geometry"), sock(inst.inputs, "Instance"))
    L.new(sock(rot.outputs, "Value", "VECTOR"), sock(inst.inputs, "Rotation"))
    L.new(sock(mul.outputs, "Vector"), sock(inst.inputs, "Scale"))
    join = N.new("GeometryNodeJoinGeometry")
    L.new(sock(gi.outputs, "Geometry"), sock(join.inputs, "Geometry"))
    L.new(sock(inst.outputs, "Instances"), sock(join.inputs, "Geometry"))
    L.new(sock(join.outputs, "Geometry"), sock(go.inputs, "Geometry"))
    return ng

ng = group()
done = []
meshes = []
for root in cfg["roots"]:
    r = bpy.data.objects.get(root)
    for o in ([r, *r.children_recursive] if r is not None else []):
        if o.type == "MESH" and any(m is not None and "foliage" in m.name for m in o.data.materials):
            meshes.append(o.name)
for k, name in enumerate(meshes):
    ob = bpy.data.objects[name]
    sx, sy, sz = ob.matrix_world.to_scale()
    area = sum(p.area for p in ob.data.polygons) * ((sx * sy * sz) ** (2.0 / 3.0))
    density_world = min(cfg["density"], cfg["max_leaves"] / max(area, 1e-6))
    mine = ng.copy()  # per object: Blender 5 keeps no per-modifier input values
    mine.name = f"ai_db_foliage_{{name}}"
    items = {{s.name: s for s in mine.interface.items_tree if getattr(s, "in_out", "") == "INPUT"}}
    items["Density"].default_value = density_world * ((sx * sy * sz) ** (2.0 / 3.0))  # per local unit area
    leaf = cfg["leaf_m"]
    items["Leaf Size"].default_value = (leaf / sx, leaf / sy, leaf / sz)
    items["Seed"].default_value = k
    mod = ob.modifiers.get("ai_db_foliage") or ob.modifiers.new("ai_db_foliage", "NODES")
    mod.node_group = mine
    if ob.active_material is not None:
        leaf_ob = bpy.data.objects["ai_db_leaf"]
        if not leaf_ob.data.materials:
            leaf_ob.data.materials.append(ob.active_material)
    done.append({{"name": name, "leaves": int(density_world * area)}})
result = {{"foliage": done}}
"""

#: a tree grown from its skeleton (modelling guides: trunk tapering and splitting at the crown,
#: recursive limbs - momentum + upward pull + jitter, thinner and shorter per level, side shoots,
#: kept inside a crown envelope - leaf clumps at the tips with gaps), skinned with tapered cones,
#: fitted to the placement's height and footprint. Leaves come from _FOLIAGE on the clumps.
_TREE = """
import bpy, bmesh, json, math, random
from mathutils import Vector, Matrix
cfg = json.loads({payload!r})
from mathutils import noise
rng = random.Random(cfg["seed"])
# every tree its own size and shape inside its placement box (park rows were clones)
H = cfg["dims"][2] * rng.uniform(0.82, 1.0)
RX, RY = cfg["dims"][0] / 2 * rng.uniform(0.85, 1.0), cfg["dims"][1] / 2 * rng.uniform(0.85, 1.0)
# the crown starts low (28-36 % of the height): a full head on a short trunk, not a column of
# balls on a pole (the aidb7 park's "lollipops")
crown_lo = cfg.get("crown_start", H * rng.uniform(0.28, 0.36))
cz = (crown_lo + H) / 2
segs, tips, shoots = [], [], []

def jitter(s):
    return Vector((rng.uniform(-s, s), rng.uniform(-s, s), rng.uniform(-s, s)))

def in_crown(p):
    return (p.x / RX) ** 2 + (p.y / RY) ** 2 + ((p.z - cz) / ((H - crown_lo) / 2)) ** 2

def grow(p, d, length, r, depth):
    n = 4
    for i in range(n):
        d = (d + jitter(0.28) + Vector((0, 0, 0.12 - 0.06 * depth))).normalized()  # upward pull, droop when thin
        if depth and in_crown(p) > 0.85:  # bend back into the crown envelope
            d = (d + (Vector((0, 0, cz)) - p).normalized() * 0.6).normalized()
        q = p + d * (length / n)
        r2 = r * (0.86 if i < n - 1 else 0.8)
        segs.append((p.copy(), q.copy(), r, r2))
        if depth >= 1 and i in (1, 2) and rng.random() < 0.55 and depth < 3:  # side shoots along the limb
            side = (d.cross(Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), 0.3))) + d * 0.4).normalized()
            grow(q, side, length * 0.45, r2 * 0.55, depth + 1)
        p, r = q, r2
    if depth < 3:
        for _ in range(rng.choice((2, 2, 3))):
            axis = d.orthogonal().normalized()
            axis.rotate(Matrix.Rotation(rng.uniform(0, math.tau), 3, d))
            child = d.copy()
            child.rotate(Matrix.Rotation(math.radians(rng.uniform(25, 50)), 3, axis))
            grow(p, child, length * rng.uniform(0.6, 0.75), r * 0.62, depth + 1)
    if depth >= 2:
        tips.append(p.copy())
    elif depth == 1:
        shoots.append(p.copy())

# trunk: slight lean and wobble up to the crown, then 3-4 limbs
r0 = cfg.get("trunk_r", max(0.06, 0.035 * H))
p, d = Vector((0, 0, 0)), Vector((rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 1)).normalized()
steps = 5
for i in range(steps):
    q = p + (d + jitter(0.05)).normalized() * (crown_lo / steps)
    segs.append((p.copy(), q.copy(), r0 * (1.35 if i == 0 else 1.0) * (1 - 0.06 * i), r0 * (1 - 0.06 * (i + 1))))
    p = q
limb = (H - crown_lo) * 0.5
nl = rng.choice((3, 4, 4, 5))
for k in range(nl):  # spreading limbs: the branch structure shows under the crown
    ang = k / nl * math.tau + rng.uniform(-0.35, 0.35)
    spread = rng.uniform(1.0, 1.5)
    ld = Vector((math.cos(ang) * spread, math.sin(ang) * spread, 1.0)).normalized()
    grow(p.copy(), ld, limb * rng.uniform(0.85, 1.1), r0 * 0.7, 1)

def skin(name, items, mat_slot=None):
    bm = bmesh.new()
    for a, b, ra, rb in items:
        v = b - a
        L = v.length
        if L < 1e-5:
            continue
        rot = v.to_track_quat("Z", "Y").to_matrix().to_4x4()
        m = Matrix.Translation((a + b) / 2) @ rot
        bmesh.ops.create_cone(bm, cap_ends=False, segments=8 if ra > 0.03 else 5, radius1=ra, radius2=rb, depth=L, matrix=m)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    return me

root = bpy.data.objects.new(cfg["name"], None)
bpy.context.scene.collection.objects.link(root)
root["asset"] = cfg["name"]  # one asset: the validator checks its parts together, not against each other
wood = bpy.data.objects.new(cfg["name"] + "_wood", skin(cfg["name"] + "_wood", segs))
# the crown: leaf masses of 2-4 irregular, flattened lobes round each twig end (the leaf cards of
# the foliage pass cover them), a few big lobes filling the core so it reads as one crown with
# a bumpy outline; nothing low in the crown, so the limbs show where they fork
bm = bmesh.new()
cr = max(0.18, 0.08 * min(RX, RY) * 2 + 0.03 * H) * cfg.get("clump", 1.0)
floor_z = crown_lo + 0.15 * (H - crown_lo)

def lobe(c, rad, flat):
    m = Matrix.Translation(c) @ Matrix.Rotation(rng.uniform(0, math.tau), 4, "Z") @ Matrix.Diagonal((1.0, 1.0, flat, 1.0))
    vs = bmesh.ops.create_icosphere(bm, subdivisions=2, radius=rad, matrix=m)["verts"]
    off = Vector((rng.uniform(0, 50), rng.uniform(0, 50), rng.uniform(0, 50)))
    for v in vs:  # lumpy, not a ball: coherent noise pushes the surface in and out
        d = v.co - c
        v.co = c + d * (1.0 + 0.32 * noise.noise(d / rad * 1.6 + off))

anchors = [t for t in tips if t.z > floor_z] + [q for q in shoots if q.z > floor_z + 0.1 * (H - crown_lo)]
for t in anchors:
    if rng.random() < 0.15:  # gaps: sky and branches show through
        continue
    for _ in range(rng.choice((2, 3, 3, 4))):
        rad = cr * rng.uniform(0.55, 1.0)
        lobe(t + jitter(cr * 0.6) + Vector((0, 0, rad * 0.2)), rad, rng.uniform(0.6, 0.8))
core = min(RX, RY, (H - crown_lo) / 2)
for _ in range(rng.choice((3, 4, 5))):
    c = Vector((rng.uniform(-0.35, 0.35) * RX, rng.uniform(-0.35, 0.35) * RY, cz + rng.uniform(0.0, 0.25) * (H - cz)))
    lobe(c, core * rng.uniform(0.5, 0.65), rng.uniform(0.7, 0.85))
me = bpy.data.meshes.new(cfg["name"] + "_leaves")
bm.to_mesh(me)
bm.free()
leaves = bpy.data.objects.new(cfg["name"] + "_leaves", me)
for ob in (wood, leaves):
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = root
# fit: height to H, crown within the footprint
pts = [Vector(c) for ob in (wood, leaves) for c in ((ob.matrix_world @ Vector(v.co)) for v in ob.data.vertices)]
zmax = max(p.z for p in pts)
xr = max(abs(p.x) for p in pts) or 1.0
yr = max(abs(p.y) for p in pts) or 1.0
sz = H / zmax
for ob in (wood, leaves):
    ob.scale = (min(sz, RX / xr), min(sz, RY / yr), sz)
root.location = cfg["location"]
root.rotation_euler[2] = math.radians(cfg.get("rot_z", 0.0))
result = {{"segments": len(segs), "tips": len(tips)}}
"""

_TIDY = """
import bpy, json
ids = set(json.loads({payload!r}))
ours = ("Room_", "floor_line_", "ground_beyond", "ai_db_fill", "Room_Planks", "ai_db_daylight", "Moulding_")
lights = {{i + "_light" for i in ids}}
gone = []

def drop(o):
    for c in list(o.children_recursive):
        gone.append(c.name)
        bpy.data.objects.remove(c, do_unlink=True)
    gone.append(o.name)
    bpy.data.objects.remove(o, do_unlink=True)

for o in list(bpy.context.scene.objects):
    if o.name not in bpy.data.objects:
        continue
    factory_cube = (o.name == "Cube" and o.type == "MESH" and len(o.data.vertices) == 8 and o.get("ai_db_id") is None
                    and all(abs(d - 2.0) < 1e-4 for d in o.dimensions) and o.location.length < 1e-4)
    # any root an earlier ai-db build tagged belongs to the scene being replaced; untagged
    # (user-made) objects are kept
    if factory_cube or o.name.startswith(ours) or o.name in lights or (o.get("ai_db_id") is not None and o.parent is None) \
            or o.get("ai_db_terrain"):
        drop(o)
result = {{"removed": gone[:50], "count": len(gone)}}
"""

_ROOM_MATERIAL = """
import bpy, math
mat = bpy.data.materials.get("ai_db_room") or bpy.data.materials.new("ai_db_room")
mat.diffuse_color = (0.82, 0.8, 0.76, 1.0)
if mat.use_nodes and mat.node_tree.nodes.get("Principled BSDF"):
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.82, 0.8, 0.76, 1.0)
n = 0
for o in bpy.context.scene.objects:
    if o.type == "MESH" and (o.name.startswith("Room_") or o.name == "floor") and not any(o.data.materials):
        o.data.materials.append(mat)
        n += 1
smooth = 0
for o in bpy.context.scene.objects:  # curved walls and their mouldings: one surface, not facets
    if o.type == "MESH" and (o.name.startswith("Room_") or o.get("moulding")):
        o.data.shade_smooth()
        o.data.set_sharp_from_angle(angle=math.radians(SMOOTH_DEG))
        smooth += 1
result = {"room_surfaces": n, "smooth_shaded": smooth}
""".replace("SMOOTH_DEG", str(_ROOM_SMOOTH_ANGLE_DEG))


#: category -> (generator family, params from the placement's [w, d, h]). Parametric furniture
#: with its own part materials (fabric seat, wooden legs) instead of a textured box.
_GENERATED: dict[str, tuple[str, Any]] = {
    "sofa": ("sofa", lambda w, d, h: {"width": w, "depth": d}),
    "armchair": ("armchair", lambda w, d, h: {"width": w, "depth": d}),
    "dining_chair": ("dining_chair", lambda w, d, h: {"seat_width": w * 0.9, "seat_depth": d * 0.85}),
    "chair": ("dining_chair", lambda w, d, h: {"seat_width": w * 0.9, "seat_depth": d * 0.85}),
    "dining_table": ("table", lambda w, d, h: {"width": w, "depth": d, "height": h}),
    "coffee_table": ("table", lambda w, d, h: {"width": w, "depth": d, "height": h}),
    "side_table": ("table", lambda w, d, h: {"width": w, "depth": d, "height": h}),
    "conference_table": ("table", lambda w, d, h: {"width": w, "depth": d, "height": h}),
    "desk": ("desk", lambda w, d, h: {"width": w, "depth": d, "height": h}),
    "bookshelf": ("bookshelf", lambda w, d, h: {"width": w, "depth": d, "height": h}),
    # the bed generator sizes by mattress (twin/full/queen/king): pick the one the width fits
    "bed": ("bed", lambda w, d, h: {"size": "twin" if w < 1.2 else "full" if w < 1.5 else "queen" if w < 1.8 else "king"}),
}
#: setting.style -> the generator preset a family starts from (a seed then varies it)
_STYLE_PRESETS: dict[str, dict[str, str]] = {
    "classic": {"sofa": "chesterfield_like", "armchair": "wingback_like", "bed": "tufted_queen"},
    "modern": {"sofa": "mid_century", "armchair": "modern_lounge", "bed": "platform_king"},
    "scandinavian": {"sofa": "compact_loveseat", "armchair": "modern_lounge", "dining_chair": "scandi_slat",
                     "bed": "wooden_twin"},
    "rustic": {"sofa": "lounge_deep", "armchair": "club_chair", "bed": "wooden_twin"},
    "industrial": {"sofa": "lounge_deep", "armchair": "club_chair"},
}


def variation_key(spec: SceneSpec) -> list[Any]:
    """What makes one scene's furniture differ from another's: setting + object list."""
    return [spec.setting.kind, list(spec.setting.bounds_m), spec.setting.style,
            sorted([o.id, o.category] for o in spec.objects)]


def _variation(spec: SceneSpec, family: str, base_id: str, key: list[Any] | None = None) -> dict[str, Any]:
    """Seed (and style preset) for a generated piece, so scenes do not all get the generator's
    default sofa. The seed comes from the scene (``key``: the first build's ``variation_key``)
    and the object's base id: a different scene gets a different sofa, a rebuild or edit_scene
    of the same scene keeps it (edit_scene passes the stored key: the edited object list used
    to reseed every untouched piece), and the copies of one spec object (armchair_1,
    armchair_2) stay a matching pair."""
    sig = json.dumps([*(key if key is not None else variation_key(spec)), base_id])
    out: dict[str, Any] = {"seed": int(hashlib.sha1(sig.encode()).hexdigest()[:7], 16)}  # < 2**31: a C int
    from ai_db_scene.materials import style_palette

    preset = _STYLE_PRESETS.get(style_palette(spec.setting.style) or "", {}).get(family)
    if preset:
        out["preset"] = preset
    return out


#: recipes with their own finish (painted steel, wood, cardboard loads): a plain material of
#: these kinds on them means "made of that", not "repaint it all" (qwen aidb7 warehouse: "metal"
#: racks and forklift came out chrome, loads and all)
_OWN_FINISH: dict[str, tuple[str, ...]] = {
    "pallet_rack": ("metal", "steel", "chrome"), "forklift": ("metal", "steel", "chrome"),
    "safety_barrier": ("metal", "steel", "chrome"), "conveyor": ("metal", "steel", "chrome"),
    "shelving_unit": ("metal", "steel", "chrome"), "workbench": ("metal", "steel", "chrome", "wood"),
    "pallet": ("wood",), "crate": ("wood",), "trash_can": ("metal", "steel")}


#: what makes a setting a warehouse / stock room: its crates are cartons
WAREHOUSE_CATEGORIES = ("pallet_rack", "forklift", "pallet", "conveyor", "loading_dock", "shelving_unit")
WAREHOUSE_STYLES = ("industrial", "warehouse", "factory", "logistics", "storage")


def context_recipe(spec: SceneSpec, category: str, ospec: Any) -> str | None:
    """The recipe a category gets in this setting instead of its own (recipes._KINDS), else None:
    a crate in a warehouse is a cardboard carton unless the spec makes it wooden (a wood variant or
    tint, or 'wood' in its id; a bare 'wood' is the material every crate gets: aidb7 review, the
    qwen warehouse's crates read as wooden props on the pallets); an outdoor bin a park litter bin."""
    if category == "trash_can" and spec.setting.kind == "outdoor":
        return "litter_bin"
    if category != "crate":
        return None
    cats = {o.category for o in spec.objects}
    style = str(spec.setting.style or "").lower()
    if not cats & set(WAREHOUSE_CATEGORIES) and not any(w in style for w in WAREHOUSE_STYLES):
        return None
    mat = str(getattr(ospec, "material", None) or "")
    oid = str(getattr(ospec, "id", "")).lower()
    if (mat.startswith("wood") and mat != "wood") or "wood" in oid:
        return None
    return "cardboard_box"


def _plain_finish(category: str, material: str | None) -> bool:
    """Is ``material`` a plain kind (no tint) that the category's recipe already shows?"""
    if not material or "#" in material or category not in _OWN_FINISH:
        return False
    return material.split(":")[0] in _OWN_FINISH[category]


GEOMETRY_MODES = ("generated", "boxes")
#: recipe categories built with a per-object seed (recipes._VARIANTS): every rock its own shape
_SEEDED_RECIPES = ("rock",)
#: parts a spec material must not repaint on generated furniture (wooden legs, the books)
_KEEP_PARTS = {**{c: ("leg", "arm_frame", "arm_rib", "base_rail", "plinth", "pillow") for c in ("sofa", "armchair")},
               "bookshelf": ("book",),
               # beach furniture: a "fabric" lounger is its cushions on a teak frame, a parasol's canopy
               "sun_lounger": ("leg", "rail", "end_", "slat", "back_frame", "back_strut"),
               "parasol": ("base", "pole", "hub", "finial"),
               # a "wood" park bench is its slats on the cast-iron frames; a bin keeps its dark liner
               "bench": ("frame_",), "trash_can": ("liner", "mouth"),
               "fountain": ("water",),  # a "stone" fountain is stone round its water
               # lamps: a "brass lamp" is brass metalwork under a fabric shade and a bulb
               **{c: ("shade", "bulb", "diffuser") for c in ("floor_lamp", "table_lamp", "pendant_lamp",
                                                             "wall_sconce", "ceiling_light")}}

#: after import_file into collection "ai_db_asset_<id>": one root named <id> (an empty parenting
#: the file's roots), scaled uniformly to fit the placement, base centre on the floor point,
#: turned to the placement's yaw. Files come at any scale and origin; the layout does not.
_ADOPT = """
import bpy, json
from mathutils import Vector
ent = json.loads({payload!r})
coll = bpy.data.collections.get(ent["collection"])
objs = list(coll.objects) if coll else []
roots = [o for o in objs if o.parent not in objs]
root = bpy.data.objects.new(ent["name"], None)
(coll or bpy.context.scene.collection).objects.link(root)
for o in roots:
    o.parent = root
bpy.context.view_layer.update()
pts = [o.matrix_world @ Vector(c) for o in objs if o.type == "MESH" for c in o.bound_box]
if pts:
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    size = hi - lo
    s = min(ent["dims"][i] / size[i] for i in range(3) if size[i] > 1e-6)
    for o in roots:  # recentre the content on the root's origin: XY centre, base at z=0
        o.location -= Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    root.scale = (s, s, s)
root.location = ent["location"]
root.rotation_euler[2] = ent["rot_z"]
result = {{"root": root.name, "parts": len(objs)}}
"""


_RESKIN = """
import bpy, sys, json
mod = next(m for n, m in sys.modules.items() if n.endswith(".materials") and hasattr(m, "resolve_material"))
done = {{}}
for ent in json.loads({payload!r}):
    root = bpy.data.objects.get(ent["name"])
    if root is None:
        continue
    mat = mod.resolve_material(ent["material"])
    parts = [o for o in [root, *root.children_recursive] if o.type == "MESH"
             and not any(k in o.name.lower() for k in ent["keep"])]
    for o in parts:
        o.data.materials.clear()
        o.data.materials.append(mat)
    done[ent["name"]] = len(parts)
result = {{"reskinned": done}}
"""


_WEAR = """
import json
for ent in json.loads(PAYLOAD):
    root = bpy.data.objects.get(ent["name"])
    if root is None:
        continue
    copies = {}
    for o in [root, *root.children_recursive]:
        for slot in (o.material_slots if o.type == "MESH" else ()):
            m = slot.material
            if m is None or not m.use_nodes or not any(n.type == "BSDF_PRINCIPLED" for n in m.node_tree.nodes):
                continue
            if m.name not in copies:  # library materials are shared: wear a copy
                worn = m.copy()
                worn.name = f"{m.name}_{ent['preset']}"
                copies[m.name] = add_grunge(worn, amount=ent["amount"], scale=ent["scale"])
            slot.material = copies[m.name]
    worn_names.append(ent["name"])
result = {"worn": worn_names}
"""


def _wear_command(wear: list[dict[str, Any]]) -> dict[str, Any]:
    """run_code: the pbr-shader-builder skill's library, then its grunge on each listed object."""
    from ai_db_scene import skills_dir

    text = (Path(skills_dir()) / "pbr-shader-builder" / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if m is None:
        raise AiDbError("pbr-shader-builder skill has no code block")
    code = m.group(1) + f"PAYLOAD = {json.dumps(wear)!r}\nworn_names = []\n" + _WEAR
    return {"type": "run_code", "params": {"code": code}}


def build_layout_commands(
    spec: SceneSpec, placements: list[Placement], asset_map: dict[str, Any] | None = None,
    geometry: str = "generated", providers: list[Any] | None = None, variation: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Blender batch for room shell + one object per placement. Returns (commands, created names).

    Per placement, the first that applies: an ``asset_map`` entry; with ``geometry="generated"``
    a model file from an asset provider (see providers.py; none unless configured), parametric
    furniture for the categories in ``_GENERATED`` (unless a user recipe was saved for it with
    overwrite=true), a part recipe; else a textured box.
    """
    import math

    from ai_db_scene import terrain, vegetation
    from ai_db_scene.furniture_variants import CATEGORY_FAMILY, fit_command, uses_families
    from ai_db_scene.furniture_variants import piece as family_piece
    from ai_db_scene.providers import AssetFile, find_asset

    asset_map = asset_map or {}
    provided: dict[str, AssetFile | None] = {}
    placements = terrain.settle(spec, placements)  # on the shore slope objects stand on the ground

    def provider_file(category: str) -> AssetFile | None:
        if geometry != "generated" or not providers:
            return None
        if category not in provided:  # one lookup (and download) per category per build
            provided[category] = find_asset(providers, category)
        return provided[category]

    priors = load_priors()
    heroes = {o.id for o in spec.objects if o.hero}
    # the style's palette counts as a choice: it wins over generator/category defaults
    materials = {o.id: o.material or style_material(spec.setting.style, o.category) for o in spec.objects}
    details = {o.id: o.detail for o in spec.objects}
    cmds = _room_commands(spec, placements) + _interior_wall_commands(spec)
    names = terrain.object_names(spec) or [c["params"].get("name", "room_shell") for c in cmds
                                           if c["type"] != "apply_material"]
    cmds += moulding_commands(spec, placements)
    if placements:
        reserve = [pl.id for pl in placements] + [
            a["asset_name"] for pl in placements
            if (a := _lookup_asset(asset_map, pl, _SUFFIX.sub("", pl.id))) is not None]
        reserve += [f.asset_name for pl in placements
                    if (f := provider_file(pl.category)) is not None and f.asset_name]
        if spec.setting.kind != "single_asset":  # shell surfaces and the fill are addressed by name too
            reserve += ["Room_Floor", "Room_Walls", "floor", "ai_db_fill", *terrain.object_names(spec)]
        cmds.insert(0, {"type": "run_code", "params": {"code": _RESERVE.format(payload=json.dumps(reserve))}})
    off = frame_offset(spec)
    reskin: list[dict[str, Any]] = []
    tag: list[dict[str, Any]] = []
    fit: list[dict[str, Any]] = []
    by_id = {o.id: o for o in spec.objects}
    # pieces without a design family hash the key as before: the marker leaves them unchanged
    plain_key = variation[:-1] if variation is not None and uses_families(variation) else variation
    for pl in placements:
        base = _SUFFIX.sub("", pl.id) if pl.id not in {o.id for o in spec.objects} else pl.id
        prior = priors.get(pl.category)
        props = {
            "ai_db_id": pl.id,
            "category": pl.category,
            "hero": base in heroes,
            "support": prior.support if prior else "floor",
            "detail": details.get(base, "standard"),
        }
        if pl.facing_target:
            props["facing_target"] = pl.facing_target
        asset = _lookup_asset(asset_map, pl, base)
        found = None if asset else provider_file(pl.category)
        if found is not None and found.kind == "blend":
            asset = {"filepath": found.filepath, "asset_name": found.asset_name}
        if found is not None:
            props["asset_source"] = found.source
        if found is not None and found.kind == "gltf":
            loc = _shift(pl.location, off)
            coll = f"ai_db_asset_{pl.id}"
            cmds.append({"type": "import_file", "params": {"filepath": found.filepath, "collection": coll}})
            cmds.append({"type": "run_code", "params": {"code": _ADOPT.format(payload=json.dumps({
                "name": pl.id, "collection": coll, "dims": list(pl.dims),
                "location": [loc[0], loc[1], loc[2] - pl.dims[2] / 2],
                "rot_z": math.radians(pl.rotation_deg_z)}))}})
            tag.append({"name": pl.id, "location": loc, "rot_z": 0.0, "props": props})
            if materials.get(base):
                reskin.append({"name": pl.id, "material": material_for(pl.category, materials[base]),
                               "keep": list(_KEEP_PARTS.get(pl.category, ()))})
        elif asset:
            cmds.append({"type": "append_asset", "params": {
                "filepath": asset["filepath"], "asset_name": asset["asset_name"], "location": _shift(pl.location, off)}})
            tag.append({"name": pl.id, "asset_name": asset["asset_name"], "location": _shift(pl.location, off),
                        "rot_z": math.radians(pl.rotation_deg_z), "props": props})
        elif geometry == "generated" and pl.category == "tree" and not user_overrides("tree"):
            loc = _shift(pl.location, off)
            seed = (_variation(spec, "tree", base, plain_key)["seed"] + zlib.crc32(pl.id.encode())) % (2**31)  # every tree differs
            cmds.append({"type": "run_code", "params": {"code": _TREE.format(payload=json.dumps({
                "name": pl.id, "dims": list(pl.dims), "seed": seed,
                "location": [loc[0], loc[1], loc[2] - pl.dims[2] / 2], "rot_z": pl.rotation_deg_z}))}})
            cmds.append({"type": "apply_material", "params": {"name": f"{pl.id}_wood", "material": TREE_BARK_MATERIAL}})
            cmds.append({"type": "apply_material", "params": {
                "name": f"{pl.id}_leaves", "material": material_for("tree", materials.get(base))}})
            tag.append({"name": pl.id, "location": loc, "rot_z": 0.0, "props": props})
        elif geometry == "generated" and (vegetation.has_plant(pl.category) or pl.category in terrain.OVER_WATER) \
                and not user_overrides(pl.category):  # palms, ferns, tropical bushes; boardwalks on posts
            loc = _shift(pl.location, off)
            seed = (_variation(spec, pl.category, base, plain_key)["seed"] + zlib.crc32(pl.id.encode())) % (2**31)
            chosen = materials.get(base)
            if pl.category in terrain.OVER_WATER:
                cmds += terrain.boardwalk_commands(spec, pl, off, seed, material_for(pl.category, chosen)
                                                   if chosen else None)
            else:
                cmds += vegetation.plant_commands(pl.category, pl.id, pl.dims, [loc[0], loc[1], loc[2] - pl.dims[2] / 2],
                                                  pl.rotation_deg_z, seed, material_for(pl.category, chosen)
                                                  if chosen else None)
            tag.append({"name": pl.id, "location": loc, "rot_z": 0.0, "props": props})
        elif geometry == "generated" and pl.category in _GENERATED and not user_overrides(pl.category):
            loc = _shift(pl.location, off)
            ospec = by_id.get(base)
            if uses_families(variation) and pl.category in CATEGORY_FAMILY:  # a design family (furniture_variants)
                got = family_piece(variation or [], pl.category, pl.dims, spec.setting.style,
                                   ospec.variant if ospec else None, ospec.room if ospec else None)
                gen = {"family": got["family"], "params": got["params"], "seed": got["seed"]}
                props["design"] = got["design"]
                fit.append({"name": pl.id, "dims": list(pl.dims)})
                if ospec is not None and ospec.material is None and got["family"] in ("sofa", "armchair"):
                    materials[base] = None  # the family's upholstery beats the style palette's one colour
            else:
                family, params = _GENERATED[pl.category]
                gen = {"family": family, "params": params(*pl.dims), **_variation(spec, family, base, plain_key)}
            if gen["family"] == "table" and ospec is not None and ospec.shape:  # "round tables": a round top
                gen["params"] = {**gen["params"], "top_shape": ospec.shape}
            cmds.append({"type": "generate_asset", "params": {
                **gen, "name": pl.id,
                "location": [loc[0], loc[1], loc[2] - pl.dims[2] / 2],  # base centre
                "rotation": [0.0, 0.0, pl.rotation_deg_z]}})
            tag.append({"name": pl.id, "location": loc, "rot_z": 0.0, "props": props})
            if materials.get(base):  # the spec chose a material: it wins over the generator's
                reskin.append({"name": pl.id, "material": material_for(pl.category, materials[base]),
                               "keep": list(_KEEP_PARTS.get(pl.category, ()))})
        elif geometry == "generated" and has_recipe(pl.category):
            loc = _shift(pl.location, off)
            kind = context_recipe(spec, pl.category, by_id.get(base))
            cmds.append({"type": "build_asset", "params": {"spec": recipe_spec(
                pl.category, pl.id, pl.dims, seed=pl.id if pl.category in _SEEDED_RECIPES else None, kind=kind)}})
            if kind == "cardboard_box" and _plain_finish(pl.category, materials.get(base)):
                materials[base] = None  # a warehouse "wood" crate: a carton in its own cardboard
            if pl.category in _SEEDED_RECIPES and materials.get(base) == "stone":
                materials[base] = None  # plain stone: keep each rock's own tones
            if _plain_finish(pl.category, materials.get(base)):
                materials[base] = None  # "metal" racks: the recipe's painted steel, not chrome all over
            cmds.append({"type": "modify_object", "params": {
                "name": pl.id, "location": [loc[0], loc[1], loc[2] - pl.dims[2] / 2],
                "rotation": [0.0, 0.0, pl.rotation_deg_z]}})
            tag.append({"name": pl.id, "location": loc, "rot_z": 0.0, "props": props})
            if materials.get(base):
                reskin.append({"name": pl.id, "material": material_for(pl.category, materials[base]),
                               "keep": list(_KEEP_PARTS.get(pl.category, ()))})
        else:
            cmds.append({"type": "create_primitive", "params": {
                "primitive": "cube", "name": pl.id, "location": _shift(pl.location, off),
                "scale": [pl.dims[0] / 2, pl.dims[1] / 2, pl.dims[2] / 2]}})
            if pl.rotation_deg_z:
                cmds.append({"type": "modify_object", "params": {
                    "name": pl.id, "rotation": [0.0, 0.0, pl.rotation_deg_z]}})
            tag.append({"name": pl.id, "location": _shift(pl.location, off), "rot_z": 0.0, "props": props})
            cmds.append({"type": "apply_material", "params": {
                "name": pl.id, "material": material_for(pl.category, materials.get(base))}})
        names.append(pl.id)
    _strip_shelf_loads(spec, cmds)
    if fit:
        cmds.append(fit_command(fit))
    if tag:
        cmds.append({"type": "run_code", "params": {"code": _FINALIZE.format(payload=json.dumps(tag))}})
    if reskin:
        cmds.append({"type": "run_code", "params": {"code": _RESKIN.format(payload=json.dumps(reskin))}})
    if spec.setting.kind != "single_asset":
        st = spec.setting
        indoor_style = st.style if st.kind == "indoor" else None  # outdoor ground stays ground
        floor = st.floor_material or style_material(indoor_style, "floor") or FLOOR_MATERIALS.get(st.kind, "plaster")
        surfaces = [("floor" if st.kind != "indoor" and st.shape == "rect" else "Room_Floor", floor)]
        if terrain.terrain_of(spec) is not None:
            surfaces = []  # the terrain shades itself (sand, grass, ... with the wet shore band)
        elif st.kind == "outdoor":
            surfaces.append(("ground_beyond", floor))
        if st.kind == "indoor":
            surfaces.append(("Room_Walls", interior_wall(st.wall_material or style_material(st.style, "wall"))))
        for obj, mat in surfaces:
            mat_s = material_for("", mat)
            if mat_s.startswith("wood") and "scale=" not in mat_s:  # floor-sized figure, not room-sized swirls
                mat_s += f":scale={SCENE_PLANK_GRAIN_SCALE}"
            cmds.append({"type": "apply_material", "params": {"name": obj, "material": mat_s}})
        cmds += _floor_and_rug_detail(spec, placements, floor)
        # anything else of the shell (ceiling, trim) gets the neutral room material
        cmds.append({"type": "run_code", "params": {"code": _ROOM_MATERIAL}})
    ids = {o.id for o in spec.objects}
    wear = [{"name": pl.id, "preset": chosen[4:], "amount": p[1], "scale": p[2]}
            for pl in placements
            if (chosen := materials.get(pl.id if pl.id in ids else _SUFFIX.sub("", pl.id)))
            and (p := pbr_preset(chosen))]
    if spec.setting.kind != "single_asset":
        wear += [{"name": obj, "preset": mat[4:], "amount": p[1], "scale": p[2]}
                 for obj, mat in surfaces if (p := pbr_preset(mat))]
    if wear:
        cmds.append(_wear_command(wear))
    if spec.setting.kind == "outdoor":  # a park's paved plaza round its fountain (terrain.plaza_commands)
        pave, _note = terrain.plaza_material(spec.setting.floor_material)
        cmds += terrain.plaza_commands(spec, placements, off, material_for("", pave))
    cmds += lighting_commands(spec, placements, off)
    plants = [pl.id for pl in placements if pl.category in FOLIAGE_CATEGORIES
              and not (vegetation.has_plant(pl.category) and not user_overrides(pl.category))]  # grown leaves
    if plants and SCENE_FOLIAGE_DENSITY > 0 and geometry == "generated":
        cmds.append({"type": "run_code", "params": {"code": _FOLIAGE.format(payload=json.dumps({
            "roots": plants, "density": SCENE_FOLIAGE_DENSITY, "max_leaves": SCENE_FOLIAGE_MAX_LEAVES,
            "leaf_m": SCENE_FOLIAGE_LEAF_M}))}})
    cmds.append(camera_command(spec, placements, off))
    shift_cmd = camera_shift_command(spec, placements, off)
    if shift_cmd:
        cmds.append(shift_cmd)
    return cmds, names


#: built-in stock of storage recipes (recipes._rack, shelving_unit): dropped when the spec puts
#: its own objects on the shelves, or the racked pallets stood inside the recipe's loads
_SHELF_STOCK = ("load_", "box_")


def _strip_shelf_loads(spec: SceneSpec, cmds: list[dict[str, Any]]) -> None:
    from ai_db_scene.layout import SHELF_LEVELS

    cats = {o.id: o.category for o in spec.objects}
    loaded = {r.b for r in spec.relations if r.rel == "on" and cats.get(r.b) in SHELF_LEVELS}
    for c in cmds:
        asset = c.get("params", {}).get("spec") if c.get("type") == "build_asset" else None
        name = str(asset.get("asset")) if isinstance(asset, dict) else ""
        if isinstance(asset, dict) and (name in loaded or _SUFFIX.sub("", name) in loaded):
            asset["parts"] = [q for q in asset.get("parts", []) if not str(q.get("id", "")).startswith(_SHELF_STOCK)]


def floor_area_m2(spec: SceneSpec) -> float:
    """Floor area of the setting: rectangle, ellipse inscribed in the bounds, or polygon."""
    import math

    st = spec.setting
    w, d, _h = st.bounds_m
    if st.shape == "polygon" and len(st.points) >= 3:
        pts = st.points
        return abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts)))) / 2
    if st.shape == "ellipse":
        return math.pi * w * d / 4
    return w * d


def _room_centre(spec: SceneSpec, off: tuple[float, float, float]) -> tuple[float, float]:
    st = spec.setting
    if st.shape == "polygon" and st.points:
        xs, ys = [p[0] for p in st.points], [p[1] for p in st.points]
        return ((min(xs) + max(xs)) / 2 + off[0], (min(ys) + max(ys)) / 2 + off[1])
    w, d, _h = st.bounds_m
    return (w / 2 + off[0], d / 2 + off[1])


def indoor_exposure(area_m2: float) -> float:
    """Exposure (stops) for a room: windows light a small room well, the middle of a big one
    much less, so each doubling of floor area over the reference adds a fraction of a stop."""
    import math

    lo, hi = SCENE_EXPOSURE_RANGE
    stops = SCENE_EXPOSURE_INDOOR + SCENE_EXPOSURE_PER_DOUBLING * math.log2(max(area_m2, 1.0) / SCENE_REF_FLOOR_M2)
    return round(min(max(stops, lo), hi), 2)


_AREA_SHAPE = """
import bpy
o = bpy.data.objects.get({name!r})
if o is not None and o.type == "LIGHT" and o.data.type == "AREA":
    o.data.shape = "RECTANGLE"
    o.data.size, o.data.size_y = {sx!r}, {sy!r}
result = {{"fill": o.name if o else None}}
"""


def lighting_commands(spec: SceneSpec, placements: list[Placement], off: tuple[float, float, float]) -> list[dict[str, Any]]:
    """Daylight (sky world, sun, a portal per window), an interior fill, and a warm point light in
    every lamp. A lone sun left interiors flat and dark (judges' second most common complaint),
    and window light alone does not reach the middle of a large room: indoors the exposure grows
    with the floor area and a soft ceiling area light (energy by floor area) fills the room."""
    indoor = spec.setting.kind == "indoor"
    area = floor_area_m2(spec)
    if spec.setting.kind == "outdoor":
        return outdoor_lighting_commands(spec, placements, off)
    cmds: list[dict[str, Any]] = [{"type": "daylight_rig", "params": {
        "sun_elevation": 40.0, "sun_azimuth": 135.0, "strength": SCENE_SUN_ENERGY, "sky": "nishita",
        "window_portals": indoor, "sky_strength": 0.8, "name": "ai_db_daylight",
        "exposure": indoor_exposure(area) if indoor else SCENE_EXPOSURE_OUTDOOR}}]
    if indoor:
        w, d, h = spec.setting.bounds_m
        lo, hi = SCENE_FILL_W_RANGE
        cx, cy = _room_centre(spec, off)
        cmds.append({"type": "add_light", "params": {
            "light_type": "AREA", "name": "ai_db_fill", "energy": round(min(max(SCENE_FILL_W_PER_M2 * area, lo), hi), 1),
            "color": [1.0, 0.96, 0.9], "location": [cx, cy, h - SCENE_FILL_CEILING_GAP_M],
            "rotation": [0.0, 0.0, 0.0]}})  # shines along -Z: down
        cmds.append({"type": "run_code", "params": {"code": _AREA_SHAPE.format(
            name="ai_db_fill", sx=round(w * SCENE_FILL_SIZE_FRACTION, 3), sy=round(d * SCENE_FILL_SIZE_FRACTION, 3))}})
    for pl in placements:
        watts = SCENE_LAMP_W.get(pl.category)
        if watts is None:
            continue
        x, y, z = _shift(pl.location, off)
        cmds.append({"type": "add_light", "params": {
            "light_type": "POINT", "name": f"{pl.id}_light", "energy": watts, "color": [1.0, 0.82, 0.62],
            "location": [x, y, z + pl.dims[2] / 2 - 0.15]}})
    return cmds


def outdoor_sun_azimuth(spec: SceneSpec, placements: list[Placement], off: tuple[float, float, float]) -> float:
    """Degrees toward the sun (CCW from +X): behind the build camera, a little to its side, so the
    fronts it looks at are sunlit (a sun in the lens left the beach's fronts in shade)."""
    import math

    if not placements:
        return 135.0
    yaw = camera_poses(spec, placements, off)[0]["rotation"][2]
    view = 90.0 + yaw  # rotation [90, 0, yaw] looks along +Y turned by yaw
    return round(math.fmod(view + 180.0 + SCENE_SUN_BEHIND_CAMERA_DEG + 360.0, 360.0), 1)


#: the outdoor rig's sun colour and the view look (daylight_rig sets a neutral sun, no look)
_OUTDOOR_RIG = """
import bpy
scene = bpy.context.scene
sun = next((o for o in scene.objects if o.type == "LIGHT" and o.data.type == "SUN" and o.name.startswith("ai_db_daylight")), None)
if sun is not None:
    sun.data.color = {color!r}
try:
    scene.view_settings.look = {look!r}
except TypeError:
    pass
result = {{"sun": sun.name if sun else None, "look": scene.view_settings.look}}
"""


def outdoor_lighting_commands(spec: SceneSpec, placements: list[Placement],
                              off: tuple[float, float, float]) -> list[dict[str, Any]]:
    """A clear sunny day: a warm high sun behind the camera, hard shadows, a weak sky fill, AgX with
    more contrast; warm point lights in street lamps. aidb7 beach and park were dusky: the sky at
    0.8 swamped a 4 W/m² sun, the sand rendered grey and the lagoon's turquoise did not show."""
    cmds: list[dict[str, Any]] = [{"type": "daylight_rig", "params": {
        "sun_elevation": SCENE_SUN_ELEVATION_OUTDOOR, "sun_azimuth": outdoor_sun_azimuth(spec, placements, off),
        "strength": SCENE_SUN_ENERGY_OUTDOOR, "sky": "nishita", "window_portals": False,
        "sky_strength": SCENE_SKY_STRENGTH_OUTDOOR, "name": "ai_db_daylight", "exposure": SCENE_EXPOSURE_OUTDOOR}},
        {"type": "run_code", "params": {"code": _OUTDOOR_RIG.format(color=list(SCENE_SUN_COLOR_OUTDOOR),
                                                                    look=SCENE_VIEW_LOOK_OUTDOOR)}}]
    for pl in placements:
        watts = SCENE_LAMP_W.get(pl.category)
        if watts is None:
            continue
        x, y, z = _shift(pl.location, off)
        cmds.append({"type": "add_light", "params": {
            "light_type": "POINT", "name": f"{pl.id}_light", "energy": watts, "color": [1.0, 0.82, 0.62],
            "location": [x, y, z + pl.dims[2] / 2 - 0.15]}})
    return cmds


def _inside_polygon(x: float, y: float, pts: Any) -> bool:
    inside = False
    for i in range(len(pts)):
        (x0, y0), (x1, y1) = pts[i - 1], pts[i]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def footprint(spec: SceneSpec, off: tuple[float, float, float], n: int = 48) -> list[list[float]]:
    """The room outline in the Blender frame, counter-clockwise (an ellipse as n points)."""
    import math

    st = spec.setting
    w, d, _h = st.bounds_m
    if st.shape == "polygon" and len(st.points) >= 3:
        pts = [[float(p[0]) + off[0], float(p[1]) + off[1]] for p in st.points]
        area = sum(pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1] for i in range(len(pts)))
        return pts if area > 0 else pts[::-1]
    if st.shape == "ellipse":
        return [[w / 2 * (1 + math.cos(2 * math.pi * k / n)) + off[0], d / 2 * (1 + math.sin(2 * math.pi * k / n)) + off[1]]
                for k in range(n)]
    return [[off[0], off[1]], [w + off[0], off[1]], [w + off[0], d + off[1]], [off[0], d + off[1]]]


def camera_poses(spec: SceneSpec, placements: list[Placement], off: tuple[float, float, float]) -> list[dict[str, Any]]:
    """set_camera params for every candidate (spot x eye height, then a higher one), best first:
    from the open side, looking at the fronts of the seating and at the heroes and features
    (camera.py), the lens wide enough to take them in. Level camera with a lens shift.
    The old pick (the clearest corner, farthest from the heroes) stood behind the lobby's
    seating looking at a blank wall (Marc's review of the aidb5 round lobby)."""
    import math

    from ai_db_scene.camera import best_views, faced_features

    h = spec.setting.bounds_m[2]
    indoor = spec.setting.kind == "indoor"
    eye = SCENE_CAMERA_HEIGHT_M if indoor else 2.5
    if not indoor and spec.setting.water is not None:
        # by the water a dune-top eye: the plants between the lens and the shore drop below the
        # frame and the shallows show their colour (at 2.5 m the sea was a grey reflection of the
        # sky and a bush filled the bottom corner: qwen aidb7 beach_cove)
        eye = SCENE_CAMERA_WATER_EYE_M
    heights = [eye, round(min(eye + 0.8, h - 0.3), 2) if indoor else eye + 2.0]
    faced_eye = None
    if indoor and faced_features(placements):
        # a living room shot to show the sofa AND the TV it faces stands in a corner of the TV
        # wall, past the end of the group: the tripod goes up so the near armchair's back sits low
        # in the frame instead of filling it (f7e0879 demo: the TV cut at the frame's edge)
        eye_f = round(min(SCENE_CAMERA_FACED_EYE_M, h - 0.5), 2)
        top = best_views(spec, placements, eye_f, faced=True)[:1]
        if top and top[0].shows_faced and top[0].score > 0:  # (a cramped room may have no such view)
            faced_eye = eye_f
            heights.insert(0, eye_f)
    poses = []
    for height in dict.fromkeys(heights):
        for v in best_views(spec, placements, height, faced=height == faced_eye):
            dx, dy, dz = v.aim[0] - v.x, v.aim[1] - v.y, v.aim[2] - height
            lens = round(max(v.lens_mm, 16.0) if indoor else max(v.lens_mm, 24.0), 1)
            # Level camera (pitch 90): verticals stay vertical (two-point perspective, as interior
            # photographers shoot); a vertical lens shift brings the subject into frame instead of
            # tilting, which made walls lean in. Blender's shift is a fraction of the sensor width.
            shift = dz / max(math.hypot(dx, dy), 0.5) * lens / 36.0
            shift = max(-SCENE_CAMERA_MAX_SHIFT, min(SCENE_CAMERA_MAX_SHIFT, shift))
            poses.append({"location": _shift((v.x, v.y, height), off), "rotation": [90.0, 0.0, round(v.yaw_deg, 2)],
                          "name": "ai_db_camera", "lens": lens, "shift_y": round(shift, 4), "expect": list(v.seen)})
    return poses


#: the last build's spec and placements: set_camera says when a model's own camera looks at a
#: wall or at the backs of the seating (scene_tools)
_LAST_BUILD: dict[str, Any] = {}


def camera_hint(location: Any, rotation: Any, lens: Any = 50.0) -> str | None:
    """Why a set_camera pose (Blender frame) is a poor view of the last build, with a better one."""
    from ai_db_scene.camera import view_issue

    spec, placements = _LAST_BUILD.get("spec"), _LAST_BUILD.get("placements")
    if spec is None or not placements or spec.setting.kind in ("single_asset", "tabletop"):
        return None
    try:
        lens_mm = float(lens) if lens is not None else 50.0
    except (TypeError, ValueError):
        lens_mm = 50.0
    return view_issue(spec, placements, location, rotation, lens_mm, off=frame_offset(spec))


def camera_command(spec: SceneSpec, placements: list[Placement], off: tuple[float, float, float]) -> dict[str, Any]:
    """An eye-level camera in the room corner with the clearest view, aimed at the heroes (else
    at everything). The old camera looked in from above the walls: walls and framing were the
    judges' most common complaints."""
    if spec.setting.kind in ("single_asset", "tabletop") or not placements:
        # small subjects: a close three-quarter view that frames them, not a room corner
        direction = [0.8, -1.0, 0.5] if spec.setting.kind == "single_asset" else [0.6, -1.0, 0.7]
        return {"type": "frame_camera", "params": {"direction": direction, "margin": 1.2, "lens": 50}}
    pose = camera_poses(spec, placements, off)[0]
    return {"type": "set_camera", "params": {k: v for k, v in pose.items() if k not in ("shift_y", "expect")}}


def camera_shift_command(spec: SceneSpec, placements: list[Placement], off: tuple[float, float, float]) -> \
        dict[str, Any] | None:
    """The lens shift of the chosen pose (set_camera has no shift parameter)."""
    if spec.setting.kind in ("single_asset", "tabletop") or not placements:
        return None
    shift = camera_poses(spec, placements, off)[0]["shift_y"]
    return {"type": "run_code", "params": {"code": (
        "import bpy\ncam = bpy.data.objects.get('ai_db_camera')\n"
        f"if cam is not None:\n    cam.data.shift_y = {shift!r}\nresult = {{'shift_y': {shift!r}}}")}}


#: is the scene camera inside the room, and which targets does it see (in frame, not hidden behind
#: other objects: rays from the camera to the centre and corners of each target's box)? Failing,
#: every candidate pose is scored and the best kept.
_COVERAGE = """
import bpy, json, math
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
cfg = json.loads({payload!r})
scene = bpy.context.scene
cam = scene.camera
roots = {{o.get("ai_db_id"): o for o in scene.objects if o.get("ai_db_id") in cfg["targets"]}}

def owner(o):
    while o is not None and o.get("ai_db_id") is None:
        o = o.parent
    return o

def box_points(o):
    cs = [m.matrix_world @ Vector(c) for m in [o, *o.children_recursive] if m.type == "MESH" for c in m.bound_box]
    if not cs:
        return [o.matrix_world.translation.copy()]
    lo = Vector([min(c[i] for c in cs) for i in range(3)])
    hi = Vector([max(c[i] for c in cs) for i in range(3)])
    mid = (lo + hi) / 2
    return [mid] + [mid + 0.7 * (Vector((x, y, z)) - mid) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]

points = {{t: box_points(o) for t, o in roots.items()}}

def inside(loc):
    fp = cfg["footprint"]
    if fp is None:
        return True
    x, y, ok = loc[0], loc[1], False
    for i in range(len(fp)):
        (x0, y0), (x1, y1) = fp[i - 1], fp[i]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            ok = not ok
    return ok and 0.0 < loc[2] < cfg["height"]

def score():
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    origin = cam.matrix_world.translation.copy()
    seen = []
    for t, pts in points.items():
        n = 0
        for p in pts:
            co = world_to_camera_view(scene, cam, p)
            if not (0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0 and co.z > 0.0):
                continue
            v = p - origin
            hit, _l, _n, _i, obj, _m = scene.ray_cast(dg, origin, v.normalized(), distance=max(v.length - 0.02, 0.0))
            if not hit or owner(obj) is roots[t]:
                n += 1
        if n * 3 >= len(pts):
            seen.append(t)
    return inside(origin), seen

def pose(p):
    cam.location = p["location"]
    cam.rotation_euler = [math.radians(a) for a in p["rotation"]]
    cam.data.lens = p["lens"]
    cam.data.shift_y = p.get("shift_y", 0.0)

if cam is None:
    result = {{"skipped": "no scene camera"}}
else:
    # the poses come best view first: take the first one in which nothing hides the targets it
    # frames ("expect": what camera.py has in frame). Asking every pose for every target took
    # the 6th pose of the aidb7 living room, from behind the sofa: a view of the sofa's front
    # from the TV side cannot have the TV in it.
    def good(got, p):
        exp = [t for t in p.get("expect", list(points)) if t in points] or list(points)
        miss = sum(1 for t in exp if t not in got[1])
        return got[0] and miss <= (0 if len(exp) <= 3 else max(1, len(exp) // 8))

    best = (*score(), 0)
    tried = 1
    if not good(best, cfg["poses"][0]):
        for k, p in enumerate(cfg["poses"][1:], 1):
            pose(p)
            got = (*score(), k)
            tried += 1
            if (got[0], len(got[1])) > (best[0], len(best[1])):
                best = got
            if good(got, p):
                best = got
                break
        pose(cfg["poses"][best[2]])
    result = {{"inside": best[0], "visible_heroes": best[1], "targets": sorted(points), "tried": tried, "pose": best[2],
               "expect": [t for t in cfg["poses"][best[2]].get("expect", list(points)) if t in points]}}
"""


def coverage_targets(spec: SceneSpec, placements: list[Placement], n: int = 3) -> list[str]:
    """Hero placement ids, or the n largest objects (openings and lamps' lights aside)."""
    heroes = {o.id for o in spec.objects if o.hero}
    hero_ids = [p.id for p in placements if p.id in heroes or _SUFFIX.sub("", p.id) in heroes]
    if hero_ids:
        return hero_ids
    if spec.setting.kind == "outdoor":
        # what camera.py frames (a beach's shore group, a fountain and its benches): the largest
        # objects were the palms behind the camera, and the check walked to a pose looking inland
        from ai_db_scene.camera import focus_weights

        w = focus_weights(spec, placements)
        return [k for k, _w in sorted(w.items(), key=lambda kv: -kv[1])[:max(n, 4)]]
    solid = [p for p in placements if p.category not in _OPENING_KINDS]
    return [p.id for p in sorted(solid, key=lambda p: -(p.dims[0] * p.dims[1] * p.dims[2]))[:n]]


def camera_check(spec: SceneSpec, placements: list[Placement]) -> dict[str, Any] | None:
    """Run _COVERAGE for a room build (None for close-up kinds framed by frame_camera)."""
    if spec.setting.kind in ("single_asset", "tabletop") or not placements:
        return None
    off = frame_offset(spec)
    payload = json.dumps({
        "poses": camera_poses(spec, placements, off), "targets": coverage_targets(spec, placements),
        "footprint": footprint(spec, off) if spec.setting.kind == "indoor" else None,
        "height": spec.setting.bounds_m[2]})
    return dict(BlenderClient().run_code(_COVERAGE.format(payload=payload), timeout=120.0))


def camera_issue(cam: dict[str, Any]) -> dict[str, Any] | None:
    if "inside" not in cam:
        return None
    # targets the chosen view frames ("expect") and does not see: hidden behind something. A TV
    # behind a camera that looks at the sofa's front is out of frame by choice, not an issue
    missing = sorted(set(cam.get("expect") or cam.get("targets", [])) - set(cam.get("visible_heroes", [])))
    if cam["inside"] and not missing:
        return None
    where = "" if cam["inside"] else "camera outside the room; "
    return {"check": "camera", "severity": "warning", "objects": missing, "fix": None,
            "message": f"{where}hidden from the camera ({cam.get('tried')} spots tried): {', '.join(missing) or '-'}"}


# --- handlers ------------------------------------------------------------------


def validator_args(spec: SceneSpec) -> dict[str, Any]:
    """A SceneSpec in the shape the Blender addon's validate_scene reads.

    The validator knows nothing of SceneSpec: given one directly it found no room bounds,
    priors or usable relations, so out_of_bounds/scale never ran.
    """
    priors = load_priors()
    out: dict[str, Any] = {
        "relations": [{"a": r.a, "rel": r.rel, "b": r.b} for r in spec.relations
                      if r.rel in ("facing", "in_front_of")],
        "priors": {o.category: {"min": list(priors[o.category].min_m), "max": list(priors[o.category].max_m)}
                   for o in spec.objects if o.category in priors},
    }
    if spec.setting.kind in ("indoor", "outdoor") and spec.setting.shape != "polygon":
        out["room_bounds"] = list(spec.setting.bounds_m)  # centred: see frame_offset
    return out


_LIST_IDS = """
import bpy
result = [{"name": o.name, "ai_db_id": o.get("ai_db_id")} for o in bpy.context.scene.objects]
"""


def missing_objects(spec: SceneSpec, present_ids: list[str]) -> list[dict[str, Any]]:
    """Spec objects with fewer instances in the scene than ``count`` (by ai_db_id tag)."""
    have: dict[str, int] = {}
    ids = {o.id for o in spec.objects}
    for pid in present_ids:
        base = pid if pid in ids else _SUFFIX.sub("", pid)
        have[base] = have.get(base, 0) + 1
    issues = []
    for o in spec.objects:
        n = have.get(o.id, 0)
        if n < o.count:
            issues.append({"check": "missing_objects", "severity": "blocking", "objects": [o.id],
                           "message": f"{o.id}: {n} of {o.count} in the scene", "fix": None})
    return issues


def _handle_validate_scene(args: dict[str, Any]) -> Any:
    from ai_db_scene.tools.scene_tools import _forward

    raw = args.get("spec")
    if not (isinstance(raw, dict) and "setting" in raw):
        return _forward("validate_scene", args)  # validator-native spec or none
    spec, errors = _parse_spec(raw)
    if spec is None:
        return {"status": "invalid", "errors": errors}
    # Explicit args (checks, room_bounds, priors, ...) win over the spec, as in the validator.
    res = _forward("validate_scene", {**{k: v for k, v in args.items() if k != "spec"},
                                      "spec": validator_args(spec)})
    if res.get("status") != "success":
        return res
    from ai_db_scene.blender_client import BlenderClient, BlenderError
    try:
        objs = BlenderClient().run_code(_LIST_IDS)
    except BlenderError as e:
        return {"status": "error", "message": str(e)}
    extra = missing_objects(spec, [o["ai_db_id"] for o in objs if o.get("ai_db_id")])
    result = res["result"]
    if extra:
        result["issues"] = list(result.get("issues", [])) + extra
        result["ok"] = False
        stats = result.setdefault("stats", {})
        stats["blocking"] = stats.get("blocking", 0) + len(extra)
    return res



def _room_kits(brief: str) -> list[dict[str, Any]]:
    from ai_db_scene.room_kits import kits_for

    return kits_for(brief)

def _lesson_checklist(categories: set[str], brief: str, project: str) -> list[str]:
    try:
        from ai_db_scene import feedback

        return feedback.checklist(categories, brief, project)
    except Exception:  # noqa: BLE001 - an unreadable lesson file must not break the schema
        return []


def _handle_get_scene_spec_schema(args: dict[str, Any]) -> Any:
    priors = load_priors()
    exemplars: list[dict[str, Any]] = []
    try:
        runs = _memory().search_exemplars(args.get("brief", ""), min_score=4.0, top_k=_MAX_EXEMPLARS)
        exemplars = [{"brief": r.brief, "spec": r.spec, "score": r.score} for r in runs if r.spec]
    except Exception:  # noqa: BLE001  memory unavailable must not block schema discovery
        exemplars = []
    # Sizes only for the categories the brief is likely to need; every name stays listed, so the
    # vocabulary is complete but the reply is ~60% smaller (ai-db runs spent the most tokens here).
    brief_words = {w.rstrip("s") for w in re.findall(r"[a-z]+", str(args.get("brief", "")).lower())}
    relevant = {n for n in priors if any(part.rstrip("s") in brief_words for part in n.split("_"))}
    if not relevant:
        relevant = set(priors)  # no brief: everything, as before
    return {
        "schema": spec_schema(),
        "categories": {
            n: {"typical_m": list(p.typical_m), "support": p.support, "can_support_others": p.can_support_others}
            for n, p in sorted(priors.items()) if n in relevant
        },
        "other_categories": sorted(set(priors) - relevant),
        "relations": list(RELATIONS),
        "setting_kinds": list(SETTING_KINDS),
        "styles": sorted(STYLE_PALETTES),
        "exemplars": exemplars[:_MAX_EXEMPLARS] or [{"brief": "a small reading corner", "spec": _EXAMPLE_SPEC}],
        # what rooms of the types the brief names usually hold (complete the room, the brief wins)
        "room_kits": _room_kits(str(args.get("brief", ""))),
        # problems users reported on these objects before (record_feedback): verify each one
        "lessons": _lesson_checklist(relevant, str(args.get("brief", "")), args.get("project", "default")),
        "next": "submit_scene_spec(spec) to check it, then build_scene(spec)",
    }


_EXAMPLE_SPEC: dict[str, Any] = {
    "setting": {"kind": "indoor", "bounds_m": [4.0, 3.5, 2.7], "floor_material": "wood:oak"},
    "objects": [{"id": "armchair", "category": "armchair", "hero": True, "material": "velvet:#3d5a80"},
                {"id": "lamp", "category": "floor_lamp"}, {"id": "shelf", "category": "bookshelf"},
                {"id": "rug", "category": "rug"}],
    "relations": [{"a": "shelf", "rel": "against_wall", "b": "wall_north"},
                  {"a": "armchair", "rel": "facing", "b": "shelf"},
                  {"a": "lamp", "rel": "beside", "b": "armchair"},
                  {"a": "rug", "rel": "centered_in", "b": "room"}]}


def _handle_submit_scene_spec(args: dict[str, Any]) -> Any:
    spec, errors = _spec_from_args(args)
    if spec is None:
        return {"status": "invalid", "errors": errors, "hint": _INVALID_HINT}
    return {"status": "ok", "spec_id": remember_spec(spec), "spec": spec.to_dict(),
            **({"notes": list(spec.notes)} if spec.notes else {}),
            "next": "build_scene() (no arguments) builds this spec"}


def _handle_solve_layout(args: dict[str, Any]) -> Any:
    spec, errors = _spec_from_args(args, remembered=True)
    if spec is None:
        return {"status": "invalid", "errors": errors, "hint": _INVALID_HINT}
    kw: dict[str, Any] = {}
    if args.get("seed") is not None:
        kw["seed"] = int(args["seed"])
    try:
        pls = solve_layout(spec, **kw)
    except LayoutError as e:
        return {"status": "conflict", "message": str(e), "conflicts": e.conflicts}
    from ai_db_scene.layout import room_doors

    return {"status": "ok", "placements": [placement_to_dict(p) for p in pls],
            **({"notes": list(spec.notes)} if spec.notes else {}),
            **({"doors": room_doors(spec)} if spec.setting.rooms else {})}


def _handle_apply_layout(args: dict[str, Any]) -> Any:
    spec, errors = _spec_from_args(args)
    if spec is None:
        return {"status": "invalid", "errors": errors}
    try:
        pls = parse_placements(args.get("placements"))
    except (KeyError, TypeError, ValueError) as e:
        return _err(f"bad placements: {e!r}")
    geometry = args.get("geometry", "generated")
    if geometry not in GEOMETRY_MODES:
        return _err(f"geometry must be one of {list(GEOMETRY_MODES)}")
    try:
        from ai_db_scene.providers import providers_from_config

        providers = providers_from_config()
    except AiDbConfigError as e:  # a bad plugins.scene section: say so instead of ignoring it
        return _err(str(e))
    cmds, names = build_layout_commands(spec, pls, args.get("asset_map"), geometry, providers,
                                        args.get("variation"))
    try:
        results = BlenderClient().batch(cmds)
    except BlenderError as e:
        return {"status": "error", "message": str(e), "error_type": e.error_type or type(e).__name__, "hint": e.hint}
    results = _retry_legacy_materials(cmds, results)
    failed = [{"index": i, "type": cmds[i]["type"], "message": r.get("message")}
              for i, r in enumerate(results) if not r.get("ok", r.get("status") == "success")]
    return {"status": "partial" if failed else "success", "created": names, "commands": len(cmds), "failed": failed}


def _handle_record_run(args: dict[str, Any]) -> Any:
    outcome = args.get("outcome", "success")
    if outcome not in RUN_OUTCOMES:
        return _err(f"outcome must be one of {list(RUN_OUTCOMES)}")
    fields = {k: args[k] for k in (
        "brief", "spec", "placements", "actions", "validation_reports", "judge_scores",
        "token_cost", "wall_time_s", "project") if k in args}
    run = RunRecord(outcome=outcome, **fields)
    return {"status": "saved", "run_id": _memory().record_run(run)}


def _handle_search_exemplars(args: dict[str, Any]) -> Any:
    runs = _memory().search_exemplars(
        args["brief"], min_score=float(args.get("min_score", 4.0)),
        top_k=int(args.get("top_k", 3)), project=args.get("project"))
    return {"count": len(runs), "exemplars": [r.to_dict() | {"score": r.score} for r in runs]}


_SPEC = {"type": ["object", "string"],
         "description": "SceneSpec {setting, objects, relations} (see get_scene_spec_schema); a JSON string also works"}
_SPEC_ID = {"type": "string", "description": ("a spec_id from submit_scene_spec / build_scene; with neither spec "
                                              "nor spec_id the last valid spec is used")}
#: the spec's own fields, accepted at the top level when ``spec`` is left out
_SPEC_FIELDS: dict[str, Any] = {
    "spec": _SPEC,
    "setting": {"type": "object", "description": "instead of spec: the spec's setting at the top level"},
    "objects": {"type": "array", "items": {"type": "object"}, "description": "instead of spec: the spec's objects"},
    "relations": {"type": "array", "items": {"type": "object"}, "description": "instead of spec: the spec's relations"},
    "lighting": {"type": "object"}, "camera": {"type": "object"},
}

def _issues(res: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((res or {}).get("result") or {}).get("issues", []))


def _handle_build_scene(args: dict[str, Any]) -> Any:
    """spec -> layout (repairing conflicts) -> build -> validate -> apply the validator's fixes;
    the result compact, verdict and next first (see build_report)."""
    args, lifted = lift_build_args(args)
    res = _build_scene(args)
    if lifted and isinstance(res.get("repairs"), list):
        res["repairs"] = [f"spec: {lifted} are build_scene arguments, not spec fields: read as such", *res["repairs"]]
    return present_build(res)


#: build_scene arguments a model put inside the spec (qwen3-14b-local-aidb6: brief, project,
#: poly_budget, exposure_check in 6 of 7 scenes' first call, each refused as an unknown field)
_BUILD_ARGS = ("brief", "project", "prompt", "seed", "geometry", "poly_budget", "exposure_check", "judge",
               "render", "render_path", "physics", "snapshot", "clear", "asset_map")


def lift_build_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """(args with build arguments found inside the spec moved to the top level, their names)."""
    raw, err = spec_arg(args)
    if err or not isinstance(raw, dict):
        return args, []
    lifted = [k for k in _BUILD_ARGS if k in raw]
    if not lifted:
        return args, []
    spec = {k: v for k, v in raw.items() if k not in lifted}
    out = {**{k: raw[k] for k in lifted}, **{k: v for k, v in args.items() if k not in SPEC_KEYS}}
    return {**out, "spec": spec}, lifted


def _build_scene(args: dict[str, Any]) -> dict[str, Any]:
    from ai_db_scene.constants import BUILD_FIX_ROUNDS, BUILD_MAX_REPAIRS
    from ai_db_scene.normalize import room_shape_from_brief, table_shape_from_brief
    from ai_db_scene.repair import solve_with_repairs
    from ai_db_scene.terrain import water_from_brief

    raw, raw_err = spec_arg(args)
    brief_notes: list[str] = []
    if raw_err is None and isinstance(raw, dict):
        brief = args.get("brief") or args.get("prompt")
        raw0 = raw
        raw, shape_note = room_shape_from_brief(raw, brief)  # "a round lobby" with no setting.shape: an ellipse
        raw, water_note = water_from_brief(raw, brief)  # "turquoise water" with water left implicit: a lagoon
        raw, table_note = table_shape_from_brief(raw, brief)  # "four round tables": round tops
        brief_notes = [n for n in (shape_note, water_note, table_note) if n]
        if raw is not raw0:
            args = {**{k: v for k, v in args.items() if k not in SPEC_KEYS}, "spec": raw}
    spec, errors = _spec_from_args(args, remembered=True)
    if spec is None:
        return {"status": "invalid", "errors": errors, "hint": _INVALID_HINT}
    if brief_notes:
        spec = replace(spec, notes=(*brief_notes, *spec.notes))
    spec_id = remember_spec(spec)
    asked = spec  # before repairs: what the model asked for
    # furniture variants follow the first build's spec; edit_scene passes the stored key back
    variation = args.get("variation") if isinstance(args.get("variation"), list) else None
    if variation is None:  # a new scene: design families, seeded by the build (furniture_variants)
        from ai_db_scene.furniture_variants import variation_marker

        variation = [*variation_key(spec), variation_marker(int(args.get("seed", 0)))]
    render_opts = None
    if args.get("render") is not None:
        render_opts, render_err = render_options(args["render"])
        if render_opts is None:
            return {"status": "invalid", "errors": [render_err]}
    from ai_db_scene.terrain import apply_brief_defaults

    spec, terrain_notes = apply_brief_defaults(spec, args.get("brief") or args.get("prompt"))  # 'beach': sand + sea
    seed = int(args.get("seed", 0))
    fixed = solve_with_repairs(spec.to_dict(), seed, BUILD_MAX_REPAIRS)
    data = fixed.data
    repairs: list[str] = [*spec.notes, *terrain_notes, *fixed.repairs]  # what validation coerced first
    if fixed.errors or fixed.spec is None:  # a repair must never produce an invalid spec; report it if it does
        return {"status": "invalid", "errors": fixed.errors, "repairs": repairs}
    if fixed.error is not None or fixed.placements is None:
        e = fixed.error
        return {"status": "conflict", "message": str(e), "conflicts": e.conflicts if e else [], "repairs": repairs,
                "spec": data, "hint": fixed.hint}
    spec, placements = fixed.spec, fixed.placements
    _LAST_BUILD.update(spec=spec, placements=placements)
    if set(spec.camera) - {"focus"}:  # qwen aidb5/aidb6 specs carried camera position/rotation/fov
        repairs.append(f"camera: {sorted(set(spec.camera) - {'focus'})} ignored: the build places its camera on the "
                       "open side, looking at the fronts of the seating and the main objects (camera.focus picks "
                       "what it looks at; set_camera after the build overrides it)")
    if args.get("clear"):
        try:
            BlenderClient().request("clear_scene", {"keep_camera": False, "purge_orphans": True})
            BlenderClient().request("run_code", {"code": _PURGE})
        except BlenderError as e:
            return {"status": "error", "message": f"clear_scene failed: {e}", "repairs": repairs}
    else:
        # without clear: build_scene replaces the previous ai-db scene - drop Blender's untouched
        # factory cube, the room shell and helpers of an earlier build, and every object an earlier
        # build tagged (benchmark agents left all of these: thousands of false clashes); objects
        # the user made themselves are kept
        try:
            BlenderClient().request("run_code", {"code": _TIDY.format(payload=json.dumps(
                [p.id for p in placements]))})
        except BlenderError:
            pass
    built = _handle_apply_layout({"spec": data, "placements": [placement_to_dict(p) for p in placements],
                                  "asset_map": args.get("asset_map"), "geometry": args.get("geometry", "generated"),
                                  "variation": variation})
    if built.get("status") not in ("success", "partial"):
        return {**built, "repairs": repairs, "spec": data}
    report = _handle_validate_scene({"spec": data})
    first = _issues(report)
    applied: list[str] = []
    for _ in range(BUILD_FIX_ROUNDS):
        fixes = [i for i in _issues(report) if isinstance(i.get("fix"), dict) and i["fix"].get("handler")]
        if not fixes:
            break
        for issue in fixes:
            try:
                BlenderClient().request(issue["fix"]["handler"], issue["fix"].get("params") or {})
                applied.append(f"{issue.get('check')}: {issue.get('message')}")
                turned = keep_turn(data, issue)  # the stored spec turns it too: a rebuild keeps the fix
                if turned:
                    repairs.append(turned)
            except BlenderError as e:
                applied.append(f"fix failed for {issue.get('check')}: {e}")
        report = _handle_validate_scene({"spec": data})
    left = _issues(report)
    # a relation the repair loop had to drop is not "built": the model treated it as success and
    # judges named exactly those (sofa not facing the TV, chairs unpaired)
    # judges named exactly those (sofa not facing the TV, chairs unpaired); qwen aidb6's cafe
    # replied DONE over three dropped table relations, so a dropped relation blocks "done" and
    # comes with a step that fits
    left += [dropped_issue(lbl, data, fixed.spec) for lbl in fixed.dropped]
    from ai_db_scene import constants as _c

    camera: dict[str, Any] | None = None
    try:
        camera = camera_check(spec, placements)
    except Exception as e:  # noqa: BLE001 - a check must never fail a finished build
        camera = {"error": str(e)}
    cam_issue = camera_issue(camera) if camera else None
    if cam_issue:
        left.append(cam_issue)
    exposure: dict[str, Any] | None = None
    if args.get("exposure_check", _c.BUILD_EXPOSURE_CHECK):
        try:
            exposure = exposure_check(outdoor=spec.setting.kind == "outdoor")
        except Exception as e:  # noqa: BLE001 - a check must never fail a finished build
            exposure = {"error": str(e)}
        exp_issue = exposure_issue(exposure)
        if exp_issue:
            left.append(exp_issue)
    try:
        from ai_db_scene import ergonomics

        left += ergonomics.check(spec, placements, centred=False)  # solver placements: x in [0, W]
        from ai_db_scene import realism

        left += realism.check(spec, placements)
        if args.get("brief"):
            left += realism.check_brief(str(args["brief"]), placements)
        left += count_issues(asked, placements)
    except Exception as e:  # noqa: BLE001 - a check must never fail a finished build
        left.append({"check": "ergonomics", "severity": "warning", "message": f"not checked: {e}"})
    try:
        from ai_db_scene import feedback

        left += feedback.apply(left, placements, args.get("project", "default"))
    except Exception as e:  # noqa: BLE001 - a lesson must never fail a finished build
        left.append({"check": "lesson_error", "severity": "warning", "message": f"lessons not applied: {e}"})
    clean = not fixed.dropped and not any(i.get("severity") == "blocking" for i in left)
    out = {"status": "built" if clean else "built_with_issues",
           "objects": len(placements), "created": built.get("created"), "failed_commands": built.get("failed"),
           "repairs": repairs, "fixes_applied": applied, "issues_found": len(first),
           "issues_left": left[:20], "spec": data if repairs else None, "spec_id": spec_id}
    if camera is not None:
        out["camera"] = camera
    if render_opts is not None:
        try:
            out["final_render"] = final_render(render_opts)
        except (BlenderError, OSError, TypeError, ValueError) as e:
            out["final_render"] = {"error": str(e)}
    if exposure is not None:
        out["exposure"] = exposure
    from ai_db_scene.constants import BUILD_POLY_BUDGET

    budget = args.get("poly_budget", BUILD_POLY_BUDGET)
    if budget is not None:
        try:
            out["polys"] = poly_report(int(budget))
        except (BlenderError, KeyError, TypeError, ValueError) as e:
            out["polys"] = {"error": str(e)}
    if args.get("physics"):
        try:
            out["physics"] = BlenderClient().request("run_code", physics_commands(spec, placements)["params"])["result"]
        except (BlenderError, KeyError, TypeError) as e:
            out["physics"] = {"error": str(e)}
    if args.get("judge"):
        import tempfile
        import uuid

        out["critique"] = critique(left, repairs, built.get("failed"), out.get("polys"))
        target = Path(args.get("render_path") or Path(tempfile.gettempdir()) / "ai_db_judge" / f"{uuid.uuid4().hex}.png")
        try:
            out["render"] = judge_render(target)
        except BlenderError as e:
            out["render_error"] = str(e)
    # every build is a run: exemplar search (get_scene_spec_schema, search_exemplars) learns
    # from successes without anyone calling record_run (4 calls in 23 benchmark runs)
    blocking = sum(1 for i in left if i.get("severity") == "blocking")
    try:
        out["run_id"] = _memory().record_run(RunRecord(
            brief=str(args.get("prompt") or ""), spec=data, placements=[placement_to_dict(p) for p in placements],
            validation_reports=[{"issues_found": len(first), "issues_left": left[:20], "fixes": applied,
                                 "repairs": repairs}],
            judge_scores=[{"overall": max(1.0, 5.0 - blocking), "source": "validator"}],
            outcome="success" if not blocking else "partial", project=args.get("project", "default")))
    except Exception as e:  # noqa: BLE001 - recording must not fail a finished build
        out["record_error"] = str(e)
    if args.get("snapshot", True):
        # the spec travels with the version: edit_scene starts from it
        try:
            out["scene_id"] = _memory().snapshot_scene(
                prompt=str(args.get("prompt") or "build_scene"), project=args.get("project", "default"),
                operation_type=args.get("operation_type", "build"),
                metadata={"spec": data, "repairs": repairs, "build": build_options({**args, "variation": variation}),
                          **(args.get("metadata") or {})},
                tags=["build_scene"], parent_scene_id=args.get("parent_scene_id"))
        except Exception as e:  # noqa: BLE001 - the scene is built; say why it was not saved
            out["snapshot_error"] = str(e)
    hint = sparse_hint(spec, placements)
    if hint:
        out["sparse"] = hint
    out["verdict"], out["next"] = build_verdict(out, brief_given=bool(args.get("brief")))
    return out


#: a room whose furniture covers less than this share of its floor reads as empty (bench
#: layout_sanity's fill band starts at 0.10): the verdict suggests a smaller room or more content
SPARSE_ROOM_FILL = 0.08


def sparse_hint(spec: SceneSpec, placements: list[Placement]) -> str | None:
    """A hint for a room far too big for its content (qwen aidb7 round lobby: 6 % of a 10 m room),
    with the bounds that would hold it at ~20 %, else None."""
    import math

    if spec.setting.kind != "indoor" or not placements:
        return None
    priors = load_priors()
    foot = sum(p.dims[0] * p.dims[1] for p in placements
               if p.category != "rug" and getattr(priors.get(p.category), "support", "floor") == "floor")
    area = floor_area_m2(spec)
    if area <= 0 or foot / area >= SPARSE_ROOM_FILL:
        return None
    w, d, _h = spec.setting.bounds_m
    k = math.sqrt(foot / 0.2 / area)
    sw, sd = max(3.0, round(w * k * 2) / 2), max(3.0, round(d * k * 2) / 2)
    return (f"the room is mostly empty: the furniture covers {foot / area:.0%} of its floor. A smaller room "
            f"(edit_scene step {{op: 'update_setting', set: {{bounds_m: [{sw:g}, {sd:g}, {spec.setting.bounds_m[2]:g}]}}}}) "
            f"or more of what the "
            f"brief implies would read better")


_TURN_MSG = re.compile(r"rotate it (-?\d+) deg")


def turn_step(issue: dict[str, Any]) -> dict[str, Any] | None:
    """The edit_scene rotate step an issue asks for ("... rotate it 180 deg"), else None."""
    m = _TURN_MSG.search(str(issue.get("message", "")))
    objs = issue.get("objects") or []
    if not m or not objs or not isinstance(objs[0], str):
        return None
    return {"op": "rotate", "id": objs[0], "deg": int(m.group(1))}


def keep_turn(data: dict[str, Any], issue: dict[str, Any]) -> str | None:
    """A faces_wall fix the build applied in Blender, written into the stored spec as the
    object's ``turn_deg`` (single objects only): the qwen aidb6 TV faced the wall again on every
    rebuild because only Blender knew of the turn."""
    step = turn_step(issue)
    obj = next((o for o in data.get("objects", []) if step and o.get("id") == step["id"]), None)
    if step is None or obj is None or obj.get("count", 1) != 1 or obj.get("grid") or obj.get("seats"):
        return None
    obj["turn_deg"] = int(obj.get("turn_deg", 0) + step["deg"]) % 360
    return f"'{obj['id']}' faced a wall: turned {step['deg']} deg (turn_deg {obj['turn_deg']} in the stored spec)"


def dropped_issue(label: str, data: dict[str, Any], spec: SceneSpec | None) -> dict[str, Any]:
    """The blocking issue of a relation the repair loop dropped, with one edit_scene step that
    keeps A near what it was related to: beside the object that kept the same relation to B
    (two tables in_front_of one counter: the second beside the first), else beside B itself."""
    a, rel, b = (label.split(" ") + ["", "", ""])[:3]
    kept = [r for r in data.get("relations", []) if r.get("rel") == rel and r.get("b") == b and r.get("a") != a]
    others = {r.get("a") for r in data.get("relations", []) if r.get("b") == a}  # what stands relative to A
    peer = next((r["a"] for r in kept if r["a"] not in others), None)
    ids = {o.id for o in spec.objects} if spec is not None else set()
    if rel == "beside" or b not in ids:
        step: dict[str, Any] | None = None
    else:
        step = {"op": "move", "id": a, "to": {"beside": peer or b, "gap_m": 0.6}}
    msg = (f"relation '{label}' could not be kept (dropped so the scene could be laid out): the scene does not have it")
    out: dict[str, Any] = {"check": "dropped_relation", "severity": "blocking", "objects": [a], "message": msg}
    if step is not None:
        out["suggest"] = step
    return out


def count_issues(asked: SceneSpec, placements: list[Placement]) -> list[dict[str, Any]]:
    """Objects the spec's own ``count`` (or grid) asks for that the build placed fewer of (the
    repair loop places one fewer when the room is full): the model would believe they exist."""
    issues = []
    for o in asked.objects:
        pat = re.compile(rf"{re.escape(o.id)}(?:_\d+)?")
        got = sum(1 for p in placements if pat.fullmatch(p.id))
        if got < o.count:
            issues.append({"check": "count", "severity": "warning", "objects": [o.id],
                           "message": (f"the spec asks for {o.count} '{o.id}' ({o.category}), {got} were placed "
                                       "(no room for the rest): use a grid, a bigger setting or fewer")})
    return issues


#: verdict/next tail when build_scene had no brief: it could not compare counts with the request
_NO_BRIEF = ("no brief given, so counts against the request were not checked")
_NO_BRIEF_NEXT = (" Before replying DONE, check created against every object and count the request asks for "
                  "and add what is missing with edit_scene; pass brief (the request text) to build_scene and it "
                  "checks this itself.")


#: the edit_scene ops a verdict names (edits.OPS): never one that does not exist
_VERDICT_OPS = ("move", "rotate", "update_object", "add_object", "remove_object", "add_relation")


def build_verdict(out: dict[str, Any], brief_given: bool = True) -> tuple[str, str]:
    """(verdict, next): a one-line stopping rule for the calling model.

    Weak models never stopped after a good build (mimo: 21-57 calls of renders and bounds
    checks, 3 of 6 scenes timed out) or took built_with_issues as done. The verdict says done,
    or names the single most important thing left; next says what to call, or to stop.
    """
    left = [i for i in out.get("issues_left") or [] if isinstance(i, dict)]
    failed = out.get("failed_commands") or []
    order = ([i for i in left if i.get("severity") == "blocking"]
             + [i for i in left if i.get("check") == "dropped_relation"]
             + [i for i in left if "the brief asks" in str(i.get("message", "")) or i.get("check") == "count"])
    sid = out.get("scene_id")
    edit = f"edit_scene({{scene_id: '{sid}', steps: [...]}})" if sid else "build_scene with the fixed spec"
    unsaved = ("" if sid or not out.get("snapshot_error") else
               " The scene is built but was not saved (snapshot_error): call snapshot_scene to save it; "
               "edit_scene needs a scene_id.")
    if failed:
        f = failed[0]
        thing = f"{len(failed)} build command(s) failed, first: {f.get('type')}: {f.get('message')}"
    elif order:
        thing = str(order[0].get("message", order[0].get("check")))
    else:
        rest = f"; {len(left)} advisory warning(s) you may ignore" if left else ""
        verdict = f"done: {out.get('objects')} objects built, nothing blocking{rest}"
        where = f" and saved as {sid}" if sid else ""
        sparse = f" Optional, once: {out['sparse']}; otherwise stop." if out.get("sparse") else ""
        if not brief_given:
            return f"{verdict}; {_NO_BRIEF}", (f"The scene is built{where}; do not rebuild, re-render or "
                                               f"re-validate it.{_NO_BRIEF_NEXT}{sparse}{unsaved}")
        return verdict, (f"The scene is finished{where}. Stop here and reply DONE: do not rebuild, "
                         f"re-render or re-validate it.{sparse}{unsaved}")
    more = len(order) + len(failed) - 1
    verdict = f"one thing to fix: {thing}" + (f" (+{more} more)" if more > 0 else "")
    if not brief_given:
        verdict += f"; {_NO_BRIEF}"
    first = order[0] if order and not failed else {}
    if first.get("check") == "dropped_relation":
        # not optional: the model replied DONE over dropped relations (qwen aidb6 cafe)
        dropped = [i for i in order if i.get("check") == "dropped_relation"]
        steps = [i["suggest"] for i in dropped if isinstance(i.get("suggest"), dict)]
        how = (f"edit_scene({{scene_id: '{sid}', steps: {json.dumps(steps)}}})" if sid and steps else edit)
        return verdict, (f"The scene is not done: {len(dropped)} requested relation(s) are missing. Place them "
                         f"with {how} (or other moves that fit, or a bigger setting with update_setting); the "
                         f"stored spec no longer has them, so an edit that leaves them out accepts the layout "
                         f"as it is.{unsaved}")
    step = turn_step(first) if first else None
    if step is not None and sid:  # "rotate it 180 deg": the op that does it (update_object rotation was refused)
        edit = f"edit_scene({{scene_id: '{sid}', steps: [{json.dumps(step)}]}})"
    return verdict, (f"Fix only that with {edit} (ops: {', '.join(_VERDICT_OPS)}), then stop. If it does not "
                     f"matter for the brief, reply DONE.{unsaved}")


#: faces per tagged object (its whole hierarchy, modifiers applied) and, over the budget, a
#: Decimate modifier on the heaviest non-hero objects, each cut just enough (ratio floor
#: BUILD_DECIMATE_MIN_RATIO) until the scene total fits. Untagged meshes (the room shell) count
#: towards the total but are never decimated. Only an object's dense parts (>= min_part faces) take
#: the cut: collapsing a 64-face tube frame folded it into big mirror-chrome triangles across the
#: armchair's side (qwen aidb7 lobby, poly_budget 50000).
_POLY_BUDGET = """
import bpy, json
budget, floor, min_part = json.loads({payload!r})
dg = bpy.context.evaluated_depsgraph_get()

def faces(o):
    return len(o.evaluated_get(dg).data.polygons) if o.type == "MESH" else 0

def parts(root):
    return [o for o in [root, *root.children_recursive] if o.type == "MESH"]

tagged = [o for o in bpy.context.scene.objects if o.get("ai_db_id") is not None]
inside = {{p.name for o in tagged for p in parts(o)}}
per = {{o.name: sum(faces(p) for p in parts(o)) for o in tagged}}
total = sum(per.values()) + sum(faces(o) for o in bpy.context.scene.objects if o.name not in inside)
before, changed = total, []
if budget is not None and total > budget:
    for o in sorted((o for o in tagged if not o.get("hero")), key=lambda o: -per[o.name]):
        if total <= budget or per[o.name] == 0:
            break
        dense = [p for p in parts(o) if faces(p) >= min_part]
        heavy = sum(faces(p) for p in dense)
        if not heavy:
            continue
        ratio = max(floor, (heavy - (total - budget)) / heavy)
        for p in dense:
            mod = p.modifiers.get("ai_db_decimate") or p.modifiers.new("ai_db_decimate", "DECIMATE")
            mod.ratio = ratio
        dg.update()
        now = sum(faces(p) for p in parts(o))
        changed.append({{"object": o.name, "ratio": round(ratio, 3), "faces_before": per[o.name], "faces_after": now}})
        total -= per[o.name] - now
        per[o.name] = now
result = {{"faces_total": total, "faces_before": before, "budget": budget, "over_budget": budget is not None and total > budget,
          "faces_per_object": dict(sorted(per.items(), key=lambda kv: -kv[1])), "decimated": changed}}
"""


def poly_report(budget: int | None) -> dict[str, Any]:
    """Faces per object after a build; with a budget, decimate down to it (see _POLY_BUDGET)."""
    from ai_db_scene.constants import BUILD_DECIMATE_MIN_PART_FACES, BUILD_DECIMATE_MIN_RATIO

    return dict(BlenderClient().run_code(_POLY_BUDGET.format(
        payload=json.dumps([budget, BUILD_DECIMATE_MIN_RATIO, BUILD_DECIMATE_MIN_PART_FACES])), timeout=120.0))


#: rigid bodies on every mesh of the scene: active (convex hull) for the parts of objects the
#: spec marks dynamic, passive for everything else: furniture as convex hulls, the floor as a box,
#: the walls as their mesh (a box around a room's walls would fill the room).
_PHYSICS = """
import bpy, json
dynamic = set(json.loads({payload!r}))
scene = bpy.context.scene
if scene.rigidbody_world is None:
    with bpy.context.temp_override(scene=scene):
        bpy.ops.rigidbody.world_add()

def owner(o):
    while o is not None and o.get("ai_db_id") is None:
        o = o.parent
    return o

counts = {{"active": 0, "passive": 0}}
for o in list(scene.objects):
    if o.type != "MESH":
        continue
    root = owner(o)
    active = root is not None and root.get("ai_db_id") in dynamic
    if o.rigid_body is None:
        with bpy.context.temp_override(object=o, active_object=o, selected_objects=[o]):
            bpy.ops.rigidbody.object_add()
    rb = o.rigid_body
    rb.type = "ACTIVE" if active else "PASSIVE"
    rb.collision_shape = ("CONVEX_HULL" if root is not None else
                          "MESH" if o.name.startswith("Room_Wall") else "BOX")
    counts["active" if active else "passive"] += 1
result = counts
"""


def physics_commands(spec: SceneSpec, placements: list[Placement]) -> dict[str, Any]:
    """run_code adding the rigid bodies of _PHYSICS; dynamic ids are the placements of dynamic objects."""
    ids = {o.id for o in spec.objects}
    dynamic = {o.id for o in spec.objects if o.dynamic}
    payload = [pl.id for pl in placements if (pl.id if pl.id in ids else _SUFFIX.sub("", pl.id)) in dynamic]
    return {"type": "run_code", "params": {"code": _PHYSICS.format(payload=json.dumps(payload))}}


#: validator check -> the edit_scene step that usually clears it (a hint, not an order)
_CRITIQUE_FIX: dict[str, str] = {
    "missing_objects": "update_setting {set: {bounds_m: larger}} or update_object {set: {count: fewer}}",
    "out_of_bounds": "update_setting {set: {bounds_m: larger}} or remove_relation that pushes it out",
    "interpenetration": "add_relation beside / in_front_of with gap_m, or remove_object one of them",
    "floating": "add_relation {a: <id>, rel: on, b: <support>} or update_object {set: {category: a floor one}}",
    "sunk": "remove_relation on, or update_object {set: {category: ...}} with a matching support",
    "facing": "remove_relation against_wall that fixes its yaw, or move the target (add_relation in_front_of)",
    "scale": "update_object {set: {category: the right one}}",
    "door_clearance": "remove_relation against_wall near the door, or update_setting {set: {bounds_m: larger}}",
    "camera": ("hidden from every camera spot: add_relation against_wall for the tall object in front of it, "
               "or update_setting {set: {bounds_m: larger}}"),
    "exposure": ("a black render: the camera may sit in a wall or behind furniture (move it, or add lamps); "
                 "blown out: fewer lights or an outdoor kind without sky"),
}


#: categories that make a room a working floor: painted lines mark its aisles and walkways
_WAREHOUSE = ("pallet_rack", "conveyor", "forklift", "loading_dock", "shelving_unit")


def _safety_lines(spec: SceneSpec, placements: list[Placement]) -> list[dict[str, Any]]:
    """Yellow floor lines in an indoor rect room with warehouse equipment: along both long sides
    of every rack/conveyor (the aisle edges) and a walkway border inside the walls."""
    st = spec.setting
    if st.kind != "indoor" or st.shape != "rect" or st.rooms or \
            not any(p.category in _WAREHOUSE for p in placements):
        return []
    from ai_db_scene.ergonomics import _footprint

    w, d, _ = st.bounds_m
    lw, z, gap = SCENE_LINE_WIDTH_M, 0.002, 0.15
    strips: list[tuple[float, float, float, float]] = []  # centre x, centre y, length x, length y
    ox, oy, _ = frame_offset(spec)  # placements -> the room shell's centred frame
    for p in placements:
        if p.category not in ("pallet_rack", "conveyor"):
            continue
        x0, x1, y0, y1 = _footprint(p)
        x0, x1, y0, y1 = x0 + ox, x1 + ox, y0 + oy, y1 + oy
        if x1 - x0 >= y1 - y0:  # long along x: lines in front of and behind it
            strips += [((x0 + x1) / 2, y0 - gap, x1 - x0, lw), ((x0 + x1) / 2, y1 + gap, x1 - x0, lw)]
        else:
            strips += [(x0 - gap, (y0 + y1) / 2, lw, y1 - y0), (x1 + gap, (y0 + y1) / 2, lw, y1 - y0)]
    inset = SCENE_WALKWAY_INSET_M
    strips += [(0, -d / 2 + inset, w - 2 * inset, lw), (0, d / 2 - inset, w - 2 * inset, lw),
               (-w / 2 + inset, 0, lw, d - 2 * inset), (w / 2 - inset, 0, lw, d - 2 * inset)]
    cmds: list[dict[str, Any]] = []
    for k, (cx, cy, lx, ly) in enumerate(strips):
        if lx <= 0 or ly <= 0 or abs(cx) + lx / 2 > w / 2 or abs(cy) + ly / 2 > d / 2:
            continue  # a line would run into a wall
        name = f"floor_line_{k}"
        cmds.append({"type": "create_primitive", "params": {
            "primitive": "cube", "name": name, "location": [cx, cy, z], "scale": [lx / 2, ly / 2, z]}})
        cmds.append({"type": "apply_material", "params": {"name": name, "material": SCENE_LINE_MATERIAL}})
    return cmds


def _floor_and_rug_detail(spec: SceneSpec, placements: list[Placement], floor: str) -> list[dict[str, Any]]:
    """Real geometry where a texture alone reads as fake: a wood floor of separate bevelled planks
    (rect rooms; planks would poke through curved walls) and carpet pile on rugs."""
    cmds: list[dict[str, Any]] = []
    st = spec.setting
    if SCENE_PLANK_FLOOR and st.kind == "indoor" and st.shape == "rect" and floor.startswith("wood") \
            and not st.rooms and pbr_preset(floor) is None:
        w, d, _ = st.bounds_m
        cmds.append({"type": "wood_plank_floor", "params": {
            "name": "Room_Planks", "size": [w, d], "material": floor.split(":proj")[0] + ":proj=uv:scale=" + str(SCENE_PLANK_GRAIN_SCALE),
            "location": [0.0, 0.0, SCENE_PLANK_TOP_M - 0.015], "thickness": 0.015}})
    cmds += _safety_lines(spec, placements)
    if SCENE_RUG_PILE_DENSITY > 0:
        cmds += [{"type": "carpet_pile", "params": {"name": pl.id, "density": SCENE_RUG_PILE_DENSITY}}
                 for pl in placements if pl.category == "rug"]
    return cmds


def _retry_legacy_materials(cmds: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """An addon loaded before the concrete/stone/foliage/plastic/paint/rubber kinds rejects them:
    re-apply those objects' materials as their flat colour instead of leaving them untextured."""
    retry = []
    for i, (c, r) in enumerate(zip(cmds, results)):
        if c["type"] == "apply_material" and not r.get("ok", r.get("status") == "success") \
                and "unknown material kind" in str(r.get("message")):
            fb = legacy_fallback(str(c["params"].get("material", "")))
            if fb:
                retry.append((i, {**c, "params": {**c["params"], "material": fb}}))
    if not retry:
        return results
    try:
        again = BlenderClient().batch([c for _, c in retry])
    except BlenderError:
        return results
    out = list(results)
    for (i, _), r in zip(retry, again):
        out[i] = r
    return out


def critique(left: list[dict[str, Any]], repairs: list[str], failed: list[dict[str, Any]] | None,
             polys: dict[str, Any] | None) -> list[dict[str, Any]]:
    """What a calling agent should look at after a build, worst first: validator issues still
    open (missing objects included), spec repairs build_scene made, failed commands, a poly
    budget it could not reach. Each item: source, severity, objects, message, suggest."""
    out = [{"source": "validator", "severity": i.get("severity", "warning"), "check": i.get("check"),
            "objects": i.get("objects") or [], "message": i.get("message", ""),
            "suggest": _CRITIQUE_FIX.get(str(i.get("check")))} for i in left]
    out += [{"source": "repair", "severity": "warning", "objects": [], "message": r,
             "suggest": "edit the spec so it solves without this repair"} for r in repairs]
    out += [{"source": "build", "severity": "blocking", "objects": [], "message": f"{f.get('type')}: {f.get('message')}",
             "suggest": None} for f in failed or []]
    if polys and polys.get("over_budget"):
        out.append({"source": "polys", "severity": "warning", "objects": [],
                    "message": f"{polys['faces_total']} faces, budget {polys['budget']} (heroes are never decimated)",
                    "suggest": "update_object {set: {detail: draft}} or {hero: false} on heavy objects"})
    rank = {"blocking": 0, "warning": 1}
    return sorted(out, key=lambda c: rank.get(str(c["severity"]), 2))


#: probe renders from the scene camera (Cycles, tiny, few samples; the scene's render settings are
#: restored after) and view exposure steps until the mean display luminance is in band. A step
#: assumes display ~ linear^(1/2.2): k times brighter on screen is 2.2*log2(k) stops.
_EXPOSURE = """
import bpy, json, math, os, tempfile
import numpy as np
cfg = json.loads({payload!r})
scene = bpy.context.scene
r, cy, vs = scene.render, scene.cycles, scene.view_settings
if scene.camera is None:
    result = {{"skipped": "no scene camera"}}
else:
    saved = (r.resolution_x, r.resolution_y, r.resolution_percentage, r.engine, r.filepath,
             r.image_settings.file_format, cy.samples, cy.use_denoising)
    path = os.path.join(tempfile.gettempdir(), "ai_db_exposure_probe_%d.png" % os.getpid())
    start, rounds, stats = vs.exposure, 0, None
    try:
        r.resolution_x, r.resolution_y = cfg["res"]
        r.resolution_percentage, r.engine, r.filepath = 100, "CYCLES", path
        r.image_settings.file_format = "PNG"
        cy.samples, cy.use_denoising = cfg["samples"], False
        lo, hi = cfg["band"]
        while True:
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images.load(path, check_existing=False)
            try:
                px = np.empty(len(img.pixels), dtype=np.float32)
                img.pixels.foreach_get(px)
            finally:
                bpy.data.images.remove(img)
            lum = px.reshape(-1, 4)[:, :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            stats = {{"mean": round(float(lum.mean()), 3),
                     "dark_frac": round(float((lum < cfg["clip"][0]).mean()), 3),
                     "bright_frac": round(float((lum > cfg["clip"][1]).mean()), 3)}}
            too_bright = stats["bright_frac"] > cfg["max_bright"]
            if (lo <= stats["mean"] <= hi and not too_bright) or rounds >= cfg["rounds"]:
                break
            step = 2.2 * math.log2(cfg["target"] / max(stats["mean"], 0.005))
            if too_bright:
                step = min(step, -0.5)
            step = max(-cfg["max_step"], min(cfg["max_step"], step))
            new = max(start - cfg["max_total"], min(start + cfg["max_total"], vs.exposure + step))
            if abs(new - vs.exposure) < 0.05:
                break  # at the limit: more rounds would render the same image
            vs.exposure = new
            rounds += 1
    finally:
        (r.resolution_x, r.resolution_y, r.resolution_percentage, r.engine, r.filepath,
         r.image_settings.file_format, cy.samples, cy.use_denoising) = saved
        if os.path.exists(path):
            os.remove(path)
    result = {{"rounds": rounds, **(stats or {{}}), "adjusted_by": round(vs.exposure - start, 2),
              "exposure": round(vs.exposure, 2)}}
"""


def exposure_check(outdoor: bool = False) -> dict[str, Any]:
    """Probe the scene camera's exposure and fix it (see _EXPOSURE). Adds ``in_band``. Outdoors
    in the sun the band is brighter (BUILD_EXPOSURE_BAND_OUTDOOR)."""
    from ai_db_scene import constants as c

    band = c.BUILD_EXPOSURE_BAND_OUTDOOR if outdoor else c.BUILD_EXPOSURE_BAND
    payload = json.dumps({
        "res": list(c.BUILD_EXPOSURE_PROBE_RESOLUTION), "samples": c.BUILD_EXPOSURE_PROBE_SAMPLES,
        "band": list(band), "target": c.BUILD_EXPOSURE_TARGET_OUTDOOR if outdoor else c.BUILD_EXPOSURE_TARGET,
        "clip": list(c.BUILD_EXPOSURE_CLIP),
        "max_bright": c.BUILD_EXPOSURE_MAX_BRIGHT_FRAC, "rounds": c.BUILD_EXPOSURE_MAX_ROUNDS,
        "max_step": c.BUILD_EXPOSURE_MAX_STEP, "max_total": c.BUILD_EXPOSURE_MAX_TOTAL})
    res = dict(BlenderClient(timeout=300.0).run_code(_EXPOSURE.format(payload=payload), timeout=300.0))
    if "mean" in res:
        lo, hi = band
        res["in_band"] = lo <= res["mean"] <= hi and res.get("bright_frac", 0.0) <= c.BUILD_EXPOSURE_MAX_BRIGHT_FRAC
    return res


def exposure_issue(exp: dict[str, Any]) -> dict[str, Any] | None:
    """A validator-style warning when the probe is still out of band after the adjustments."""
    if exp.get("in_band", True):
        return None
    dark = exp.get("mean", 0.0) < 0.3
    return {"check": "exposure", "severity": "warning", "objects": [], "fix": None,
            "message": (f"render too {'dark' if dark else 'bright'} after {exp.get('rounds')} exposure rounds: "
                        f"mean {exp.get('mean')}, {exp.get('dark_frac')} clipped dark, "
                        f"{exp.get('bright_frac')} clipped bright")}


#: a procedural sky world (Nishita, no HDRI) whose sun matches the daylight rig's sun lamp
#: (azimuth 135, elevation 40 when there is none): what the windows show and let in agrees with
#: the shadows. The sky's own sun disc stays off: the lamp is the direct light.
_SKY = """
import bpy, math
from mathutils import Vector
scene = bpy.context.scene
sun = next((o for o in scene.objects if o.type == "LIGHT" and o.data.type == "SUN"), None)
if sun is not None:
    d = -(sun.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0)))
    el, az = math.asin(max(-1.0, min(1.0, d.z))), math.atan2(d.y, d.x)
else:
    el, az = math.radians(40.0), math.radians(135.0)
world = scene.world or bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
nt = world.node_tree
st = next((n for n in nt.nodes if n.type == "TEX_SKY"), None)
if st is None:
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    st = nt.nodes.new("ShaderNodeTexSky")
    nt.links.new(st.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
for kind in ("NISHITA", "MULTIPLE_SCATTERING", "SINGLE_SCATTERING"):
    try:
        st.sky_type = kind
        break
    except (TypeError, ValueError):
        continue
st.sun_elevation, st.sun_rotation = el, math.pi / 2 - az
try:
    st.sun_disc = False
except AttributeError:
    pass
bg = next((n for n in nt.nodes if n.type == "BACKGROUND"), None)
if bg is not None:
    bg.inputs["Strength"].default_value = max(bg.inputs["Strength"].default_value, {strength!r})
result = {{"sky": st.sky_type.lower(), "sun_elevation": round(math.degrees(el), 1),
          "sun_azimuth": round(math.degrees(az), 1), "sun": sun.name if sun else None}}
"""


def render_options(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """build_scene's ``render`` argument -> (options, error)."""
    from ai_db_scene.constants import BUILD_RENDER_RESOLUTION, BUILD_RENDER_SAMPLES

    if not isinstance(raw, dict):
        return None, "render must be an object {engine, samples, sky, resolution, path}"
    unknown = set(raw) - {"engine", "samples", "sky", "resolution", "path"}
    if unknown:
        return None, f"render: unknown keys {sorted(unknown)}"
    engine = raw.get("engine", "cycles")
    if engine not in ("cycles", "eevee"):
        return None, "render.engine must be cycles|eevee"
    samples = raw.get("samples", BUILD_RENDER_SAMPLES)
    if not isinstance(samples, int) or isinstance(samples, bool) or not 1 <= samples <= 4096:
        return None, "render.samples must be an integer 1..4096"
    res = raw.get("resolution", list(BUILD_RENDER_RESOLUTION))
    if not (isinstance(res, list) and len(res) == 2 and all(isinstance(v, int) and 16 <= v <= 8192 for v in res)):
        return None, "render.resolution must be [w, h] in pixels (16..8192)"
    if not isinstance(raw.get("sky", False), bool):
        return None, "render.sky must be true|false"
    return {"engine": engine, "samples": samples, "resolution": res, "sky": raw.get("sky", False),
            "path": raw.get("path")}, None


def final_render(opts: dict[str, Any]) -> dict[str, Any]:
    """The optional nicer render: sky world first if asked, then the camera at full settings."""
    import tempfile
    import uuid

    from ai_db_scene.constants import BUILD_RENDER_SKY_STRENGTH

    out: dict[str, Any] = {"engine": opts["engine"], "samples": opts["samples"]}
    if opts["sky"]:
        out["sky"] = BlenderClient().run_code(_SKY.format(strength=BUILD_RENDER_SKY_STRENGTH))
    path = Path(opts["path"] or Path(tempfile.gettempdir()) / "ai_db_render" / f"{uuid.uuid4().hex}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    res = BlenderClient(timeout=1800.0).request("render", {
        "filepath": str(path), "resolution": opts["resolution"], "samples": opts["samples"],
        "engine": opts["engine"], "denoise": True}, timeout=1800.0)
    out["path"] = str(res.get("filepath") or path) if isinstance(res, dict) else str(path)
    return out


def judge_render(path: Path) -> str:
    """Render the build's eye-level camera to ``path`` (small and quick: a probe, not a final)."""
    from ai_db_scene.constants import BUILD_JUDGE_RESOLUTION, BUILD_JUDGE_SAMPLES

    path.parent.mkdir(parents=True, exist_ok=True)
    res = BlenderClient(timeout=300.0).request("render", {
        "filepath": str(path), "resolution": list(BUILD_JUDGE_RESOLUTION), "samples": BUILD_JUDGE_SAMPLES,
        "denoise": False})
    return str(res.get("filepath") or path) if isinstance(res, dict) else str(path)


#: build_scene options a version stores so edit_scene rebuilds it the same way (a rebuild with
#: another seed or geometry mode changed every untouched piece of furniture)
#: build_scene arguments a version keeps: edit_scene rebuilds with them (the brief keeps its counts checked)
_BUILD_OPTIONS = ("seed", "geometry", "asset_map", "poly_budget", "physics", "variation", "brief")


def build_options(args: dict[str, Any]) -> dict[str, Any]:
    return {k: args[k] for k in _BUILD_OPTIONS if args.get(k) is not None}


def _stored_spec(scene_id: str) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    """The spec a version was built from, with its build options and project.

    A version without a spec of its own (a fork_scene copy, a manual snapshot_scene saved as
    the next version) takes the nearest ancestor's: fork-then-edit used to fail with "no
    stored spec".
    """
    memory = _memory()
    scene = memory.restore_scene(scene_id)
    project = scene.project
    seen: set[str] = set()
    while scene is not None and scene.scene_id not in seen:
        seen.add(scene.scene_id)
        meta = scene.prompt_context.metadata if scene.prompt_context else {}
        spec = meta.get("spec") if isinstance(meta, dict) else None
        if isinstance(spec, dict):
            raw = meta.get("build")
            build: dict[str, Any] = raw if isinstance(raw, dict) else {}
            inferred = None
            if "variation" not in build:  # saved before versions stored their build options
                from ai_db_scene.legacy_build import infer_build_options

                guessed, inferred = infer_build_options(spec, scene)
                build = {**guessed, **build}
            return spec, None, {**build, "project": project, "spec_from": scene.scene_id,
                                **({"build_inferred": inferred} if inferred else {})}
        scene = memory.backend.load_scene(scene.parent_scene_id) if scene.parent_scene_id else None
    return None, (f"scene {scene_id} has no stored spec (it was not made by build_scene/edit_scene); "
                  "pass the spec to build_scene first"), {}


def _handle_edit_scene(args: dict[str, Any]) -> Any:
    """Apply ordered edit steps to a built scene's spec, rebuild it, save the next version."""
    from ai_db_scene.edits import apply_steps

    scene_id = args.get("scene_id")
    if not scene_id:
        return _err("scene_id is required")
    try:
        spec, why, stored = _stored_spec(scene_id)
    except ValueError as e:
        return _err(str(e))
    if spec is None:
        return _err(why or "no stored spec")
    # rebuild as the version was built (seed, geometry, assets) and keep it in its project,
    # unless the call says otherwise
    opts = {k: args[k] if args.get(k) is not None else stored.get(k) for k in _BUILD_OPTIONS}
    project = args.get("project") or stored.get("project") or "default"
    data, done, errors = apply_steps(spec, args.get("steps"))
    if data is None:
        return {"status": "invalid", "errors": errors, "applied_before_error": done}
    parsed, errors = _parse_spec(data)
    if parsed is None:
        return {"status": "invalid", "errors": errors, "steps_applied": done}
    safety = None
    try:  # keep manual Blender edits made since the last version before the scene is cleared
        safety = _memory().snapshot_scene(prompt=f"before edit of {scene_id}", project=project,
                                          operation_type="pre_edit", tags=["pre_edit"])
    except Exception:  # noqa: BLE001 - no Blender or empty scene: nothing to keep
        safety = None
    out = _handle_build_scene({**{k: v for k, v in opts.items() if v is not None}, "spec": data,
                               "clear": True, "snapshot": True,
                               "prompt": args.get("prompt") or "; ".join(done), "project": project,
                               "operation_type": "edit", "parent_scene_id": scene_id,
                               "metadata": {"steps": args.get("steps")}})
    return present_build({**out, "steps_applied": done, "previous_scene_id": scene_id, "safety_snapshot": safety,
                          **({"build_inferred": stored["build_inferred"]} if stored.get("build_inferred") else {})})


def export_root() -> Path:
    """The one folder export_scene writes into: $AI_DB_SCENE_EXPORT_DIR, plugins.scene.export_dir
    or SCENE_EXPORT_DIR."""
    import os

    from ai_db_scene.constants import SCENE_EXPORT_DIR

    configured = os.environ.get("AI_DB_SCENE_EXPORT_DIR")
    if not configured:
        from ai_db.config import load_config
        from ai_db_scene.config import scene_config

        try:
            configured = scene_config(load_config(check_env=False)).export_dir
        except AiDbConfigError:
            configured = None
    return Path(configured or SCENE_EXPORT_DIR).expanduser().resolve()


def export_path(path: str, fmt: str | None, root: Path) -> tuple[Path, str]:
    """Resolve ``path`` (relative to ``root``, or absolute inside it) and the format.

    Raises ValueError for a path that leaves ``root`` once resolved ("..", symlinks) or an
    unknown format; a missing extension is added from the format.
    """
    from ai_db_scene.constants import SCENE_EXPORT_FORMATS

    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    fmt = fmt or next((f for f, ext in SCENE_EXPORT_FORMATS.items() if ext == suffix), None)
    if fmt not in SCENE_EXPORT_FORMATS:
        raise ValueError(f"format must be one of {list(SCENE_EXPORT_FORMATS)} (or a path ending in one)")
    if suffix != SCENE_EXPORT_FORMATS[fmt]:
        p = p.with_name(p.name + SCENE_EXPORT_FORMATS[fmt])
    target = (p if p.is_absolute() else root / p).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"{path!r} is outside the export folder {root} (set plugins.scene.export_dir)")
    return target, fmt


def _handle_export_scene(args: dict[str, Any]) -> Any:
    from ai_db_scene.constants import SCENE_EXPORT_TIMEOUT_S

    root = export_root()
    try:
        target, fmt = export_path(str(args.get("path") or ""), args.get("format"), root)
    except ValueError as e:
        return _err(str(e))
    # a stored version or the live scene, exported by a separate Blender and imported back
    from ai_db_scene.export import export_live, export_version

    verify = args.get("verify", True) is not False
    try:
        if args.get("scene_id"):
            report = export_version(_memory(), str(args["scene_id"]), target, fmt, verify=verify,
                                    timeout=SCENE_EXPORT_TIMEOUT_S)
        else:
            report = export_live(BlenderClient(), target, fmt, verify=verify, timeout=SCENE_EXPORT_TIMEOUT_S)
    except BlenderError as e:
        return {"status": "error", "message": str(e), "error_type": e.error_type or type(e).__name__, "hint": e.hint}
    except AiDbError as e:
        return _err(str(e))
    rt = report.get("roundtrip")
    return {"status": "success" if rt is None or not rt["mismatch_count"] else "exported_with_mismatches",
            **report}


PIPELINE_TOOLS: list[tuple[str, str, dict[str, Any], Any]] = [
    ("build_scene",
     ("START HERE: one call builds the whole scene. build_scene({spec: {setting, objects, relations}}) "
      "validates the spec (odd materials and relations are coerced, with notes), lays it out (repairing "
      "conflicts, each change reported), builds it with real furniture, materials, daylight, lamps and an "
      "eye-level camera, validates it, applies the validator's own fixes and saves it (scene_id). Read the "
      "result's verdict and next: when the verdict says done, stop. build_scene() with no arguments rebuilds "
      "the last valid spec. No need for submit_scene_spec, solve_layout or apply_layout first"),
     {"type": "object",
      "properties": {
          **_SPEC_FIELDS,
          "spec_id": _SPEC_ID,
          "seed": {"type": "integer"},
          "geometry": {"enum": list(GEOMETRY_MODES)},
          "asset_map": {"type": "object", "description": "id|category -> {filepath, asset_name}"},
          "clear": {"type": "boolean", "description": "empty the Blender scene first (default false)"},
          "snapshot": {"type": "boolean", "description": "save the result as a scene version with its spec (default true)"},
          "prompt": {"type": "string", "description": "what this build is, for the scene history"},
          "project": {"type": "string"},
          "poly_budget": {"type": "integer", "minimum": 1,
                          "description": ("total faces; reports faces per object and decimates the heaviest "
                                          "non-hero objects until the scene fits (report: polys)")},
          "physics": {"type": "boolean",
                      "description": "add rigid bodies: passive structure and furniture, active objects with dynamic: true"},
          "judge": {"type": "boolean",
                    "description": ("render the build camera (render) and return a structured critique (validator issues, "
                                    "missing objects, repairs, failed commands) with suggested edit_scene steps")},
          "render_path": {"type": "string", "description": "judge: where to write the render (default: a temp file)"},
          "render": {"type": "object",
                     "description": ("a nicer final render after the build (default: none, fast): {engine: cycles|eevee, "
                                     "samples, sky: true (procedural sky world aligned with the daylight sun), "
                                     "resolution: [w, h], path}; report: final_render.path"),
                     "properties": {"engine": {"enum": ["cycles", "eevee"]}, "samples": {"type": "integer", "minimum": 1},
                                    "sky": {"type": "boolean"}, "resolution": {"type": "array", "items": {"type": "integer"}},
                                    "path": {"type": "string"}}},
          "brief": {"type": "string", "description": "pass the user's request text: build_scene then checks the "
                                                      "objects it asks for and their counts ('twelve desks'); "
                                                      "without it counts are not checked"},
          "exposure_check": {"type": "boolean",
                             "description": ("default true: a tiny probe render from the camera; the view exposure is "
                                             "adjusted until the image is neither black nor blown out (report: exposure)")}}},
     _handle_build_scene),
    ("get_scene_spec_schema",
     "SceneSpec JSON schema + category vocabulary with typical dims + relations + up to 3 exemplar specs from past successful runs",
     {"type": "object", "properties": {"brief": {"type": "string", "description": "Brief, to retrieve similar exemplars"},
                                       "project": {"type": "string", "description": "lessons of this project (and global ones)"}}},
     _handle_get_scene_spec_schema),
    ("submit_scene_spec",
     ("Optional (build_scene validates too): check a SceneSpec; returns exact errors, or ok + the normalized "
      "spec + a spec_id. The valid spec is remembered: build_scene() with no arguments builds it"),
     {"type": "object", "properties": {**_SPEC_FIELDS}},
     _handle_submit_scene_spec),
    ("solve_layout",
     ("Low-level, rarely needed: place every object of a SceneSpec; returns placements or the conflicting "
      "relations. build_scene already does this (and repairs conflicts): do not loop solve_layout, call build_scene"),
     {"type": "object", "properties": {**_SPEC_FIELDS, "spec_id": _SPEC_ID, "seed": {"type": "integer"}}},
     _handle_solve_layout),
    ("edit_scene",
     ("Refactor a scene made by build_scene in ordered steps: add_object {object: {id, category}, relations}, "
      "remove_object {id}, update_object {id, set: count|material|hero|grid|category|detail}, "
      "move {id, to: {against_wall: wall_south} | {beside: sofa, gap_m: 0.5}} (replaces the object's own "
      "relations; positions always come from relations), add_relation {relation: {a, rel, b}}, "
      "remove_relation {a, rel, b}, update_setting {set: bounds_m|style|floor_material|wall_material|shape|points|kind}, "
      "rotate {id, deg: 90|180|270} (turns it from the way its relations face it). "
      "Common aliases (delete, add, target/name for id, flat fields) are accepted. Applies them to the scene's stored "
      "spec, rebuilds the scene (same seed: untouched objects mostly stay), validates, and saves the result as the "
      "next version (restore_scene on an earlier id undoes it). The current Blender scene is snapshotted first"),
     {"type": "object",
      "properties": {
          "scene_id": {"type": "string", "description": "a version saved by build_scene or edit_scene"},
          "steps": {"type": "array", "items": {"type": "object"},
                    "description": ("e.g. [{op: add_object, object: {id: lamp, category: floor_lamp}, relations: [{a: lamp, rel: beside, b: sofa}]}, "
                                    "{op: update_object, id: sofa, set: {material: velvet:#5b2a4a}}, {op: move, id: table, to: {in_front_of: sofa}}]")},
          "seed": {"type": "integer", "description": "default: the seed the version was built with"},
          "geometry": {"enum": list(GEOMETRY_MODES), "description": "default: the version's own"},
          "brief": {"type": "string", "description": ("the user's request: counts are checked against it "
                                                      "(default: the brief the version was built with)")},
          "prompt": {"type": "string"},
          "project": {"type": "string", "description": "default: the version's project"}},
      "required": ["scene_id", "steps"]},
     _handle_edit_scene),
    ("apply_layout",
     ("Low-level, rarely needed (build_scene does all of this): "
      "create the room shell (floor + walls if indoor) and one object per placement in Blender as one batch; "
      "placeholder cubes unless asset_map maps an object id / category to {filepath, asset_name}. "
      "Tags custom props category/hero/support"),
     {"type": "object",
      "properties": {
          **_SPEC_FIELDS,
          "placements": {"type": "array", "items": {"type": "object"}, "description": "From solve_layout"},
          "asset_map": {"type": "object", "description": "id|category -> {filepath, asset_name}"},
          "geometry": {"enum": list(GEOMETRY_MODES),
                       "description": "generated (default): parametric furniture where a generator exists; boxes: placeholders only"}},
      "required": ["placements"]},
     _handle_apply_layout),
    ("record_run",
     "Persist a pipeline run record (brief, spec, placements, actions, validation reports, judge scores, cost, outcome)",
     {"type": "object",
      "properties": {
          "brief": {"type": "string"}, "spec": {"type": "object"},
          "placements": {"type": "array"}, "actions": {"type": "array"},
          "validation_reports": {"type": "array"}, "judge_scores": {"type": "array"},
          "token_cost": {"type": "number"}, "wall_time_s": {"type": "number"},
          "outcome": {"type": "string", "enum": list(RUN_OUTCOMES)}, "project": {"type": "string"}},
      "required": ["brief"]},
     _handle_record_run),
    ("export_scene",
     ("Export the current Blender scene, or a stored version (`scene_id`), to glb, gltf, usd, usda, usdc or "
      "usdz (format from `format` or the path's extension). `path` is relative to the export folder "
      "(plugins.scene.export_dir); paths outside it are refused. Returns the file size and object count; the "
      "file is imported back and checked (roundtrip: missing objects, transforms, materials)"),
     {"type": "object",
      "properties": {"path": {"type": "string", "description": "e.g. living_room.glb"},
                     "format": {"enum": ["glb", "gltf", "usd", "usda", "usdc", "usdz"]},
                     "scene_id": {"type": "string", "description": "export this stored version instead of the live scene"},
                     "verify": {"type": "boolean", "default": True,
                                "description": "import the file back and compare (roundtrip)"}},
      "required": ["path"]},
     _handle_export_scene),
    ("search_exemplars",
     "Find past successful, judge-approved runs similar to a brief (never returns failed or low-score runs)",
     {"type": "object",
      "properties": {"brief": {"type": "string"}, "min_score": {"type": "number", "default": 4.0},
                     "top_k": {"type": "integer", "default": 3}, "project": {"type": "string"}},
      "required": ["brief"]},
     _handle_search_exemplars),
]


def _handle_resolve_asset(args: dict[str, Any]) -> Any:
    from ai_db_scene.assets import AssetResolver
    from ai_db_scene.blender_client import try_get_blender_client

    client = try_get_blender_client()
    if client is None:
        return _err("Blender is not running or its socket server is not started.")
    try:
        cands = AssetResolver(client).resolve_asset(args["category"], args.get("style"), int(args.get("top_k", 3)))
    except BlenderError as e:
        return {"status": "error", "message": str(e), "error_type": e.error_type or type(e).__name__, "hint": e.hint}
    return {"status": "success", "candidates": [c.to_dict() for c in cands]}


PIPELINE_TOOLS.append(
    ("resolve_asset",
     ("Find Blender library assets for a category/style, ranked by name/tag match, with measured dimensions "
      "(appends candidates to measure them). Pass the result to apply_layout's asset_map"),
     {"type": "object",
      "properties": {"category": {"type": "string"}, "style": {"type": "string"},
                     "top_k": {"type": "integer", "default": 3}},
      "required": ["category"]},
     _handle_resolve_asset)
)
