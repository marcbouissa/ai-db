"""Grass with blades where the eye lands: around the scene's objects and in the camera's foreground.

A grass ground was a shaded plane (a noise texture), flat as a carpet next to a bench or under
the camera (Marc's review list: "grass has no blades"). On a grass terrain the build now grows
real blades - each a thin, tapering, bent strip - but only in patches that matter: within
PATCH_M of every object and along the camera's line of sight up to FOREGROUND_M. Blades stand
only on the ground itself (not on a plaza, a path or in water: a ray from above must hit the
ground first), thinning out toward a patch's edge. The count is capped (MAX_BLADES) so a render
stays cheap; one mesh holds them all. build_scene grows them after its checks (they change neither
the layout nor the camera, and 45k blades made the checks' ray casts take a minute).
"""

from __future__ import annotations

import json
import math
from typing import Any

PATCH_M = 3.0  # blades this far round every object
FOREGROUND_M = 9.0  # and along the camera's view this far
DENSITY_PER_M2 = 260.0
MAX_BLADES = 45000
BLADE_H_M = (0.05, 0.13)
BLADE_W_M = (0.004, 0.008)
GRASS_MATERIAL = "foliage:#56762f"
NAME = "grass_blades"

_GRASS = """
import bpy, bmesh, json, math, random
from mathutils import Vector
from mathutils.bvhtree import BVHTree
cfg = json.loads({payload!r})
rng = random.Random(cfg["seed"])
old = bpy.data.objects.get(cfg["name"])
if old is not None:
    bpy.data.objects.remove(old, do_unlink=True)
ground = bpy.data.objects.get(cfg["ground"])
made = 0
if ground is not None:
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene

    def tree(o):  # one BVH per object (scene.ray_cast rebuilt them per ray: 20 ms a blade)
        return BVHTree.FromObject(o, dg), o.matrix_world.copy(), o.matrix_world.inverted()

    floor = tree(ground)
    # what covers the ground: a plaza, a path, the water, the objects' own feet
    covers = [tree(o) for o in scene.objects if o.type == "MESH" and o is not ground and o.name != cfg["name"]
              and (o.name.startswith(("plaza_", "path_")) or "water" in o.name.lower())]

    def cast(t, x, y):
        bvh, mw, inv = t
        hit = bvh.ray_cast(inv @ Vector((x, y, 60.0)), (inv.to_3x3() @ Vector((0, 0, -1))).normalized())
        if hit[0] is None:
            return None, None
        return mw @ hit[0], (mw.to_3x3() @ hit[1]).normalized()
    bm = bmesh.new()
    pts = []
    def keep(f):  # thinning out toward a patch's edge: no visible rim of lawn round a patch
        return f < 0.55 or rng.random() < (1.0 - f) / 0.45
    for cx, cy, r in cfg["patches"]:
        n = int(cfg["density"] * math.pi * r * r)
        for _ in range(n):
            a, d = rng.uniform(0, 2 * math.pi), r * math.sqrt(rng.random())
            if keep(d / r):
                pts.append((cx + d * math.cos(a), cy + d * math.sin(a)))
    (x0, y0), (x1, y1), w = cfg["view"]
    if w > 0:
        L = math.hypot(x1 - x0, y1 - y0)
        n = int(cfg["density"] * L * w * 0.6)
        for _ in range(n):  # a wedge widening away from the camera
            t = rng.random() ** 0.7
            s = rng.uniform(-0.5, 0.5) * w * (0.25 + 0.75 * t)
            if not keep(max(t, abs(s) / (0.5 * w * (0.25 + 0.75 * t) + 1e-9))):
                continue
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            pts.append((x0 + ux * L * t - uy * s, y0 + uy * L * t + ux * s))
    rng.shuffle(pts)
    seen = set()
    for x, y in pts:
        if made >= cfg["max"]:
            break
        key = (round(x * 40), round(y * 40))
        if key in seen:
            continue
        seen.add(key)
        at, nrm = cast(floor, x, y)
        if at is None or nrm.z < 0.6:
            continue
        if any((c := cast(t, x, y)[0]) is not None and c.z > at.z - 0.01 for t in covers):
            continue
        h = rng.uniform(*cfg["h"]) * (0.7 if rng.random() < 0.3 else 1.0)
        wd = rng.uniform(*cfg["w"])
        az = rng.uniform(0, 2 * math.pi)
        lean = rng.uniform(0.05, 0.45)
        side = Vector((math.cos(az), math.sin(az), 0.0))
        bend = Vector((-math.sin(az), math.cos(az), 0.0)) * lean
        base = Vector((at.x, at.y, at.z - 0.004))
        mid = base + Vector((0, 0, h * 0.55)) + bend * h * 0.25
        tip = base + Vector((0, 0, h)) + bend * h * 0.7
        v = [bm.verts.new(base - side * wd / 2), bm.verts.new(base + side * wd / 2),
             bm.verts.new(mid + side * wd * 0.35), bm.verts.new(mid - side * wd * 0.35), bm.verts.new(tip)]
        bm.faces.new((v[0], v[1], v[2], v[3]))
        bm.faces.new((v[3], v[2], v[4]))
        made += 1
    me = bpy.data.meshes.new(cfg["name"])
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(cfg["name"], me)
    scene.collection.objects.link(ob)
    ob["category"] = "ground"  # part of the terrain: no checklist object, no clash
result = {{"blades": made}}
"""


def grass_commands(spec: Any, placements: list[Any], off: tuple[float, float, float],
                   camera: dict[str, Any] | None, seed: int = 0) -> list[dict[str, Any]]:
    """run_code growing the blades, then their material (nothing unless the ground is grass)."""
    from ai_db_scene.materials import material_for
    from ai_db_scene.terrain import terrain_of

    ter = terrain_of(spec)
    if ter is None or ter.ground != "grass":
        return []
    patches = [[round(p.location[0] + off[0], 3), round(p.location[1] + off[1], 3),
                round(PATCH_M + max(p.dims[0], p.dims[1]) / 2, 3)] for p in placements]
    view: list[Any] = [[0.0, 0.0], [0.0, 0.0], 0.0]
    if camera is not None:
        x, y = camera["location"][0], camera["location"][1]
        yaw = math.radians(camera["rotation"][2])
        dx, dy = -math.sin(yaw), math.cos(yaw)
        view = [[round(x, 3), round(y, 3)], [round(x + dx * FOREGROUND_M, 3), round(y + dy * FOREGROUND_M, 3)],
                round(FOREGROUND_M * 0.9, 3)]
    payload = {"name": NAME, "ground": f"{ter.ground}_ground", "patches": patches, "view": view,
               "density": DENSITY_PER_M2, "max": MAX_BLADES, "h": list(BLADE_H_M), "w": list(BLADE_W_M),
               "seed": seed}
    return [{"type": "run_code", "params": {"code": _GRASS.format(payload=json.dumps(payload))}},
            {"type": "apply_material", "params": {"name": NAME, "material": material_for("", GRASS_MATERIAL)}}]
