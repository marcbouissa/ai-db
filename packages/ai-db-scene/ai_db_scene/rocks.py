"""Boulders: irregular faceted rocks, not flattened domes.

The rock recipe was a squashed sphere with sphere lumps pressed in: it read as a smooth dome or a
pile of eggs (Marc's review list: "rocks as flattened domes"). A generated rock is a convex hull
of jittered points on a squashed ellipsoid, cut by a few cleavage planes (the big flat faces of a
broken boulder), its faces split and nudged by noise so no face is a perfect plane, its foot sunk a
little into the ground, flat-shaded with a sharp-edge bevel look. Every rock is its own shape
(seeded by its id) and one of a few stone tones.
"""

from __future__ import annotations

import json
import zlib
from typing import Any

#: stone tones a rock picks from (a plain "stone" material keeps them)
TONES = ("stone:#6b665e", "stone:#7a746a", "stone:#5a554e", "stone:#837d72", "stone:#646058")
#: of the height below the placement's base (terrain.settle beds it in by 0.15 already): a boulder sits
#: in the ground, not on it, also where the shore bed drops away under the shallows
SINK = 0.1

_ROCK = """
import bpy, bmesh, json, math, random
from mathutils import Vector, noise
cfg = json.loads({payload!r})
rng = random.Random(cfg["seed"])
name = cfg["name"]
for nm in (name + "_stone", name):
    old = bpy.data.objects.get(nm)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
bm = bmesh.new()
n = 48
golden = math.pi * (3 - math.sqrt(5))
squash = rng.uniform(0.55, 0.8)  # boulders are broader than tall
for k in range(n):
    z = 1 - 2 * (k + 0.5) / n
    r = math.sqrt(max(0.0, 1 - z * z))
    a = k * golden + rng.uniform(-0.25, 0.25)
    j = rng.uniform(0.78, 1.12)
    bm.verts.new((r * math.cos(a) * j, r * math.sin(a) * j * rng.uniform(0.8, 1.0), z * j * squash))
bmesh.ops.convex_hull(bm, input=bm.verts[:])
# cleavage planes: a broken boulder has a few big flat faces
for _ in range(rng.randint(2, 4)):
    a, e = rng.uniform(0, 2 * math.pi), rng.uniform(-0.2, 0.9)
    no = Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=no * rng.uniform(0.55, 0.8), plane_no=no, clear_outer=True)
    cut = [g for g in res["geom_cut"] if isinstance(g, bmesh.types.BMEdge)]
    if cut:
        bmesh.ops.holes_fill(bm, edges=cut, sides=0)
# the foot: flat where it meets the ground
geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, -squash * 0.55), plane_no=(0, 0, -1), clear_outer=True)
cut = [g for g in res["geom_cut"] if isinstance(g, bmesh.types.BMEdge)]
if cut:
    bmesh.ops.holes_fill(bm, edges=cut, sides=0)
bmesh.ops.triangulate(bm, faces=bm.faces[:])
bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
off = Vector((rng.uniform(0, 50), rng.uniform(0, 50), rng.uniform(0, 50)))
for v in bm.verts:  # no face a perfect plane: chips and dents
    v.co += v.co.normalized() * 0.06 * noise.noise(v.co * 2.5 + off)
lo = min(v.co.z for v in bm.verts)
hi = max(v.co.z for v in bm.verts)
xs = [v.co.x for v in bm.verts]
ys = [v.co.y for v in bm.verts]
sx, sy, sz = cfg["dims"]
kx, ky, kz = sx / (max(xs) - min(xs)), sy / (max(ys) - min(ys)), sz / (hi - lo)
cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
for v in bm.verts:
    v.co = Vector(((v.co.x - cx) * kx, (v.co.y - cy) * ky, (v.co.z - lo) * kz - cfg["sink"] * sz))
me = bpy.data.meshes.new(name + "_stone")
bm.to_mesh(me)
bm.free()
for p in me.polygons:
    p.use_smooth = False
# an asset root (empty) with the stone under it, as a recipe builds it: the validator checks
# assets as a whole (validate_asset), and a bedded rock reaches into the ground by design
ob = bpy.data.objects.new(name, None)
ob.empty_display_type = "PLAIN_AXES"
ob["asset"] = name
bpy.context.scene.collection.objects.link(ob)
stone = bpy.data.objects.new(name + "_stone", me)
bpy.context.scene.collection.objects.link(stone)
stone.parent = ob
ob.location = cfg["location"]
ob.rotation_euler = (0.0, 0.0, math.radians(cfg["rot_z"]))
result = {{"rock": ob.name, "stone": stone.name, "faces": len(me.polygons)}}
"""


def tone(pid: str, seed: int = 0) -> str:
    return TONES[(zlib.crc32(pid.encode()) + seed) % len(TONES)]


def rock_commands(pid: str, dims: Any, base: list[float], rot_z: float, seed: int,
                  material: str | None = None) -> list[dict[str, Any]]:
    """run_code building rock ``pid`` (base centre ``base``, Blender frame), then its stone."""
    payload = {"name": pid, "dims": [float(d) for d in dims], "location": [float(v) for v in base],
               "rot_z": float(rot_z), "seed": int(seed), "sink": SINK}
    return [{"type": "run_code", "params": {"code": _ROCK.format(payload=json.dumps(payload))}},
            {"type": "apply_material", "params": {"name": f"{pid}_stone", "material": material or tone(pid, seed)}}]
