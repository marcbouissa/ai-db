"""Generated plants for outdoor scenes: palm tree, fern, tropical bush.

Each is grown in Blender from a seed (run_code, like the broadleaf tree in scene_pipeline): a
palm's curved, ringed trunk and arching fronds with V-set leaflets; a fern's rosette of pinnate
fronds; a tropical bush's broad, folded leaves on stalks. Every plant is fitted to the
placement's exact [w, d, h] (base at z=0, footprint centred), so it never changes the layout.
"""

from __future__ import annotations

import json
from typing import Any

#: category -> generator kind
GENERATED_PLANTS = {"palm_tree": "palm", "fern": "fern", "tropical_bush": "tropical_bush"}
#: part suffix -> material when the spec names none (the spec's material goes on the leaves)
PART_MATERIALS = {
    "palm": {"trunk": "stone:#7a6a55", "leaves": "foliage:#4f7d2e", "nuts": "wood:dark"},
    "fern": {"leaves": "foliage:#3f6f2a", "base": "foliage:#3a4a26"},
    "tropical_bush": {"leaves": "foliage:#2f6b2c", "stems": "foliage:#4f7a35"},
}


def has_plant(category: str) -> bool:
    return category in GENERATED_PLANTS


def plant_commands(category: str, name: str, dims: tuple[float, float, float] | list[float],
                   base: list[float], rot_z_deg: float, seed: int, leaf_material: str | None) -> list[dict[str, Any]]:
    """run_code growing the plant, then its part materials (``leaf_material``: the spec's)."""
    kind = GENERATED_PLANTS[category]
    payload = {"name": name, "kind": kind, "dims": [float(v) for v in dims], "seed": int(seed),
               "location": [float(v) for v in base], "rot_z": float(rot_z_deg)}
    cmds: list[dict[str, Any]] = [{"type": "run_code", "params": {
        "code": "import json\nCFG = json.loads(" + repr(json.dumps(payload)) + ")\n" + _PLANT_CODE}}]
    for part, mat in PART_MATERIALS[kind].items():
        chosen = leaf_material if part == "leaves" and leaf_material else mat
        cmds.append({"type": "apply_material", "params": {
            "name": f"{name}_{part}", "material": chosen if chosen.startswith("color") or ":proj=" in chosen
            else f"{chosen}:proj=box"}})
    return cmds


_PLANT_CODE = r'''
import bpy, bmesh, math, random
from mathutils import Vector, Matrix

rng = random.Random(CFG["seed"])
W, D, H = CFG["dims"]
R = min(W, D) / 2
UP = Vector((0, 0, 1))

def ring(bm, centre, tangent, radius, n):
    t = tangent.normalized()
    a = t.orthogonal().normalized()
    b = t.cross(a).normalized()
    return [bm.verts.new(centre + (a * math.cos(2 * math.pi * k / n) + b * math.sin(2 * math.pi * k / n)) * radius)
            for k in range(n)]

def bridge(bm, lo, hi):
    n = len(lo)
    for k in range(n):
        bm.faces.new((lo[k], lo[(k + 1) % n], hi[(k + 1) % n], hi[k]))

def tube(bm, pts, radii, n=8, cap=True):
    loops = []
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])
        loops.append(ring(bm, p, t if t.length > 1e-9 else UP, radii[i], n))
    for lo, hi in zip(loops, loops[1:]):
        bridge(bm, lo, hi)
    if cap:
        bm.faces.new(list(reversed(loops[0])))
        bm.faces.new(loops[-1])

def ribbon(bm, pts, widths, side_of):
    """A flat strip along pts (a stalk / rachis seen from any angle: two crossed strips)."""
    for twist in (0.0, math.pi / 2):
        left, right = [], []
        for i, p in enumerate(pts):
            t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
            s = side_of(t)
            s = (Matrix.Rotation(twist, 3, t) @ s).normalized()
            left.append(bm.verts.new(p + s * widths[i] / 2))
            right.append(bm.verts.new(p - s * widths[i] / 2))
        for i in range(len(pts) - 1):
            bm.faces.new((left[i], right[i], right[i + 1], left[i + 1]))

def side_vec(t):
    s = t.cross(UP)
    return s.normalized() if s.length > 1e-6 else Vector((1, 0, 0))

def leaflet(bm, base, direction, length, width, droop, fold):
    """A narrow leaf: base, two edge points a third along, a midrib crease, the tip."""
    d = direction.normalized()
    s = side_vec(d)
    up = s.cross(d).normalized()
    mid = base + d * length * 0.45 - UP * droop * 0.3
    tip = base + d * length - UP * droop
    v0 = bm.verts.new(base)
    v1 = bm.verts.new(mid + s * width / 2 - up * fold)
    v2 = bm.verts.new(mid + up * fold * 0.4)
    v3 = bm.verts.new(mid - s * width / 2 - up * fold)
    v4 = bm.verts.new(tip)
    bm.faces.new((v0, v1, v2))
    bm.faces.new((v0, v2, v3))
    bm.faces.new((v1, v4, v2))
    bm.faces.new((v2, v4, v3))

def frond(bm_leaf, bm_stalk, start, az, elev, length, steps, droop_step, leaf_len, leaf_w, leaf_angle, pairs=1,
          stalk_w=0.03, taper="palm"):
    """A pinnate frond: a rachis arching under its weight, leaflets in a V along both sides
    (``pairs`` per rachis step); ``taper`` palm: longest mid-frond, fern: longest near the base."""
    pts, dirs = [start.copy()], []
    p, e = start.copy(), elev
    for i in range(steps):
        d = Vector((math.cos(e) * math.cos(az), math.cos(e) * math.sin(az), math.sin(e)))
        dirs.append(d)
        p = p + d * (length / steps)
        pts.append(p.copy())
        e -= droop_step * (0.6 + 0.8 * i / steps)
    ribbon(bm_stalk, pts, [stalk_w * (1 - 0.7 * i / steps) for i in range(len(pts))], side_vec)
    for i in range(2, steps + 1):
        for k in range(pairs):
            f = (k + 1) / pairs
            u = (i - 1 + f) / steps
            if taper == "fern":
                shape = (1 - u) ** 0.55 * min(1.0, 0.35 + u * 4)
            else:
                shape = math.sin(math.pi * min(u * 1.05, 1.0)) ** 0.6
            ll = leaf_len * shape * rng.uniform(0.85, 1.1) + leaf_len * 0.06
            d = dirs[i - 1]
            at = pts[i - 1].lerp(pts[i], f)
            s = side_vec(d)
            for sgn in (-1, 1):
                ld = (s * sgn * math.cos(leaf_angle) - UP * math.sin(leaf_angle) * 0.6 + d * 0.45).normalized()
                leaflet(bm_leaf, at, ld, ll, leaf_w * ll, ll * 0.25, leaf_w * ll * 0.15)

def blade(bm, base, direction, length, width, droop):
    """A broad leaf: an elliptic blade folded along the midrib, drooping toward the tip."""
    d = direction.normalized()
    s = side_vec(d)
    rows, cols = 7, 2
    grid = []
    for i in range(rows + 1):
        u = 0.03 + 0.94 * i / rows  # never a zero-width row: no degenerate faces
        half = width / 2 * (math.sin(math.pi * u) ** 0.75)
        c = base + d * length * u - UP * droop * u * u
        row = []
        for j in range(-cols, cols + 1):
            v = j / cols
            row.append(bm.verts.new(c + s * half * v + UP * (0.12 * half * (1 - abs(v)) - 0.08 * half * v * v)))
        grid.append(row)
    for i in range(rows):
        for j in range(2 * cols):
            bm.faces.new((grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]))

def mesh_obj(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob

name = CFG["name"]
for suffix in ("", "_trunk", "_leaves", "_nuts", "_base", "_stems"):
    old = bpy.data.objects.get(name + suffix)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
parts = []
kind = CFG["kind"]
if kind == "palm":
    trunk_h = H * rng.uniform(0.72, 0.8)
    lean = trunk_h * rng.uniform(0.08, 0.18)
    lean_az = rng.uniform(0, 2 * math.pi)
    def path(t):
        return Vector((lean * t ** 1.8 * math.cos(lean_az), lean * t ** 1.8 * math.sin(lean_az), trunk_h * t))
    r0 = max(0.07, 0.027 * H)
    n = max(12, int(trunk_h / 0.12))
    pts, radii = [], []
    for i in range(n + 1):
        t = i / n
        r = r0 * (1 - 0.32 * t) * (1 + 0.5 * max(0.0, 1 - t / 0.07))  # flared foot, tapering up
        radii.append(r * (1.07 if i % 2 else 0.95))  # the leaf-scar rings of the bark
        pts.append(path(t))
    bm = bmesh.new()
    tube(bm, pts, radii, 12)
    top = pts[-1]
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=r0 * 1.15, matrix=Matrix.Translation(top + UP * r0 * 0.3))
    parts.append(mesh_obj(name + "_trunk", bm))
    bl, bs = bmesh.new(), bmesh.new()
    nf = rng.randint(11, 15)
    flen = R * rng.uniform(1.0, 1.15)
    for k in range(nf):
        az = 2 * math.pi * k / nf + rng.uniform(-0.2, 0.2)
        elev = rng.uniform(-0.15, 0.8) if k % 3 else rng.uniform(0.6, 1.05)
        frond(bl, bl, top + UP * r0 * 0.5, az, elev, flen * rng.uniform(0.85, 1.05), 16,
              rng.uniform(0.08, 0.13), flen * 0.36, 0.11, math.radians(rng.uniform(25, 40)), pairs=2,
              stalk_w=r0 * 0.35)
    parts.append(mesh_obj(name + "_leaves", bl))
    bs.free()
    bn = bmesh.new()
    for k in range(rng.randint(3, 6)):
        a = rng.uniform(0, 2 * math.pi)
        c = top + Vector((math.cos(a) * r0 * 1.2, math.sin(a) * r0 * 1.2, -r0 * rng.uniform(0.3, 1.2)))
        bmesh.ops.create_icosphere(bn, subdivisions=2, radius=max(0.06, 0.012 * H), matrix=Matrix.Translation(c))
    parts.append(mesh_obj(name + "_nuts", bn))
elif kind == "fern":
    bl = bmesh.new()
    nf = rng.randint(18, 26)
    for k in range(nf):  # a shuttlecock: inner fronds upright, outer ones arching out and down
        az = 2 * math.pi * k / nf + rng.uniform(-0.15, 0.15)
        elev = rng.uniform(0.75, 1.3) if k % 3 else rng.uniform(0.45, 0.8)
        frond(bl, bl, Vector((0, 0, 0.04 * H)), az, elev, R * rng.uniform(1.0, 1.3), 20,
              rng.uniform(0.07, 0.12), R * 0.2, 0.2, math.radians(rng.uniform(4, 14)), pairs=2,
              stalk_w=0.012, taper="fern")
    parts.append(mesh_obj(name + "_leaves", bl))
    bb = bmesh.new()
    rad = max(0.05, 0.12 * R)
    bmesh.ops.create_icosphere(bb, subdivisions=2, radius=rad,
                               matrix=Matrix.Translation((0, 0, 0.0)) @ Matrix.Diagonal((1.0, 1.0, 0.35, 1.0)))
    parts.append(mesh_obj(name + "_base", bb))
else:  # tropical_bush: broad leaves on stalks from one clump
    bl, bs = bmesh.new(), bmesh.new()
    nl = rng.randint(24, 32)
    for k in range(nl):  # three tiers: low leaves spreading out, a middle ring, a few upright inside
        az = 2 * math.pi * k / nl * 2.618 + rng.uniform(-0.25, 0.25)
        tier = k % 3
        inner = tier == 2 and k % 2 == 0
        elev = rng.uniform(0.95, 1.2) if inner else (rng.uniform(0.3, 0.6) if tier == 0 else rng.uniform(0.6, 0.95))
        stalk = H * (rng.uniform(0.35, 0.5) if inner else rng.uniform(0.15, 0.3) if tier == 0 else rng.uniform(0.25, 0.42))
        d = Vector((math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev)))
        end = d * stalk
        tube(bs, [Vector((0, 0, 0)), d * stalk * 0.5 + Vector((0, 0, 0.02)), end], [0.02, 0.016, 0.012], 5)
        out = Vector((d.x, d.y, 0.25)).normalized() if not inner else Vector((d.x, d.y, 0.9)).normalized()
        ln = R * rng.uniform(0.75, 1.0) * (0.75 if inner else 1.0)
        blade(bl, end, out, ln, ln * rng.uniform(0.5, 0.65), ln * rng.uniform(0.25, 0.5))
    parts.append(mesh_obj(name + "_leaves", bl))
    parts.append(mesh_obj(name + "_stems", bs))

def closed(ob, thickness):
    """Leaf sheets as thin closed shells (Solidify, applied): manifold, so validate_scene's mesh
    fixes (fill holes) never paste faces over open leaves."""
    m = ob.modifiers.new("ai_db_solid", "SOLIDIFY")
    m.thickness, m.offset, m.use_rim = thickness, 0.0, True
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    old = ob.data
    ob.modifiers.remove(m)
    ob.data = me
    me.name = old.name
    bpy.data.meshes.remove(old)

for ob in parts:
    if ob.name.endswith(("_leaves", "_stems")):
        closed(ob, max(0.003, 0.0012 * H))
for ob in parts:  # drooping tips rest on the ground instead of reaching into it
    for v in ob.data.vertices:
        if v.co.z < 0.004 * H:
            v.co.z = 0.004 * H
    ob.data.update()
root = bpy.data.objects.new(name, None)
bpy.context.scene.collection.objects.link(root)
root["asset"] = name  # one asset: the validator checks its parts together
for ob in parts:
    ob.parent = root
bpy.context.view_layer.update()
pts = [ob.matrix_world @ Vector(c) for ob in parts for c in ob.bound_box]
lo = Vector([min(p[i] for p in pts) for i in range(3)])
hi = Vector([max(p[i] for p in pts) for i in range(3)])
size = hi - lo
sc = Vector([CFG["dims"][i] / max(size[i], 1e-6) for i in range(3)])
for ob in parts:  # exact footprint and height: centred on the placement, base on the ground
    ob.scale = sc
    ob.location = Vector((-(lo.x + hi.x) / 2 * sc.x, -(lo.y + hi.y) / 2 * sc.y, -lo.z * sc.z))
root.location = CFG["location"]
root.rotation_euler[2] = math.radians(CFG["rot_z"])
result = {"plant": name, "parts": [ob.name for ob in parts]}
'''
