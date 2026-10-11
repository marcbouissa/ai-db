"""Showcase: the Oval Office, built through the addon registry (the same handlers agents call).

Run:  ~/bin/blender -b --factory-startup --python eval/showcase/oval_office.py -- <addon_root> <out_dir>
Furniture is declarative build_asset specs (no hand-written bpy); only the elliptical room shell is
custom geometry, since the layout solver handles rectangular rooms only.
Approximate public dimensions: oval 10.9 m x 8.8 m, ceiling 5.6 m. Long axis = Y; desk end = +Y.
"""
import importlib
import json
import math
import os
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
ADDON_ROOT, OUT = argv[0], argv[1]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ADDON_ROOT)

import bmesh  # noqa: E402
import bpy  # noqa: E402

_pkg = next(d for d in os.listdir(ADDON_ROOT) if os.path.isfile(os.path.join(ADDON_ROOT, d, "registry.py")))
registry = importlib.import_module(f"{_pkg}.registry")

A, B, H = 8.8 / 2, 10.9 / 2, 5.6   # semi-axes x, y; ceiling height
LOG = []


def run(handler, **params):
    res = registry.execute([{"handler": handler, "params": params}], undo=False)[0]
    LOG.append({"handler": handler, "ok": res.get("ok"), "error": res.get("error")})
    if not res.get("ok"):
        raise RuntimeError(f"{handler}: {res.get('error')} {res.get('details', '')}")
    return res.get("result")


def place(name, x, y, rz=0.0, z=0.0):
    run("modify_object", name=name, location=[x, y, z], rotation=[0, 0, rz])


def asset(spec, x, y, rz=0.0, z=0.0):
    r = run("build_asset", spec=spec)
    name = r["name"] if isinstance(r, dict) else r
    place(name, x, y, rz, z)
    return name


# ---------------------------------------------------------------- room shell (custom geometry)
bpy.ops.wm.read_factory_settings(use_empty=True)


def ellipse_pt(t, ra=A, rb=B):
    return ra * math.cos(t), rb * math.sin(t)


def wall_ring(name, ra, rb, z0, z1, thick, arcs=None):
    """Watertight elliptical wall (inner semi-axes ra, rb; grows outward by `thick`).
    arcs=None -> closed ring; else list of (t_start, t_end) open arcs, each capped at both ends."""
    n = 192
    bm = bmesh.new()

    def strip(ta, tb, closed):
        steps = n if closed else max(2, int(n * (tb - ta) / (2 * math.pi)) + 1)
        cols = []
        for k in range(steps if closed else steps + 1):
            t = ta + (tb - ta) * k / steps
            x, y = ellipse_pt(t, ra, rb)
            nx, ny = x / ra**2, y / rb**2
            ln = math.hypot(nx, ny)
            ox, oy = x + nx / ln * thick, y + ny / ln * thick
            cols.append([bm.verts.new(v) for v in ((x, y, z0), (ox, oy, z0), (ox, oy, z1), (x, y, z1))])
        m = len(cols)
        for k in range(m if closed else m - 1):
            a, b = cols[k], cols[(k + 1) % m]
            for j in range(4):
                bm.faces.new((a[j], b[j], b[(j + 1) % 4], a[(j + 1) % 4]))
        if not closed:
            bm.faces.new(cols[0])
            bm.faces.new(list(reversed(cols[-1])))

    if arcs is None:
        strip(0.0, 2 * math.pi, True)
    else:
        for ta, tb in arcs:
            strip(ta, tb, False)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob["category"], ob["support"] = "wall", "floor"
    return ob


def arcs_excluding(gaps):
    """Open arcs covering the full ellipse except the given (t_center, half_width) gaps."""
    gaps = sorted(((c - w) % (2 * math.pi), (c + w) % (2 * math.pi)) for c, w in gaps)
    out, start = [], gaps[-1][1] - 2 * math.pi
    for a, b in gaps:
        out.append((start, a))
        start = b
    return out


def disc(name, ra, rb, z, thick):
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=1, depth=thick, location=(0, 0, z))
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = (ra, rb, 1)
    bpy.ops.object.transform_apply(scale=True)
    return ob


WIN_T = [math.pi / 2 + d for d in (-0.42, 0.0, 0.42)]           # three windows behind the desk
DOOR_T = [math.pi / 2 + 1.05, math.pi / 2 - 1.05, -math.pi / 2 + 0.95, -math.pi / 2 - 0.95]
FIRE_T = -math.pi / 2
wall = wall_ring("Wall_Oval", A, B, 0.0, H, 0.3)
# window openings: boolean cutters through the wall
for i, t in enumerate(WIN_T):
    x, y = ellipse_pt(t)
    ln = math.hypot(x / A**2, y / B**2)
    cx, cy = x + x / A**2 / ln * 0.15, y + y / B**2 / ln * 0.15    # same centre as the window frame
    bpy.ops.mesh.primitive_cube_add(location=(cx, cy, 0.75 + 1.45))
    cut = bpy.context.active_object
    cut.dimensions = (1.42, 1.6, 2.92)
    cut.rotation_euler[2] = math.atan2(y / B**2, x / A**2) - math.pi / 2
    mod = wall.modifiers.new(f"win{i}", "BOOLEAN")
    mod.operation, mod.solver, mod.object = "DIFFERENCE", "EXACT", cut
    bpy.context.view_layer.objects.active = wall
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cut)
floor = disc("Floor", A + 0.3, B + 0.3, -0.05, 0.1)
floor["category"], floor["support"] = "floor", "ground"
ceiling = disc("Ceiling", A + 0.3, B + 0.3, H + 0.05, 0.1)
ceiling["category"], ceiling["support"] = "ceiling", "wall"
cornice = wall_ring("Cornice", A - 0.182, B - 0.182, H - 0.35, H - 0.002, 0.18)
TRIM_GAPS = [(t, 0.085) for t in WIN_T] + [(t, 0.085) for t in DOOR_T] + [(FIRE_T, 0.17)]
chair_rail = wall_ring("Chair_Rail", A - 0.032, B - 0.032, 0.9, 0.96, 0.03, arcs=arcs_excluding(TRIM_GAPS))
baseboard = wall_ring("Baseboard", A - 0.032, B - 0.032, 0.0, 0.2, 0.03, arcs=arcs_excluding(TRIM_GAPS))
medallion = disc("Ceiling_Medallion", 1.6, 1.3, H - 0.02, 0.06)
for ob in (cornice, chair_rail, baseboard, medallion):
    ob["category"], ob["support"] = "trim", "wall"

run("set_material", name="Wall_Oval", material_name="Wall_Cream", color=[0.86, 0.79, 0.62, 1], roughness=0.8)
run("apply_wood", name="Floor", variant="oak")
run("set_material", name="Ceiling", material_name="Ceiling_White", color=[0.96, 0.95, 0.92, 1], roughness=0.9)
for n in ("Cornice", "Chair_Rail", "Baseboard", "Ceiling_Medallion"):
    run("set_material", name=n, material_name="Trim_White", color=[0.97, 0.96, 0.93, 1], roughness=0.5)

# ---------------------------------------------------------------- windows, curtains, doors (asset specs)
def on_wall(t, inset=0.0):
    x, y = ellipse_pt(t)
    nx, ny = x / A**2, y / B**2
    ln = math.hypot(nx, ny)
    yaw = math.degrees(math.atan2(ny, nx)) - 90.0   # local -Y (front) faces the room centre
    return x - nx / ln * inset, y - ny / ln * inset, yaw


WINDOW = {"asset": "window", "category": "window", "support": "wall", "bbox_m": [1.4, 0.12, 2.9],
          "parts": [
              {"id": "frame", "shape": "frame", "mode": "border", "size": [1.4, 2.9, 0.12], "rotation_deg": [90, 0, 0], "material": "color:#f4f1ea"},
              {"id": "glass", "shape": "box", "size": [1.3, 0.02, 2.8], "at": "frame.center",
               "material": "glass:clear"},
              {"id": "mullion_v", "shape": "box", "size": [0.04, 0.06, 2.8], "at": "frame.center",
               "material": "color:#f4f1ea"},
              {"id": "mullion_h", "shape": "box", "size": [1.3, 0.06, 0.04], "at": "frame.center",
               "count": 3, "pattern": "linear", "spacing": [0, 0, 0.7], "centered": True,
               "material": "color:#f4f1ea"}]}
CURTAIN = {"asset": "curtain", "category": "curtain", "support": "wall", "bbox_m": [0.55, 0.14, 3.4],
           "parts": [{"id": "drape", "shape": "box", "size": [0.55, 0.06, 3.4], "material": "color:#c79a2b"},
                     {"id": "folds", "shape": "cylinder", "size": [0.08, 0.08, 3.4], "at": "drape.front",
                      "count": 4, "pattern": "linear", "spacing": [0.13, 0, 0], "centered": True,
                      "material": "color:#c79a2b"}]}
DOOR = {"asset": "door", "category": "door", "support": "wall", "bbox_m": [1.1, 0.1, 2.6],
        "parts": [{"id": "casing", "shape": "frame", "mode": "border", "size": [1.3, 2.75, 0.1], "rotation_deg": [90, 0, 0], "material": "color:#f4f1ea"},
                  {"id": "leaf", "shape": "box", "size": [1.05, 0.05, 2.6], "at": "casing.center",
                   "material": "color:#efeadf"},
                  {"id": "panel", "shape": "frame", "mode": "border", "size": [0.8, 1.0, 0.03], "rotation_deg": [90, 0, 0], "at": "leaf.front",
                   "count": 2, "pattern": "linear", "spacing": [0, 0, 1.2], "centered": True,
                   "material": "color:#f6f2e8"},
                  {"id": "knob", "shape": "sphere", "size": [0.07, 0.07, 0.07], "at": "leaf.front",
                   "offset": [0.4, -0.03, 0], "material": "metal:gold"}]}

for i, t in enumerate(WIN_T):
    x, y, yaw = on_wall(t, -0.15)
    asset({**WINDOW, "asset": f"Window_{i}"}, x, y, yaw, 0.75)
    for side in (-1, 1):
        cx, cy, cyaw = on_wall(t + side * 0.105, 0.2)
        asset({**CURTAIN, "asset": f"Curtain_{i}_{'L' if side < 0 else 'R'}"}, cx, cy, cyaw, 0.55)
for i, t in enumerate(DOOR_T):
    x, y, yaw = on_wall(t, 0.16)
    asset({**DOOR, "asset": f"Door_{i}"}, x, y, yaw)

# ---------------------------------------------------------------- fireplace (north end, -Y)
FIREPLACE = {"asset": "fireplace", "category": "fireplace", "support": "floor", "bbox_m": [2.2, 0.5, 1.5],
             "parts": [
                 {"id": "jamb", "shape": "box", "size": [0.6, 0.35, 0.9], "at": [0.8, 0, 0], "mirror": "x",
                  "material": "color:#f2efe8"},
                 {"id": "surround", "shape": "box", "size": [2.2, 0.35, 0.4], "at": [0, 0, 0.9],
                  "material": "color:#f2efe8"},
                 {"id": "firebox", "shape": "box", "size": [1.0, 0.3, 0.9], "at": [0, 0.04, 0],
                  "material": "color:#151210"},
                 {"id": "mantel", "shape": "rounded_slab", "size": [2.4, 0.5, 0.1], "attach": "surround.top",
                  "material": "color:#f6f3ec"},
                 {"id": "hearth", "shape": "box", "size": [2.0, 0.6, 0.04], "at": [0, -0.3, 0],
                  "material": "color:#8b8178"}]}
fx, fy, fyaw = on_wall(FIRE_T, 0.42)
asset(FIREPLACE, fx, fy, fyaw)

# ---------------------------------------------------------------- furniture (declarative specs)
RUG = {"asset": "Oval_Rug", "category": "rug", "support": "floor", "bbox_m": [6.6, 8.4, 0.02],
       "parts": [{"id": "rug", "shape": "cylinder", "size": [6.6, 8.4, 0.015], "at": [0, 0, 0],
                  "material": "color:#d9c48f"},
                 {"id": "border", "shape": "ring", "outer_radius": 1.0, "inner_radius": 0.94, "size": [6.6, 8.4, 0.02], "at": [0, 0, 0],
                  "material": "color:#2f3f6b"}]}
asset(RUG, 0.0, 0.0, 0.0, 0.001)
RUG_Z = 0.022

RESOLUTE = {"asset": "Resolute_Desk", "category": "desk", "support": "floor", "bbox_m": [1.83, 1.22, 0.78],
            "default_material": "wood:dark",
            "parts": [
                {"id": "ped_l", "shape": "box", "size": [0.55, 1.1, 0.7], "at": [-0.62, 0, 0], "bevel": 0.01},
                {"id": "ped_r", "shape": "box", "size": [0.55, 1.1, 0.7], "at": [0.62, 0, 0], "bevel": 0.01},
                {"id": "kneehole", "shape": "box", "size": [0.7, 0.06, 0.55], "at": [0, 0.5, 0.15]},
                {"id": "top", "shape": "rounded_slab", "size": [1.83, 1.22, 0.06], "at": [0, 0, 0.7]},
                {"id": "leather", "shape": "box", "size": [1.4, 0.8, 0.012], "attach": "top.top",
                 "material": "fabric:leather"},
                {"id": "panels", "shape": "frame", "mode": "border", "size": [0.42, 0.5, 0.02], "rotation_deg": [90, 0, 0], "at": "ped_l.front",
                 "mirror": "x"},
                {"id": "plinth", "shape": "box", "size": [1.83, 1.2, 0.06], "at": [0, 0, 0],
                 "material": "wood:dark"}]}
desk = asset(RESOLUTE, 0.0, 3.4, 180.0, RUG_Z)

DESK_CHAIR = {"asset": "Desk_Chair", "category": "office_chair", "support": "floor", "bbox_m": [0.7, 0.7, 1.25],
              "parts": [
                  {"id": "base", "shape": "cylinder", "size": [0.6, 0.6, 0.05], "at": [0, 0, 0],
                   "material": "metal:chrome"},
                  {"id": "column", "shape": "cylinder", "size": [0.06, 0.06, 0.38], "attach": "base.top",
                   "material": "metal:chrome"},
                  {"id": "seat", "shape": "rounded_slab", "size": [0.6, 0.58, 0.14], "attach": "column.top",
                   "material": "fabric:leather"},
                  {"id": "back", "shape": "rounded_slab", "size": [0.58, 0.14, 0.75], "attach": "seat.back_edge",
                   "material": "fabric:leather"}]}
asset(DESK_CHAIR, 0.0, 4.5, 180.0, RUG_Z)

SOFA = {"asset": "sofa", "category": "sofa", "support": "floor", "bbox_m": [2.3, 0.95, 0.85],
        "parts": [
            {"id": "base", "shape": "rounded_slab", "size": [2.3, 0.9, 0.22], "at": [0, 0, 0.1],
             "material": "color:#e6dcc0"},
            {"id": "leg", "shape": "tapered_cylinder", "size": [0.05, 0.05, 0.1], "attach": "base.bottom_corners",
             "inset": 0.06, "material": "wood:dark"},
            {"id": "cushion", "shape": "rounded_slab", "size": [0.98, 0.7, 0.16], "attach": "base.top",
             "count": 2, "pattern": "linear", "spacing": [1.0, 0, 0], "centered": True, "offset": [0, -0.08, 0],
             "material": "color:#e6dcc0"},
            {"id": "back", "shape": "rounded_slab", "size": [2.3, 0.2, 0.5], "attach": "base.back_edge",
             "material": "color:#e6dcc0"},
            {"id": "arm", "shape": "rounded_slab", "size": [0.18, 0.9, 0.3], "attach": "base.top",
             "offset": [1.06, 0, 0], "mirror": "x", "material": "color:#e6dcc0"}]}
asset({**SOFA, "asset": "Sofa_West"}, -1.45, -0.6, -90.0, RUG_Z)
asset({**SOFA, "asset": "Sofa_East"}, 1.45, -0.6, 90.0, RUG_Z)

COFFEE = {"asset": "Coffee_Table", "category": "coffee_table", "support": "floor", "bbox_m": [1.3, 0.7, 0.45],
          "default_material": "wood:walnut",
          "parts": [{"id": "top", "shape": "rounded_slab", "size": [1.3, 0.7, 0.05], "at": [0, 0, 0.4]},
                    {"id": "leg", "shape": "tapered_cylinder", "size": [0.05, 0.05, 0.4],
                     "attach": "top.bottom_corners", "inset": 0.08}]}
asset(COFFEE, 0.0, -0.6, 90.0, RUG_Z)

ARMCHAIR = {"asset": "armchair", "category": "armchair", "support": "floor", "bbox_m": [0.85, 0.85, 1.05],
            "parts": [
                {"id": "seat", "shape": "rounded_slab", "size": [0.7, 0.7, 0.2], "at": [0, 0, 0.25],
                 "material": "color:#7d1f2a"},
                {"id": "leg", "shape": "tapered_cylinder", "size": [0.045, 0.045, 0.25],
                 "attach": "seat.bottom_corners", "inset": 0.05, "material": "wood:dark"},
                {"id": "back", "shape": "rounded_slab", "size": [0.7, 0.16, 0.6], "attach": "seat.back_edge",
                 "material": "color:#7d1f2a"},
                {"id": "arm", "shape": "rounded_slab", "size": [0.12, 0.7, 0.25], "attach": "seat.top",
                 "offset": [0.33, 0, 0], "mirror": "x", "material": "color:#7d1f2a"}]}
asset({**ARMCHAIR, "asset": "Armchair_W"}, -1.4, -3.25, -60.0, RUG_Z)
asset({**ARMCHAIR, "asset": "Armchair_E"}, 1.4, -3.25, 60.0, RUG_Z)

SIDE = {"asset": "side_table", "category": "side_table", "support": "floor", "bbox_m": [0.5, 0.5, 0.6],
        "default_material": "wood:walnut",
        "parts": [{"id": "top", "shape": "cylinder", "size": [0.5, 0.5, 0.04], "at": [0, 0, 0.56]},
                  {"id": "stem", "shape": "lathe", "profile": [[0.03, 0], [0.05, 0.1], [0.03, 0.3], [0.05, 0.56]],
                   "at": [0, 0, 0]},
                  {"id": "foot", "shape": "cylinder", "size": [0.32, 0.32, 0.03], "at": [0, 0, 0]}]}
LAMP = {"asset": "table_lamp", "category": "table_lamp", "support": "floor", "bbox_m": [0.4, 0.4, 0.7],
        "parts": [{"id": "body", "shape": "lathe",
                   "profile": [[0.0, 0], [0.11, 0], [0.13, 0.15], [0.08, 0.35], [0.02, 0.42]],
                   "at": [0, 0, 0], "material": "metal:gold"},
                  {"id": "shade", "shape": "tapered_cylinder", "size": [0.4, 0.4, 0.28], "at": [0, 0, 0.42],
                   "material": "fabric:linen"}]}
for sx in (-1, 1):
    asset({**SIDE, "asset": f"Side_Table_{'W' if sx < 0 else 'E'}"}, sx * 1.55, 0.95, 0.0, RUG_Z)
    asset({**LAMP, "asset": f"Lamp_{'W' if sx < 0 else 'E'}"}, sx * 1.55, 0.95, 0.0, 0.6 + RUG_Z + 0.001)

FLAG = {"asset": "flag", "category": "flag", "support": "floor", "bbox_m": [0.9, 0.5, 2.6],
        "parts": [{"id": "stand", "shape": "cylinder", "size": [0.4, 0.4, 0.06], "at": [0, 0, 0],
                   "material": "metal:gold"},
                  {"id": "pole", "shape": "cylinder", "size": [0.04, 0.04, 2.45], "attach": "stand.top",
                   "material": "metal:gold"},
                  {"id": "finial", "shape": "sphere", "size": [0.09, 0.09, 0.09], "attach": "pole.top",
                   "material": "metal:gold"},
                  {"id": "cloth", "shape": "box", "size": [0.75, 0.03, 1.3], "bevel": 0.01, "at": "pole.center",
                   "offset": [0.4, 0.0, 0.25], "material": "fabric:velvet"}]}
def flag(name, col, toward_x):
    spec = json.loads(json.dumps(FLAG))
    spec["asset"] = name
    spec["parts"][3]["material"] = col
    spec["parts"][3]["offset"] = [0.4 * toward_x, 0.0, 0.25]   # cloth hangs toward the desk
    return spec


# rotated 180 deg: local +x is world -x
asset(flag("Flag_US", "color:#b22234", -1), -1.7, 4.1, 180.0)
asset(flag("Flag_President", "color:#1f2f6b", 1), 1.7, 4.1, 180.0)



# ---------------------------------------------------------------- lighting + camera
run("add_light", light_type="SUN", name="Sun", energy=4.0, location=[0, 12, 9])
run("modify_object", name="Sun", rotation=[-50, 0, 180])
run("add_light", light_type="AREA", name="Window_Bounce", energy=500.0, location=[0, 3.8, 3.0])
run("modify_object", name="Window_Bounce", rotation=[70, 0, 180])
run("add_light", light_type="AREA", name="Ceiling_Fill", energy=250.0, location=[0, 0, H - 0.4])
run("setup_environment", color=[0.75, 0.82, 0.95], strength=0.25)
run("set_camera", name="Camera", location=[0.0, -4.7, 1.65], rotation=[86, 0, 0], lens=18.0)
bpy.context.scene.camera = bpy.data.objects["Camera"]
desk_obj = bpy.data.objects[desk]
desk_obj["hero"] = True

# ---------------------------------------------------------------- validate + fix loop
room = [2 * A + 0.7, 2 * B + 0.7, H + 0.2]
reports = []
for it in range(3):
    rep = run("validate_scene", room_bounds=room)
    blocking = [i for i in rep["issues"] if i["severity"] == "blocking"]
    reports.append({"iteration": it, "ok": rep["ok"], "blocking": len(blocking),
                    "warnings": len(rep["issues"]) - len(blocking),
                    "issues": [{k: i[k] for k in ("check", "severity", "objects", "message")} for i in rep["issues"]]})
    fixes = [i["fix"] for i in blocking if i.get("fix")]
    if not fixes:
        break
    registry.execute([{"handler": f["handler"], "params": f["params"]} for f in fixes], undo=False)

# ---------------------------------------------------------------- renders
sc = bpy.context.scene
for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
    try:
        sc.render.engine = eng
        break
    except TypeError:
        continue
sc.view_settings.view_transform = "AgX"
try:
    sc.view_settings.look = "AgX - Medium High Contrast"
except TypeError:
    pass
sc.view_settings.exposure = -0.4
run("render", filepath=os.path.join(OUT, "oval_office_main.png"), resolution=[1600, 900])
views = {
    "desk_from_sofas": ([0.0, -1.2, 1.5], [88, 0, 0], 24),
    "fireplace_from_desk": ([0.0, 2.6, 1.6], [86, 0, 180], 22),
    "overhead": ([0.0, -0.3, H - 0.25], [0, 0, 0], 12),
}
for nm, (loc, rot, lens) in views.items():
    run("set_camera", name="Camera", location=loc, rotation=rot, lens=lens)
    run("render", filepath=os.path.join(OUT, f"oval_office_{nm}.png"), resolution=[1200, 800])

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "oval_office.blend"))
with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump({"validation": reports, "handler_calls": len(LOG),
               "failed_calls": [entry for entry in LOG if not entry["ok"]],
               "objects": len(bpy.data.objects)}, f, indent=1)
print("SHOWCASE_DONE", json.dumps({"iterations": reports and [(r["iteration"], r["blocking"], r["warnings"]) for r in reports]}))
sys.stdout.flush()
os._exit(0)
