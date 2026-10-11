"""Oval Office FRONTIER BASELINE v2 (authored by the lead model, Opus; NOT a small-model run).

Purpose: the strongest reference a big model achieves with the addon tools, against which small-model
runs (Sonnet/Haiku/Qwen via build_scene) are compared. Every call goes through the registry, as an agent
would call it over the socket.

Run:  ~/bin/blender -b --factory-startup --python eval/showcase/oval_office_v2.py -- <addon_root> <out_dir> [final]
"""
import importlib
import json
import math
import os
import sys

import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
ADDON_ROOT, OUT = argv[0], argv[1]
FINAL = len(argv) > 2 and argv[2] == "final"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ADDON_ROOT)

import bpy  # noqa: E402

_pkg = next(d for d in os.listdir(ADDON_ROOT) if os.path.isfile(os.path.join(ADDON_ROOT, d, "registry.py")))
registry = importlib.import_module(f"{_pkg}.registry")
imaging = importlib.import_module(f"{_pkg}.imaging")

W, D, H = 8.8, 10.9, 5.6           # interior oval (x, y) and ceiling height; desk end = +Y
A, B = W / 2, D / 2
LOG: list[dict] = []


def run(handler, **params):
    res = registry.execute([{"handler": handler, "params": params}], undo=False)[0]
    LOG.append({"handler": handler, "ok": res.get("ok"), "error": res.get("error")})
    if not res.get("ok"):
        raise RuntimeError(f"{handler}: {res.get('error')} {res.get('details', '')}")
    return res.get("result")


def recipe(name):
    with open(os.path.join(ADDON_ROOT, "recipes", f"{name}.json")) as f:
        return json.load(f)


def asset(spec, x, y, yaw=0.0, z=0.0, **over):
    """yaw: Blender Z rotation in degrees; asset front (-Y) then points along (sin yaw, -cos yaw)."""
    spec = {**spec, **over}
    r = run("build_asset", spec=spec)
    name = r["name"] if isinstance(r, dict) else r
    run("modify_object", name=name, location=[x, y, z], rotation=[0, 0, yaw])
    return name


def yaw_facing(x, y, tx, ty):
    """Yaw so the asset front (-Y) points from (x,y) to (tx,ty)."""
    return math.degrees(math.atan2(tx - x, -(ty - y)))


def t_of(angle):          # ellipse parameter as a fraction CCW from +X
    return (angle / (2 * math.pi)) % 1.0


def ellipse_pt(angle, inset=0.0):
    x, y = A * math.cos(angle), B * math.sin(angle)
    nx, ny = x / A**2, y / B**2
    ln = math.hypot(nx, ny)
    return x - nx / ln * inset, y - ny / ln * inset


def ellipse_pt_nrm(angle, inset=0.0):
    x, y = A * math.cos(angle), B * math.sin(angle)
    nx, ny = x / A**2, y / B**2
    ln = math.hypot(nx, ny)
    yaw = math.degrees(math.atan2(-nx, ny))
    return x - nx / ln * inset, y - ny / ln * inset, yaw


# ---------------------------------------------------------------- procedural textures
def make_us_flag(path, width=1024, height=540):
    img = np.zeros((height, width, 4), dtype=np.float32)
    stripe_h = height / 13.0
    red = [0.698, 0.133, 0.204, 1.0]
    white = [0.98, 0.98, 0.98, 1.0]
    for i in range(13):
        y0 = int(round(i * stripe_h))
        y1 = int(round((i + 1) * stripe_h))
        img[y0:y1, :] = red if (i % 2 == 0) else white

    canton_w = int(width * 0.40)
    canton_h = int(round(7 * stripe_h))
    navy = [0.078, 0.141, 0.302, 1.0]
    img[:canton_h, :canton_w] = navy

    star_r = canton_h * 0.034
    y_coords = np.linspace(canton_h * 0.09, canton_h * 0.91, 9)
    for r_idx, cy in enumerate(y_coords):
        n_stars = 6 if (r_idx % 2 == 0) else 5
        x_pad = canton_w * 0.08 if n_stars == 6 else canton_w * 0.16
        x_coords = np.linspace(x_pad, canton_w - x_pad, n_stars)
        for cx in x_coords:
            y_min = max(0, int(cy - star_r * 1.5))
            y_max = min(canton_h, int(cy + star_r * 1.5) + 1)
            x_min = max(0, int(cx - star_r * 1.5))
            x_max = min(canton_w, int(cx + star_r * 1.5) + 1)
            Y, X = np.ogrid[y_min:y_max, x_min:x_max]
            mask = np.hypot(X - cx, Y - cy) < star_r
            img[y_min:y_max, x_min:x_max][mask] = white

    bimg = bpy.data.images.new("us_flag_tex", width, height, alpha=True)
    bimg.pixels.foreach_set(img[::-1].ravel())
    bimg.alpha_mode = "STRAIGHT"
    bimg.filepath_raw = os.path.abspath(path)
    bimg.file_format = "PNG"
    bimg.save()
    bpy.data.images.remove(bimg)


def make_presidential_flag(path, seal_path, width=1024, height=600):
    flag = np.zeros((height, width, 4), dtype=np.float32)
    navy = [0.06, 0.11, 0.25, 1.0]
    flag[:, :] = navy

    s_img = bpy.data.images.load(os.path.abspath(seal_path), check_existing=True)
    sw_orig, sh_orig = s_img.size
    pixels = np.array(s_img.pixels[:]).reshape(sh_orig, sw_orig, 4)
    sw, sh = 420, 420
    y_idx = np.linspace(0, sh_orig - 1, sh).astype(int)
    x_idx = np.linspace(0, sw_orig - 1, sw).astype(int)
    sub = pixels[np.ix_(y_idx, x_idx)]
    y0 = (height - sh) // 2
    x0 = (width - sw) // 2
    alpha = sub[:, :, 3:4]
    flag[y0:y0 + sh, x0:x0 + sw, :3] = sub[:, :, :3] * alpha + flag[y0:y0 + sh, x0:x0 + sw, :3] * (1.0 - alpha)

    bimg = bpy.data.images.new("pres_flag_tex", width, height, alpha=True)
    bimg.pixels.foreach_set(flag[::-1].ravel())
    bimg.alpha_mode = "STRAIGHT"
    bimg.filepath_raw = os.path.abspath(path)
    bimg.file_format = "PNG"
    bimg.save()
    bpy.data.images.remove(bimg)


bpy.ops.wm.read_factory_settings(use_empty=True)

# ---------------------------------------------------------------- room shell
UP = math.pi / 2
WIN = [UP - 0.42, UP, UP + 0.42]            # South windows
NICHE = [0.85, -0.15]                       # East wall: two arched niche bookcases
DOOR = [0.35]                               # East wall: 1 Rose Garden door between the bookcases
FIRE = -UP                                  # North wall: fireplace

openings = ([{"kind": "window", "t": t_of(a), "width": 1.3, "height": 2.9, "sill": 0.75} for a in WIN]
            + [{"kind": "niche", "t": t_of(a), "width": 1.2, "height": 2.7, "sill": 0.85, "depth": 0.22, "arch": True}
               for a in NICHE]
            + [{"kind": "door", "t": t_of(DOOR[0]), "width": 1.3, "height": 2.8}]
            + [{"kind": "niche", "t": t_of(FIRE), "width": 2.4, "height": 1.6, "sill": 0.0, "depth": 0.05, "arch": False}])

room = run("room_shell", shape="ellipse", size=[W, D], height=H, wall_thickness=0.35, openings=openings,
           trim={"baseboard": 0.22, "chair_rail": 0.92}, name="Oval", segments=192)
ops = room["openings"]
win_ops = [o for o in ops if o["kind"] == "window"]
niche_ops = [o for o in ops if o["kind"] == "niche" and o["sill"] > 0.5]
door_ops = [o for o in ops if o["kind"] == "door"]

# Walls: dual-tone wainscoting (white base below chair rail, cream damask wallpaper above)
run("apply_material", name=room["walls"], material="damask:wallpaper:#ece2c6:#ddd0ad")
wall_mat = bpy.data.objects[room["walls"]].data.materials[0]
nt = wall_mat.node_tree
bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
geom = nt.nodes.new("ShaderNodeNewGeometry")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(geom.outputs["Position"], sep.inputs["Vector"])
comp = nt.nodes.new("ShaderNodeMath")
comp.operation = "GREATER_THAN"
comp.inputs[1].default_value = 0.92
nt.links.new(sep.outputs["Z"], comp.inputs[0])

mix_col = nt.nodes.new("ShaderNodeMix")
mix_col.data_type = "RGBA"
nt.links.new(comp.outputs["Value"], mix_col.inputs["Factor"])
mix_col.inputs[6].default_value = (0.97, 0.96, 0.93, 1.0)          # satin white wainscoting base
if bsdf.inputs["Base Color"].is_linked:
    prev_link = bsdf.inputs["Base Color"].links[0]
    nt.links.new(prev_link.from_socket, mix_col.inputs[7])
else:
    mix_col.inputs[7].default_value = bsdf.inputs["Base Color"].default_value
nt.links.new(mix_col.outputs[2], bsdf.inputs["Base Color"])

mix_rough = nt.nodes.new("ShaderNodeMix")
mix_rough.data_type = "FLOAT"
nt.links.new(comp.outputs["Value"], mix_rough.inputs["Factor"])
mix_rough.inputs[2].default_value = 0.35                             # wainscot satin finish
if bsdf.inputs["Roughness"].is_linked:
    prev_r = bsdf.inputs["Roughness"].links[0]
    nt.links.new(prev_r.from_socket, mix_rough.inputs[3])
else:
    mix_rough.inputs[3].default_value = 0.75
nt.links.new(mix_rough.outputs[0], bsdf.inputs["Roughness"])

run("wallpaper_emboss", name=room["walls"], pattern="damask", depth=0.0015, tile=3.0, axis="y")

for o in room["objects"]:
    if o.endswith("_Ceiling"):
        run("apply_material", name=o, material="plaster:#f4f1ea")
    elif "Baseboard" in o or "Chair" in o or "Crown" in o:
        run("apply_material", name=o, material="plaster:#f7f4ee")
floor_name = next(o for o in room["objects"] if o.endswith("_Floor"))
bpy.data.objects[floor_name].hide_render = True          # replaced by real planks
run("wood_plank_floor", name="Floor_Planks", size=[W + 0.8, D + 0.8], pattern="herringbone",
    plank_width=0.09, plank_length=0.55, material="wood:oak:proj=uv", location=[0, 0, -0.016])

# Crown moulding with dentils along ceiling
crown = asset({"asset": "Crown_Moulding", "category": "trim", "support": "wall",
               "parts": [{"id": "crown", "shape": "cornice", "plan": "ellipse", "width": W - 0.02,
                          "length": D - 0.02, "profile": "crown_classic", "profile_width": 0.32,
                          "profile_height": 0.42, "dentil": True, "material": "plaster:#f7f4ee"}]},
              0, 0, 0, H - 0.46)

# Ceiling: plaster relief border ring + central stylised seal relief
seal_png = os.path.join(OUT, "seal.png")
run("make_stylised_seal", path=seal_png, size_px=1024)
asset({"asset": "Ceiling_Medallion", "category": "trim", "support": "ceiling",
       "parts": [{"id": "disc", "shape": "cylinder", "size": [2.6, 2.1, 0.04], "material": "plaster:#f4f1ea"}]},
      0, 0, 0, H - 0.058)
run("plaster_relief", name="Ceiling_Medallion_disc", pattern="seal_stylised", depth=0.018)

# ---------------------------------------------------------------- windows & curtains
WINDOW = {"asset": "window", "category": "window", "support": "wall",
          "parts": [{"id": "frame", "shape": "frame", "mode": "border", "size": [1.3, 2.9, 0.12],
                     "rotation_deg": [90, 0, 0], "material": "plaster:#f7f4ee"},
                    {"id": "glass", "shape": "box", "size": [1.22, 0.012, 2.82], "at": "frame.center",
                     "material": "glass:clear:thin=1"},
                    {"id": "mullion_v", "shape": "box", "size": [0.035, 0.05, 2.82], "at": "frame.center",
                     "count": 2, "pattern": "linear", "spacing": [0.41, 0, 0], "centered": True,
                     "material": "plaster:#f7f4ee"},
                    {"id": "mullion_h", "shape": "box", "size": [1.22, 0.05, 0.035], "at": "frame.center",
                     "count": 5, "pattern": "linear", "spacing": [0, 0, 0.47], "centered": True,
                     "material": "plaster:#f7f4ee"}]}
GOLD_DAMASK = "damask:fabric:#d9ae45:#c79a32"
for i, o in enumerate(win_ops):
    cx, cy, cz = o["center"]
    asset({**WINDOW, "asset": f"Window_{i}"}, cx, cy, o["yaw"], o["sill"])
    yaw = math.radians(o["yaw"])
    fx, fy = math.sin(yaw), -math.cos(yaw)                   # front (into room)
    curtain = recipe("swag_curtain_set")
    asset(curtain, cx + fx * 0.38, cy + fy * 0.38, o["yaw"], 0.0, asset=f"Curtains_{i}",
          bbox_m=[1.50, 0.16, H - 1.25],
          materials={"drape": GOLD_DAMASK, "valance": GOLD_DAMASK, "rod": "brass", "finial": "brass"})

# Exterior backdrop beyond windows
asset({"asset": "Garden", "category": "backdrop", "support": "ground",
       "parts": [{"id": "lawn", "shape": "box", "size": [30, 12, 0.1], "material": "color:#4f7a3a"},
                 {"id": "trees", "shape": "box", "size": [30, 0.5, 9], "at": [0, 5, 0],
                  "material": "fabric:canvas:#3d6b35"}]}, 0, B + 9, 0, -0.1)

# ---------------------------------------------------------------- single East door (anti-glow backing)
DOOR_ASSET = {"asset": "Door_East", "category": "door", "support": "wall",
              "parts": [{"id": "casing", "shape": "frame", "mode": "border", "size": [1.3, 2.8, 0.1],
                         "rotation_deg": [90, 0, 0], "material": "color:#f4f1ea"},
                        {"id": "leaf", "shape": "raised_panel", "size": [1.08, 0.05, 2.63], "at": "casing.center",
                         "material": "color:#f2eee4"},
                        {"id": "knob", "shape": "sphere", "size": [0.07, 0.07, 0.07], "at": "leaf.front",
                         "offset": [0.42, -0.03, 0], "material": "brass"},
                        # Opaque light-blocking backing box: stops Nishita sky/sun bleed
                        {"id": "backing", "shape": "box", "size": [1.20, 0.03, 2.70], "at": [0, 0.12, 0],
                         "material": "color:#151515"}]}
for i, o in enumerate(door_ops):
    cx, cy, _ = o["center"]
    asset({**DOOR_ASSET, "asset": f"Door_{i}"}, cx, cy, o["yaw"])

# ---------------------------------------------------------------- arched dome niche bookshelves
w_bk, h_bk = 1.05, 2.60
r_bk = w_bk / 2.0
h_spring = h_bk - r_bk
arch_pts = [[r_bk * math.cos(a), 0.0, h_spring + r_bk * math.sin(a)] for a in np.linspace(0, math.pi, 21)]

def make_arched_bookshelf(name):
    return {"asset": name, "category": "shelf", "support": "wall", "default_material": "plaster:#f7f4ee",
            "parts": [{"id": "back", "shape": "box", "size": [w_bk, 0.02, h_spring], "at": [0, 0.08, 0.0]},
                      {"id": "jamb", "shape": "box", "size": [0.035, 0.16, h_spring], "at": [r_bk - 0.02, 0.0, 0.0], "mirror": "x"},
                      {"id": "arch_casing", "shape": "sweep", "path": arch_pts, "profile": "circle", "radius": 0.025, "at": [0, 0.0, 0.0]},
                      {"id": "shelf_0", "shape": "box", "size": [w_bk - 0.06, 0.16, 0.025], "at": [0, 0.0, 0.38]},
                      {"id": "shelf_1", "shape": "box", "size": [w_bk - 0.06, 0.16, 0.025], "at": [0, 0.0, 0.76]},
                      {"id": "shelf_2", "shape": "box", "size": [w_bk - 0.06, 0.16, 0.025], "at": [0, 0.0, 1.14]},
                      {"id": "shelf_3", "shape": "box", "size": [w_bk - 0.06, 0.16, 0.025], "at": [0, 0.0, 1.52]},
                      {"id": "shelf_4", "shape": "box", "size": [w_bk - 0.06, 0.16, 0.025], "at": [0, 0.0, 1.90]},
                      {"id": "books_0", "shape": "box", "size": [0.90, 0.12, 0.28], "at": [0, -0.01, 0.40], "material": "leather:#6b1f1f"},
                      {"id": "books_1", "shape": "box", "size": [0.90, 0.12, 0.28], "at": [0, -0.01, 0.78], "material": "leather:#203a5a"},
                      {"id": "books_2", "shape": "box", "size": [0.90, 0.12, 0.28], "at": [0, -0.01, 1.16], "material": "leather:#2d4a22"},
                      {"id": "books_3", "shape": "box", "size": [0.90, 0.12, 0.28], "at": [0, -0.01, 1.54], "material": "leather:#5a3311"}]}

for i, o in enumerate(niche_ops):
    cx, cy, _ = o["center"]
    yaw = math.radians(o["yaw"])
    fx, fy = math.sin(yaw), -math.cos(yaw)
    asset(make_arched_bookshelf(f"Niche_Shelves_{i}"), cx - fx * 0.02, cy - fy * 0.02, o["yaw"], o["sill"])

# ---------------------------------------------------------------- realistic fireplace with depth
FIREPLACE = {"asset": "Fireplace", "category": "fireplace", "support": "floor",
             "parts": [
                 # Green marble hearth slab on floor
                 {"id": "hearth", "shape": "box", "size": [2.2, 0.52, 0.04], "at": [0, -0.16, 0.0], "material": "marble:verde"},
                 # Recessed firebox cavity
                 {"id": "firebox_back", "shape": "box", "size": [1.1, 0.04, 1.12], "at": [0, 0.12, 0.0], "material": "color:#151312"},
                 {"id": "firebox_side", "shape": "box", "size": [0.04, 0.24, 1.12], "at": [0.55, 0.0, 0.0], "mirror": "x", "material": "color:#151312"},
                 {"id": "fireback", "shape": "raised_panel", "size": [0.85, 0.03, 0.95], "at": [0, 0.10, 0.0], "material": "metal:steel:finish=brushed"},
                 # Cast iron grate, brass andirons & firewood logs
                 {"id": "grate", "shape": "box", "size": [0.85, 0.20, 0.08], "at": [0, 0.0, 0.04], "material": "color:#121212"},
                 {"id": "andiron", "shape": "turned_leg", "size": [0.07, 0.07, 0.38], "at": [0.38, -0.06, 0.0], "mirror": "x", "material": "brass"},
                 {"id": "log_0", "shape": "cylinder", "size": [0.12, 0.12, 0.65], "rotation_deg": [0, 85, 12], "at": [0, 0.0, 0.14], "material": "wood:dark"},
                 {"id": "log_1", "shape": "cylinder", "size": [0.10, 0.10, 0.60], "rotation_deg": [0, 92, -15], "at": [0.02, -0.02, 0.24], "material": "wood:dark"},
                 # Marble mantel jambs with fluted pilasters
                 {"id": "jamb", "shape": "raised_panel", "size": [0.55, 0.32, 1.12], "at": [0.82, 0.0, 0.0], "mirror": "x", "material": "marble:carrara"},
                 {"id": "flute", "shape": "box", "size": [0.20, 0.03, 0.95], "at": [0.82, -0.16, 0.08], "mirror": "x", "material": "marble:carrara"},
                 # Lintel and carved frieze tablet
                 {"id": "lintel", "shape": "raised_panel", "size": [2.20, 0.32, 0.34], "at": [0, 0.0, 1.12], "material": "marble:carrara"},
                 {"id": "center_tablet", "shape": "raised_panel", "size": [0.50, 0.04, 0.22], "at": [0, -0.16, 1.18], "material": "marble:carrara"},
                 # Projecting mantel shelf with ogee edge
                 {"id": "mantel_shelf", "shape": "cornice", "profile": "ogee", "plan": "rect", "width": 2.10, "length": 0.32,
                  "profile_width": 0.06, "profile_height": 0.10, "at": [0, 0.0, 1.46], "material": "marble:carrara"},
                 # Gilded clock & brass candlesticks
                 {"id": "clock", "shape": "box", "size": [0.24, 0.12, 0.26], "at": [0, 0.0, 1.56], "material": "gold"},
                 {"id": "candlestick", "shape": "turned_leg", "size": [0.08, 0.08, 0.30], "at": [0.80, 0.0, 1.56], "mirror": "x", "material": "brass"}]}

fx, fy = ellipse_pt(FIRE, 0.45)
asset(FIREPLACE, fx, fy, yaw_facing(fx, fy, 0, 0))
px, py = ellipse_pt(FIRE, 0.08)
asset(recipe("framed_portrait"), px, py, yaw_facing(px, py, 0, 0), 2.1, asset="Portrait_Fireplace",
      bbox_m=[1.1, 0.08, 1.4], materials={"canvas": "color:#3b2d22"})

# Additional historical portraits hung flush against the elliptical wall
for k, ang in enumerate([0.08, math.pi - 0.15, math.pi + 0.35, -0.35]):
    x, y, yaw = ellipse_pt_nrm(ang, 0.08)
    asset(recipe("framed_portrait"), x, y, yaw, 1.85, asset=f"Portrait_{k}",
          bbox_m=[0.85, 0.08, 1.15], materials={"canvas": "color:#2e2a24"})

# ---------------------------------------------------------------- resolute desk (visitor paneling + seal, no front drawers)
RUG_Z = 0.016

def resolute_desk_spec():
    d = recipe("resolute_desk")
    parts = []
    for p in d["parts"]:
        pid = p["id"]
        # Move pulls to the rear (+Y) for President's drawers; NO pulls on front (-Y)
        if pid in ("pull", "center_pull"):
            p_back = dict(p)
            p_back["at"] = [p["at"][0], -p["at"][1], p["at"][2]]
            parts.append(p_back)
        elif pid in ("ped_panel0", "ped_panel1", "center_drawer"):
            # Front decorative raised panels (visitor side)
            parts.append(p)
            # President drawers on rear (+Y)
            p_back = dict(p)
            p_back["id"] = pid + "_back"
            p_back["at"] = [p["at"][0], -p["at"][1], p["at"][2]]
            parts.append(p_back)
        elif pid == "kneehole_back":
            # FDR carved modesty door panel on the FRONT (-Y) facing the room
            p_front = dict(p)
            p_front["id"] = "kneehole_fdr_panel"
            p_front["at"] = [0, -0.375, 0.1]
            parts.append(p_front)
        elif pid == "medallion":
            # Presidential coat of arms seal medallion mounted on FDR front door panel
            p_front = dict(p)
            p_front["at"] = [0, -0.362, 0.38]
            p_front["rotation_deg"] = [-90, 0, 0]
            parts.append(p_front)
        elif pid == "medallion_ring":
            p_front = dict(p)
            p_front["at"] = [0, -0.366, 0.245]
            p_front["rotation_deg"] = [-90, 0, 0]
            parts.append(p_front)
        else:
            parts.append(p)
    d["parts"] = parts
    return d

desk = asset(resolute_desk_spec(), 0.0, 3.55, 0.0, RUG_Z, asset="Resolute_Desk", bbox_m=[1.83, 1.22, 0.78],
             default_material="wood:mahogany")
run("decal", name="Resolute_Desk_medallion", image_path=seal_png, size=0.24)

# President's executive chair
DESK_CHAIR = {"asset": "Desk_Chair", "category": "office_chair", "support": "floor",
              "parts": [{"id": "base", "shape": "cylinder", "size": [0.62, 0.62, 0.05], "material": "wood:mahogany"},
                        {"id": "column", "shape": "turned_leg", "size": [0.1, 0.1, 0.36], "attach": "base.top",
                         "material": "wood:mahogany"},
                        {"id": "seat", "shape": "cushion", "size": [0.62, 0.6, 0.14], "attach": "column.top",
                         "material": "leather:#5a1a14"},
                        {"id": "back", "shape": "tufted_panel", "size": [0.62, 0.14, 0.82],
                         "attach": "seat.back_edge", "material": "leather:#5a1a14"}]}
asset(DESK_CHAIR, 0.0, 4.40, 0.0, RUG_Z)

# Busts on pedestals flanking desk group
for side in (-1, 1):
    x, y = side * 3.05, 2.2
    asset(recipe("bust_on_pedestal"), x, y, yaw_facing(x, y, 0, 0), 0.0, asset=f"Bust_{side}",
          bbox_m=[0.42, 0.42, 1.75])

# ---------------------------------------------------------------- guest armchairs (sculpted round front legs + straight splayed rear legs)
GUEST_CHAIR = {"asset": "Guest_Armchair", "category": "chair", "support": "floor", "default_material": "wood:mahogany",
               "parts": [
                   # Sculpted round front legs with brass casters
                   {"id": "leg_f", "shape": "turned_leg", "size": [0.05, 0.05, 0.40], "at": [0.24, -0.20, 0.02], "mirror": "x"},
                   {"id": "caster_f", "shape": "cylinder", "size": [0.03, 0.03, 0.02], "at": [0.24, -0.20, 0.01], "mirror": "x", "material": "brass"},
                   # Straight square rear legs raked back
                   {"id": "leg_r", "shape": "box", "size": [0.045, 0.045, 0.42], "rotation_deg": [-7, 0, 0], "at": [0.21, 0.20, 0.0], "mirror": "x"},
                   # Seat frame & striped upholstery
                   {"id": "seat_frame", "shape": "cornice", "profile": "ogee", "plan": "rect", "width": 0.58, "length": 0.50,
                    "profile_width": 0.03, "profile_height": 0.06, "at": [0, 0, 0.38]},
                   {"id": "seat_cushion", "shape": "cushion", "size": [0.55, 0.48, 0.08], "at": [0, 0, 0.42], "material": "stripes:#b8862f:#f0dfb0"},
                   {"id": "back_post", "shape": "box", "size": [0.04, 0.04, 0.50], "rotation_deg": [-5, 0, 0], "at": [0.21, 0.21, 0.44], "mirror": "x"},
                   {"id": "back_panel", "shape": "cushion", "size": [0.44, 0.05, 0.42], "rotation_deg": [-5, 0, 0], "at": [0, 0.22, 0.65],
                    "material": "stripes:#b8862f:#f0dfb0"},
                   {"id": "crest_rail", "shape": "cornice", "profile": "cyma_reversa", "plan": "rect", "width": 0.48, "length": 0.05,
                    "profile_width": 0.025, "profile_height": 0.05, "at": [0, 0.23, 0.88]},
                   {"id": "arm_rest", "shape": "rounded_slab", "size": [0.05, 0.34, 0.025], "corner_radius": 0.01, "at": [0.24, -0.02, 0.62], "mirror": "x"},
                   {"id": "arm_post", "shape": "turned_leg", "size": [0.035, 0.035, 0.18], "at": [0.24, -0.17, 0.44], "mirror": "x"}]}

for k, (x, y) in enumerate([(-0.55, 2.35), (0.55, 2.35), (-1.45, 2.65), (1.45, 2.65)]):
    asset(GUEST_CHAIR, x, y, yaw_facing(x, y, x * 0.6, 3.55), RUG_Z, asset=f"Guest_Chair_{k}")

# ---------------------------------------------------------------- gravity-draped flags & authentic textures
us_flag_png = os.path.join(OUT, "us_flag.png")
pres_flag_png = os.path.join(OUT, "presidential_flag.png")
make_us_flag(us_flag_png)
make_presidential_flag(pres_flag_png, seal_png)

def gravity_flag_spec(name):
    return {"asset": name, "category": "flag", "support": "floor",
            "parts": [{"id": "stand", "shape": "lathe", "profile": [[0, 0], [0.2, 0], [0.18, 0.05], [0.05, 0.09],
                                                                     [0.03, 0.12], [0, 0.12]], "material": "brass"},
                      {"id": "pole", "shape": "cylinder", "size": [0.035, 0.035, 2.85], "attach": "stand.top",
                       "material": "wood:walnut"},
                      {"id": "finial", "shape": "lathe", "profile": [[0, 0], [0.04, 0.02], [0.02, 0.08], [0.05, 0.12],
                                                                      [0, 0.2]], "attach": "pole.top", "material": "gold"},
                      # Naturally draped flag cloth gathered vertically along pole
                      {"id": "cloth", "shape": "flag_cloth", "width": 0.34, "height": 1.75, "side": "left",
                       "fold_count": 5, "fold_depth": 0.07, "droop": 0.75, "at": "pole.center",
                       "offset": [0.16, 0, 0.42], "material": "fabric:canvas"},
                      {"id": "fringe", "shape": "fringe", "length": 0.34, "strand_length": 0.06,
                       "attach": "cloth.bottom", "material": "gold"}]}

# Two authentic ceremonial flags flanking the central window behind the desk
us_x, us_y = ellipse_pt(UP + 0.22, 0.45)
pres_x, pres_y = ellipse_pt(UP - 0.22, 0.45)

asset(gravity_flag_spec("Flag_US"), us_x, us_y, yaw_facing(us_x, us_y, 0, 3.55))
imaging.apply_image_texture("Flag_US_cloth", us_flag_png, projection="uv")

asset(gravity_flag_spec("Flag_President"), pres_x, pres_y, yaw_facing(pres_x, pres_y, 0, 3.55))
imaging.apply_image_texture("Flag_President_cloth", pres_flag_png, projection="uv")

# ---------------------------------------------------------------- rug with seal
asset({"asset": "Oval_Rug", "category": "rug", "support": "floor",
       "parts": [{"id": "rug", "shape": "cylinder", "size": [6.2, 7.9, 0.012], "material": "fabric:canvas:#e7cfa8"},
                 {"id": "band", "shape": "ring", "outer_radius": 1.0, "inner_radius": 0.965, "size": [5.6, 7.2, 0.014],
                  "material": "fabric:canvas:#b98d58"},
                 {"id": "medallion", "shape": "ring", "outer_radius": 1.0, "inner_radius": 0.9, "size": [2.4, 1.9, 0.014],
                  "material": "fabric:canvas:#c99a5c"},
                 {"id": "seal", "shape": "cylinder", "size": [1.7, 1.35, 0.013], "material": "fabric:canvas:#e7cfa8"}]},
      0.0, 0.15, 0.0, 0.001)
run("decal", name="Oval_Rug_seal", image_path=seal_png, size=1.0)
run("carpet_pile", name="Oval_Rug_rug", height=0.008, density=2500.0, tint="#e7cfa8")

# ---------------------------------------------------------------- seating group (4-legged side tables, open underneath)
SIDE_TABLE = {"asset": "Side_Table", "category": "table", "support": "floor", "default_material": "wood:mahogany",
              "parts": [{"id": "top", "shape": "rounded_slab", "size": [0.65, 0.48, 0.03], "corner_radius": 0.02, "at": [0, 0, 0.60]},
                        {"id": "apron", "shape": "box", "size": [0.58, 0.41, 0.09], "at": [0, 0, 0.54]},
                        {"id": "pull", "shape": "sphere", "size": [0.015, 0.015, 0.015], "at": [0, -0.21, 0.54], "material": "brass"},
                        {"id": "leg_f", "shape": "turned_leg", "size": [0.045, 0.045, 0.54], "at": [0.26, -0.17, 0.0], "mirror": "x"},
                        {"id": "leg_r", "shape": "turned_leg", "size": [0.045, 0.045, 0.54], "at": [0.26, 0.17, 0.0], "mirror": "x"},
                        {"id": "shelf", "shape": "box", "size": [0.50, 0.33, 0.015], "at": [0, 0, 0.16]}]}

CREAM_DAMASK = "damask:fabric:#eee2c2:#e1d1a6"
for side in (-1, 1):
    sx = side * 1.55
    asset(recipe("roll_arm_sofa"), sx, -0.55, yaw_facing(sx, -0.55, 0, -0.55), RUG_Z,
          asset=f"Sofa_{'W' if side < 0 else 'E'}", bbox_m=[2.25, 0.95, 0.88],
          materials={k: CREAM_DAMASK for k in ("arm", "back", "back_cushion", "deck", "seat_cushion", "skirt")})
    for k, ty in enumerate((0.90, -2.00)):
        tx = side * 1.55
        asset(SIDE_TABLE, tx, ty, yaw_facing(tx, ty, 0, ty), RUG_Z, asset=f"Side_Table_{'W' if side < 0 else 'E'}{k}")
    LAMP = {"asset": f"Lamp_{'W' if side < 0 else 'E'}", "category": "table_lamp", "support": "surface",
            "parts": [{"id": "body", "shape": "lathe", "profile": [[0, 0], [0.08, 0], [0.12, 0.12], [0.1, 0.3],
                                                                   [0.04, 0.4], [0.02, 0.5], [0, 0.5]],
                       "material": "ceramic:#e9e3d3"},
                      {"id": "shade", "shape": "tapered_cylinder", "radius": 0.24, "radius_top": 0.16, "height": 0.34,
                       "at": [0, 0, 0.48], "material": "fabric:linen:#f3ead8"}]}
    asset(LAMP, side * 1.55, 0.90, 0.0, RUG_Z + 0.63)
    run("add_light", light_type="POINT", name=f"Lamp_Light_{side}", energy=25.0, color=[1.0, 0.82, 0.6],
        location=[side * 1.55, 0.90, RUG_Z + 1.25])

COFFEE = {"asset": "Coffee_Table", "category": "coffee_table", "support": "floor", "default_material": "wood:mahogany",
          "parts": [{"id": "top", "shape": "rounded_slab", "size": [1.25, 0.7, 0.05], "corner_radius": 0.03, "at": [0, 0, 0.42]},
                    {"id": "edge", "shape": "cornice", "plan": "rect", "width": 1.25, "length": 0.7, "profile": "ogee",
                     "profile_width": 0.03, "profile_height": 0.04, "at": [0, 0, 0.38]},
                    {"id": "leg", "shape": "turned_leg", "size": [0.06, 0.06, 0.38], "attach": "top.bottom_corners",
                     "inset": 0.07}]}
asset(COFFEE, 0.0, -0.55, 90.0, RUG_Z)

ARMCHAIR = {"asset": "armchair", "category": "armchair", "support": "floor",
            "parts": [{"id": "deck", "shape": "box", "size": [0.8, 0.78, 0.22], "at": [0, 0, 0.14],
                       "material": GOLD_DAMASK},
                      {"id": "skirt", "shape": "skirt", "size": [0.82, 0.8, 0.14], "at": [0, 0, 0.0],
                       "material": GOLD_DAMASK},
                      {"id": "seat", "shape": "cushion", "size": [0.6, 0.62, 0.15], "attach": "deck.top",
                       "offset": [0, -0.04, 0], "material": GOLD_DAMASK},
                      {"id": "back", "shape": "cushion", "size": [0.8, 0.2, 0.72], "attach": "deck.back_edge",
                       "material": GOLD_DAMASK},
                      {"id": "arm", "shape": "rolled_arm", "size": [0.12, 0.74, 0.3], "attach": "deck.top",
                       "offset": [0.34, 0, 0], "mirror": "x", "material": GOLD_DAMASK}]}
for side in (-1, 1):
    x, y = side * 1.35, -3.35
    asset({**ARMCHAIR, "asset": f"Armchair_{'W' if side < 0 else 'E'}"}, x, y, yaw_facing(x, y, 0, -1.2), RUG_Z)

# ---------------------------------------------------------------- light + camera
run("daylight_rig", sun_elevation=28.0, sun_azimuth=95.0, strength=3.5, sky="nishita", window_portals=True,
    sky_strength=0.5, exposure=0.3)
run("add_light", light_type="AREA", name="Ceiling_Bounce", energy=90.0, location=[0, 0, H - 0.3],
    color=[1.0, 0.95, 0.88], rotation=[180, 0, 0])

CAM = {"photo_match": ([0.0, -4.85, 1.15], [87, 0, 0], 16),
       "desk_close": ([0.0, -0.4, 1.45], [86, 0, 0], 22),
       "fireplace": ([0.0, 2.2, 1.55], [84, 0, 180], 18),
       "overhead": ([0.0, -0.2, H - 0.3], [0, 0, 0], 11)}
run("set_camera", name="Camera", location=CAM["photo_match"][0], rotation=CAM["photo_match"][1],
    lens=CAM["photo_match"][2])
bpy.context.scene.camera = bpy.data.objects["Camera"]
bpy.data.objects[desk]["hero"] = True

# ---------------------------------------------------------------- validate + fix loop
reports = []
for it in range(3):
    rep = run("validate_scene", room_bounds=[W + 1.0, D + 1.0, H + 0.3])
    blocking = [i for i in rep["issues"] if i["severity"] == "blocking"]
    reports.append({"iteration": it, "blocking": len(blocking), "warnings": len(rep["issues"]) - len(blocking),
                    "issues": [{k: i[k] for k in ("check", "severity", "objects", "message")} for i in rep["issues"]]})
    fixes = [i["fix"] for i in blocking if i.get("fix")]
    if not fixes:
        break
    registry.execute([{"handler": f["handler"], "params": f["params"]} for f in fixes], undo=False)

# ---------------------------------------------------------------- renders
engine, samples, res = ("cycles", 160, [1800, 1200]) if FINAL else ("eevee", None, [1200, 800])
for nm, (loc, rot, lens) in CAM.items():
    run("set_camera", name="Camera", location=loc, rotation=rot, lens=lens)
    run("render", filepath=os.path.join(OUT, f"v2_{nm}.png"), resolution=res, samples=samples, engine=engine,
        denoise=True, look="AgX - Medium High Contrast")
    if not FINAL and nm != "photo_match":
        continue

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "oval_office_v2.blend"))
with open(os.path.join(OUT, "report_v2.json"), "w") as f:
    json.dump({"validation": reports, "handler_calls": len(LOG), "objects": len(bpy.data.objects),
               "failed_calls": [e for e in LOG if not e["ok"]]}, f, indent=1)
print("SHOWCASE_DONE", [(r["iteration"], r["blocking"], r["warnings"]) for r in reports])
sys.stdout.flush()
os._exit(0)
