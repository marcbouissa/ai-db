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

argv = sys.argv[sys.argv.index("--") + 1:]
ADDON_ROOT, OUT = argv[0], argv[1]
FINAL = len(argv) > 2 and argv[2] == "final"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ADDON_ROOT)

import bpy  # noqa: E402

_pkg = next(d for d in os.listdir(ADDON_ROOT) if os.path.isfile(os.path.join(ADDON_ROOT, d, "registry.py")))
registry = importlib.import_module(f"{_pkg}.registry")

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


bpy.ops.wm.read_factory_settings(use_empty=True)

# ---------------------------------------------------------------- room shell
UP = math.pi / 2
WIN = [UP - 0.42, UP, UP + 0.42]
NICHE = [UP - 0.76, UP + 0.76]
DOOR = [UP - 1.08, UP + 1.08, -UP - 0.95, -UP + 0.95]
FIRE = -UP
openings = ([{"kind": "window", "t": t_of(a), "width": 1.3, "height": 2.9, "sill": 0.75} for a in WIN]
            + [{"kind": "niche", "t": t_of(a), "width": 1.2, "height": 2.5, "sill": 0.95, "depth": 0.22}
               for a in NICHE]
            + [{"kind": "door", "t": t_of(a), "width": 1.3, "height": 2.8} for a in DOOR]
            + [{"kind": "niche", "t": t_of(FIRE), "width": 2.4, "height": 1.6, "sill": 0.0, "depth": 0.05, "arch": False}])
room = run("room_shell", shape="ellipse", size=[W, D], height=H, wall_thickness=0.35, openings=openings,
           trim={"baseboard": 0.22, "chair_rail": 0.92}, name="Oval", segments=192)
ops = room["openings"]
win_ops = [o for o in ops if o["kind"] == "window"]
niche_ops = [o for o in ops if o["kind"] == "niche" and o["sill"] > 0.5]
door_ops = [o for o in ops if o["kind"] == "door"]

run("apply_material", name=room["walls"], material="damask:wallpaper:#ece2c6:#ddd0ad")
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

# crown moulding with dentils around the ceiling line (plan = interior ellipse)
crown = asset({"asset": "Crown_Moulding", "category": "trim", "support": "wall",
               "parts": [{"id": "crown", "shape": "cornice", "plan": "ellipse", "width": W - 0.02,
                          "length": D - 0.02, "profile": "crown_classic", "profile_width": 0.32,
                          "profile_height": 0.42, "dentil": True, "material": "plaster:#f7f4ee"}]},
              0, 0, 0, H - 0.46)

# ceiling: plaster relief border ring + central stylised seal relief
seal_png = os.path.join(OUT, "seal.png")
run("make_stylised_seal", path=seal_png, size_px=1024)
asset({"asset": "Ceiling_Medallion", "category": "trim", "support": "ceiling",
       "parts": [{"id": "disc", "shape": "cylinder", "size": [2.6, 2.1, 0.04], "material": "plaster:#f4f1ea"}]},
      0, 0, 0, H - 0.058)
run("plaster_relief", name="Ceiling_Medallion_disc", pattern="seal_stylised", depth=0.018)

# ---------------------------------------------------------------- windows, curtains, doors, niches
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
# exterior garden backdrop beyond the windows
asset({"asset": "Garden", "category": "backdrop", "support": "ground",
       "parts": [{"id": "lawn", "shape": "box", "size": [30, 12, 0.1], "material": "color:#4f7a3a"},
                 {"id": "trees", "shape": "box", "size": [30, 0.5, 9], "at": [0, 5, 0],
                  "material": "fabric:canvas:#3d6b35"}]}, 0, B + 9, 0, -0.1)

DOORS = {"asset": "door", "category": "door", "support": "wall",
         "parts": [{"id": "casing", "shape": "frame", "mode": "border", "size": [1.3, 2.8, 0.1],
                    "rotation_deg": [90, 0, 0], "material": "plaster:#f7f4ee"},
                   {"id": "leaf", "shape": "raised_panel", "size": [1.08, 0.05, 2.63], "at": "casing.center",
                    "material": "plaster:#f2eee4"},
                   {"id": "knob", "shape": "sphere", "size": [0.07, 0.07, 0.07], "at": "leaf.front",
                    "offset": [0.42, -0.03, 0], "material": "brass"}]}
for i, o in enumerate(door_ops):
    cx, cy, _ = o["center"]
    asset({**DOORS, "asset": f"Door_{i}"}, cx, cy, o["yaw"])

for i, o in enumerate(niche_ops):
    cx, cy, _ = o["center"]
    yaw = math.radians(o["yaw"])
    fx, fy = math.sin(yaw), -math.cos(yaw)
    asset(recipe("bookshelf_niche_unit"), cx - fx * 0.02, cy - fy * 0.02, o["yaw"], o["sill"],
          asset=f"Niche_Shelves_{i}", bbox_m=[0.95, 0.16, 2.05], default_material="plaster:#f7f4ee",
          materials={"books0": "leather:#6b1f1f", "books2": "leather:#203a5a"})

# ---------------------------------------------------------------- fireplace + portrait above
FIREPLACE = {"asset": "Fireplace", "category": "fireplace", "support": "floor",
             "parts": [{"id": "jamb", "shape": "raised_panel", "size": [0.55, 0.32, 1.12], "at": [0.82, 0, 0],
                        "mirror": "x", "material": "marble:carrara"},
                       {"id": "lintel", "shape": "raised_panel", "size": [2.2, 0.32, 0.34], "at": [0, 0, 1.12],
                        "material": "marble:carrara"},
                       {"id": "firebox", "shape": "box", "size": [1.1, 0.28, 1.12], "at": [0, 0.03, 0],
                        "material": "color:#141110"},
                       {"id": "mantel", "shape": "extrude_profile", "profile": "ogee", "size": [2.1, 0.32, 0.12],
                        "attach": "lintel.top", "material": "marble:carrara"},
                       {"id": "hearth", "shape": "box", "size": [2.1, 0.48, 0.04], "at": [0, -0.15, 0],
                        "material": "marble:verde"}]}
fx, fy = ellipse_pt(FIRE, 0.45)
asset(FIREPLACE, fx, fy, yaw_facing(fx, fy, 0, 0))
px, py = ellipse_pt(FIRE, 0.08)
asset(recipe("framed_portrait"), px, py, yaw_facing(px, py, 0, 0), 2.0, asset="Portrait_Fireplace",
      bbox_m=[1.0, 0.08, 1.3], materials={"canvas": "color:#3b2d22"})
for k, ang in enumerate([UP - 0.95 + 0.0, UP + 0.95]):
    pass  # side portraits are hung beside the niches below
for k, ang in enumerate([0.15, math.pi - 0.15]):
    px, py = ellipse_pt(ang, 0.08)
    asset(recipe("framed_portrait"), px, py, yaw_facing(px, py, 0, 0), 1.75, asset=f"Portrait_{k}",
          bbox_m=[0.9, 0.08, 1.15], materials={"canvas": "color:#2e2a24"})

# ---------------------------------------------------------------- desk group
RUG_Z = 0.016
desk = asset(recipe("resolute_desk"), 0.0, 3.55, 0.0, RUG_Z, asset="Resolute_Desk", bbox_m=[1.83, 1.22, 0.78],
             default_material="wood:mahogany")
run("decal", name="Resolute_Desk_medallion", image_path=seal_png, size=1.0)
DESK_CHAIR = {"asset": "Desk_Chair", "category": "office_chair", "support": "floor",
              "parts": [{"id": "base", "shape": "cylinder", "size": [0.62, 0.62, 0.05], "material": "wood:mahogany"},
                        {"id": "column", "shape": "turned_leg", "size": [0.1, 0.1, 0.36], "attach": "base.top",
                         "material": "wood:mahogany"},
                        {"id": "seat", "shape": "cushion", "size": [0.62, 0.6, 0.14], "attach": "column.top",
                         "material": "leather:#5a1a14"},
                        {"id": "back", "shape": "tufted_panel", "size": [0.62, 0.14, 0.82],
                         "attach": "seat.back_edge", "material": "leather:#5a1a14"}]}
asset(DESK_CHAIR, 0.0, 4.40, 0.0, RUG_Z)
for k, (x, y) in enumerate([(-0.55, 2.35), (0.55, 2.35), (-1.45, 2.65), (1.45, 2.65)]):
    asset(recipe("chippendale_armchair"), x, y, yaw_facing(x, y, x * 0.6, 3.55), RUG_Z, asset=f"Guest_Chair_{k}",
          materials={"seat": "stripes:#b8862f:#f0dfb0"})

# flags between the windows: US (stripes + canton), presidential (navy + seal), two service flags
def flag_spec(name, cloth_mat, extra_parts=()):
    return {"asset": name, "category": "flag", "support": "floor",
            "parts": [{"id": "stand", "shape": "lathe", "profile": [[0, 0], [0.2, 0], [0.18, 0.05], [0.05, 0.09],
                                                                     [0.03, 0.12], [0, 0.12]], "material": "brass"},
                      {"id": "pole", "shape": "cylinder", "size": [0.035, 0.035, 2.75], "attach": "stand.top",
                       "material": "wood:walnut"},
                      {"id": "finial", "shape": "lathe", "profile": [[0, 0], [0.04, 0.02], [0.02, 0.08], [0.05, 0.12],
                                                                      [0, 0.2]], "attach": "pole.top", "material": "gold"},
                      {"id": "cloth", "shape": "flag_cloth", "width": 0.95, "height": 1.5, "side": "left",
                       "fold_count": 5, "fold_depth": 0.06, "droop": 0.55, "at": "pole.center",
                       "offset": [0.49, 0, 0.45], "material": cloth_mat},
                      {"id": "fringe", "shape": "fringe", "length": 0.95, "strand_length": 0.07,
                       "attach": "cloth.bottom", "material": "gold"}, *extra_parts]}


FLAGS = [("Flag_US", "stripes:#b22234:#f4f1ea:dir=u", UP + 0.21),
         ("Flag_President", "color:#1c2b5e", UP - 0.21),
         ("Flag_Army", "color:#f2efe6", UP + 0.6),
         ("Flag_Marines", "color:#b5202e", UP - 0.6)]
for name, mat, ang in FLAGS:
    x, y = ellipse_pt(ang, 0.45)
    asset(flag_spec(name, mat), x, y, yaw_facing(x, y, 0, 0))
run("decal", name="Flag_President_cloth", image_path=seal_png, size=0.55)

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

# ---------------------------------------------------------------- seating group (faces inward)
CREAM_DAMASK = "damask:fabric:#eee2c2:#e1d1a6"
for side in (-1, 1):
    sx = side * 1.55
    asset(recipe("roll_arm_sofa"), sx, -0.55, yaw_facing(sx, -0.55, 0, -0.55), RUG_Z,
          asset=f"Sofa_{'W' if side < 0 else 'E'}", bbox_m=[2.25, 0.95, 0.88],
          materials={k: CREAM_DAMASK for k in ("arm", "back", "back_cushion", "deck", "seat_cushion", "skirt")})
    for k, ty in enumerate((0.90, -2.00)):
        tx = side * 1.55
        asset(recipe("drum_side_table"), tx, ty, yaw_facing(tx, ty, 0, ty), RUG_Z,
              asset=f"Side_Table_{'W' if side < 0 else 'E'}{k}", default_material="wood:mahogany")
    LAMP = {"asset": f"Lamp_{'W' if side < 0 else 'E'}", "category": "table_lamp", "support": "surface",
            "parts": [{"id": "body", "shape": "lathe", "profile": [[0, 0], [0.08, 0], [0.12, 0.12], [0.1, 0.3],
                                                                   [0.04, 0.4], [0.02, 0.5], [0, 0.5]],
                       "material": "ceramic:#e9e3d3"},
                      {"id": "shade", "shape": "tapered_cylinder", "radius": 0.24, "radius_top": 0.16, "height": 0.34,
                       "at": [0, 0, 0.48], "material": "fabric:linen:#f3ead8"}]}
    asset(LAMP, side * 1.55, 0.90, 0.0, RUG_Z + 0.66)
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

for side in (-1, 1):
    x, y = side * 3.05, 2.2
    asset(recipe("bust_on_pedestal"), x, y, yaw_facing(x, y, 0, 0), 0.0, asset=f"Bust_{side}",
          bbox_m=[0.42, 0.42, 1.75])

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
