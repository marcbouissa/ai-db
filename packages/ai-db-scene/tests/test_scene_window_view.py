"""What the windows look out on (daylight.window_view_commands): the Nishita sky is dark navy below
the horizon, so every interior window showed a sea-like band under a bright sky (aidb8 contact
sheet, bench/_after_666586a). Camera rays now see a light, sunlit ground and a distant tree line;
the room's light is unchanged."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from ai_db_scene import daylight, shell
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

FIX = Path(__file__).parent / "fixtures"
IN = json.loads((FIX / "bench_specs_aidb7.json").read_text())


def _room():
    spec, errors = sp._spec_from_args({"spec": copy.deepcopy(IN["last/qwen/living_room"]["spec"])}, remembered=False)
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), 0, 3)
    sh = shell.plan(fixed.spec, fixed.placements)
    return shell.apply(fixed.spec, fixed.placements, sh)


def _has_view(cmds):
    return any(c["type"] == "run_code" and "ai_db_view_" in c["params"]["code"] for c in cmds)


def test_a_daylit_room_gets_the_window_view_and_a_windowless_one_does_not():
    spec, pls = _room()
    off = sp.frame_offset(spec)
    lit = sp.lighting_commands(spec, pls, off)
    assert _has_view(lit)
    # after the rig: the view is built on the rig's sky
    assert [c["type"] for c in lit].index("daylight_rig") < next(
        i for i, c in enumerate(lit) if c["type"] == "run_code" and "ai_db_view_" in c["params"]["code"])
    bare = [p for p in pls if p.category != "window"]
    assert not _has_view(sp.lighting_commands(spec, bare, off))


def test_the_view_code_compiles_and_keeps_the_room_light():
    code = daylight.window_view_commands()[0]["params"]["code"]
    compile(code, "window_view", "exec")
    assert '"Is Camera Ray"' in code  # only what the camera sees changes
    assert "clamp_result = False" in code  # sky radiance is far above 1


#: the camera's view through a window: mean luminance (display-referred, the scene's exposure and
#: view transform) of the rows below the horizon over the rows above it.
_THROUGH_WINDOW = """
import bpy, math
from mathutils import Vector
sc = bpy.context.scene
def root(o):
    while o.parent is not None:
        o = o.parent
    return o
wins = sorted({root(o) for o in sc.objects if root(o).get("category") == "window"}, key=lambda o: o.name)
w = wins[0]
pts = [m.matrix_world @ Vector(c) for m in [w, *w.children_recursive] if m.type == "MESH" for c in m.bound_box]
centre = sum(pts, Vector()) / len(pts)
yaw = w.matrix_world.to_euler().z
inward = Vector((math.sin(yaw), -math.cos(yaw), 0.0))
cam = bpy.data.objects.new("view_cam", bpy.data.cameras.new("view_cam"))
sc.collection.objects.link(cam)
cam.data.lens = 50.0
cam.location = centre + inward * 1.2
cam.rotation_euler = (math.radians(90.0), 0.0, yaw)
sc.camera = cam
sc.render.engine = "CYCLES"
sc.cycles.samples = 16
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 96, 72, 100
sc.render.image_settings.file_format = "PNG"
sc.render.filepath = PATH
bpy.ops.render.render(write_still=True)
img = bpy.data.images.load(PATH, check_existing=False)
px, (iw, ih) = list(img.pixels), img.size
def lum(rows):
    s = n = 0
    for y in rows:
        for x in range(iw // 8, iw - iw // 8):
            i = (y * iw + x) * 4
            s += 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]
            n += 1
    return s / n
# image rows start at the bottom; skip the middle (the horizon, trees and the transom bar)
result = {"below": lum(range(ih // 8, ih * 3 // 8)), "above": lum(range(ih * 5 // 8, ih * 7 // 8))}
"""


@pytest.mark.blender
def test_the_view_below_the_horizon_is_a_light_ground_not_a_dark_band(private_blender, tmp_path, monkeypatch):
    from ai_db_scene.blender_client import BlenderClient

    class Mem:
        def record_run(self, run):
            return "run-local"

        def snapshot_scene(self, **kw):
            return "scene-local"

    monkeypatch.setattr(sp, "_memory", lambda: Mem())
    spec = copy.deepcopy(IN["last/qwen/living_room"]["spec"])
    out = sp._handle_build_scene({"spec": spec, "clear": True, "snapshot": False, "exposure_check": False})
    assert out["status"] == "built", out
    client = BlenderClient(port=private_blender, timeout=300)
    seen = client.run_code(_THROUGH_WINDOW.replace("PATH", repr(str(tmp_path / "window.png"))))
    seen = seen.get("result", seen) if isinstance(seen, dict) else seen
    ratio = seen["below"] / max(seen["above"], 1e-6)
    # measured on this room: navy band 0.38 / 0.87 (ratio 0.44); light ground 0.66 / 0.87 (0.76)
    assert 0.55 <= ratio <= 1.05, seen
    assert seen["below"] > 0.5, seen  # light, not a dark band
