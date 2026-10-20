"""Daylight through a room's windows, so an interior reads like a photo, not a grey box.

The interior rig was a sun at a fixed azimuth (usually outside a window-less wall), a sky the
portals sample, and a big ceiling area fill at 8 W/m² that drowned both: flat, shadowless grey
light from above (Marc's review list: "flat grey interior lighting"). With windows (the spec's or
the shell's, shell.py) the light now comes in through them:

* the sun stands outside the main window wall, low enough to throw patches of light on the floor;
* a soft, slightly cool area light fills each window opening from inside (the sky's light, which
  a 24-sample Cycles render barely gets through small portals);
* the ceiling fill drops to a weak bounce (DAYLIT_FILL_W_PER_M2), so the room has a bright and a
  dark side, and the warm lamps show against it.

A room with no window keeps the old rig (the addon's daylight_rig adds its fill under the ceiling
when it finds no portal; the pipeline's ceiling fill stays as it was).
"""

from __future__ import annotations

import math
from typing import Any

#: W per m² of window opening of the sky light just inside a window (area light, facing in)
WINDOW_W_PER_M2 = 25.0
WINDOW_LIGHT_COLOR = (0.9, 0.95, 1.0)  # the sky: slightly cool against the warm lamps
#: ceiling fill in a room with windows: a bounce, not the main light
DAYLIT_FILL_W_PER_M2 = 1.0
SUN_ELEVATION_DEG = 32.0  # low enough to throw light patches across the floor
SUN_OFF_NORMAL_DEG = 35.0  # the sun this far round from straight out of the window wall
SUN_ENERGY = 12.0
#: exposure (stops) of a daylit room against indoor_exposure: the build's exposure probe pulled
#: every interior down 2-4 stops (to its 4-stop limit in the living room); starting near the
#: right value keeps the sun patches and the dark side of the room
DAYLIT_EXPOSURE_SHIFT = -2.5
#: lamps in a daylit room: a little stronger so the warm pools still read by day
LAMP_BOOST = 1.5


def windows(placements: list[Any]) -> list[Any]:
    return [p for p in placements if p.category == "window"]


def _inward(p: Any) -> tuple[float, float]:
    """A window's inward normal: its front (-Y turned by its yaw) faces the room."""
    yaw = math.radians(p.rotation_deg_z)
    return math.sin(yaw), -math.cos(yaw)


def sun_azimuth(placements: list[Any]) -> float | None:
    """Degrees toward the sun (CCW from +X): outside the wall with the most window area, a little
    round so the light falls in at a slant. None without windows."""
    wins = windows(placements)
    if not wins:
        return None
    by_dir: dict[tuple[int, int], float] = {}
    for p in wins:
        nx, ny = _inward(p)
        key = (round(nx * 4), round(ny * 4))  # walls of a rect room; arcs of a round one
        by_dir[key] = by_dir.get(key, 0.0) + p.dims[0] * p.dims[2]
    best = max(by_dir, key=lambda k: by_dir[k])
    sx, sy = -best[0], -best[1]  # toward the sun: out of the room through that wall
    return round((math.degrees(math.atan2(sy, sx)) + SUN_OFF_NORMAL_DEG) % 360.0, 1)


def window_light_commands(placements: list[Any], off: tuple[float, float, float]) -> list[dict[str, Any]]:
    """An area light in every window opening, just inside it, facing into the room."""
    cmds: list[dict[str, Any]] = []
    for p in windows(placements):
        nx, ny = _inward(p)
        w, d, h = p.dims
        x, y, z = (p.location[0] + off[0] + nx * (d / 2 + 0.05), p.location[1] + off[1] + ny * (d / 2 + 0.05),
                   p.location[2] + off[2])
        # an area light shines along its local -Z: tilt it level (90 deg about X) and turn it
        # so -Z points along the inward normal
        yaw = math.degrees(math.atan2(-nx, ny))
        name = f"{p.id}_daylight"
        cmds.append({"type": "add_light", "params": {
            "light_type": "AREA", "name": name, "energy": round(WINDOW_W_PER_M2 * w * h, 1),
            "color": list(WINDOW_LIGHT_COLOR), "location": [round(x, 4), round(y, 4), round(z, 4)],
            "rotation": [90.0, 0.0, round(yaw, 2)]}})
        cmds.append({"type": "run_code", "params": {"code": _AREA_RECT.format(name=name, sx=round(w * 0.9, 3),
                                                                               sy=round(h * 0.9, 3))}})
    if cmds:
        cmds.append({"type": "run_code", "params": {"code": _WINDOW_GLASS}})
    return cmds


#: window panes: clear glass blocked the sun (a refracting pane casts a black shadow for a sun
#: lamp in Cycles: no caustics), so the room got no patch of sunlight. Shadow and diffuse rays now
#: pass the pane; the camera still sees a glossy, mostly clear glass.
_WINDOW_GLASS = """
import bpy
mat = bpy.data.materials.get("ai_db_window_glass")
if mat is None:
    mat = bpy.data.materials.new("ai_db_window_glass")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    path = nt.nodes.new("ShaderNodeLightPath")
    through = nt.nodes.new("ShaderNodeMath")
    through.operation = "MAXIMUM"
    nt.links.new(path.outputs["Is Shadow Ray"], through.inputs[0])
    nt.links.new(path.outputs["Is Diffuse Ray"], through.inputs[1])
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    gloss = nt.nodes.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Roughness"].default_value = 0.02
    fres = nt.nodes.new("ShaderNodeLayerWeight")
    fres.inputs["Blend"].default_value = 0.15
    look = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fres.outputs["Fresnel"], look.inputs["Fac"])
    nt.links.new(clear.outputs[0], look.inputs[1])
    nt.links.new(gloss.outputs[0], look.inputs[2])
    final = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(through.outputs[0], final.inputs["Fac"])
    nt.links.new(look.outputs[0], final.inputs[1])
    clear2 = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(clear2.outputs[0], final.inputs[2])
    nt.links.new(final.outputs[0], out.inputs["Surface"])
    try:
        mat.surface_render_method = "BLENDED"
    except (AttributeError, TypeError):
        pass


def root(o):
    while o.parent is not None:
        o = o.parent
    return o


done = []
for o in bpy.context.scene.objects:
    if o.type == "MESH" and "glass" in o.name.lower() and root(o).get("category") == "window":
        o.data.materials.clear()
        o.data.materials.append(mat)
        done.append(o.name)
result = {"window_glass": done}
"""


_AREA_RECT = """
import bpy
o = bpy.data.objects.get({name!r})
if o is not None and o.type == "LIGHT" and o.data.type == "AREA":
    o.data.shape = "RECTANGLE"
    o.data.size, o.data.size_y = {sx!r}, {sy!r}
    o.data.spread = 2.0  # radians: a soft window glow, not a beam
result = {{"window_light": o.name if o else None}}
"""
