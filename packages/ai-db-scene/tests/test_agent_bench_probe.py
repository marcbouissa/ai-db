"""The bench plan probe cuts away only what covers the scene (aidb7 park / beach: the 2.5 m cut hid
every tree and palm canopy, only the trunks showed from above)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "agent_bench"))
import blender as bench_blender

_SCENE = """
import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
bpy.context.object.name = "floor"
bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=4, location=(1, 1, 2))
bpy.context.object.name = "trunk"
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.5, location=(1, 1, 5))
bpy.context.object.name = "canopy"
bpy.ops.mesh.primitive_cube_add(size=1, location=(-2, -2, 0.5))
bpy.context.object.name = "box"
if CEILING:
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 3.05))
    bpy.context.object.name = "Room_Ceiling"
    bpy.context.object.scale = (10, 10, 0.1)
bpy.context.view_layer.update()
result = {"ok": True}
"""


@pytest.mark.blender
def test_plan_probe_keeps_canopies_in_the_open_and_cuts_ceilings(private_blender):
    from ai_db_scene.blender_client import BlenderClient

    c = BlenderClient(port=private_blender, timeout=120)
    probe = bench_blender._PROBE_CAMERA.replace("VIEW", repr("plan"))
    clip = "\nresult['clip'] = [cam_data.clip_start, cam_data.clip_end, cam.location.z]\n"
    c.run_code(_SCENE.replace("CEILING", "False"))
    got = c.run_code(probe + clip)
    assert got["clipped"] is False
    start, end, z = got["clip"]
    assert z - start > 6.5 and z - end < 0  # the canopy's top (6.5 m) and the floor are in range
    c.run_code(_SCENE.replace("CEILING", "True"))
    got = c.run_code(probe + clip)
    assert got["clipped"] is True
    start, _end, z = got["clip"]
    assert z - start == pytest.approx(2.5, abs=0.01)  # under a ceiling: everything above 2.5 m cut
