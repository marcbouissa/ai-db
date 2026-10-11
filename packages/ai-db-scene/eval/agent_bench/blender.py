"""One fresh headless Blender per run, and what the harness does with the result
(untimed, outside the agent's budget): save the .blend, render probe views, count objects."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import Any

BLENDER = os.path.expanduser("~/bin/blender")
BLENDER_MCP = os.path.expanduser("~/GitRepos/blender_mcp")


def start(port: int, log_path: str, blend: str | None = None) -> subprocess.Popen:
    log = open(log_path, "ab")  # noqa: SIM115 - lives as long as the process
    proc = subprocess.Popen(
        [BLENDER, "-b", *([blend] if blend else ["--factory-startup"]), "-P", "tools/serve_headless.py", "--",
         "--port", str(port), "--from-repo"],
        cwd=BLENDER_MCP, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + 90
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return proc
        if proc.poll() is not None:
            raise RuntimeError(f"Blender exited early (see {log_path})")
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"Blender did not listen on :{port} within 90 s")


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def client(port: int) -> Any:
    from ai_db_scene.blender_client import BlenderClient

    return BlenderClient(port=port, timeout=300.0)


_INVENTORY = """
import bpy
objs = []
for o in bpy.context.scene.objects:
    objs.append({"name": o.name, "type": o.type, "category": str(o.get("category") or ""),
                 "parent": o.parent.name if o.parent else None,
                 "mats": len([m for m in getattr(o.data, "materials", []) if m]) if o.data is not None and hasattr(o.data, "materials") else 0,
                 "tris": sum(len(p.vertices) - 2 for p in o.data.polygons) if o.type == "MESH" else 0})
result = {"objects": objs, "camera": bpy.context.scene.camera.name if bpy.context.scene.camera else None,
          "lights": sum(1 for o in bpy.context.scene.objects if o.type == "LIGHT")}
"""

_PROBE_CAMERA = """
import bpy
from mathutils import Vector
scene = bpy.context.scene
meshes = [o for o in scene.objects if o.type == "MESH" and o.visible_get()]
lo = Vector((min(min((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in meshes) for i in range(3)))
hi = Vector((max(max((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in meshes) for i in range(3)))
center = (lo + hi) / 2
floor = max(lo.z, 0.0)
cam_data = bpy.data.cameras.get("bench_probe") or bpy.data.cameras.new("bench_probe")
cam = bpy.data.objects.get("bench_probe") or bpy.data.objects.new("bench_probe", cam_data)
if cam.name not in scene.collection.objects:
    scene.collection.objects.link(cam)
if VIEW == "interior":
    # inside one corner at eye height, looking at the centre: works for rooms and open scenes
    cam_data.type, cam_data.lens, cam_data.clip_start = "PERSP", 16, 0.05
    w, d = hi.x - lo.x, hi.y - lo.y
    def box(o):
        cs = [o.matrix_world @ Vector(c) for c in o.bound_box]
        return [min(c[i] for c in cs) for i in range(3)], [max(c[i] for c in cs) for i in range(3)]
    # furniture only: skip room-sized meshes (walls, floor, ceiling)
    props = [b for b in map(box, meshes) if (b[1][0] - b[0][0]) < 0.6 * w and (b[1][1] - b[0][1]) < 0.6 * d]
    def clearance(p):
        gaps = [max(b[0][0] - p.x, p.x - b[1][0], b[0][1] - p.y, p.y - b[1][1], 0.0) for b in props
                if b[0][2] <= p.z <= b[1][2] + 0.3 or b[1][2] > p.z - 0.3]
        return min(gaps, default=99.0)
    corners = [Vector((lo.x + fx * w, lo.y + fy * d, floor + 1.6)) for fx in (0.1, 0.9) for fy in (0.1, 0.9)]
    cam.location = max(corners, key=clearance)  # the corner farthest from any furniture
    target = Vector((center.x, center.y, floor + 0.8))
else:
    # plan: straight down, everything above 2.5 m cut away (ceilings, wall tops, roofs)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(hi.x - lo.x, hi.y - lo.y) * 1.1
    cam.location = Vector((center.x, center.y, floor + 30.0))
    cam_data.clip_start, cam_data.clip_end = 30.0 - 2.5, 30.0 + 1.0
    target = Vector((center.x, center.y, floor))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam
result = {"camera": cam.name}
"""


def finish(port: int, out_dir: str, save: bool = True) -> dict[str, Any]:
    """Save the scene, render the agent's own camera view and two probe views; inventory."""
    c = client(port)
    inv = c.run_code(_INVENTORY)
    renders: dict[str, str] = {}
    agent_cam = inv.get("camera")
    if save:
        c.request("save_file", {"filepath": os.path.join(out_dir, "scene.blend")}, timeout=120)
    meshes = [o for o in inv["objects"] if o["type"] == "MESH"]
    if not meshes:
        return {"inventory": inv, "renders": renders}
    views = ([("agent_camera", None)] if agent_cam else []) + [("probe_interior", "interior"), ("probe_plan", "plan")]
    for label, view in views:
        if view is not None:
            c.run_code(_PROBE_CAMERA.replace("VIEW", repr(view)))
        path = os.path.join(out_dir, f"{label}.png")
        c.request("render", {"filepath": path, "resolution": [960, 540], "samples": 24}, timeout=600)
        renders[label] = path
    return {"inventory": inv, "renders": renders}
