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


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _listener_pid(port: int) -> int | None:
    """The pid of the process holding the IPv4 listening socket on ``port`` (Linux /proc)."""
    inode = None
    with open("/proc/net/tcp", encoding="ascii") as f:
        for line in f.readlines()[1:]:
            parts = line.split()
            if parts[1].endswith(f":{port:04X}") and parts[3] == "0A":  # 0A = LISTEN
                inode = parts[9]
                break
    if inode is None:
        return None
    target = f"socket:[{inode}]"
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.readlink(f"/proc/{pid}/fd/{fd}") == target:
                    return int(pid)
        except OSError:
            continue
    return None


def _pgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def _sandboxed(cmd: list[str], port: int, out_dir: str, sock_dir: str) -> list[str]:
    """``cmd`` (Blender) in a bubblewrap sandbox for runs driven by a cloud model: Blender's
    run_code runs the model's Python, which could otherwise read the private repos or query local
    services (the ai-db daemon). No network (its port reaches the host through a unix socket
    pair of socat relays), /home hidden except Blender, the addon, the texture cache and this
    run's output folder, its own pid namespace (no /proc/<pid>/root way out)."""
    home = os.path.expanduser("~")
    blender_dir = os.path.dirname(os.path.realpath(BLENDER))
    sock = os.path.join(sock_dir, "blender.sock")
    args = ["bwrap", "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/home", "--tmpfs", "/mnt", "--tmpfs", "/media",
            "--bind", "/tmp", "/tmp"]
    for private in sorted(d for d in os.listdir("/tmp") if d.startswith("claude-")):  # agent scratchpads
        args += ["--tmpfs", os.path.join("/tmp", private)]
    args += ["--dir", home, "--ro-bind", blender_dir, blender_dir, "--ro-bind", BLENDER_MCP, BLENDER_MCP]
    cache = os.path.join(home, ".cache", "ai-db")
    if os.path.isdir(cache):
        args += ["--ro-bind", cache, cache]
    args += ["--bind", out_dir, out_dir, "--bind", sock_dir, sock_dir, "--setenv", "HOME", home,
             "--unshare-net", "--unshare-pid", "--die-with-parent", "--chdir", BLENDER_MCP]
    inner = (f"socat UNIX-LISTEN:{sock},fork TCP:127.0.0.1:{port} & exec "
             + " ".join(_q(c) for c in [os.path.realpath(BLENDER), *cmd[1:]]))
    outer = (f"socat TCP-LISTEN:{port},bind=127.0.0.1,reuseaddr,fork UNIX-CONNECT:{sock} & exec "
             + " ".join(_q(c) for c in args) + " sh -c " + _q(inner))
    return ["sh", "-c", outer]


def _q(s: str) -> str:
    import shlex

    return shlex.quote(s)


def _answers(port: int) -> bool:
    """Blender itself answers on the port (a relay can accept before Blender listens)."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall(b'{"type": "get_scene_info", "params": {}}\n')
            s.settimeout(10)
            return bool(s.recv(1))
    except OSError:
        return False


def start(port: int, log_path: str, blend: str | None = None, sandbox_dir: str | None = None) -> subprocess.Popen:
    """A fresh Blender listening on ``port``. Fails rather than hand back another run's Blender:
    the port must be free before, and the process listening on it must be in the new process's
    session (start_new_session: its process group is the Popen pid). ``sandbox_dir``: run it in
    the sandbox (_sandboxed) with write access to that folder only."""
    if _port_open(port):
        raise RuntimeError(f"port {port} is already in use: refusing to share another run's Blender")
    log = open(log_path, "ab")  # noqa: SIM115 - lives as long as the process
    cmd = [BLENDER, "-b", *([blend] if blend else ["--factory-startup"]), "-P", "tools/serve_headless.py", "--",
           "--port", str(port), "--from-repo"]
    if sandbox_dir is not None:
        import tempfile

        cmd = _sandboxed(cmd, port, sandbox_dir, tempfile.mkdtemp(prefix="aidb-bench-sock-"))
    proc = subprocess.Popen(cmd, cwd=BLENDER_MCP, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            stop(proc)  # the rest of its session (a relay) too
            raise RuntimeError(f"Blender exited early (see {log_path})")
        if _port_open(port):
            owner = _listener_pid(port)
            # our Blender (or its relay) in this session bound it, and Blender answers
            if owner is not None and _pgid(owner) == proc.pid and (sandbox_dir is None or _answers(port)):
                return proc
        time.sleep(0.5)
    stop(proc)
    raise RuntimeError(f"Blender did not listen on :{port} within 90 s")


def stop(proc: subprocess.Popen) -> None:
    """Blender and everything in its session (the sandbox's relays)."""
    import signal

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()


def client(port: int) -> Any:
    from ai_db_scene.blender_client import BlenderClient

    return BlenderClient(port=port, timeout=300.0)


_INVENTORY = """
import bpy
from mathutils import Vector

def _meshes(o):
    return ([o] if o.type == "MESH" else []) + [c for c in o.children_recursive if c.type == "MESH"]

def _size(o):
    # extent of the object and all its mesh descendants in the root's own (yaw) frame, in metres:
    # comparable with the category priors (width, depth, height) whatever the rotation
    rot = o.matrix_world.to_quaternion().inverted()
    pts = [rot @ (m.matrix_world @ Vector(c)) for m in _meshes(o) for c in m.bound_box]
    if not pts:
        return None
    return [round(max(p[i] for p in pts) - min(p[i] for p in pts), 4) for i in range(3)]

objs = []
for o in bpy.context.scene.objects:
    objs.append({"name": o.name, "type": o.type, "category": str(o.get("category") or ""),
                 "parent": o.parent.name if o.parent else None,
                 "mats": len([m for m in getattr(o.data, "materials", []) if m]) if o.data is not None and hasattr(o.data, "materials") else 0,
                 "tris": sum(len(p.vertices) - 2 for p in o.data.polygons) if o.type == "MESH" else 0,
                 "faces": len(o.data.polygons) if o.type == "MESH" else 0,
                 "size_m": _size(o) if o.parent is None else None})
result = {"objects": objs, "camera": bpy.context.scene.camera.name if bpy.context.scene.camera else None,
          "lights": sum(1 for o in bpy.context.scene.objects if o.type == "LIGHT")}
"""

_PROBE_CAMERA = """
import bpy
from mathutils import Vector
scene = bpy.context.scene
meshes = [o for o in scene.objects if o.type == "MESH" and o.visible_get()]

def ground_like(o):
    # terrain, water, the ground to the horizon: framing those showed an empty plane (outdoor scenes)
    ext = [max((o.matrix_world @ Vector(c))[i] for c in o.bound_box) - min((o.matrix_world @ Vector(c))[i] for c in o.bound_box)
           for i in range(2)]
    return (o.get("ai_db_terrain") or str(o.get("category") or "") in ("ground", "water")
            or o.name.startswith(("ground_beyond", "Ground", "ground", "Plane")) and max(ext) > 30.0 or max(ext) > 60.0)

meshes = [o for o in meshes if not ground_like(o)] or meshes
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
