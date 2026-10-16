"""Export a scene to glTF or USD in a separate headless Blender, and check the file by importing it back.

The live Blender session is never changed: a separate headless Blender (``$AI_DB_BLENDER_EXE``,
else the live session's own executable, else ``blender`` on PATH) opens an empty scene, loads
the geometry (a stored version's .blend, a fork's ancestor's, or the live scene written to a
temporary .blend), applies the version's own transforms, exports, then imports the file into
another empty scene and compares mesh objects, world transforms and materials with what it
exported.

Geometry-nodes instances (tree leaves: ~120k per park) are realized before the export: the USD
exporter wrote one prim per instance (17 GB of memory for one park, never finishing) or, with
instancing, dropped them; glTF dropped them silently.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ai_db.errors import AiDbError
from ai_db_scene.blender_client import _LINK_HOME, _indent

#: USD prim names are letters, digits and "_" (unicode letters allowed, no leading digit): the
#: exporter rewrites every other character to "_", so names are compared in that form
_SAFE_NAME = """
def safe(name, usd):
    if not usd:
        return name
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    return "_" + out if out[:1].isdigit() else out
"""

_SCRIPT = """
import bpy, json, math, sys
job = json.load(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8"))
fmt, path, usd = job["format"], job["path"], job["format"].startswith("usd")
TOL, ROT_TOL = job["tol"], job["rot_tol_deg"]
""" + _SAFE_NAME + """
def state(objects):
    out = {}
    for o in objects:
        if o.type != "MESH":
            continue
        loc, rot, scl = o.matrix_world.decompose()
        # materials faces use (a slot no face uses is not exported: glTF drops it)
        used = {p.material_index for p in o.data.polygons}
        mats = sorted({safe(m.name, usd) for i, m in enumerate(o.data.materials) if m is not None and i in used})
        out[safe(o.name, usd)] = {"location": list(loc), "rotation": list(rot), "scale": list(scl),
                                  "materials": mats, "vertices": len(o.data.vertices)}
    return out

def empty_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene

# 1. the version, in an empty scene
scene = empty_scene()
with bpy.data.libraries.load(job["blend"], link=False) as (src, dst):
    dst.objects = list(src.objects)
for o in dst.objects:
    if o is None:
        continue
    o.use_fake_user = False
""" + _indent(_LINK_HOME, 4) + """
moved = 0
for name, t in job["transforms"].items():  # a fork's transforms differ from its geometry's
    o = bpy.data.objects.get(name)
    if o is None:
        continue
    rot = [math.radians(a) for a in t["rotation"]]
    if (any(abs(a - b) > TOL for a, b in zip(o.location, t["location"]))
            or any(abs(a - b) > math.radians(ROT_TOL) for a, b in zip(o.rotation_euler, rot))
            or any(abs(a - b) > TOL for a, b in zip(o.scale, t["scale"]))):
        o.location, o.rotation_euler, o.scale = t["location"], rot, t["scale"]
        moved += 1
cam = job.get("camera")
if cam and cam in bpy.data.objects and bpy.data.objects[cam].type == "CAMERA":
    scene.camera = bpy.data.objects[cam]
# geometry-nodes instances become real geometry (see the module docstring)
realize = None
for o in scene.objects:
    if o.type == "MESH" and any(m.type == "NODES" for m in o.modifiers):
        if realize is None:
            realize = bpy.data.node_groups.new("ai_db_realize", "GeometryNodeTree")
            realize.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            realize.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            gin, gout = realize.nodes.new("NodeGroupInput"), realize.nodes.new("NodeGroupOutput")
            node = realize.nodes.new("GeometryNodeRealizeInstances")
            realize.links.new(gin.outputs[0], node.inputs[0])
            realize.links.new(node.outputs[0], gout.inputs[0])
        o.modifiers.new("ai_db_realize", "NODES").node_group = realize
bpy.context.view_layer.update()
exported = state(scene.objects)

# 2. export
if fmt in ("glb", "gltf"):
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB" if fmt == "glb" else "GLTF_SEPARATE",
                              export_cameras=True, export_lights=True, export_apply=True)
else:
    bpy.ops.wm.usd_export(filepath=path, export_cameras=True, export_lights=True, export_materials=True,
                          merge_parent_xform=True)  # prims named after their objects

# 3. import it back into another empty scene and compare
report = {"objects": len(scene.objects), "meshes": len(exported), "applied_transforms": moved}
if job["verify"]:
    scene = empty_scene()
    if fmt in ("glb", "gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        bpy.ops.wm.usd_import(filepath=path)
    bpy.context.view_layer.update()
    back = state(scene.objects)
    # USD never nests meshes: an object with children comes back as an empty of its name holding
    # its mesh under the mesh's data name; that mesh is the object
    for o in scene.objects:
        key, owner = safe(o.name, usd), o.parent
        if (o.type == "MESH" and key not in exported and owner is not None and owner.type == "EMPTY"
                and safe(owner.name, usd) in exported and safe(owner.name, usd) not in back and key in back):
            back[safe(owner.name, usd)] = back.pop(key)
    problems = []
    for name in sorted(exported.keys() - back.keys()):
        problems.append({"name": name, "problem": "missing"})
    for name in sorted(back.keys() - exported.keys()):
        problems.append({"name": name, "problem": "extra"})
    for name in sorted(exported.keys() & back.keys()):
        a, b = exported[name], back[name]
        if any(abs(x - y) > TOL for x, y in zip(a["location"], b["location"])):
            problems.append({"name": name, "problem": "location", "exported": a["location"], "imported": b["location"]})
        dot = min(1.0, abs(sum(x * y for x, y in zip(a["rotation"], b["rotation"]))))
        if math.degrees(2 * math.acos(dot)) > ROT_TOL:
            problems.append({"name": name, "problem": "rotation", "exported": a["rotation"], "imported": b["rotation"]})
        if any(abs(x - y) > TOL * max(1.0, abs(x)) for x, y in zip(a["scale"], b["scale"])):
            problems.append({"name": name, "problem": "scale", "exported": a["scale"], "imported": b["scale"]})
        if a["materials"] != b["materials"]:
            problems.append({"name": name, "problem": "materials", "exported": a["materials"], "imported": b["materials"]})
    report["roundtrip"] = {"imported_meshes": len(back), "mismatches": problems[:50],
                           "mismatch_count": len(problems)}
with open(job["out"], "w", encoding="utf-8") as f:
    json.dump(report, f)
"""


def blender_executable(client: Any = None) -> str:
    """The Blender to run exports with: ``$AI_DB_BLENDER_EXE``, the live session's own binary,
    else ``blender`` on PATH."""
    exe = os.environ.get("AI_DB_BLENDER_EXE")
    if not exe and client is not None:
        try:
            exe = str(client.run_code("import bpy\nresult = bpy.app.binary_path", timeout=30.0) or "")
        except Exception:  # noqa: BLE001 - a Blender that cannot answer: try PATH
            exe = ""
    exe = exe or shutil.which("blender") or ""
    if not exe or not Path(exe).exists():
        raise AiDbError("no Blender executable for the export: set AI_DB_BLENDER_EXE or start Blender "
                        "with the ai-db addon")
    return exe


def export_blend(blend: Path, target: Path, fmt: str, *, transforms: dict[str, Any] | None = None,
                 camera: str | None = None, verify: bool = True, exe: str | None = None,
                 timeout: float = 300.0, tol: float = 1e-3, rot_tol_deg: float = 0.05) -> dict[str, Any]:
    """Export the objects of ``blend`` (written by ``write_blend``) to ``target`` as ``fmt``
    (glb, gltf, usd, usda, usdc, usdz); ``transforms`` (name -> location, rotation in degrees,
    scale) override the stored ones. With ``verify`` the file is imported back: ``roundtrip``
    lists every mesh object that is missing or extra or whose world transform or materials
    changed."""
    exe = exe or blender_executable()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ai_db_export_") as tmp:
        job_file, out_file, script = Path(tmp) / "job.json", Path(tmp) / "out.json", Path(tmp) / "export.py"
        job_file.write_text(json.dumps({
            "blend": str(blend), "path": str(target), "format": fmt, "verify": verify, "out": str(out_file),
            "tol": tol, "rot_tol_deg": rot_tol_deg, "camera": camera, "transforms": transforms or {},
        }), encoding="utf-8")
        script.write_text(_SCRIPT, encoding="utf-8")
        try:
            proc = subprocess.run([exe, "-b", "--factory-startup", "--python-exit-code", "1", "--python",
                                   str(script), "--", str(job_file)],
                                  capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as e:
            raise AiDbError(f"export to {target.name} took longer than {timeout:.0f} s") from e
        if proc.returncode != 0 or not out_file.exists():
            tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-8:])
            raise AiDbError(f"export to {target.name} failed (Blender exit {proc.returncode}): {tail}")
        report: dict[str, Any] = json.loads(out_file.read_text(encoding="utf-8"))
    files = [target] + ([target.with_suffix(".bin")] if fmt == "gltf" else [])
    return {"path": str(target), "format": fmt, "bytes": sum(f.stat().st_size for f in files if f.exists()),
            **report}


def export_version(memory: Any, scene_id: str, target: Path, fmt: str, **kw: Any) -> dict[str, Any]:
    """Export stored version ``scene_id`` (its geometry, a fork's from its ancestor, with the
    version's own transforms); see ``export_blend``."""
    scene = memory._load(scene_id)
    blend = memory._require_geometry(scene)
    transforms = {o.name: {"location": list(o.location), "rotation": list(o.rotation), "scale": list(o.scale)}
                  for o in scene.objects.values()}
    report = export_blend(blend, target, fmt, transforms=transforms, exe=kw.pop("exe", None) or blender_executable(
        memory._blender), camera=scene.camera.name if scene.camera else None, **kw)
    return {"scene_id": scene_id, "geometry_from": blend.stem, **report}


def export_live(client: Any, target: Path, fmt: str, **kw: Any) -> dict[str, Any]:
    """Export the live scene: written to a temporary .blend (as a snapshot is), then exported by
    a separate Blender; see ``export_blend``."""
    exe = kw.pop("exe", None) or blender_executable(client)
    camera = client.run_code("import bpy\nc = bpy.context.scene.camera\nresult = c.name if c else None",
                             timeout=30.0)
    with tempfile.TemporaryDirectory(prefix="ai_db_live_") as tmp:
        blend = Path(tmp) / "live.blend"
        client.write_blend(str(blend))
        return export_blend(blend, target, fmt, exe=exe, camera=camera, **kw)
