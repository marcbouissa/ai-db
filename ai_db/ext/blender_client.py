"""Single client for the blender_ox socket server (TCP, newline-delimited JSON).

Protocol: see blender_mcp/PROTOCOL.md. One request per connection; supports
single commands and ``{"batch": [...]}``. Zero cost when Blender is inactive.
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any

from ai_db.constants import BLENDER_OX_HOST, BLENDER_OX_PORT, BLENDER_OX_TIMEOUT_S
from ai_db.scene.models import CameraState, MaterialState, ObjectState, SceneState


class BlenderError(RuntimeError):
    """Blender returned an error (or the transport failed)."""

    def __init__(
        self,
        message: str,
        error_type: str = "",
        hint: str = "",
        response: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.hint = hint
        self.response = response or {}


class BlenderUnavailable(BlenderError):
    """Blender socket is not reachable / timed out / sent garbage."""


def default_port() -> int:
    raw = os.environ.get("AI_DB_BLENDER_PORT")
    try:
        return int(raw) if raw else BLENDER_OX_PORT
    except ValueError:
        return BLENDER_OX_PORT


def _ok(res: dict[str, Any]) -> bool:
    return res.get("status") == "success" or res.get("ok") == 1


def _error_from(res: dict[str, Any]) -> BlenderError:
    msg = str(res.get("message") or res.get("error") or res.get("e") or "Blender error")
    return BlenderError(msg, str(res.get("error_type") or ""), str(res.get("hint") or ""), res)


class BlenderClient:
    """TCP client for the blender_ox addon."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout: float = BLENDER_OX_TIMEOUT_S,
    ):
        self.host = host or os.environ.get("AI_DB_BLENDER_HOST", BLENDER_OX_HOST)
        self.port = port if port is not None else default_port()
        self.timeout = timeout

    def _roundtrip(self, payload: Any, timeout: float | None = None) -> dict[str, Any]:
        t = timeout if timeout is not None else self.timeout
        try:
            with socket.create_connection((self.host, self.port), timeout=t) as s:
                s.settimeout(t)
                s.sendall(json.dumps(payload).encode("utf-8") + b"\n")
                buf = b""
                while b"\n" not in buf:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
        except (OSError, TimeoutError) as e:
            raise BlenderUnavailable(
                f"Cannot reach Blender at {self.host}:{self.port} ({type(e).__name__}: {e})"
            ) from e
        line = buf.strip().splitlines()[0] if buf.strip() else b""
        if not line:
            raise BlenderUnavailable("Empty response from Blender")
        try:
            res = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise BlenderUnavailable(f"Invalid JSON from Blender: {e}") from e
        if not isinstance(res, dict):
            raise BlenderUnavailable("Unexpected response shape from Blender")
        return res

    def request(
        self, type_: str, params: dict[str, Any] | None = None, timeout: float | None = None
    ) -> Any:
        """Run one handler; return its ``result``. Raises BlenderError on failure."""
        res = self._roundtrip({"type": type_, "params": params or {}}, timeout)
        if not _ok(res):
            raise _error_from(res)
        return res.get("result")

    def batch(
        self, commands: list[dict[str, Any]], timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """Run commands as one undo step. Returns per-command response dicts (does not raise on partial failure)."""
        res = self._roundtrip({"batch": commands}, timeout)
        results = res.get("results")
        if not isinstance(results, list):
            raise _error_from(res)
        return results

    def is_available(self, timeout: float = 0.5) -> bool:
        try:
            self.request("get_scene_info", timeout=timeout)
            return True
        except BlenderError:
            return False

    def get_scene_info(self, detailed: bool = False) -> Any:
        return self.request("get_scene_info", {"detailed": detailed} if detailed else {})

    def scene_snapshot(self) -> Any:
        return self.request("scene_snapshot")

    def run_code(self, code: str, timeout: float = 30.0) -> Any:
        """Execute Python in Blender; return the ``result`` variable."""
        return self.request("run_code", {"code": code}, timeout=timeout)

    def extract_scene_state(
        self, project: str = "default", name: str = "Blender Scene"
    ) -> SceneState:
        """
        Extract detailed 3D scene state from Blender and hydrate into a SceneState model.
        Extracts mesh statistics, transforms, materials, active camera, and collection hierarchy.
        """
        extractor_code = """
import bpy
import math

objs = {}
materials = {}
collections = {}

for o in bpy.context.scene.objects:
    # Stable UUID custom property
    uid = o.get("uuid")
    if not uid:
        import uuid as _u
        uid = f"blender-{str(_u.uuid4())[:8]}"
        o["uuid"] = uid

    # Rotation in Euler degrees
    if hasattr(o, "rotation_euler"):
        rot = [round(math.degrees(a), 2) for a in o.rotation_euler]
    else:
        rot = [0.0, 0.0, 0.0]

    # Materials
    mats = []
    if hasattr(o.data, "materials"):
        for m in o.data.materials:
            if m:
                mats.append(m.name)
                if m.name not in materials:
                    materials[m.name] = {
                        "name": m.name,
                        "type": "PRINCIPLED_BSDF",
                    }

    # Mesh geometry info
    v_count = len(o.data.vertices) if o.type == "MESH" and hasattr(o.data, "vertices") else 0
    f_count = len(o.data.polygons) if o.type == "MESH" and hasattr(o.data, "polygons") else 0

    col_name = o.users_collection[0].name if o.users_collection else "Scene Collection"

    objs[uid] = {
        "uuid": uid,
        "name": o.name,
        "type": o.type,
        "collection": col_name,
        "location": [round(v, 4) for v in o.location],
        "rotation": rot,
        "scale": [round(v, 4) for v in o.scale],
        "vertices": v_count,
        "faces": f_count,
        "materials": mats,
        "mesh_data": o.data.name if hasattr(o, "data") and o.data else None,
        "parent_uuid": o.parent.get("uuid") if o.parent else None,
        "hide_viewport": o.hide_viewport,
        "hide_render": o.hide_render,
    }

# Camera info
cam = bpy.context.scene.camera
camera_data = {
    "name": "Camera",
    "location": [0.0, 0.0, 10.0],
    "rotation": [0.0, 0.0, 0.0],
    "lens": 50.0,
}
if cam:
    cam_rot = [round(math.degrees(a), 2) for a in cam.rotation_euler] if hasattr(cam, "rotation_euler") else [0.0, 0.0, 0.0]
    lens = cam.data.lens if hasattr(cam, "data") and hasattr(cam.data, "lens") else 50.0
    camera_data = {
        "name": cam.name,
        "location": [round(v, 4) for v in cam.location],
        "rotation": cam_rot,
        "lens": lens,
    }

result = {
    "timestamp": __import__("time").time(),
    "objects": objs,
    "materials": materials,
    "collections": collections,
    "camera": camera_data,
}
"""
        extracted = self.run_code(extractor_code)
        if not isinstance(extracted, dict):
            raise BlenderError(f"Unexpected response from Blender extractor: {extracted}")

        scene_id = f"blender-{int(extracted.get('timestamp', time.time()))}"
        scene = SceneState(
            scene_id=scene_id,
            project=project,
            name=name,
            timestamp=extracted.get("timestamp", time.time()),
            version=1,
            tags=["blender", "live-scene"],
        )

        for obj_uuid, obj_data in extracted.get("objects", {}).items():
            scene.objects[obj_uuid] = ObjectState(
                uuid=obj_data.get("uuid", obj_uuid),
                name=obj_data.get("name", ""),
                type=obj_data.get("type", "MESH"),
                collection=obj_data.get("collection", "Collection"),
                location=tuple(obj_data.get("location", [0.0, 0.0, 0.0])),
                rotation=tuple(obj_data.get("rotation", [0.0, 0.0, 0.0])),
                scale=tuple(obj_data.get("scale", [1.0, 1.0, 1.0])),
                mesh_data=obj_data.get("mesh_data"),
                vertices=obj_data.get("vertices", 0),
                faces=obj_data.get("faces", 0),
                materials=obj_data.get("materials", []),
                parent_uuid=obj_data.get("parent_uuid"),
                hide_viewport=obj_data.get("hide_viewport", False),
                hide_render=obj_data.get("hide_render", False),
            )

        for mat_name, mat_data in extracted.get("materials", {}).items():
            scene.materials[mat_name] = MaterialState(
                name=mat_data.get("name", mat_name),
                type=mat_data.get("type", "PRINCIPLED_BSDF"),
            )

        cam_info = extracted.get("camera", {})
        if cam_info:
            scene.camera = CameraState(
                name=cam_info.get("name", "Camera"),
                location=tuple(cam_info.get("location", [0.0, 0.0, 10.0])),
                rotation=tuple(cam_info.get("rotation", [0.0, 0.0, 0.0])),
                lens=float(cam_info.get("lens", 50.0)),
            )

        return scene

    def get_available_libraries(self) -> dict[str, Any]:
        """
        Query Blender for linked libraries (.blend files) and configured Asset Libraries.
        Returns:
            {
                "linked_libraries": [{"name": ..., "filepath": ...}],
                "asset_libraries": [{"name": ..., "path": ...}],
                "available_assets": [{"name": ..., "type": ..., "library": ...}]
            }
        """
        code = """
import bpy
import os

linked = []
for lib in bpy.data.libraries:
    linked.append({
        "name": lib.name,
        "filepath": lib.filepath,
        "is_missing": getattr(lib, "is_missing", False),
    })

asset_libs = []
try:
    prefs = bpy.context.preferences.filepaths.asset_libraries
    for al in prefs:
        path_str = bpy.path.abspath(al.path)
        asset_libs.append({
            "name": al.name,
            "path": path_str,
            "exists": os.path.exists(path_str),
        })
except Exception:
    pass

# Collect assets marked in current blend file or loaded data
local_assets = []
for obj in bpy.data.objects:
    if getattr(obj, "asset_data", None):
        local_assets.append({
            "name": obj.name,
            "type": "OBJECT",
            "description": getattr(obj.asset_data, "description", ""),
        })

for mat in bpy.data.materials:
    if getattr(mat, "asset_data", None):
        local_assets.append({
            "name": mat.name,
            "type": "MATERIAL",
            "description": getattr(mat.asset_data, "description", ""),
        })

result = {
    "linked_libraries": linked,
    "asset_libraries": asset_libs,
    "local_assets": local_assets,
    "linked_libraries_count": len(linked),
    "asset_libraries_count": len(asset_libs),
    "local_assets_count": len(local_assets),
}
"""
        res = self.run_code(code)
        if isinstance(res, dict):
            return res
        return {
            "linked_libraries": [],
            "asset_libraries": [],
            "local_assets": [],
            "linked_libraries_count": 0,
            "asset_libraries_count": 0,
            "local_assets_count": 0,
        }


def try_get_blender_client(
    host: str | None = None, port: int | None = None
) -> BlenderClient | None:
    """Return a client if Blender is listening, else None."""
    client = BlenderClient(host=host, port=port)
    return client if client.is_available(timeout=0.3) else None
