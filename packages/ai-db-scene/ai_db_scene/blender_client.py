"""Single client for the Blender addon socket server (TCP, newline-delimited JSON).

Protocol: see blender_mcp/PROTOCOL.md. One request per connection; supports
single commands and ``{"batch": [...]}``. Zero cost when Blender is inactive.
"""

from __future__ import annotations

import json
import os
import socket
import textwrap
import time
from typing import Any

from ai_db_scene.constants import BLENDER_OX_HOST, BLENDER_OX_PORT, BLENDER_OX_TIMEOUT_S
from ai_db_scene.models import CameraState, MaterialState, ObjectState, SceneState


def _indent(code: str, n: int) -> str:
    return textwrap.indent(code, " " * n)


#: Blender code: link object ``o`` into the collection ``write_blend`` recorded for it
#: (created under the scene collection when missing). Needs ``bpy`` and ``scene``.
_LINK_HOME = (
    "cname = o.get('ai_db_collection') or ''\n"
    "coll = scene.collection if cname in ('', scene.collection.name) else bpy.data.collections.get(cname)\n"
    "if coll is None:\n"
    "    coll = bpy.data.collections.new(cname)\n"
    "    scene.collection.children.link(coll)\n"
    "elif coll is not scene.collection and coll.name not in scene.collection.children:\n"
    "    scene.collection.children.link(coll)\n"
    "if o.name not in coll.objects:\n"
    "    coll.objects.link(o)\n"
)

#: Blender code: a material the load added whose name only gained a ``.NNN`` suffix next to a
#: material of the stored name (present before the load, pointers in ``before``) is that
#: material: its users are moved to the original and the copy is removed.
_DEDUP_MATERIALS = (
    "import re as _re\n"
    "for m in [m for m in bpy.data.materials if m.as_pointer() not in before]:\n"
    "    base = _re.sub(r'\\.\\d{3,}$', '', m.name)\n"
    "    old = bpy.data.materials.get(base)\n"
    "    if base != m.name and old is not None and old.as_pointer() in before:\n"
    "        m.user_remap(old)\n"
    "        bpy.data.materials.remove(m)\n"
)


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
    """TCP client for the Blender addon socket server."""

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

    def scene_snapshot(self) -> list[dict[str, Any]]:
        """Live objects as plain dicts (the Blender addon has no ``scene_snapshot`` handler)."""
        return [{"uuid": o.uuid, "name": o.name, "type": o.type, "collection": o.collection,
                 "location": list(o.location), "rotation": list(o.rotation), "scale": list(o.scale),
                 "materials": o.materials}
                for o in self.extract_scene_state().objects.values()]

    def run_code(self, code: str, timeout: float = 30.0) -> Any:
        """Execute Python in Blender; return the ``result`` variable.

        The handler replies ``{"result": <result>}``; callers get the variable itself
        (every caller reads its keys, so the wrapper used to make them see nothing).
        """
        res = self.request("run_code", {"code": code}, timeout=timeout)
        if not isinstance(res, dict) or "result" not in res:
            raise BlenderError(f"run_code returned an unexpected shape: {str(res)[:200]}")
        return res["result"]

    def write_blend(self, path: str) -> int:
        """Write every object of the current scene (with its mesh, materials, ...) to ``path``.

        Uses ``bpy.data.libraries.write``: the session's own file and path are untouched,
        unlike ``save_file``. External images are packed into the snapshot for the write (and
        unpacked again in the session): a snapshot must not depend on texture files that a
        cache clean-up or a moved asset folder deletes later. Remaining file paths are written
        absolute (relative ones would resolve against the snapshot folder). Returns the number
        of objects written.
        """
        code = (
            "import bpy, os\n"
            f"path = {path!r}\n"
            "os.makedirs(os.path.dirname(path), exist_ok=True)\n"
            "objs = set(bpy.context.scene.objects)\n"
            "for o in objs:\n"
            "    o['ai_db_collection'] = o.users_collection[0].name if o.users_collection else ''\n"
            "packed = []\n"
            "for img in bpy.data.images:\n"
            "    if img.packed_file is None and img.source in ('FILE', 'SEQUENCE', 'TILED') and img.filepath \\\n"
            "            and os.path.exists(bpy.path.abspath(img.filepath)):\n"
            "        try:\n"
            "            img.pack()\n"
            "            packed.append(img)\n"
            "        except RuntimeError:\n"
            "            pass\n"
            "try:\n"
            "    bpy.data.libraries.write(path, objs, path_remap='ABSOLUTE', fake_user=True)\n"
            "finally:\n"
            "    for img in packed:\n"
            "        img.unpack(method='REMOVE')\n"
            "result = {'objects': len(objs)}\n"
        )
        return int(self.run_code(code, timeout=120.0)["objects"])

    def load_blend(self, path: str, replace: bool, camera: str | None) -> list[str]:
        """Bring the objects stored by ``write_blend`` into the current scene.

        ``replace`` first removes every object of the scene and the data only they used, so
        the restored meshes and materials keep their stored names (a second restore used to
        add ``.001`` copies of every material, which ``diff_scenes`` then reported as changed).
        A loaded material whose name only gained a ``.NNN`` suffix next to a material of the
        stored name is replaced by that one (``additive``). Objects go back into their
        original collection (created under the scene collection when missing). Returns the
        names Blender gave the restored objects.
        """
        code = (
            "import bpy\n"
            f"path, replace, camera = {path!r}, {replace!r}, {camera!r}\n"
            "scene = bpy.context.scene\n"
            "if replace:\n"
            "    for o in list(scene.objects):\n"
            "        bpy.data.objects.remove(o, do_unlink=True)\n"
            "    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)\n"
            "before = {m.as_pointer() for m in bpy.data.materials}\n"
            "with bpy.data.libraries.load(path, link=False) as (src, dst):\n"
            "    dst.objects = list(src.objects)\n"
            + _DEDUP_MATERIALS +
            "names = []\n"
            "for o in dst.objects:\n"
            "    if o is None:\n"
            "        continue\n"
            "    o.use_fake_user = False\n"
            + _indent(_LINK_HOME, 4) +
            "    names.append(o.name)\n"
            "if camera and camera in bpy.data.objects and bpy.data.objects[camera].type == 'CAMERA':\n"
            "    scene.camera = bpy.data.objects[camera]\n"
            "result = {'names': names}\n"
        )
        return list(self.run_code(code, timeout=120.0)["names"])

    def missing_files(self) -> list[str]:
        """External files the open scene needs that do not exist (images, linked libraries)."""
        code = (
            "import bpy, os\n"
            "miss = []\n"
            "for img in bpy.data.images:\n"
            "    if img.packed_file is None and img.source in ('FILE', 'SEQUENCE', 'TILED') and img.filepath \\\n"
            "            and img.users and not os.path.exists(bpy.path.abspath(img.filepath)):\n"
            "        miss.append(bpy.path.abspath(img.filepath))\n"
            "for lib in bpy.data.libraries:\n"
            "    if not os.path.exists(bpy.path.abspath(lib.filepath)):\n"
            "        miss.append(bpy.path.abspath(lib.filepath))\n"
            "result = sorted(set(miss))\n"
        )
        return [str(m) for m in self.run_code(code)]

    def restore_object(
        self, path: str | None, name: str, location: list[float], rotation: list[float],
        scale: list[float], family: list[str] | None = None,
    ) -> dict[str, Any]:
        """Put one object back: from the .blend at ``path`` when given, then set its transform.

        ``family`` names the object's stored descendants (the parts of a generated piece of
        furniture): they come back with it, so restoring ``desk`` restores the whole desk.
        Live objects of those names are removed first so the stored ones keep their exact
        names; live children that are not part of the stored family stay attached. A part
        restored on its own is attached to its live parent (the load copies the parent too;
        that copy is dropped, or brought back when the parent is gone as well). Materials and
        meshes that only gained a ``.NNN`` suffix are folded back into the originals. Nothing
        else in the scene changes. ``rotation`` is Euler degrees.
        Returns ``{"restored": "geometry"|"transform", "objects": [names restored]}``.
        """
        code = (
            "import bpy, math, re\n"
            f"path, name, family = {path!r}, {name!r}, {list(family or [])!r}\n"
            f"loc, rot, scale = {location!r}, {rotation!r}, {scale!r}\n"
            "scene = bpy.context.scene\n"
            "how, restored = 'transform', []\n"
            "def home(o):\n"
            + _indent(_LINK_HOME, 4) +
            "if path:\n"
            "    before_objs = {o.as_pointer() for o in bpy.data.objects}\n"
            "    before = {m.as_pointer() for m in bpy.data.materials}\n"
            "    before_meshes = {m.as_pointer() for m in bpy.data.meshes}\n"
            "    with bpy.data.libraries.load(path, link=False) as (src, dst):\n"
            "        if name not in src.objects:\n"
            "            raise ValueError(f'object {name!r} is not in the stored geometry')\n"
            "        wanted = [name] + [n for n in family if n in src.objects and n != name]\n"
            "        dst.objects = list(wanted)  # the load swaps the list's names for objects in place\n"
            "    loaded = {n: o for n, o in zip(wanted, dst.objects) if o is not None}\n"
            "    mine = {o.as_pointer() for o in loaded.values()}\n"
            "    adopt = []\n"
            "    for n in loaded:\n"
            "        old = bpy.data.objects.get(n)\n"
            "        if old is not None and old.as_pointer() not in mine:\n"
            "            adopt += [(c, n, c.matrix_world.copy()) for c in old.children if c.name not in loaded]\n"
            "            bpy.data.objects.remove(old, do_unlink=True)\n"
            "    for n, o in loaded.items():\n"
            "        o.name = n\n"
            "        o.use_fake_user = False\n"
            "        home(o)\n"
            "        restored.append(n)\n"
            "    for c, n, mw in adopt:\n"
            "        c.parent = loaded[n]\n"
            "        c.matrix_world = mw\n"
            "    for o in [o for o in bpy.data.objects if o.as_pointer() not in before_objs | mine]:\n"
            "        base = re.sub(r'\\.\\d{3,}$', '', o.name)\n"
            "        live = scene.objects.get(base)\n"
            "        if live is None:\n"
            "            o.name = base\n"
            "            o.use_fake_user = False\n"
            "            home(o)\n"
            "            restored.append(base)\n"
            "            continue\n"
            "        o.user_remap(live)  # parent pointers, modifier and constraint targets\n"
            "        bpy.data.objects.remove(o, do_unlink=True)\n"
            + _indent(_DEDUP_MATERIALS, 4) +
            "    for m in [m for m in bpy.data.meshes if m.as_pointer() not in before_meshes]:\n"
            "        base = re.sub(r'\\.\\d{3,}$', '', m.name)\n"
            "        old = bpy.data.meshes.get(base)\n"
            "        if base != m.name and old is not None and old.users == 0:\n"
            "            bpy.data.meshes.remove(old)\n"
            "            m.name = base\n"
            "    how = 'geometry'\n"
            "o = scene.objects.get(name)\n"
            "if o is None:\n"
            "    raise ValueError(f'object {name!r} is not in the scene and no geometry is stored')\n"
            "o.location = loc\n"
            "o.rotation_euler = [math.radians(a) for a in rot]\n"
            "o.scale = scale\n"
            "result = {'restored': how, 'objects': restored or [name]}\n"
        )
        res = self.run_code(code, timeout=60.0)
        return {"restored": str(res["restored"]), "objects": [str(n) for n in res["objects"]]}

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

# Stable UUID custom property, given to every object first: a child read before its parent
# used to store parent_uuid None (the parent had no uuid yet), losing the hierarchy.
import uuid as _u
for o in bpy.context.scene.objects:
    if not o.get("uuid"):
        o["uuid"] = f"blender-{str(_u.uuid4())[:8]}"

for o in bpy.context.scene.objects:
    uid = o["uuid"]

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
        if not isinstance(extracted, dict) or not isinstance(extracted.get("objects"), dict):
            # Never store a snapshot we could not read: an empty scene here used to be
            # saved silently, so every live snapshot lost all its objects.
            raise BlenderError(f"Unexpected response from Blender extractor: {str(extracted)[:200]}")

        # Second resolution alone made two snapshots in the same second share an id
        # (the second overwrote the first).
        import uuid
        scene_id = f"blender-{int(extracted.get('timestamp', time.time()))}-{uuid.uuid4().hex[:6]}"
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
        if not isinstance(res, dict):
            raise BlenderError(f"Unexpected response from Blender library query: {str(res)[:200]}")
        return res


def try_get_blender_client(
    host: str | None = None, port: int | None = None
) -> BlenderClient | None:
    """Return a client if Blender is listening, else None."""
    client = BlenderClient(host=host, port=port)
    return client if client.is_available(timeout=0.3) else None
