"""export_scene: glTF and USD written by a separate headless Blender from a stored version (or
the live scene), then imported back and compared (mesh objects, world transforms, materials)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ai_db_scene.export import export_version
from ai_db_scene.memory import SceneMemory
from ai_db_scene.models import ObjectState, SceneState
from ai_db_scene.tools import scene_pipeline as sp
from ai_db_scene.tools import scene_tools

GOLDEN = Path(__file__).parent / "fixtures" / "golden"


def test_export_of_a_version_without_geometry_says_which_file_is_missing(tmp_path, monkeypatch):
    memory = SceneMemory("json_file", {"path": str(tmp_path / "m")})
    scene = SceneState(objects={"u": ObjectState(uuid="u", name="a")})
    memory.backend.save_scene(scene)
    monkeypatch.setattr(scene_tools, "_scene_memory", memory)
    monkeypatch.setenv("AI_DB_SCENE_EXPORT_DIR", str(tmp_path / "exports"))
    out = sp._handle_export_scene({"path": "a.usdc", "scene_id": scene.scene_id})
    assert out["status"] == "error" and f"{scene.scene_id}.blend is missing" in out["message"]
    assert sp.export_path("a", "usda", tmp_path)[0].suffix == ".usda"


def _live(port: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[SceneMemory, Any]:
    from ai_db_scene.blender_client import BlenderClient

    client = BlenderClient(port=port, timeout=120)
    memory = SceneMemory("sqlite_vec", {"path": str(tmp_path / "store" / "s.db")})
    memory.set_blender_client(client)
    monkeypatch.setattr(scene_tools, "_scene_memory", memory)
    monkeypatch.setenv("AI_DB_SCENE_EXPORT_DIR", str(tmp_path / "exports"))
    return memory, client


_EMPTY = "import bpy\nfor o in list(bpy.data.objects):\n    bpy.data.objects.remove(o)\nresult = 1"


@pytest.mark.blender
@pytest.mark.parametrize(("fixture", "formats"), [("indoor_living_room", ["glb", "usdc", "usda"]),
                                                  ("outdoor_park", ["glb", "usdc"])])
def test_live_export_of_a_stored_version_round_trips(fixture, formats, private_blender, tmp_path, monkeypatch):
    """The park's tree leaves (geometry-nodes instances) made the USD export take 17 GB and never
    finish; with instancing they were dropped, as glTF dropped them."""
    memory, client = _live(private_blender, tmp_path, monkeypatch)
    spec = json.loads((GOLDEN / f"{fixture}.json").read_text())["spec"]
    built = sp._handle_build_scene({"spec": spec, "clear": True, "prompt": fixture})
    assert built["status"] == "built", built
    sid = built["scene_id"]
    stored_meshes = sum(1 for o in memory.restore_scene(sid).objects.values() if o.type == "MESH")
    client.run_code(_EMPTY)
    for fmt in formats:
        out = sp._handle_export_scene({"path": f"room.{fmt}", "scene_id": sid})
        assert out["status"] == "success", out
        assert out["meshes"] == out["roundtrip"]["imported_meshes"] == stored_meshes, out
        assert (tmp_path / "exports" / f"room.{fmt}").stat().st_size == out["bytes"] > 0
    assert client.run_code("import bpy\nresult = len(bpy.data.objects)") == 0  # the session was not used


_PARTS = """
import bpy
for o in list(bpy.context.scene.objects):
    bpy.data.objects.remove(o, do_unlink=True)
oak, steel = bpy.data.materials.new('oak'), bpy.data.materials.new('brushed steel')
def part(name, loc, mats, parent=None):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    o = bpy.context.active_object
    o.name = name
    for m in mats:
        o.data.materials.append(m)
    if parent is not None:
        o.parent = parent
        o.matrix_parent_inverse = parent.matrix_world.inverted()
    return o
desk = part('desk', (1, 1, 0.75), [oak, steel])
desk.rotation_euler = (0, 0, 0.5)
part('desk_leg_0', (0.5, 0.5, 0.3), [steel], desk)
part('Stuhl für 灯 2', (3, 0, 0.4), [oak]).scale = (0.5, 0.5, 1.2)
result = 1
"""


@pytest.mark.blender
def test_live_export_of_a_fork_uses_its_own_transforms_and_keeps_names(private_blender, tmp_path, monkeypatch):
    memory, client = _live(private_blender, tmp_path, monkeypatch)
    client.run_code(_PARTS)
    v1 = memory.snapshot_scene(prompt="desk and chair")
    fork = memory.fork_scene(v1, "chair moved")
    moved = memory.backend.load_scene(fork)  # a fork stores transforms only: move one of them
    chair = next(o for o in moved.objects.values() if o.name == "Stuhl für 灯 2")
    chair.location = (4.0, 1.0, 0.4)
    memory.backend.save_scene(moved)
    client.run_code(_EMPTY)
    for fmt in ("glb", "usda"):
        out = export_version(memory, fork, tmp_path / "exports" / f"fork.{fmt}", fmt)
        assert out["geometry_from"] == v1 and out["applied_transforms"] == 1, out
        assert out["roundtrip"]["mismatches"] == [] and out["meshes"] == 3, out
    usda = (tmp_path / "exports" / "fork.usda").read_text()
    assert "xformOp:translate = (4, 1, 0.4" in usda  # the fork's chair, not v1's at (3, 0)
    assert 'def Mesh "desk_leg_0"' in usda and '"Stuhl_für_灯_2"' in usda and "brushed_steel" in usda
    # the live scene: written to a temporary file, exported the same way
    client.run_code(_PARTS)
    live = sp._handle_export_scene({"path": "live.usdc"})
    assert live["status"] == "success" and live["meshes"] == 3 and live["roundtrip"]["mismatch_count"] == 0, live
