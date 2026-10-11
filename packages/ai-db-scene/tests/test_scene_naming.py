"""Rebuilt objects keep their exact names (no ".001") even when leftovers hold those names.

A small fake of Blender's naming rules runs the pipeline's own run_code snippets (reserve,
purge, finalize) against a fake ``bpy``: a name taken anywhere in ``bpy.data.objects`` makes
Blender suffix the new object, and commands addressing an object by name hit whichever object
holds that name.
"""

from __future__ import annotations

import json
import re
import sys
import types
from typing import Any

import pytest
from ai_db_scene.tools import scene_pipeline as sp

SPEC = {
    "setting": {"kind": "indoor", "bounds_m": [6, 5, 2.8]},
    "objects": [{"id": "sofa", "category": "sofa"}, {"id": "table", "category": "coffee_table"},
                {"id": "lamp", "category": "floor_lamp"}, {"id": "box", "category": "crate"}],
    "relations": [{"a": "sofa", "rel": "against_wall", "b": "wall_north"}],
}
_SUFFIX = re.compile(r"[.][0-9]{3}$")


class FakeObj:
    def __init__(self, world: FakeWorld, name: str):
        self._world, self._name = world, ""
        self.props: dict[str, Any] = {}
        self.location: list[float] = [0.0, 0.0, 0.0]
        self.rotation_euler = [0.0, 0.0, 0.0]
        self.use_fake_user = False
        self.material: str | None = None
        self.name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:  # Blender: a taken name gets the next free ".NNN"
        self._name = self._world.free_name(value, self)

    @property
    def users(self) -> int:
        return 1 if self in self._world.scene or self in self._world.other_scene else 0

    def get(self, key: str, default: Any = None) -> Any:
        return self.props.get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        self.props[key] = value


class FakeWorld:
    def __init__(self) -> None:
        self.objects: list[FakeObj] = []
        self.scene: list[FakeObj] = []
        self.other_scene: list[FakeObj] = []

    def free_name(self, want: str, me: FakeObj | None = None) -> str:
        taken = {o.name for o in self.objects if o is not me}
        if want not in taken:
            return want
        base = _SUFFIX.sub("", want)
        n = 1
        while f"{base}.{n:03d}" in taken:
            n += 1
        return f"{base}.{n:03d}"

    def create(self, name: str) -> FakeObj:
        o = FakeObj(self, name)
        self.objects.append(o)
        self.scene.append(o)
        return o

    def by_name(self, name: str) -> FakeObj | None:
        return next((o for o in self.objects if o.name == name), None)

    def bpy(self) -> types.ModuleType:
        world = self

        class Coll:
            def __init__(self, items: Any):
                self._items = items

            def get(self, name: str) -> FakeObj | None:
                return next((o for o in self._items() if o.name == name), None)

            def __iter__(self) -> Any:
                return iter(list(self._items()))

            def remove(self, obj: FakeObj) -> None:
                for lst in (world.objects, world.scene, world.other_scene):
                    if obj in lst:
                        lst.remove(obj)

        empty = Coll(list)
        data = types.SimpleNamespace(objects=Coll(lambda: world.objects), meshes=empty, curves=empty,
                                     materials=empty, lights=empty, cameras=empty, collections=empty,
                                     node_groups=empty, images=empty)
        scene = types.SimpleNamespace(objects=Coll(lambda: world.scene))
        mod = types.ModuleType("bpy")
        mod.data = data  # type: ignore[attr-defined]
        mod.context = types.SimpleNamespace(scene=scene)  # type: ignore[attr-defined]
        return mod


class FakeClient:
    """BlenderClient stand-in that applies the commands to a FakeWorld."""

    world: FakeWorld

    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    def _run(self, code: str) -> Any:
        saved = sys.modules.get("bpy")
        sys.modules["bpy"] = self.world.bpy()
        try:
            env: dict[str, Any] = {}
            exec(code, env)  # noqa: S102 - the pipeline's own snippets, as Blender runs them
            return env.get("result")
        finally:
            if saved is None:
                sys.modules.pop("bpy", None)
            else:
                sys.modules["bpy"] = saved

    def request(self, handler: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        params = params or {}
        if handler == "clear_scene":  # like the addon: only this scene's objects go
            for o in list(self.world.scene):
                self.world.objects.remove(o)
            self.world.scene.clear()
            return {}
        if handler == "run_code":
            return {"result": self._run(params["code"])}
        return {}

    def _cmd(self, c: dict[str, Any]) -> None:
        t, p = c["type"], c["params"]
        w = self.world
        if t == "room_shell":
            w.create("Room_Floor")
            w.create("Room_Walls")
        elif t in ("create_primitive", "generate_asset", "add_light"):
            w.create(p["name"]).location = list(p.get("location") or [0, 0, 0])
        elif t == "build_asset":
            w.create(p["spec"]["asset"])
        elif t == "run_code" and "_stale" in p["code"]:  # reserve + finalize; not materials/lights
            self._run(p["code"])
        elif t in ("modify_object", "apply_material"):
            o = w.by_name(p["name"])
            if o is not None and "location" in p:
                o.location = list(p["location"])
            if o is not None and "material" in p:
                o.material = p["material"]

    def batch(self, cmds: list[dict[str, Any]], timeout: float | None = None) -> list[dict[str, Any]]:
        for c in cmds:
            self._cmd(c)
        return [{"ok": 1, "status": "success"} for _ in cmds]


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> FakeWorld:
    w = FakeWorld()
    monkeypatch.setattr(FakeClient, "world", w, raising=False)
    monkeypatch.setattr(sp, "BlenderClient", FakeClient)
    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})
    return w


def _build() -> dict[str, Any]:
    return sp._handle_build_scene({"spec": SPEC, "clear": True, "snapshot": False})


def test_rebuild_keeps_exact_names_despite_leftovers(world: FakeWorld) -> None:
    assert _build()["status"] == "built"
    first = sorted(o.name for o in world.scene)
    FakeClient().request("clear_scene")
    # leftovers clear_scene does not remove: an orphan (no users) and an object of another scene
    orphan = FakeObj(world, "sofa")
    world.objects.append(orphan)
    elsewhere = world.create("box")
    world.scene.remove(elsewhere)
    world.other_scene.append(elsewhere)
    elsewhere.location = [9.0, 9.0, 9.0]

    assert _build()["status"] == "built"  # what edit_scene does: clear + rebuild
    names = sorted(o.name for o in world.scene)
    assert names == first
    assert not [n for n in names if _SUFFIX.search(n)]
    for oid in ("sofa", "table", "lamp", "box"):
        assert world.scene and next(o for o in world.scene if o.name == oid).get("ai_db_id") == oid
    assert orphan not in world.objects  # purged: nothing used it
    # the other scene's object kept its data, under another name; commands never touched it
    assert elsewhere.name == "box_stale" and elsewhere.location == [9.0, 9.0, 9.0]
    # the recipe's modify_object moved the new crate, not the other scene's "box"
    assert next(o for o in world.scene if o.name == "box").location != [0.0, 0.0, 0.0]


def test_finalize_renames_a_suffixed_new_object(world: FakeWorld) -> None:
    """Even if a name could not be reserved, the tags land on the new object under the exact name."""
    held = world.create("lamp")
    world.scene.remove(held)
    world.other_scene.append(held)
    new = world.create("lamp")
    assert new.name == "lamp.001"
    payload = [{"name": "lamp", "location": [0, 0, 0], "rot_z": 0.0, "props": {"ai_db_id": "lamp"}}]
    FakeClient().request("run_code", {"code": sp._FINALIZE.format(payload=json.dumps(payload))})
    assert new.name == "lamp" and new.get("ai_db_id") == "lamp"
    assert held.name == "lamp_stale" and held.get("ai_db_id") is None


@pytest.mark.blender
def test_live_edit_rebuild_keeps_exact_names(private_blender: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_db_scene.blender_client import BlenderClient

    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})
    c = BlenderClient(port=private_blender, timeout=120)
    names = "import bpy\nresult = sorted(o.name for o in bpy.context.scene.objects)"
    assert _build()["status"] == "built"
    first = c.run_code(names)
    # leave an orphan object holding a rebuilt name, as interrupted edits did
    c.run_code("import bpy\nbpy.data.objects.new('sofa', None)\nresult = 1")
    assert _build()["status"] == "built"
    again = c.run_code(names)
    assert again == first
    assert {"sofa", "table", "lamp", "box"} <= set(again)
    assert not [n for n in again if _SUFFIX.search(n) and _SUFFIX.sub("", n) in {"sofa", "table", "lamp", "box"}]
