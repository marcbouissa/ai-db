"""3D scene memory: models, pluggable backends, factory, and the MCP tool surface.

Covers the `ai_db/scene` subsystem added for 3D persistence:

  - ``ai_db.scene.models``     domain DTOs and their round-trip
  - ``ai_db.scene.storage``    backend contract: JSON file + SQLite-vec
  - ``ai_db.scene.factory``    backend selection and registration
  - ``ai_db.memory.scene``     high-level API (snapshot/fork/version)
  - ``ai_db.mcp_tools``        the tools these are exposed as over MCP

Storage backends live under a ``storage`` package, which is what
``tests/test_storage.py::test_ast_no_raw_sql_keywords_in_non_storage`` keys on;
this module therefore exercises them through the abstract contract only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_db.scene.factory import SceneMemoryFactory
from ai_db.scene.models import (
    CollectionState,
    MaterialState,
    ObjectState,
    PromptContext,
    SceneState,
)
from ai_db.scene.storage.base import SceneMemoryBackend


# ==============================================================================
# Fixtures / helpers
# ==============================================================================

def _sample_scene(
    scene_id: str = "s1",
    project: str = "demo",
    prompt: str = "",
    timestamp: float = 1000.0,
) -> SceneState:
    """A small but structurally complete scene."""
    cube = ObjectState(
        uuid=f"{scene_id}-cube",
        name="Cube",
        type="MESH",
        collection="Props",
        location=(1.0, 2.0, 3.0),
        vertices=8,
        faces=6,
        materials=["Red"],
        custom_props={"role": "hero"},
    )
    scene = SceneState(
        scene_id=scene_id,
        project=project,
        name=f"Scene {scene_id}",
        timestamp=timestamp,
        objects={cube.uuid: cube},
        materials={"Red": MaterialState(name="Red")},
        collections={"Props": CollectionState(name="Props", objects=[cube.uuid])},
        tags=["test"],
    )
    if prompt:
        scene.prompt_context = PromptContext(prompt=prompt, project=project, scene_id=scene_id)
    return scene


@pytest.fixture
def json_backend(tmp_path: Path) -> SceneMemoryBackend:
    """The zero-dependency backend, so these tests need no sqlite-vec."""
    backend = SceneMemoryFactory.create("json_file", {"path": str(tmp_path / "scenes")})
    yield backend
    backend.close()


@pytest.fixture
def sqlite_backend(tmp_path: Path) -> SceneMemoryBackend:
    """The SQLite-vec backend, skipped if the extension is unavailable."""
    pytest.importorskip("sqlite_vec", reason="sqlite-vec not installed")
    try:
        backend = SceneMemoryFactory.create("sqlite_vec", {"path": str(tmp_path / "scene.db")})
    except Exception as exc:  # noqa: BLE001 - any init failure means unusable here
        pytest.skip(f"sqlite-vec backend unavailable: {exc}")
    yield backend
    backend.close()


# ==============================================================================
# Models
# ==============================================================================

class TestSceneModels:
    def test_object_state_round_trips(self):
        obj = ObjectState(name="Cube", materials=["Red"], custom_props={"k": 1})
        restored = ObjectState.from_dict(obj.to_dict())
        assert restored == obj

    def test_scene_state_round_trips(self):
        scene = _sample_scene(prompt="add a cube")
        restored = SceneState.from_dict(scene.to_dict())

        assert restored.scene_id == scene.scene_id
        assert restored.project == scene.project
        assert restored.tags == ["test"]
        assert set(restored.objects) == set(scene.objects)
        assert restored.camera == scene.camera
        assert restored.prompt_context.prompt == "add a cube"

    def test_scene_state_round_trips_object_details(self):
        scene = _sample_scene()
        restored = SceneState.from_dict(scene.to_dict())
        original = next(iter(scene.objects.values()))
        back = restored.objects[original.uuid]
        assert back.location == original.location
        assert back.materials == original.materials
        assert back.vertices == original.vertices

    def test_from_dict_tolerates_missing_optional_keys(self):
        scene = SceneState.from_dict({
            "scene_id": "s", "project": "p", "name": "n",
            "timestamp": 1.0, "version": 1,
        })
        assert scene.objects == {}
        assert scene.tags == []
        assert scene.prompt_context is None

    def test_to_dict_is_json_serializable(self):
        json.dumps(_sample_scene(prompt="x").to_dict())


# ==============================================================================
# Backend contract, run against every available backend
# ==============================================================================

@pytest.fixture(params=["json_file", "sqlite_vec"])
def any_backend(request, tmp_path: Path) -> SceneMemoryBackend:
    """Exercise the abstract contract so both backends must behave the same."""
    name = request.param
    if name == "sqlite_vec":
        pytest.importorskip("sqlite_vec", reason="sqlite-vec not installed")
    path = tmp_path / name
    suffix = "" if name == "json_file" else ".db"
    try:
        backend = SceneMemoryFactory.create(name, {"path": f"{path}{suffix}"})
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{name} backend unavailable: {exc}")
    yield backend
    backend.close()


class TestBackendContract:
    def test_initialize_creates_usable_backend(self, any_backend):
        assert any_backend.backend_name in ("json_file", "sqlite_vec")
        assert any_backend.capabilities

    def test_save_then_load_round_trips(self, any_backend):
        scene = _sample_scene()
        scene_id = any_backend.save_scene(scene)

        assert scene_id == scene.scene_id
        loaded = any_backend.load_scene(scene_id)
        assert loaded is not None
        assert loaded.name == scene.name
        assert len(loaded.objects) == 1

    def test_load_unknown_scene_returns_none(self, any_backend):
        assert any_backend.load_scene("does-not-exist") is None

    def test_list_scenes_is_newest_first(self, any_backend):
        any_backend.save_scene(_sample_scene("old", timestamp=1000.0))
        any_backend.save_scene(_sample_scene("new", timestamp=5000.0))

        listed = any_backend.list_scenes()
        ids = [s["scene_id"] for s in listed]
        assert "old" in ids and "new" in ids
        assert ids.index("new") < ids.index("old")

    def test_list_scenes_filters_by_project(self, any_backend):
        any_backend.save_scene(_sample_scene("a", project="alpha"))
        any_backend.save_scene(_sample_scene("b", project="beta"))

        alpha = [s["scene_id"] for s in any_backend.list_scenes(project="alpha")]
        assert alpha == ["a"]

    def test_list_scenes_respects_limit(self, any_backend):
        for i in range(5):
            any_backend.save_scene(_sample_scene(f"s{i}"))
        assert len(any_backend.list_scenes(limit=2)) == 2

    def test_delete_scene_removes_it(self, any_backend):
        any_backend.save_scene(_sample_scene("gone"))
        assert any_backend.delete_scene("gone") is True
        assert any_backend.load_scene("gone") is None
        assert [s["scene_id"] for s in any_backend.list_scenes()] == []

    def test_delete_missing_scene_reports_false(self, any_backend):
        assert any_backend.delete_scene("never-existed") is False

    def test_prompt_context_is_saved_with_the_scene(self, any_backend):
        scene = _sample_scene("withctx", prompt="make it blue")
        any_backend.save_scene(scene)

        history = any_backend.get_scene_history(scene_id="withctx")
        assert len(history) == 1
        assert history[0].prompt == "make it blue"
        assert history[0].scene_id == "withctx"

    def test_history_filters_by_object(self, any_backend):
        scene = _sample_scene("obj-scene", prompt="touched the cube")
        scene.prompt_context.objects_affected = ["s-obj-scene-cube"]
        any_backend.save_scene(scene)

        hit = any_backend.get_scene_history(object_uuid="s-obj-scene-cube")
        miss = any_backend.get_scene_history(object_uuid="unrelated-uuid")
        assert len(hit) == 1
        assert miss == []

    def test_search_scenes_by_prompt(self, any_backend):
        any_backend.save_scene(_sample_scene("hit", prompt="add a red cube to the scene"))
        any_backend.save_scene(_sample_scene("miss", prompt="unrelated plumbing work"))

        hits = any_backend.search_scenes_by_prompt("red cube")
        assert [s.scene_id for s in hits] == ["hit"]

    def test_search_objects_by_description(self, any_backend):
        any_backend.save_scene(_sample_scene("obs"))
        hits = any_backend.search_objects_by_description("cube")
        assert len(hits) == 1
        assert hits[0]["scene_id"] == "obs"
        assert hits[0]["object"]["name"] == "Cube"


# ==============================================================================
# JSON backend specifics
# ==============================================================================

class TestJSONFileBackend:
    def test_creates_scenes_and_history_dirs(self, json_backend, tmp_path):
        assert (tmp_path / "scenes" / "scenes").is_dir()
        assert (tmp_path / "scenes" / "history").is_dir()

    def test_scene_file_is_human_readable(self, json_backend):
        json_backend.save_scene(_sample_scene("readable"))
        raw = (json_backend.scenes_dir / "readable.json").read_text()
        assert json.loads(raw)["scene_id"] == "readable"

    def test_index_is_scoped_per_project(self, json_backend):
        json_backend.save_scene(_sample_scene("p1", project="alpha"))
        json_backend.save_scene(_sample_scene("p2", project="beta"))
        alpha = json.loads((json_backend.scenes_dir / ".index_alpha.json").read_text())
        assert set(alpha) == {"p1"}


# ==============================================================================
# Factory
# ==============================================================================

class TestSceneMemoryFactory:
    def test_create_known_backend(self, tmp_path):
        backend = SceneMemoryFactory.create("json_file", {"path": str(tmp_path / "s")})
        assert isinstance(backend, SceneMemoryBackend)
        backend.close()

    def test_unknown_backend_lists_alternatives(self, tmp_path):
        with pytest.raises(ValueError, match="json_file"):
            SceneMemoryFactory.create("nope", {"path": str(tmp_path / "s")})

    def test_register_backend_rejects_wrong_type(self):
        with pytest.raises(TypeError, match="SceneMemoryBackend"):
            SceneMemoryFactory.register_backend("bogus", object)  # type: ignore[arg-type]

    def test_register_backend_then_create(self, tmp_path):
        class Custom(SceneMemoryFactory._backends["json_file"]):
            backend_name = "custom_json"

        SceneMemoryFactory.register_backend("custom_json", Custom)
        try:
            assert "custom_json" in SceneMemoryFactory.list_backends()
            # Capabilities are read off the class, so a subclass inherits them.
            assert SceneMemoryFactory.get_backend_capabilities("custom_json") == Custom.capabilities
            backend = SceneMemoryFactory.create("custom_json", {"path": str(tmp_path / "c")})
            assert backend.backend_name == "custom_json"
            backend.close()
        finally:
            SceneMemoryFactory._backends.pop("custom_json", None)

    def test_capabilities_are_reported_without_instantiating(self, tmp_path):
        assert SceneMemoryFactory.get_backend_capabilities("json_file") == {
            "scene_storage", "prompt_history", "text_search",
        }


# ==============================================================================
# High-level SceneMemory API
# ==============================================================================

@pytest.fixture
def memory(tmp_path: Path) -> "SceneMemory":  # noqa: F821
    from ai_db.memory.scene import SceneMemory

    return SceneMemory(backend="json_file", config={"path": str(tmp_path / "mem")})


class TestSceneMemory:
    def test_snapshot_requires_a_blender_capture(self, memory):
        with pytest.raises(RuntimeError, match="capture"):
            memory.snapshot_scene(prompt="do something")

    def test_snapshot_then_restore(self, memory):
        memory.set_blender_capture(lambda: _sample_scene("live"))
        scene_id = memory.snapshot_scene(prompt="added a cube", project="demo")

        restored = memory.restore_scene(scene_id)
        assert restored.scene_id == "live"
        assert restored.project == "demo"
        assert restored.prompt_context.prompt == "added a cube"

    def test_restore_unknown_scene_raises(self, memory):
        with pytest.raises(ValueError, match="not found"):
            memory.restore_scene("nope")

    def test_fork_links_to_parent_and_bumps_version(self, memory):
        memory.set_blender_capture(lambda: _sample_scene("root"))
        root_id = memory.snapshot_scene(prompt="root scene")

        fork_id = memory.fork_scene(root_id, new_prompt="try a variant")
        fork = memory.restore_scene(fork_id)

        assert fork.parent_scene_id == root_id
        assert fork.version == 2
        assert fork.prompt_context.operation_type == "fork"
        assert "fork" in fork.tags

    def test_fork_unknown_parent_raises(self, memory):
        with pytest.raises(ValueError, match="Parent scene not found"):
            memory.fork_scene("nope", new_prompt="x")

    def test_get_scene_versions_walks_the_parent_chain(self, memory):
        memory.set_blender_capture(lambda: _sample_scene("v1"))
        v1 = memory.snapshot_scene(prompt="one")
        v2 = memory.fork_scene(v1, new_prompt="two")
        v3 = memory.fork_scene(v2, new_prompt="three")

        versions = memory.get_scene_versions(v3)
        assert [v.scene_id for v in versions] == [v3, v2, v1]

    def test_search_helpers_delegate_to_backend(self, memory):
        memory.set_blender_capture(lambda: _sample_scene("s"))
        memory.snapshot_scene(prompt="paint the cube red")

        assert [s.scene_id for s in memory.search_by_prompt("paint")] == ["s"]
        assert [h["scene_id"] if isinstance(h, dict) else h.scene_id
                for h in memory.get_history()] == ["s"]
        assert len(memory.search_objects("cube")) == 1


# ==============================================================================
# MCP tool surface
# ==============================================================================

SCENE_TOOLS = {
    "snapshot_scene", "restore_scene", "list_scenes", "get_scene_history",
    "search_scenes_by_prompt", "search_objects", "fork_scene", "get_scene_versions",
}


class TestSceneMCPTools:
    def test_registration_exposes_every_scene_tool(self):
        from ai_db.dispatcher import ServiceDispatcher
        from ai_db.mcp_tools.scene_tools import register_scene_tools

        dispatcher = ServiceDispatcher(db_path=":memory:")
        register_scene_tools(dispatcher)
        assert SCENE_TOOLS <= set(dispatcher._tools)
        assert all(dispatcher._tools[n].category == "scene" for n in SCENE_TOOLS)

    def test_tools_have_mcp_schemas(self):
        from ai_db.dispatcher import ServiceDispatcher
        from ai_db.mcp_tools.scene_tools import register_scene_tools

        dispatcher = ServiceDispatcher(db_path=":memory:")
        register_scene_tools(dispatcher)
        for name in SCENE_TOOLS:
            schema = dispatcher._tools[name].to_mcp_dict()["inputSchema"]
            assert schema["type"] == "object"

    def test_snapshot_and_list_round_trip_through_a_dispatcher(
        self, tmp_path, monkeypatch
    ):
        from ai_db.dispatcher import ServiceDispatcher
        from ai_db.mcp_tools import scene_tools

        memory = scene_tools.SceneMemory(backend="json_file", config={"path": str(tmp_path / "t")})
        memory.set_blender_capture(lambda: _sample_scene("tooled"))
        monkeypatch.setattr(scene_tools, "_scene_memory", memory)

        dispatcher = ServiceDispatcher(db_path=":memory:")
        scene_tools.register_scene_tools(dispatcher)

        saved = dispatcher.execute(
            "snapshot_scene", {"prompt": "via MCP", "project": "demo"}
        )
        assert saved["status"] == "saved"

        listed = dispatcher.execute("list_scenes", {"project": "demo"})
        assert listed["count"] == 1
        assert listed["scenes"][0]["scene_id"] == saved["scene_id"]

        loaded = dispatcher.execute("restore_scene", {"scene_id": saved["scene_id"]})
        assert loaded["status"] == "loaded"
        assert loaded["object_count"] == 1
