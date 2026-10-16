"""Persistence round-trips: versions across Blender restarts, edits, forks, restores.

Found by driving the MCP handlers against a private headless Blender (build -> snapshot ->
restart Blender -> restore -> edit -> restore version -> fork -> diff -> restore one object,
two scenes and projects in one store, unicode names, an older store, missing files). The
unit tests need no Blender; the ``blender`` ones start their own.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from ai_db_scene.memory import SceneMemory, _compare, _descendants
from ai_db_scene.models import ObjectState, SceneState
from ai_db_scene.tools import scene_pipeline as sp
from ai_db_scene.tools import scene_tools

from ai_db.errors import AiDbError

SPEC = json.loads((Path(__file__).parent / "fixtures" / "golden" / "edit_office.json").read_text())["spec"]


def _scene(names: list[str], project: str = "p") -> SceneState:
    scene = SceneState(project=project)
    for i, n in enumerate(names):
        o = ObjectState(uuid=f"u-{n}", name=n, location=(float(i), 0.0, 0.0), materials=["M"])
        scene.objects[o.uuid] = o
    return scene


class _Fake:
    """Minimal live Blender: a SceneState plus the files write_blend 'wrote'."""

    def __init__(self, scene: SceneState):
        self.scene = scene

    def extract_scene_state(self, project: str = "default", name: str = "Blender Scene") -> SceneState:
        import copy
        out = copy.deepcopy(self.scene)
        out.scene_id = SceneState().scene_id
        return out

    def write_blend(self, path: str) -> int:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"blend")
        return len(self.scene.objects)

    def load_blend(self, path: str, replace: bool, camera: str | None) -> list[str]:
        return [o.name for o in self.scene.objects.values()]

    def missing_files(self) -> list[str]:
        return []


def _memory(tmp_path: Path, scene: SceneState, backend: str = "json_file") -> tuple[SceneMemory, _Fake]:
    path = tmp_path / ("m.db" if backend == "sqlite_vec" else "m")
    memory = SceneMemory(backend, {"path": str(path)})
    fake = _Fake(scene)
    memory.set_blender_client(fake)
    return memory, fake


# --- MCP server started before Blender ------------------------------------------------------

def test_scene_memory_attaches_blender_that_starts_later(monkeypatch, tmp_path):
    """The MCP server usually starts first: its memory could then never snapshot or restore."""
    monkeypatch.setattr(scene_tools, "_scene_memory", None)
    monkeypatch.setenv("AI_DB_SCENE_MEMORY", str(tmp_path / "s.db"))
    live: list[Any] = []
    monkeypatch.setattr(scene_tools, "try_get_blender_client", lambda: live[0] if live else None)
    memory = scene_tools.get_scene_memory()
    assert memory._blender is None
    fake = _Fake(_scene(["a"]))
    live.append(fake)
    assert scene_tools.get_scene_memory() is memory and memory._blender is fake
    sid = memory.snapshot_scene(prompt="after Blender came up")
    assert memory.blend_path(sid).exists()


# --- stored rows from another plugin version --------------------------------------------------

def test_from_dict_reads_rows_of_older_and_newer_versions():
    data = _scene(["a"]).to_dict()
    data.pop("version")
    data.pop("tags")
    data["legacy"] = 1
    obj = next(iter(data["objects"].values()))
    obj.pop("hide_render")
    obj["old_key"] = "x"
    data["camera"]["sensor_fit"] = "AUTO"
    data["prompt_context"] = {"prompt": "p", "legacy": True}
    data["materials"] = {"M": {"name": "M", "future": 1}}
    scene = SceneState.from_dict(data)
    assert scene.version == 1 and scene.tags == [] and scene.objects["u-a"].name == "a"
    assert scene.prompt_context is not None and scene.prompt_context.prompt == "p"


def test_sqlite_store_with_an_old_row_and_no_run_table_still_restores(tmp_path):
    memory, _ = _memory(tmp_path, _scene(["a", "b"]), "sqlite_vec")
    sid = memory.snapshot_scene(prompt="v1")
    db_path = tmp_path / "m.db"
    memory.backend.close()
    db = sqlite3.connect(db_path)
    (raw,) = db.execute("SELECT data FROM scenes WHERE scene_id = ?", (sid,)).fetchone()
    data = json.loads(raw)
    data.pop("version")
    for o in data["objects"].values():
        o["old_key"] = 1
    db.execute("UPDATE scenes SET data = ? WHERE scene_id = ?", (json.dumps(data), sid))
    db.execute("DROP TABLE run_records")
    db.commit()
    db.close()
    reopened, _ = _memory(tmp_path, _scene(["a", "b"]), "sqlite_vec")
    report = reopened.apply_scene(sid)
    assert report["mismatches"] == [] and report["restored"] == 2


# --- sqlite: deletes cascade, upserts keep history -------------------------------------------

def test_sqlite_delete_removes_objects_and_history_and_resave_keeps_it(tmp_path):
    memory, _ = _memory(tmp_path, _scene(["a"]), "sqlite_vec")
    keep = memory.snapshot_scene(prompt="kept scene")
    gone = memory.snapshot_scene(prompt="deleted scene")
    backend = memory.backend
    backend.save_scene(backend.load_scene(keep))  # an upsert must not drop its history
    assert "kept scene" in [h.prompt for h in memory.get_history(scene_id=keep)]
    assert memory.delete_scene(gone)
    assert memory.get_history(scene_id=gone) == []
    assert backend.search_objects_by_description("a") and all(
        r["scene_id"] == keep for r in backend.search_objects_by_description("a"))


@pytest.mark.parametrize("backend", ["json_file", "sqlite_vec"])
def test_resaving_a_version_keeps_one_history_row_and_one_listing(tmp_path, backend):
    memory, _ = _memory(tmp_path, _scene(["a"]), backend)
    sid = memory.snapshot_scene(prompt="v1", project="p1")
    for _ in range(2):
        memory.backend.save_scene(memory.backend.load_scene(sid))
    assert [h.prompt for h in memory.get_history(scene_id=sid)] == ["v1"]
    moved = memory.backend.load_scene(sid)
    moved.project = "p2"
    memory.backend.save_scene(moved)
    assert [s["project"] for s in memory.list_scenes() if s["scene_id"] == sid] == ["p2"]
    assert len(memory.get_history(scene_id=sid)) == 1


# --- json store: project names that are not file names ---------------------------------------

@pytest.mark.parametrize("project", ["client/a", "rooms [draft]", "Büro 2", "50% done"])
def test_json_store_handles_any_project_name(tmp_path, project):
    memory, _ = _memory(tmp_path, _scene(["a"], project))
    sid = memory.snapshot_scene(prompt="x", project=project)
    assert [s["scene_id"] for s in memory.list_scenes(project=project)] == [sid]
    assert memory.list_scenes()[0]["project"] == project
    assert [h.scene_id for h in memory.get_history(project=project)] == [sid]


def test_json_history_does_not_leak_between_projects_and_goes_with_the_scene(tmp_path):
    memory, _ = _memory(tmp_path, _scene(["a"]))
    a = memory.snapshot_scene(prompt="in a", project="a")
    memory.snapshot_scene(prompt="in a_b", project="a_b")  # "a_*" used to match it
    assert [h.scene_id for h in memory.get_history(project="a")] == [a]
    memory.delete_scene(a)
    assert memory.get_history(scene_id=a) == []


# --- snapshots get names list_scenes can tell apart -----------------------------------------

def test_snapshots_are_named_after_their_prompt_or_spec(tmp_path):
    memory, fake = _memory(tmp_path, _scene(["a"]))
    fake.scene.name = "Blender Scene"  # what every capture is called
    brief = memory.snapshot_scene(prompt="  a cosy reading corner\nwith a lamp  ")
    spec_only = memory.snapshot_scene(prompt="build_scene", metadata={"spec": {**SPEC, "setting": {
        **SPEC["setting"], "style": "classic"}}})
    named = memory.snapshot_scene(prompt="x", name="Client office")
    version = memory.snapshot_scene(prompt="moved the desk", parent_scene_id=brief)
    long = memory.snapshot_scene(prompt="word " * 40)
    names = {s["scene_id"]: s["name"] for s in memory.list_scenes()}
    assert names[brief] == names[version] == "a cosy reading corner with a lamp"
    assert names[spec_only] == "classic indoor 5x4 m: desk, chair, shelf"
    assert names[named] == "Client office"
    assert len(names[long]) == 60 and names[long].endswith("...")
    assert memory.restore_scene(memory.fork_scene(brief, "variant")).name == "a cosy reading corner with a lamp (fork)"


def test_old_snapshots_named_blender_scene_are_listed_by_prompt(tmp_path):
    memory, _ = _memory(tmp_path, _scene(["a"]), "sqlite_vec")
    sid = memory.snapshot_scene(prompt="office for two")
    old = memory.backend.load_scene(sid)
    old.name = "Blender Scene"
    memory.backend.save_scene(old)
    assert [s["name"] for s in memory.list_scenes()] == ["office for two"]


# --- versions keep their project ---------------------------------------------------------------

def test_version_without_project_stays_in_its_parents_project(tmp_path):
    memory, fake = _memory(tmp_path, _scene(["a"]))
    v1 = memory.snapshot_scene(prompt="v1", project="client a")
    fake.scene = _scene(["a", "b"])
    v2 = memory.snapshot_scene(prompt="v2", parent_scene_id=v1)
    assert memory.restore_scene(v2).project == "client a"
    assert memory.restore_scene(memory.snapshot_scene(prompt="loose")).project == "default"


# --- geometry: only forks borrow their ancestor's ---------------------------------------------

def test_edited_version_with_missing_file_does_not_restore_its_parents_geometry(tmp_path):
    memory, fake = _memory(tmp_path, _scene(["a"]))
    v1 = memory.snapshot_scene(prompt="v1")
    fake.scene = _scene(["a", "b"])
    v2 = memory.snapshot_scene(prompt="v2", parent_scene_id=v1)
    memory.blend_path(v2).unlink()
    with pytest.raises(AiDbError, match=rf"{v2}\.blend is missing"):
        memory.apply_scene(v2)
    fork = memory.fork_scene(v1, "variant")
    assert memory.apply_scene(fork)["geometry_from"] == v1
    memory.blend_path(v1).unlink()
    with pytest.raises(AiDbError, match=rf"{v1}\.blend is missing"):  # names the real file
        memory.apply_scene(memory.fork_scene(fork, "fork of a fork"))


def test_fork_of_a_fork_keeps_one_suffix(tmp_path):
    memory, _ = _memory(tmp_path, _scene(["a"]))
    f1 = memory.fork_scene(memory.snapshot_scene(prompt="v1"), "f1")
    f2 = memory.fork_scene(f1, "f2")
    assert memory.restore_scene(f2).name.count("(fork)") == 1


# --- restore_object brings the parts with it; restore reports materials ----------------------

def test_descendants_and_material_compare():
    scene = _scene(["desk", "top", "leg", "lamp"])
    objs = {o.name: o for o in scene.objects.values()}
    objs["top"].parent_uuid = objs["desk"].uuid
    objs["leg"].parent_uuid = objs["top"].uuid
    assert [o.name for o in _descendants(scene, objs["desk"])] == ["top", "leg"]
    assert _descendants(scene, objs["lamp"]) == []
    other = ObjectState(name="desk", location=objs["desk"].location, materials=["M.001"])
    assert [m["problem"] for m in _compare(objs["desk"], other, 1e-3)] == ["materials"]
    assert _compare(objs["desk"], None, 1e-3) == [{"name": "desk", "problem": "missing"}]


def test_restore_object_passes_the_stored_family(tmp_path):
    memory, fake = _memory(tmp_path, _scene(["desk", "top", "lamp"]))
    objs = {o.name: o for o in fake.scene.objects.values()}
    objs["top"].parent_uuid = objs["desk"].uuid
    sid = memory.snapshot_scene(prompt="v1")
    calls: list[Any] = []
    fake.restore_object = lambda path, name, loc, rot, scale, family=None: calls.append(  # type: ignore[attr-defined]
        (name, family)) or {"restored": "geometry", "objects": [name, *(family or [])]}
    report = memory.restore_object(sid, "desk")
    assert calls == [("desk", ["top"])] and report["objects"] == ["desk", "top"]


# --- edit_scene rebuilds a version as it was built ---------------------------------------------

@pytest.fixture
def built(tmp_path, monkeypatch):
    """A json store with one build_scene version (seed 3, boxes, project 'office é'), no Blender."""
    memory, _fake = _memory(tmp_path, _scene(["desk"]))
    monkeypatch.setattr(scene_tools, "_scene_memory", memory)
    variation = sp.variation_key(sp._parse_spec(SPEC)[0])
    sid = memory.snapshot_scene(prompt="office", project="office é", operation_type="build", metadata={
        "spec": SPEC, "build": sp.build_options({"seed": 3, "geometry": "boxes", "variation": variation})})
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(sp, "_handle_build_scene", lambda args: calls.append(args) or {"status": "built"})
    return memory, sid, calls, variation


def test_edit_scene_rebuilds_with_the_versions_seed_geometry_variation_and_project(built):
    _, sid, calls, variation = built
    out = sp._handle_edit_scene({"scene_id": sid, "steps": [{"op": "remove_object", "id": "shelf"}]})
    assert out["status"] == "built"
    args = calls[0]
    assert (args["seed"], args["geometry"], args["variation"], args["project"]) == (3, "boxes", variation, "office é")
    assert args["parent_scene_id"] == sid
    sp._handle_edit_scene({"scene_id": sid, "steps": [{"op": "remove_object", "id": "shelf"}],
                           "seed": 7, "project": "other"})
    assert (calls[1]["seed"], calls[1]["project"]) == (7, "other")  # the call still wins


def test_edit_scene_on_a_fork_uses_the_ancestors_spec(built):
    memory, sid, calls, _variation = built
    fork = memory.fork_scene(sid, "variant")
    out = sp._handle_edit_scene({"scene_id": fork, "steps": [
        {"op": "add_object", "object": {"id": "lamp", "category": "floor_lamp"}}]})
    assert out["status"] == "built", out
    assert calls[0]["parent_scene_id"] == fork and calls[0]["seed"] == 3
    assert {o["id"] for o in calls[0]["spec"]["objects"]} == {"desk", "chair", "shelf", "lamp"}


def test_furniture_variation_follows_the_first_builds_key():
    spec = sp._parse_spec(SPEC)[0]
    edited = sp._parse_spec({**SPEC, "objects": SPEC["objects"][:2],
                             "relations": [r for r in SPEC["relations"] if "shelf" not in (r["a"], r["b"])]})[0]
    assert spec is not None and edited is not None
    key = sp.variation_key(spec)
    assert sp._variation(edited, "desk", "desk", key) == sp._variation(spec, "desk", "desk")
    assert sp._variation(edited, "desk", "desk") != sp._variation(spec, "desk", "desk")


# --- live Blender ------------------------------------------------------------------------------

def _live_memory(port: int, tmp_path: Path) -> tuple[SceneMemory, Any]:
    from ai_db_scene.blender_client import BlenderClient

    client = BlenderClient(port=port, timeout=120)
    memory = SceneMemory("sqlite_vec", {"path": str(tmp_path / "live.db")})
    memory.set_blender_client(client)
    return memory, client


_TWO_PART_DESK = """
import bpy
for o in list(bpy.context.scene.objects):
    bpy.data.objects.remove(o, do_unlink=True)
mat = bpy.data.materials.new('oak')
def part(name, loc, parent=None):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    if parent is not None:
        o.parent = parent
        o.matrix_parent_inverse = parent.matrix_world.inverted()
    return o
desk = part('desk', (1, 1, 0.75))
part('desk_leg_0', (0.5, 0.5, 0.3), desk)
part('desk_leg_1', (1.5, 0.5, 0.3), desk)
part('Stuhl für Kinder 灯', (3, 0, 0.4))
result = 1
"""

_NAMES = "import bpy\nresult = sorted(o.name for o in bpy.data.objects)"


@pytest.mark.blender
def test_live_restores_keep_material_names_and_hierarchy(private_blender, tmp_path):
    memory, client = _live_memory(private_blender, tmp_path)
    client.run_code(_TWO_PART_DESK)
    v1 = memory.snapshot_scene(prompt="desk", project="Büro ü")
    # restoring twice used to add .001 copies of every material: diff reported them all changed
    for _ in range(2):
        assert memory.apply_scene(v1)["mismatches"] == []
    v2 = memory.snapshot_scene(prompt="after restores", parent_scene_id=v1)
    assert memory.diff_scenes(v1, v2)["unchanged"]
    assert memory.restore_scene(v2).project == "Büro ü"
    mats = "import bpy\nresult = sorted(m.name for m in bpy.data.materials)"
    assert client.run_code(mats) == ["oak"]
    # a part restored alone hangs from the live parent (no hidden 'desk.001' copy)
    client.request("delete_object", {"name": "desk_leg_0"})
    assert memory.restore_object(v1, "desk_leg_0")["mismatches"] == []
    assert client.run_code(_NAMES) == sorted(["desk", "desk_leg_0", "desk_leg_1", "Stuhl für Kinder 灯"])
    assert client.run_code("import bpy\nresult = bpy.data.objects['desk_leg_0'].parent.name") == "desk"
    # the root comes back with its parts attached
    client.request("delete_object", {"name": "desk"})
    report = memory.restore_object(v1, "desk")
    assert report["mismatches"] == [] and sorted(report["objects"]) == ["desk", "desk_leg_0", "desk_leg_1"]
    assert client.run_code("import bpy\nresult = sorted(c.name for c in bpy.data.objects['desk'].children)") == [
        "desk_leg_0", "desk_leg_1"]
    assert client.run_code(_NAMES) == sorted(["desk", "desk_leg_0", "desk_leg_1", "Stuhl für Kinder 灯"])
    assert client.run_code(mats) == ["oak"]
    client.request("delete_object", {"name": "Stuhl für Kinder 灯"})
    assert memory.restore_object(v1, "Stuhl für Kinder 灯")["mismatches"] == []


@pytest.mark.blender
def test_live_snapshot_keeps_textures_whose_files_are_gone(private_blender, tmp_path):
    memory, client = _live_memory(private_blender, tmp_path)
    tex = tmp_path / "tex dir" / "wood ü.png"
    tex.parent.mkdir()
    client.run_code(f"""
import bpy
for o in list(bpy.context.scene.objects):
    bpy.data.objects.remove(o, do_unlink=True)
img = bpy.data.images.new('src', 4, 4)
img.filepath_raw = {str(tex)!r}
img.file_format = 'PNG'
img.save()
bpy.ops.mesh.primitive_cube_add()
o = bpy.context.active_object
m = bpy.data.materials.new('wood')
m.use_nodes = True
node = m.node_tree.nodes.new('ShaderNodeTexImage')
node.image = bpy.data.images.load({str(tex)!r})
o.data.materials.append(m)
result = 1
""")
    sid = memory.snapshot_scene(prompt="textured")
    packed = "import bpy\nresult = [i.packed_file is not None for i in bpy.data.images if i.name == 'wood ü.png']"
    assert client.run_code(packed) == [False]  # the session itself is left as it was
    tex.unlink()
    client.run_code("import bpy\nfor i in list(bpy.data.images):\n    bpy.data.images.remove(i)\nresult = 1")
    report = memory.apply_scene(sid)
    assert report["mismatches"] == [] and report["missing_files"] == []
    assert client.run_code(packed) == [True]
