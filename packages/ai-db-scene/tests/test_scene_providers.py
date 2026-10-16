"""Asset providers: local library index, Poly Haven client (HTTP mocked), config, build commands."""

from __future__ import annotations

import io
import json
import types
from pathlib import Path
from typing import Any

import pytest
from ai_db_scene import providers as pv
from ai_db_scene.config import scene_config
from ai_db_scene.tools import scene_pipeline as sp

from ai_db.errors import AiDbConfigError

SPEC = {
    "setting": {"kind": "outdoor", "bounds_m": [12, 10, 4]},
    "objects": [{"id": "cone", "category": "traffic_cone", "count": 2}, {"id": "bench", "category": "bench"},
                {"id": "bin", "category": "trash_can"}],
    "relations": [],
}


def _app(scene: dict[str, Any]) -> Any:
    return types.SimpleNamespace(plugins={"scene": scene})


def _library(tmp_path: Path) -> Path:
    lib = tmp_path / "assets"
    (lib / "street").mkdir(parents=True)
    (lib / "street" / "Traffic-Cone 01.glb").write_bytes(b"glb")
    (lib / "street" / "traffic_cone_with_reflective_band.gltf").write_text("{}")
    (lib / "park_bench.blend").write_bytes(b"blend")
    (lib / "notes.txt").write_text("not a model")
    (lib / "spaceship.glb").write_bytes(b"glb")  # no category
    return lib


def test_match_category_prefers_the_most_specific():
    cats = ["lamp_post_double", "street_lamp", "bench", "cabinet", "sink", "kitchen_counter"]
    assert pv.match_category("lamp-post double 02", cats) == "lamp_post_double"
    assert pv.match_category("Street_Lamp", cats) == "street_lamp"
    assert pv.match_category("kitchen_counter_oak", cats) == "kitchen_counter"
    assert pv.match_category("sink_cabinet", cats) == "cabinet"  # one word each: ties go by name
    assert pv.match_category("spaceship", cats) is None


def test_local_library_indexes_model_files_by_category(tmp_path):
    lib = pv.LocalLibrary([str(_library(tmp_path))])
    idx = lib.index()
    assert set(idx) == {"traffic_cone", "bench"}
    cone = lib.find("traffic_cone")
    assert cone is not None and cone.kind == "gltf" and cone.filepath.endswith("Traffic-Cone 01.glb")
    bench = lib.find("bench")
    assert bench is not None and bench.kind == "blend" and bench.asset_name == "park_bench"
    assert lib.find("sofa") is None


def test_build_uses_provider_assets_before_recipes(tmp_path):
    spec, _ = sp._parse_spec(SPEC)
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": SPEC})["placements"])
    lib = pv.LocalLibrary([str(_library(tmp_path))])
    cmds, names = sp.build_layout_commands(spec, pls, providers=[lib])
    imports = [c for c in cmds if c["type"] == "import_file"]
    assert [c["params"]["collection"] for c in imports] == ["ai_db_asset_cone_1", "ai_db_asset_cone_2"]
    adopt = [c["params"]["code"] for c in cmds if c["type"] == "run_code" and "ai_db_asset_" in c["params"]["code"]]
    assert len(adopt) == 2
    for code in adopt:
        compile(code, "<adopt>", "exec")
    appended = [c["params"] for c in cmds if c["type"] == "append_asset"]
    assert appended == [{"filepath": str(tmp_path / "assets" / "park_bench.blend"), "asset_name": "park_bench",
                         "location": appended[0]["location"]}]
    # the trash can has no file: its recipe is built as before
    assert [c["params"]["spec"]["category"] for c in cmds if c["type"] == "build_asset"] == ["trash_can"]
    fin = next(c["params"]["code"] for c in cmds if c["type"] == "run_code" and "ai_db_id" in c["params"]["code"])
    assert '"asset_source": "local"' in fin and set(names) >= {"cone_1", "cone_2", "bench", "bin"}
    # the appended object's name is reserved like the ids
    assert '"park_bench"' in cmds[0]["params"]["code"]
    # an explicit asset_map still wins; boxes mode uses no provider
    cmds2, _ = sp.build_layout_commands(spec, pls, {"bench": {"filepath": "/x.blend", "asset_name": "B"}},
                                        providers=[lib])
    assert [c["params"]["asset_name"] for c in cmds2 if c["type"] == "append_asset"] == ["B"]
    boxes, _ = sp.build_layout_commands(spec, pls, geometry="boxes", providers=[lib])
    assert not any(c["type"] in ("import_file", "append_asset") for c in boxes)


def test_without_providers_nothing_changes():
    spec, _ = sp._parse_spec(SPEC)
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": SPEC})["placements"])
    assert sp.build_layout_commands(spec, pls) == sp.build_layout_commands(spec, pls, providers=[])


def test_a_failing_provider_is_skipped():
    class Broken:
        name = "broken"

        def find(self, category):
            raise OSError("network down")

    class One:
        name = "one"

        def find(self, category):
            return pv.AssetFile(category, "/a.glb", "gltf")

    assert pv.find_asset([Broken(), One()], "bench").filepath == "/a.glb"
    assert pv.find_asset([Broken()], "bench") is None


# --- config --------------------------------------------------------------------------------


def test_config_defaults_are_off():
    cfg = scene_config(_app({}))
    assert cfg.asset_dirs == () and not cfg.polyhaven.enabled
    assert pv.providers_from_config(_app({})) == []


def test_config_validation_errors_are_clear(tmp_path):
    with pytest.raises(AiDbConfigError, match="license_acknowledged"):
        scene_config(_app({"polyhaven": {"enabled": True}}))
    with pytest.raises(AiDbConfigError, match="unknown keys in 'plugins.scene.polyhaven'"):
        scene_config(_app({"polyhaven": {"enabled": False, "api_key": "x"}}))
    with pytest.raises(AiDbConfigError, match="must be true or false"):
        scene_config(_app({"polyhaven": {"enabled": "yes"}}))
    with pytest.raises(AiDbConfigError, match="resolution"):
        scene_config(_app({"polyhaven": {"resolution": "8k"}}))
    with pytest.raises(AiDbConfigError, match="is not a folder"):
        scene_config(_app({"asset_dirs": [str(tmp_path / "missing")]}))
    with pytest.raises(AiDbConfigError, match="list of folder paths"):
        scene_config(_app({"asset_dirs": "assets"}))
    with pytest.raises(AiDbConfigError, match="unknown keys in 'plugins.scene'"):
        scene_config(_app({"assets": []}))


def test_providers_from_config_order_and_gate(tmp_path):
    lib = _library(tmp_path)
    on = {"asset_dirs": [str(lib)], "polyhaven": {"enabled": True, "license_acknowledged": True,
                                                  "cache_dir": str(tmp_path / "ph")}}
    got = pv.providers_from_config(_app(on))
    assert [p.name for p in got] == ["local", "polyhaven"]
    assert pv.providers_from_config(_app({"polyhaven": {"enabled": False, "license_acknowledged": True}})) == []


def test_apply_layout_reports_a_bad_scene_config(tmp_path, monkeypatch):
    from ai_db.config_template import build_template

    cfg = build_template()
    cfg["plugins"] = {"scene": {"polyhaven": {"enabled": True}}}
    path = tmp_path / "bad_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("AI_DB_CONFIG", str(path))

    class NoBlender:
        def __init__(self, *a, **k):
            raise AssertionError("Blender must not be called with a bad config")

    monkeypatch.setattr(sp, "BlenderClient", NoBlender)
    placements = sp._handle_solve_layout({"spec": SPEC})["placements"]
    out = sp._handle_apply_layout({"spec": SPEC, "placements": placements})
    assert out["status"] == "error" and "license_acknowledged" in out["message"]


# --- Poly Haven (no network: every request goes to a fake) -----------------------------------

MODELS = {
    "traffic_cone_01": {"name": "Traffic Cone", "tags": ["street", "road"], "categories": ["props"]},
    "park_bench_02": {"name": "Wooden Bench", "tags": ["park"], "categories": ["furniture"]},
    "cone_marker_large_rusty": {"name": "Cone Marker", "tags": ["traffic"], "categories": ["props"]},
}


def _files(mid: str, includes: dict[str, str] | None = None) -> dict[str, Any]:
    inc = {rel: {"url": f"https://dl.example/{mid}/{rel}", "size": 3} for rel in (includes or {})}
    return {"gltf": {"1k": {"gltf": {"url": f"https://dl.example/{mid}/{mid}_1k.gltf", "include": inc}}}}


class FakeHttp:
    def __init__(self, files: dict[str, Any] | None = None):
        self.calls: list[str] = []
        self.files = files or {}

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if url.endswith("/assets?t=models"):
            return json.dumps(MODELS).encode()
        if "/files/" in url:
            return json.dumps(self.files[url.rsplit("/", 1)[1]]).encode()
        return b"data:" + url.encode()


def test_polyhaven_picks_downloads_and_caches(tmp_path):
    http = FakeHttp({"traffic_cone_01": _files("traffic_cone_01", {"textures/cone_diff_1k.jpg": "",
                                                                   "traffic_cone_01.bin": ""})})
    ph = pv.PolyHaven(str(tmp_path / "ph"), fetch=http)
    assert ph.best_model("traffic_cone") == "traffic_cone_01"  # name match beats a tag match
    assert ph.best_model("bench") == "park_bench_02"
    assert ph.best_model("sofa") is None
    found = ph.find("traffic_cone")
    assert found is not None and found.kind == "gltf" and found.source == "polyhaven:traffic_cone_01"
    root = tmp_path / "ph" / "traffic_cone_01" / "1k"
    assert Path(found.filepath) == root / "traffic_cone_01_1k.gltf" and Path(found.filepath).is_file()
    assert (root / "textures" / "cone_diff_1k.jpg").read_bytes().startswith(b"data:")
    n = len(http.calls)
    again = pv.PolyHaven(str(tmp_path / "ph"), fetch=http).find("traffic_cone")
    assert again == found
    # the model list comes from the disk cache; only the file listing is asked again
    assert [c for c in http.calls[n:] if c.endswith("models")] == [] and len(http.calls) - n == 1


def test_polyhaven_refuses_include_paths_outside_its_folder(tmp_path):
    http = FakeHttp({"traffic_cone_01": _files("traffic_cone_01", {"../../escape.txt": ""})})
    with pytest.raises(ValueError, match="unsafe include path"):
        pv.PolyHaven(str(tmp_path / "ph"), fetch=http).find("traffic_cone")
    assert not (tmp_path / "escape.txt").exists()


def test_polyhaven_http_sends_a_user_agent(monkeypatch):
    seen: list[Any] = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen.append((req, timeout))
        return Resp(b"{}")

    monkeypatch.setattr(pv.urllib.request, "urlopen", fake_urlopen)
    assert pv._http_get("https://api.polyhaven.com/assets?t=models") == b"{}"
    req, timeout = seen[0]
    assert req.get_header("User-agent", "").startswith("ai-db-scene/") and timeout and timeout > 0


def test_polyhaven_default_cache_follows_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert pv.PolyHaven().cache == tmp_path / "xdg" / "ai-db" / "polyhaven"


# --- live: import a real glTF and fit it ------------------------------------------------------

@pytest.mark.blender
def test_live_gltf_asset_is_fitted_to_its_placement(private_blender, tmp_path, monkeypatch):
    from ai_db_scene.blender_client import BlenderClient

    c = BlenderClient(port=private_blender, timeout=120)
    lib = tmp_path / "lib"
    lib.mkdir()
    # a 2 x 1 x 3 m block whose origin is off-centre, exported as the library's "bench"
    c.run_code("import bpy\nfor o in list(bpy.data.objects): bpy.data.objects.remove(o)\n"
               "bpy.ops.mesh.primitive_cube_add(location=(5, 5, 7))\nbpy.context.object.scale = (1, 0.5, 1.5)\n"
               "result = 1")
    c.request("export_file", {"filepath": str(lib / "old_bench.glb")})
    c.request("clear_scene", {"keep_camera": False})
    spec_raw = {"setting": {"kind": "outdoor", "bounds_m": [8, 8, 4]},
                "objects": [{"id": "bench", "category": "bench"}], "relations": []}
    spec, _ = sp._parse_spec(spec_raw)
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": spec_raw})["placements"])
    cmds, _ = sp.build_layout_commands(spec, pls, providers=[pv.LocalLibrary([str(lib)])])
    assert any(cmd["type"] == "import_file" for cmd in cmds)
    res = c.batch(cmds)
    assert all(r.get("ok") for r in res), [r for r in res if not r.get("ok")]
    got = c.run_code(
        "import bpy\nfrom mathutils import Vector\nr = bpy.data.objects['bench']\n"
        "pts = [o.matrix_world @ Vector(p) for o in r.children_recursive if o.type == 'MESH' for p in o.bound_box]\n"
        "lo = [min(p[i] for p in pts) for i in range(3)]\nhi = [max(p[i] for p in pts) for i in range(3)]\n"
        "result = {'lo': lo, 'hi': hi, 'id': r.get('ai_db_id'), 'src': r.get('asset_source')}")
    pl = pls[0]
    w, d, h = pl.dims
    size = [got["hi"][i] - got["lo"][i] for i in range(3)]
    assert got["id"] == "bench" and got["src"] == "local"
    assert abs(got["lo"][2]) < 1e-3  # on the ground
    assert max(s / lim for s, lim in zip(sorted(size[:2]) + [size[2]], sorted([w, d]) + [h])) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("stem, cat", [
    ("chairCushion", "chair"), ("tableCoffee", "coffee_table"), ("loungeSofa", "sofa"),
    ("loungeChair", "armchair"), ("bookcaseOpen", "bookshelf"), ("lampRoundFloor", "floor_lamp"),
    ("Bed_Double", "bed"), ("chairDesk", "office_chair"), ("cabinetTelevision", "tv_stand"),
    ("televisionModern", "tv"), ("kitchenFridge", "refrigerator"), ("lampSquareCeiling", "ceiling_light"),
    ("sideTableDrawers", "side_table"), ("NightStand_1", "nightstand"), ("Table_Large", "dining_table"),
    ("Couch_Medium", "sofa"), ("TVStand", "tv_stand"), ("trashcan", "trash_can"), ("pottedPlant", "potted_plant"),
    ("spaceshipHull", None),
])
def test_pack_filenames_map_to_categories(stem, cat):
    from ai_db_scene.priors import categories

    assert pv.match_category(stem, categories()) == cat


def test_split_words():
    assert pv.split_words("lampRoundFloor") == ["lamp", "round", "floor"]
    assert pv.split_words("TVStand_02") == ["tv", "stand", "02"]


def test_user_aliases_override_and_extend(tmp_path):
    lib = tmp_path / "pack"
    lib.mkdir()
    for n in ("loungeChair.glb", "stoolBar.glb", "coatRack.glb"):
        (lib / n).write_bytes(b"glb")
    idx = pv.LocalLibrary([str(lib)], aliases={"loungeChair": "sofa", "coat_rack": "wardrobe"}).index()
    assert {c: [Path(f.filepath).name for f in fs] for c, fs in idx.items()} == {
        "sofa": ["loungeChair.glb"], "bar_stool": ["stoolBar.glb"], "wardrobe": ["coatRack.glb"]}  # a bar stool now


def test_aliases_config_validated():
    from ai_db_scene.config import scene_config

    from ai_db.config import parse_config
    from ai_db.config_template import build_template
    from ai_db.errors import AiDbConfigError

    raw = build_template()
    raw.setdefault("plugins", {})["scene"] = {"aliases": {"loungeChair": "sofa"}}
    assert scene_config(parse_config(raw, check_env=False)).aliases == (("loungeChair", "sofa"),)
    raw["plugins"]["scene"] = {"aliases": {"x": "spaceship"}}
    with pytest.raises(AiDbConfigError, match="spaceship"):
        scene_config(parse_config(raw, check_env=False))
