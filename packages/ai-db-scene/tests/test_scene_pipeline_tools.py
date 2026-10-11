from __future__ import annotations

import ast
import json
import re

import pytest
from ai_db_scene.memory import SceneMemory
from ai_db_scene.models import RunRecord
from ai_db_scene.tools import scene_pipeline as sp
from ai_db_scene.tools import scene_tools

SPEC = {
    "setting": {"kind": "indoor", "bounds_m": [6, 5, 2.8], "style": "scandinavian"},
    "objects": [
        {"id": "sofa", "category": "sofa", "hero": True},
        {"id": "table", "category": "coffee_table"},
        {"id": "lamp", "category": "floor_lamp"},
    ],
    "relations": [
        {"a": "sofa", "rel": "against_wall", "b": "wall_north"},
        {"a": "table", "rel": "in_front_of", "b": "sofa", "gap_m": 0.45},
        {"a": "lamp", "rel": "beside", "b": "sofa"},
    ],
}


@pytest.fixture
def mem(tmp_path, monkeypatch):
    m = SceneMemory("json_file", {"path": str(tmp_path / "m")})
    monkeypatch.setattr(scene_tools, "_scene_memory", m)
    return m


def test_tools_in_table():
    names = [n for n, *_ in scene_tools.scene_tool_table()]
    for n in ("get_scene_spec_schema", "submit_scene_spec", "solve_layout", "apply_layout",
              "record_run", "search_exemplars", "resolve_asset"):
        assert names.count(n) == 1


def test_schema_tool_with_exemplars(mem):
    for i in range(5):
        mem.record_run(RunRecord(brief=f"living room {i}", spec=SPEC, judge_scores=[{"overall": 4.5}]))
    mem.record_run(RunRecord(brief="living room bad", spec=SPEC, outcome="failed", judge_scores=[{"overall": 5}]))
    out = sp._handle_get_scene_spec_schema({"brief": "living room"})
    assert out["schema"]["required"] == ["setting", "objects"]
    assert "sofa" in out["categories"] and len(out["categories"]["sofa"]["typical_m"]) == 3
    assert "against_wall" in out["relations"]
    assert len(out["exemplars"]) == 3 and all(e["score"] >= 4 for e in out["exemplars"])


def test_submit_spec():
    ok = sp._handle_submit_scene_spec({"spec": SPEC})
    assert ok["status"] == "ok" and ok["spec"]["objects"][0]["id"] == "sofa"
    bad = sp._handle_submit_scene_spec({"spec": {"setting": {}, "objects": []}})
    assert bad["status"] == "invalid" and bad["errors"]


def test_solve_and_conflict():
    out = sp._handle_solve_layout({"spec": SPEC, "seed": 1})
    assert out["status"] == "ok" and {p["id"] for p in out["placements"]} == {"sofa", "table", "lamp"}
    assert sp._handle_solve_layout({"spec": SPEC, "seed": 1}) == out  # deterministic
    tiny = json.loads(json.dumps(SPEC))
    tiny["setting"]["bounds_m"] = [1, 1, 2.8]
    res = sp._handle_solve_layout({"spec": tiny})
    assert res["status"] == "conflict" and res["conflicts"]


def test_build_commands_placeholder_and_mapped():
    spec, _ = sp._parse_spec(SPEC)
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": SPEC})["placements"])
    cmds, names = sp.build_layout_commands(spec, pls)
    assert names[0] == "room_shell" and cmds[1]["type"] == "room_shell"
    assert names[1:] == [p.id for p in pls]
    # before anything is created, every name the batch will use is reserved
    reserve = cmds[0]["params"]["code"]
    assert cmds[0]["type"] == "run_code" and all(f'"{p.id}"' in reserve for p in pls)
    compile(reserve, "<reserve>", "exec")
    # sofa and coffee table have generators (real furniture); the floor lamp is a box
    gen = {c["params"]["name"]: c["params"] for c in cmds if c["type"] == "generate_asset"}
    assert set(gen) == {"sofa", "table"} and gen["sofa"]["family"] == "sofa"
    sofa = next(p for p in pls if p.id == "sofa")
    assert gen["sofa"]["params"] == {"width": sofa.dims[0], "depth": sofa.dims[1]}
    assert gen["sofa"]["location"][2] == pytest.approx(0.0)  # base centre on the floor
    # the floor lamp has a part recipe: a real lamp scaled to its placement, not a box
    lamp_asset = next(c["params"]["spec"] for c in cmds if c["type"] == "build_asset")
    assert lamp_asset["asset"] == "lamp" and lamp_asset["bbox_m"] == list(next(p.dims for p in pls if p.id == "lamp"))
    assert sum(c["type"] == "create_primitive" for c in cmds) == 0
    boxes, _ = sp.build_layout_commands(spec, pls, geometry="boxes")
    assert sum(c["type"] == "create_primitive" for c in boxes) == 3
    assert {c["params"]["name"] for c in boxes if c["type"] == "apply_material"} >= {"lamp", "sofa", "table"}
    fin = next(c for c in cmds if c["type"] == "run_code" and "ai_db_id" in c["params"]["code"])
    assert '"hero": true' in fin["params"]["code"]
    compile(fin["params"]["code"], "<fin>", "exec")
    # every placeholder gets a category material; room surfaces and a camera are added
    mats = {c["params"]["name"]: c["params"]["material"] for c in cmds if c["type"] == "apply_material"}
    assert "lamp" not in mats  # generated pieces and recipes bring their own materials
    # scandinavian style: its palette paints the room
    assert mats["Room_Floor"] == "wood:pine:proj=box" and mats["Room_Walls"] == "plaster:#f4f1ea:proj=box"
    assert any(c["type"] == "run_code" and "ai_db_room" in c["params"]["code"] for c in cmds)
    cam = cmds[-1]
    assert cam["type"] == "set_camera" and cam["params"]["location"][2] == pytest.approx(1.6)
    x, y, _ = cam["params"]["location"]  # inside the (centred) room, at a corner
    assert abs(x) < 3 and abs(y) < 2.5 and abs(x) > 1.5 and abs(y) > 1.0
    assert any(c["type"] == "daylight_rig" for c in cmds)
    assert any(c["type"] == "add_light" and c["params"]["name"] == "lamp_light" for c in cmds)
    cmds2, _ = sp.build_layout_commands(spec, pls, {"sofa": {"filepath": "/l.blend", "asset_name": "Sofa1"}})
    app = [c for c in cmds2 if c["type"] == "append_asset"]
    assert len(app) == 1 and app[0]["params"]["asset_name"] == "Sofa1"
    # no rotation command / floor for single_asset
    one = json.loads(json.dumps(SPEC))
    one["setting"]["kind"] = "single_asset"
    one["relations"] = []
    assert sp._room_commands(sp._parse_spec(one)[0]) == []


def test_style_palette_with_explicit_materials_winning():
    raw = json.loads(json.dumps(SPEC))
    raw["setting"]["style"] = "industrial"
    raw["setting"]["wall_material"] = "plaster:#ffffff"
    raw["objects"][1]["material"] = "marble:verde"
    spec, _ = sp._parse_spec(raw)
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": raw})["placements"])
    cmds, _ = sp.build_layout_commands(spec, pls, geometry="boxes")
    mats = {c["params"]["name"]: c["params"]["material"] for c in cmds if c["type"] == "apply_material"}
    assert mats["Room_Floor"] == "color:#6e6e6a" and mats["Room_Walls"] == "plaster:#ffffff:proj=box"
    assert mats["sofa"] == "leather:proj=box" and mats["table"] == "marble:verde:proj=box"
    assert mats["lamp"] == "metal:steel:proj=box"
    raw["setting"]["style"] = "Industrial loft"  # free text naming a style picks its palette
    spec, _ = sp._parse_spec(raw)
    cmds, _ = sp.build_layout_commands(spec, pls, geometry="boxes")
    assert {c["params"]["name"]: c["params"]["material"] for c in cmds if c["type"] == "apply_material"} == mats
    raw["setting"]["style"] = "cozy bohemian"  # stored specs use free text: valid, no palette
    assert sp._handle_submit_scene_spec({"spec": raw})["status"] != "invalid"
    raw["setting"]["style"] = 7
    bad = sp._handle_submit_scene_spec({"spec": raw})
    assert bad["status"] == "invalid" and any("setting.style" in e for e in bad["errors"])
    assert "rustic" in sp._handle_get_scene_spec_schema({})["styles"]


class FakeClient:
    sent = None

    def batch(self, cmds, timeout=None):
        FakeClient.sent = cmds
        return [{"ok": 1, "status": "success"} for _ in cmds[:-1]] + [{"ok": 0, "status": "error", "message": "boom"}]


def test_apply_layout_one_batch(monkeypatch):
    monkeypatch.setattr(sp, "BlenderClient", FakeClient)
    pls = sp._handle_solve_layout({"spec": SPEC})["placements"]
    out = sp._handle_apply_layout({"spec": SPEC, "placements": pls})
    assert out["status"] == "partial" and out["created"][0] == "room_shell"
    assert out["failed"][0]["type"] == "set_camera" and len(FakeClient.sent) == out["commands"]
    assert sp._handle_apply_layout({"spec": {}, "placements": []})["status"] == "invalid"


def test_record_and_search_tools(mem):
    r = sp._handle_record_run({"brief": "red barn", "spec": SPEC, "judge_scores": [{"overall": 4.8}]})
    assert r["status"] == "saved" and mem.get_run(r["run_id"]).brief == "red barn"
    sp._handle_record_run({"brief": "red barn", "outcome": "failed", "judge_scores": [{"overall": 5}]})
    res = sp._handle_search_exemplars({"brief": "red barn"})
    assert res["count"] == 1 and res["exemplars"][0]["score"] == 4.8
    assert sp._handle_record_run({"brief": "x", "outcome": "bogus"})["status"] == "error"


def _parsed(raw=SPEC):
    spec, errors = sp._parse_spec(raw)
    assert spec is not None, errors
    return spec


def test_apply_puts_objects_inside_the_centred_room():
    """room_shell centres a sized room on the origin; placements must follow it."""
    spec = _parsed()
    placements = sp.parse_placements(sp._handle_solve_layout({"spec": SPEC})["placements"])
    cmds, _ = sp.build_layout_commands(spec, placements)
    w, d, _h = spec.setting.bounds_m
    for c in cmds:
        if c["type"] == "create_primitive":
            x, y, _z = c["params"]["location"]
            assert -w / 2 <= x <= w / 2 and -d / 2 <= y <= d / 2, c["params"]["name"]


def test_polygon_rooms_keep_the_layout_frame():
    raw = {**SPEC, "relations": SPEC["relations"][1:], "setting": {"kind": "indoor", "bounds_m": [6, 5, 2.8], "shape": "polygon",
                               "points": [[0, 0], [6, 0], [6, 5], [0, 5]]}}
    assert sp.frame_offset(_parsed(raw)) == (0.0, 0.0, 0.0)


def test_validator_args_translate_a_scene_spec():
    args = sp.validator_args(_parsed())
    assert args["room_bounds"] == [6.0, 5.0, 2.8]
    assert {"a": "table", "rel": "in_front_of", "b": "sofa"} in args["relations"]
    assert all(r["rel"] in ("facing", "in_front_of") for r in args["relations"])
    assert set(args["priors"]) == {"sofa", "coffee_table", "floor_lamp"}


def test_missing_objects_counts_instances():
    raw = {**SPEC, "objects": [*SPEC["objects"], {"id": "chair", "category": "dining_chair", "count": 2}]}
    issues = sp.missing_objects(_parsed(raw), ["sofa", "table", "chair_1"])
    assert {i["objects"][0]: i["message"] for i in issues} == {
        "lamp": "lamp: 0 of 1 in the scene", "chair": "chair: 1 of 2 in the scene"}
    assert all(i["severity"] == "blocking" for i in issues)


def _lights(raw):
    spec = _parsed(raw)
    return sp.lighting_commands(spec, [], sp.frame_offset(spec))


def _sized(w, d, kind="indoor"):
    return {**SPEC, "relations": SPEC["relations"][1:], "setting": {"kind": kind, "bounds_m": [w, d, 3.0]}}


def test_indoor_exposure_and_fill_scale_with_floor_area():
    small, big = _lights(_sized(4, 3)), _lights(_sized(20, 16))
    rig_s, rig_b = small[0]["params"], big[0]["params"]
    assert rig_s["window_portals"] and rig_b["window_portals"]  # windows keep their portals
    assert rig_s["exposure"] < rig_b["exposure"] <= 1.5
    fill_s = next(c["params"] for c in small if c["type"] == "add_light")
    fill_b = next(c["params"] for c in big if c["type"] == "add_light")
    assert fill_s["light_type"] == fill_b["light_type"] == "AREA"
    assert fill_s["energy"] < fill_b["energy"]
    assert fill_b["energy"] == pytest.approx(min(8.0 * 20 * 16, 2500.0))
    # centre of the (origin-centred) room, just under the ceiling
    assert fill_b["location"] == pytest.approx([0.0, 0.0, 3.0 - 0.05])
    shape = next(c["params"]["code"] for c in big if c["type"] == "run_code")
    assert "ai_db_fill" in shape and "10.0" in shape and "8.0" in shape
    compile(shape, "<fill>", "exec")


def test_outdoor_keeps_lower_exposure_and_no_fill():
    cmds = _lights(_sized(30, 30, kind="outdoor"))
    assert cmds[0]["params"]["exposure"] < 0 and not cmds[0]["params"]["window_portals"]
    assert not any(c["type"] == "add_light" and c["params"]["light_type"] == "AREA" for c in cmds)


def test_floor_area_by_shape():
    ell = {**SPEC, "relations": [], "setting": {"kind": "indoor", "bounds_m": [4, 4, 3], "shape": "ellipse"}}
    poly = {**SPEC, "relations": [], "setting": {"kind": "indoor", "bounds_m": [6, 5, 3], "shape": "polygon",
                                                 "points": [[0, 0], [6, 0], [6, 5], [0, 5]]}}
    assert sp.floor_area_m2(_parsed(ell)) == pytest.approx(3.14159 * 4, rel=1e-4)
    assert sp.floor_area_m2(_parsed(poly)) == pytest.approx(30.0)
    fill = next(c["params"] for c in _lights(poly) if c["type"] == "add_light")
    assert fill["location"][:2] == pytest.approx([3.0, 2.5])  # polygon rooms keep the layout frame


_PBR_SPEC = {
    "setting": {"kind": "indoor", "bounds_m": [5, 4, 2.7], "floor_material": "pbr:worn_concrete"},
    "objects": [{"id": "cabinet", "category": "cabinet", "material": "pbr:rusty_steel"},
                {"id": "crate", "category": "crate", "count": 2, "material": "pbr:weathered_wood"},
                {"id": "lamp", "category": "floor_lamp"}],
    "relations": [{"a": "cabinet", "rel": "against_wall", "b": "wall_north"}],
}


def test_pbr_presets_resolve_to_a_base_plus_the_skill_grunge():
    from ai_db_scene.materials import material_error, material_for

    assert material_error("pbr:worn_metal") is None
    assert "allowed" in material_error("pbr:shiny")
    assert material_for("cabinet", "pbr:rusty_steel") == "metal:rust:proj=box"
    spec, _ = sp._parse_spec(_PBR_SPEC)
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": _PBR_SPEC})["placements"])
    cmds, _ = sp.build_layout_commands(spec, pls, geometry="boxes")
    mats = {c["params"]["name"]: c["params"]["material"] for c in cmds if c["type"] == "apply_material"}
    assert mats["Room_Floor"] == "color:#8a8782" and mats["cabinet"] == "metal:rust:proj=box"
    wear = [c for c in cmds if c["type"] == "run_code" and "add_grunge(worn" in c["params"]["code"]]
    assert len(wear) == 1
    code = wear[0]["params"]["code"]
    compile(code, "<wear>", "exec")
    payload = json.loads(ast.literal_eval(re.search(r"^PAYLOAD = (.*)$", code, re.MULTILINE).group(1)))
    assert {e["name"] for e in payload} == {"cabinet", "crate_1", "crate_2", "Room_Floor"}
    assert "lamp" not in code.split("PAYLOAD")[1]


@pytest.mark.blender
def test_live_pbr_preset_builds_worn_materials(private_blender, mem, monkeypatch):
    from ai_db_scene.blender_client import BlenderClient

    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})
    res = sp._handle_build_scene({"spec": _PBR_SPEC, "clear": True, "snapshot": False})
    assert res["status"] == "built", res
    c = BlenderClient(port=private_blender, timeout=120)
    got = c.run_code(
        "import bpy\n"
        "def mats(n):\n"
        "    root = bpy.data.objects[n]\n"
        "    return sorted({s.material.name for o in [root, *root.children_recursive]\n"
        "                   for s in o.material_slots if s.material})\n"
        "result = {n: mats(n) for n in ('cabinet', 'crate_1', 'Room_Floor', 'lamp')}\n"
    )
    assert all(m.endswith("_rusty_steel") for m in got["cabinet"]), got
    assert got["crate_1"] and all(m.endswith("_weathered_wood") for m in got["crate_1"]), got
    assert all(m.endswith("_worn_concrete") for m in got["Room_Floor"]), got
    assert not any("_rusty" in m or "_weathered" in m for m in got["lamp"]), got


def test_export_path_stays_inside_the_export_folder(tmp_path):
    root = tmp_path / "exports"
    root.mkdir()
    assert sp.export_path("room.glb", None, root) == (root / "room.glb", "glb")
    assert sp.export_path("sub/room", "usdz", root) == (root / "sub" / "room.usdz", "usdz")
    assert sp.export_path(str(root / "a.gltf"), None, root)[1] == "gltf"
    (root / "escape").symlink_to(tmp_path)
    for bad in ("../room.glb", str(tmp_path / "room.glb"), "escape/room.glb", "", "room.fbx"):
        with pytest.raises(ValueError):
            sp.export_path(bad, None, root)


def test_export_scene_refuses_outside_paths_without_blender(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_DB_SCENE_EXPORT_DIR", str(tmp_path / "exports"))
    out = sp._handle_export_scene({"path": "../../etc/x.glb"})
    assert out["status"] == "error" and "outside the export folder" in out["message"]


@pytest.mark.blender
def test_live_export_scene_writes_glb_and_usd(private_blender, tmp_path, monkeypatch):
    from ai_db_scene.blender_client import BlenderClient

    monkeypatch.setenv("AI_DB_SCENE_EXPORT_DIR", str(tmp_path / "exports"))
    c = BlenderClient(port=private_blender, timeout=120)
    c.request("clear_scene", {"keep_camera": False, "purge_orphans": True})
    c.request("create_primitive", {"primitive": "cube", "name": "box", "location": [0, 0, 0.5]})
    glb = sp._handle_export_scene({"path": "scene.glb"})
    assert glb["status"] == "success" and glb["objects"] >= 1 and glb["meshes"] == 1, glb
    assert (tmp_path / "exports" / "scene.glb").stat().st_size == glb["bytes"] > 0
    usd = sp._handle_export_scene({"path": "scene", "format": "usdz"})
    assert usd["status"] == "success" and usd["bytes"] > 0, usd


@pytest.mark.blender
def test_live_poly_budget_decimates_heaviest_non_hero(private_blender, mem, monkeypatch):
    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})
    res = sp._handle_build_scene({"spec": SPEC, "clear": True, "snapshot": False, "poly_budget": 10**9})
    polys = res["polys"]
    assert polys["decimated"] == [] and not polys["over_budget"], polys
    assert set(polys["faces_per_object"]) == {"sofa", "table", "lamp"}
    budget = polys["faces_total"] - polys["faces_per_object"]["table"] // 2
    cut = sp.poly_report(budget)
    assert cut["decimated"] and "sofa" not in {d["object"] for d in cut["decimated"]}, cut
    assert cut["faces_total"] < polys["faces_total"]
    floor = sp.poly_report(1)  # unreachable: every non-hero at the ratio floor, hero untouched
    assert floor["over_budget"] and floor["faces_per_object"]["sofa"] == polys["faces_per_object"]["sofa"]


def test_physics_command_marks_only_dynamic_instances():
    spec = dict(SPEC, objects=[*SPEC["objects"], {"id": "ball", "category": "floor_lamp", "count": 2, "dynamic": True}])
    parsed, errors = sp._parse_spec(spec)
    assert parsed is not None, errors
    assert parsed.to_dict()["objects"][-1]["dynamic"] is True
    pls = sp.solve_layout(parsed)
    code = sp.physics_commands(parsed, pls)["params"]["code"]
    compile(code, "<physics>", "exec")
    assert '["ball_1", "ball_2"]' in code and "sofa" not in code
    assert sp._parse_spec(dict(SPEC, objects=[{"id": "a", "category": "sofa", "dynamic": "yes"}]))[0] is None


@pytest.mark.blender
def test_live_physics_adds_rigid_bodies(private_blender, mem, monkeypatch):
    from ai_db_scene.blender_client import BlenderClient

    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})
    spec = dict(SPEC, objects=[*SPEC["objects"][:2], {"id": "lamp", "category": "floor_lamp", "dynamic": True}])
    res = sp._handle_build_scene({"spec": spec, "clear": True, "snapshot": False, "physics": True})
    assert res["physics"]["active"] >= 1 and res["physics"]["passive"] >= 3, res.get("physics")
    got = BlenderClient(port=private_blender, timeout=120).run_code(
        "import bpy\n"
        "def kinds(n):\n"
        "    r = bpy.data.objects[n]\n"
        "    return sorted({o.rigid_body.type for o in [r, *r.children_recursive] if o.rigid_body})\n"
        "result = {n: kinds(n) for n in ('sofa', 'lamp', 'Room_Floor')}\n")
    assert got == {"sofa": ["PASSIVE"], "lamp": ["ACTIVE"], "Room_Floor": ["PASSIVE"]}, got


def test_critique_orders_blocking_first_and_suggests_steps():
    left = [{"check": "facing", "severity": "warning", "objects": ["sofa"], "message": "sofa misses tv"},
            {"check": "missing_objects", "severity": "blocking", "objects": ["chair"], "message": "chair: 2 of 4"}]
    out = sp.critique(left, ["dropped relation lamp beside sofa"], [{"type": "generate_asset", "message": "boom"}],
                      {"over_budget": True, "faces_total": 900, "budget": 500})
    assert [c["source"] for c in out] == ["validator", "build", "validator", "repair", "polys"]
    assert out[0]["check"] == "missing_objects" and "bounds_m" in out[0]["suggest"]
    assert sp.critique([], [], None, None) == []


@pytest.mark.blender
def test_live_judge_renders_and_critiques(private_blender, mem, tmp_path):
    spec = dict(SPEC, objects=[*SPEC["objects"], {"id": "chair", "category": "dining_chair", "count": 2}])
    res = sp._handle_build_scene({"spec": spec, "clear": True, "snapshot": False, "judge": True,
                                  "render_path": str(tmp_path / "judge.png")})
    assert "render_error" not in res, res.get("render_error")
    assert (tmp_path / "judge.png").stat().st_size > 0
    assert isinstance(res["critique"], list)
    assert all({"source", "severity", "objects", "message", "suggest"} <= set(c) for c in res["critique"])


def test_multi_room_build_has_inner_walls_with_door_gaps():
    from test_scene_layout import _HOUSE

    spec, errors = sp._parse_spec(_HOUSE)
    assert spec is not None, errors
    cmds, names = sp.build_layout_commands(spec, sp.solve_layout(spec), geometry="boxes")
    walls = [c["params"] for c in cmds if c["type"] == "create_primitive" and c["params"]["name"].startswith("wall_")]
    # living|bed+bath wall at x=5 split by two doors into 3 slabs; bed|bath wall at y=3 by one door into 2
    assert sorted(w["name"] for w in walls) == ["wall_bed_north_0", "wall_bed_north_1", "wall_living_east_0",
                                                "wall_living_east_1", "wall_living_east_2"]
    assert all(n in names for n in ("wall_bed_north_0", "sofa", "sink"))
    assert sp._handle_solve_layout({"spec": _HOUSE})["doors"]


def test_multi_room_lintels_and_walls_facing_empty_space():
    from test_scene_layout import _HOUSE

    spec, errors = sp._parse_spec(_HOUSE)
    assert spec is not None, errors
    cmds, _ = sp.build_layout_commands(spec, sp.solve_layout(spec), geometry="boxes")
    lintels = {c["params"]["name"]: c["params"] for c in cmds
               if c["type"] == "create_primitive" and c["params"]["name"].startswith("lintel_")}
    assert set(lintels) == {"lintel_door_living_bed", "lintel_door_living_bath", "lintel_door_bed_bath"}
    p = lintels["lintel_door_living_bed"]  # spans the door width, from the door top to the ceiling
    assert abs(p["scale"][1] - 0.45) < 1e-9 and abs(p["location"][2] - (2.1 + 2.7) / 2) < 1e-9
    # two rooms that leave a gap inside the bounds: the edges facing it get walls too
    gap = {"setting": {"kind": "indoor", "bounds_m": [10, 4, 2.7], "rooms": [
        {"id": "a", "offset_m": [0, 0], "size_m": [4, 4]}, {"id": "b", "offset_m": [6, 0], "size_m": [4, 4]}]},
        "objects": [{"id": "sofa", "category": "sofa", "room": "a"}]}
    spec, errors = sp._parse_spec(gap)
    assert spec is not None, errors
    cmds, _ = sp.build_layout_commands(spec, sp.solve_layout(spec), geometry="boxes")
    walls = {c["params"]["name"]: c["params"] for c in cmds
             if c["type"] == "create_primitive" and c["params"]["name"].startswith("wall_")}
    assert set(walls) == {"wall_a_east_0", "wall_b_west_0"}
    assert abs(walls["wall_b_west_0"]["location"][0] - (6 - 5)) < 1e-9  # x=6 in the layout frame


@pytest.mark.blender
def test_live_multi_room_builds(private_blender, mem):
    from test_scene_layout import _HOUSE

    res = sp._handle_build_scene({"spec": _HOUSE, "clear": True, "snapshot": False, "geometry": "boxes"})
    assert res["status"] in ("built", "built_with_issues") and not res["failed_commands"], res
    assert "wall_living_east_2" in res["created"]
    assert not [i for i in res["issues_left"] if i.get("check") in ("missing_objects", "out_of_bounds")], res["issues_left"]


def test_user_recipe_with_overwrite_beats_generator():
    from ai_db_scene import user_recipes as ur

    legs = [{"id": f"leg_{i}", "shape": "box", "size": [0.06, 0.06, 0.94], "at": [x, y, 0], "material": "wood:oak"}
            for i, (x, y) in enumerate(((-0.45, -0.45), (0.45, -0.45), (-0.45, 0.45), (0.45, 0.45)))]
    parts = legs + [{"id": "top", "shape": "box", "size": [1.0, 1.0, 0.06], "at": [0, 0, 0.94], "material": "wood:oak"}]
    spec, _ = sp._parse_spec(SPEC)
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": SPEC})["placements"])
    table = next(p for p in pls if p.id == "table")

    def kinds() -> tuple[set[str], set[str]]:
        cmds, _ = sp.build_layout_commands(spec, pls)
        gen = {c["params"]["name"] for c in cmds if c["type"] == "generate_asset"}
        built = {c["params"]["spec"]["asset"] for c in cmds if c["type"] == "build_asset"}
        return gen, built

    ur.create_recipe(table.category, "plain", parts)  # no overwrite: the generator still wins
    gen, built = kinds()
    assert gen == {"sofa", "table"} and "table" not in built
    ur.create_recipe(table.category, "plain", parts, overwrite=True)
    gen, built = kinds()
    assert gen == {"sofa"} and "table" in built


def test_exposure_issue_only_out_of_band():
    assert sp.exposure_issue({"in_band": True, "mean": 0.4}) is None
    assert sp.exposure_issue({"error": "no blender"}) is None
    dark = sp.exposure_issue({"in_band": False, "mean": 0.02, "rounds": 3, "dark_frac": 0.9, "bright_frac": 0.0})
    assert dark and dark["check"] == "exposure" and "dark" in dark["message"]
    assert "bright" in sp.exposure_issue({"in_band": False, "mean": 0.9, "rounds": 3})["message"]
    assert sp._CRITIQUE_FIX["exposure"]


@pytest.mark.blender
def test_live_exposure_check_brings_dark_render_into_band(private_blender, mem):
    from ai_db_scene.blender_client import BlenderClient

    res = sp._handle_build_scene({"spec": SPEC, "clear": True, "snapshot": False})
    exp = res["exposure"]
    assert {"rounds", "mean", "dark_frac", "bright_frac", "adjusted_by"} <= set(exp), exp
    assert exp["in_band"], exp
    # a far too dark view: the check raises the exposure and leaves the render settings alone
    BlenderClient().run_code("import bpy\nbpy.context.scene.view_settings.exposure = -12.0\n"
                             "bpy.context.scene.render.resolution_x = 1234\nresult = 1")
    fixed = sp.exposure_check()
    assert fixed["rounds"] >= 2 and fixed["adjusted_by"] >= 3.9, fixed
    assert BlenderClient().run_code("import bpy\nresult = bpy.context.scene.render.resolution_x") == 1234


def test_camera_spots_stay_inside_curved_and_polygon_rooms():
    import math

    oval = dict(SPEC, setting={"kind": "indoor", "bounds_m": [10, 7, 4], "shape": "ellipse"},
                relations=[{"a": "sofa", "rel": "against_wall", "b": "wall", "at_deg": 90}, *SPEC["relations"][1:]])
    spec, errors = sp._parse_spec(oval)
    assert spec, errors
    for x, y in sp._camera_spots(spec):
        assert ((x - 5) / 5) ** 2 + ((y - 3.5) / 3.5) ** 2 < 1  # a bbox corner is outside the oval
    poly = dict(SPEC, setting={"kind": "indoor", "bounds_m": [6, 6, 3], "shape": "polygon",
                               "points": [[0, 0], [6, 0], [6, 3], [3, 3], [3, 6], [0, 6]]},
                relations=[{"a": "sofa", "rel": "against_wall", "b": "wall_1"}, *SPEC["relations"][1:]])
    spec, errors = sp._parse_spec(poly)
    assert spec, errors
    assert all(sp._inside_polygon(x, y, spec.setting.points) for x, y in sp._camera_spots(spec))
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": oval})["placements"])
    spec, _ = sp._parse_spec(oval)
    poses = sp.camera_poses(spec, pls, sp.frame_offset(spec))
    assert len(poses) == 8 and len({p["location"][2] for p in poses}) == 2
    assert sp.camera_command(spec, pls, sp.frame_offset(spec))["params"] == poses[0]
    fp = sp.footprint(spec, sp.frame_offset(spec))
    assert all(math.isclose((x / 5) ** 2 + (y / 3.5) ** 2, 1) for x, y in fp)


def test_camera_issue_and_targets():
    assert sp.camera_issue({"inside": True, "targets": ["sofa"], "visible_heroes": ["sofa"], "tried": 1}) is None
    issue = sp.camera_issue({"inside": False, "targets": ["sofa"], "visible_heroes": [], "tried": 8})
    assert issue and issue["objects"] == ["sofa"] and "outside" in issue["message"]
    assert sp.camera_issue({"error": "x"}) is None
    spec, _ = sp._parse_spec(dict(SPEC, objects=[dict(o, hero=False) for o in SPEC["objects"]]))
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": spec.to_dict()})["placements"])
    assert len(sp.coverage_targets(spec, pls)) == 3


@pytest.mark.blender
def test_live_camera_check_moves_off_a_blocked_view(private_blender, mem):
    from ai_db_scene.blender_client import BlenderClient

    res = sp._handle_build_scene({"spec": SPEC, "clear": True, "snapshot": False, "exposure_check": False})
    assert res["camera"]["inside"] and res["camera"]["visible_heroes"] == ["sofa"], res["camera"]
    # a wall-sized slab right in front of the camera hides the sofa: another spot is chosen
    BlenderClient().run_code(
        "import bpy\nfrom mathutils import Vector\ncam = bpy.context.scene.camera\n"
        "fwd = cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))\n"
        "bpy.ops.mesh.primitive_cube_add(size=1, location=cam.location + fwd * 0.6)\n"
        "b = bpy.context.active_object\nb.scale = (2.0, 2.0, 2.0)\nb.rotation_euler = cam.rotation_euler\n"
        "result = list(cam.location)")
    spec, _ = sp._parse_spec(SPEC)
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": SPEC})["placements"])
    cam = sp.camera_check(spec, pls)
    assert cam["tried"] > 1 and cam["pose"] > 0 and cam["visible_heroes"] == ["sofa"], cam


def test_render_options_validated():
    opts, err = sp.render_options({"sky": True})
    assert err is None and opts["engine"] == "cycles" and opts["sky"] is True and len(opts["resolution"]) == 2
    for bad in ({"engine": "luxcore"}, {"samples": 0}, {"samples": True}, {"sky": "yes"},
                {"resolution": [10, 10]}, {"denoise": True}, "cycles"):
        assert sp.render_options(bad)[0] is None, bad
    assert sp._handle_build_scene({"spec": SPEC, "render": {"engine": "x"}})["status"] == "invalid"


@pytest.mark.blender
@pytest.mark.parametrize("engine", ["cycles", "eevee"])
def test_live_final_render_with_sky(private_blender, mem, tmp_path, engine):
    from ai_db_scene.blender_client import BlenderClient

    target = tmp_path / f"final_{engine}.png"
    res = sp._handle_build_scene({"spec": SPEC, "clear": True, "snapshot": False, "exposure_check": False,
                                  "render": {"engine": engine, "samples": 4, "sky": True, "resolution": [160, 90],
                                             "path": str(target)}})
    fr = res["final_render"]
    assert "error" not in fr, fr
    assert target.stat().st_size > 0 and fr["path"] == str(target)
    assert fr["sky"]["sun"] and abs(fr["sky"]["sun_elevation"] - 40.0) < 1.0, fr
    assert abs(fr["sky"]["sun_azimuth"] - 135.0) < 1.0, fr
    assert BlenderClient().run_code(
        "import bpy\nresult = [n.type for n in bpy.context.scene.world.node_tree.nodes].count('TEX_SKY')") == 1


_OVAL = {
    "setting": {"kind": "indoor", "bounds_m": [10, 7, 3.6], "shape": "ellipse", "style": "classic"},
    "objects": [
        {"id": "desk", "category": "desk", "hero": True},
        {"id": "door", "category": "door"},
        {"id": "door_b", "category": "door"},
        {"id": "window", "category": "window"},
        {"id": "window_l", "category": "window"},
        {"id": "window_r", "category": "window"},
    ],
    "relations": [
        {"a": "desk", "rel": "centered_in", "b": "room"},
        {"a": "door", "rel": "against_wall", "b": "wall", "at_deg": 270},
        {"a": "door_b", "rel": "against_wall", "b": "wall", "at_deg": 0},
        {"a": "window", "rel": "against_wall", "b": "wall", "at_deg": 90},
        {"a": "window_l", "rel": "against_wall", "b": "wall", "at_deg": 115},
        {"a": "window_r", "rel": "against_wall", "b": "wall", "at_deg": 65},
    ],
}


def test_mouldings_validated_and_defaulted_by_style():
    spec, errors = sp._parse_spec(_OVAL)
    assert spec, errors
    assert spec.setting.moulding_list() == ("crown", "baseboard")
    assert "mouldings" not in spec.to_dict()["setting"]
    for bad, msg in ((["cornice"], "not allowed"), (["crown", "crown"], "duplicates"), ("crown", "expected a list")):
        _s, errs = sp._parse_spec(dict(_OVAL, setting={**_OVAL["setting"], "mouldings": bad}))
        assert any(msg in e for e in errs), errs
    _s, errs = sp._parse_spec(dict(SPEC, setting={"kind": "outdoor", "bounds_m": [6, 5, 3], "mouldings": ["crown"]}))
    assert any("indoor" in e for e in errs), errs
    none, _ = sp._parse_spec(dict(_OVAL, setting={**_OVAL["setting"], "mouldings": []}))
    assert none.setting.moulding_list() == () and none.to_dict()["setting"]["mouldings"] == []
    assert sp.moulding_commands(none, []) == []


def test_moulding_commands_gap_at_doors_only_where_heights_cross():
    import json as _json

    data = dict(_OVAL, setting={**_OVAL["setting"], "mouldings": ["crown", "chair_rail", "baseboard"]})
    spec, _ = sp._parse_spec(data)
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": data})["placements"])
    (cmd,) = sp.moulding_commands(spec, pls)
    code = cmd["params"]["code"]
    jobs = {j["kind"]: j for j in _json.loads(code.split("jobs = json.loads(")[1].split(")\n")[0][1:-1])}
    assert [c for _p, c in jobs["crown"]["paths"]] == [True]  # nothing reaches the crown
    assert len(jobs["baseboard"]["paths"]) == 2 and not any(c for _p, c in jobs["baseboard"]["paths"])  # 2 doors
    assert len(jobs["chair_rail"]["paths"]) == 5  # 2 doors + 3 windows (sill under the rail top)
    assert jobs["crown"]["width"] > 0.1  # a 3.6 m room: heavier crown
    for x, y in jobs["baseboard"]["paths"][0][0]:
        assert (x / 5) ** 2 + (y / 3.5) ** 2 < 1.0  # inside the wall face


@pytest.mark.blender
def test_live_classic_oval_mouldings(private_blender, mem, tmp_path):
    from ai_db_scene.blender_client import BlenderClient

    data = dict(_OVAL, setting={**_OVAL["setting"], "mouldings": ["crown", "chair_rail", "baseboard"]})
    res = sp._handle_build_scene({"spec": data, "clear": True, "snapshot": False, "judge": True,
                                  "render_path": str(tmp_path / "oval.png")})
    assert not res["failed_commands"], res["failed_commands"]
    got = BlenderClient().run_code(
        "import bpy\nresult = {}\nfor o in bpy.data.objects:\n    k = o.get('moulding')\n"
        "    if k:\n        result.setdefault(k, []).append(len(o.data.polygons))")
    assert len(got["crown"]) == 1 and len(got["baseboard"]) == 2 and len(got["chair_rail"]) == 5, got
    assert all(n > 20 for faces in got.values() for n in faces), got


_OVAL_OFFICE = dict(_OVAL, objects=[*_OVAL["objects"][:3], {"id": "window", "category": "window", "count": 3}],
                    relations=[*_OVAL["relations"][:3], {"a": "window", "rel": "against_wall", "b": "wall", "at_deg": 90}])


@pytest.mark.parametrize("shape", ["ellipse", "polygon"])
def test_copies_at_one_at_deg_spread_along_a_curved_wall(shape):
    import math

    data = _OVAL_OFFICE if shape == "ellipse" else dict(
        _OVAL_OFFICE, setting={"kind": "indoor", "bounds_m": [10, 8, 3.6], "shape": "polygon",
                               "points": [[0, 0], [10, 0], [10, 5], [8, 8], [2, 8], [0, 5]]})
    res = sp._handle_solve_layout({"spec": data})
    assert res.get("status") == "ok", res
    spec, _ = sp._parse_spec(data)
    pls = sp.parse_placements(res["placements"])
    wins = [o for o in sp._openings(spec, pls) if o["kind"] == "window"]
    assert len(wins) == 3
    for i, a in enumerate(wins):
        for b in wins[i + 1:]:
            assert math.dist(a["at"], b["at"]) >= a["width"] + 0.3  # side by side, not stacked
    assert all(o["wall"] == "wall" for o in wins)


@pytest.mark.blender
def test_live_oval_office_openings_are_cut(private_blender, mem):
    from ai_db_scene.blender_client import BlenderClient

    res = sp._handle_build_scene({"spec": _OVAL_OFFICE, "clear": True, "snapshot": False, "exposure_check": False})
    assert not res["failed_commands"] and not res["repairs"], res
    spec, _ = sp._parse_spec(_OVAL_OFFICE)
    pls = sp.parse_placements(sp._handle_solve_layout({"spec": _OVAL_OFFICE})["placements"])
    ops = [{"at": o["at"], "z": o["sill"] + o["height"] / 2} for o in sp._openings(spec, pls)]
    assert len(ops) == 5
    # a ray from the room centre through each opening's middle leaves the walls without hitting them
    blocked = BlenderClient().run_code(
        "import bpy, json\nfrom mathutils import Vector\nwalls = bpy.data.objects['Room_Walls']\n"
        "inv = walls.matrix_world.inverted()\nout = []\n"
        f"for op in json.loads({json.dumps(json.dumps(ops))}):\n"
        "    end = Vector((op['at'][0], op['at'][1], op['z']))\n"
        "    d = Vector((end.x, end.y, 0)).normalized()\n"
        "    start = end - d * 0.6\n"
        "    hit, *_ = walls.ray_cast(inv @ start, (inv.to_3x3() @ d).normalized(), distance=1.2)\n"
        "    out.append(hit)\nresult = out")
    assert blocked == [False] * 5, blocked


def test_new_material_kinds_fall_back_on_an_older_addon(monkeypatch):
    from ai_db_scene.materials import legacy_fallback, material_error

    assert material_error("paint:#e0a000") is None and material_error("foliage") is None
    assert legacy_fallback("paint:#e0a000:proj=box") == "color:#e0a000"
    assert legacy_fallback("stone:proj=box") == "color:#7d7a72" and legacy_fallback("wood:oak") is None
    sent = []

    class Old:
        def batch(self, cmds, timeout=None):
            sent.append(cmds)
            return [{"ok": True} for _ in cmds]

    monkeypatch.setattr(sp, "BlenderClient", Old)
    cmds = [{"type": "apply_material", "params": {"name": "car", "material": "paint:#7a1f1f:proj=box"}},
            {"type": "apply_material", "params": {"name": "t", "material": "wood:oak:proj=box"}}]
    res = [{"ok": False, "message": "unknown material kind 'paint' (have [...])"}, {"ok": True}]
    assert sp._retry_legacy_materials(cmds, res) == [{"ok": True}, {"ok": True}]
    assert sent == [[{"type": "apply_material", "params": {"name": "car", "material": "color:#7a1f1f"}}]]


def test_generated_furniture_varies_by_scene_but_pairs_match():
    import json as _json
    from pathlib import Path

    raw = _json.loads((Path(__file__).parent / "fixtures" / "golden" / "indoor_living_room.json").read_text())["spec"]
    spec, _ = sp._parse_spec(raw)
    v1, v1b = sp._variation(spec, "sofa", "sofa"), sp._variation(spec, "sofa", "sofa")
    assert v1 == v1b and 0 <= v1["seed"] < 2**31            # a rebuild keeps the same sofa; a C int
    raw2 = {**raw, "setting": {**raw["setting"], "bounds_m": [7.0, 6.0, 2.8]}}
    spec2, _ = sp._parse_spec(raw2)
    assert sp._variation(spec2, "sofa", "sofa")["seed"] != v1["seed"]  # another scene, another sofa
    raw3 = {**raw, "setting": {**raw["setting"], "style": "classic"}}
    spec3, _ = sp._parse_spec(raw3)
    assert sp._variation(spec3, "sofa", "sofa")["preset"] == "chesterfield_like"
