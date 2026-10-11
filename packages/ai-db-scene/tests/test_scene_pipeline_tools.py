from __future__ import annotations

import json

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
    assert names[0] == "room_shell" and cmds[0]["type"] == "room_shell"
    assert names[1:] == [p.id for p in pls]
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
    assert mats["Room_Floor"] == "wood:oak:proj=box" and mats["Room_Walls"] == "plaster:proj=box"
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
