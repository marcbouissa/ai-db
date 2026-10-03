from __future__ import annotations

import json

import pytest

from ai_db.mcp_tools import scene_pipeline as sp
from ai_db.mcp_tools import scene_tools
from ai_db.memory.scene import SceneMemory
from ai_db.scene.models import RunRecord

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
    assert names[:5] == ["floor", "wall_north", "wall_south", "wall_east", "wall_west"]
    assert names[5:] == [p.id for p in pls]
    assert sum(c["type"] == "create_primitive" for c in cmds) == 5 + 3
    assert cmds[-1]["type"] == "run_code" and '"hero": true' in cmds[-1]["params"]["code"]
    compile(cmds[-1]["params"]["code"], "<fin>", "exec")
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
    assert out["status"] == "partial" and out["created"][0] == "floor"
    assert out["failed"][0]["type"] == "run_code" and len(FakeClient.sent) == out["commands"]
    assert sp._handle_apply_layout({"spec": {}, "placements": []})["status"] == "invalid"


def test_record_and_search_tools(mem):
    r = sp._handle_record_run({"brief": "red barn", "spec": SPEC, "judge_scores": [{"overall": 4.8}]})
    assert r["status"] == "saved" and mem.get_run(r["run_id"]).brief == "red barn"
    sp._handle_record_run({"brief": "red barn", "outcome": "failed", "judge_scores": [{"overall": 5}]})
    res = sp._handle_search_exemplars({"brief": "red barn"})
    assert res["count"] == 1 and res["exemplars"][0]["score"] == 4.8
    assert sp._handle_record_run({"brief": "x", "outcome": "bogus"})["status"] == "error"
