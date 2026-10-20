"""A brief given to a build_scene call that came back invalid still counts for the retry.

qwen3-14b-local-aidb8 round_lobby (fixtures/bench_aidb8/qwen_round_lobby_calls.json): the first
build_scene carried the brief "A round hotel lobby ..." and a spec without setting.kind (invalid);
the retry resent only the fixed spec, so the lobby came out rectangular and the counts went
unchecked. The brief now stays pending until a build succeeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ai_db_scene.tools import scene_pipeline as sp

CALLS = json.loads((Path(__file__).parent / "fixtures" / "bench_aidb8" / "qwen_round_lobby_calls.json")
                   .read_text())["calls"]
BRIEF = ("A round hotel lobby with a reception desk, four armchairs around a low table, two potted plants "
         "and a pendant lamp.")
QUIET = {"snapshot": False, "exposure_check": False, "poly_budget": None}


@pytest.fixture
def fake_blender(monkeypatch):
    class Fake:
        def batch(self, cmds, timeout=None):
            return [{"ok": 1, "status": "success"} for _ in cmds]

        def request(self, handler, params, timeout=None):
            return {}

    monkeypatch.setattr(sp, "BlenderClient", Fake)
    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})
    monkeypatch.setattr(sp, "_PENDING_BRIEF", [])
    monkeypatch.setattr(sp, "_REMEMBERED", {})


def _tool(name: str):
    return next(h for n, _d, _s, h in sp.PIPELINE_TOOLS if n == name)


def test_the_retry_of_an_invalid_build_keeps_the_round_brief(fake_blender, monkeypatch):
    built_specs: list[dict] = []
    from ai_db_scene import repair  # _build_scene imports solve_with_repairs at call time

    orig = repair.solve_with_repairs

    def spy(data, *a, **k):
        built_specs.append(data)
        return orig(data, *a, **k)

    monkeypatch.setattr(repair, "solve_with_repairs", spy)
    build = _tool("build_scene")
    first = build({**CALLS[0], "brief": BRIEF, **QUIET})
    assert first["status"] == "invalid" and "setting.kind" in first["errors"][0]
    second = build({**CALLS[1], **QUIET})
    assert second["status"] == "built", second
    assert built_specs[-1]["setting"]["shape"] == "ellipse"
    assert any("earlier call's brief" in r for r in second["repairs"])
    assert "no brief given" not in second["verdict"]
    # built: the brief is spent, a later unrelated build is not checked against it
    assert sp._PENDING_BRIEF == []


def test_a_build_with_its_own_brief_ignores_the_pending_one(fake_blender):
    _tool("get_scene_spec_schema")({"brief": BRIEF})
    assert sp._PENDING_BRIEF == [BRIEF]
    out = _tool("build_scene")({**CALLS[1], "brief": "a square office lobby", **QUIET})
    assert out["status"] == "built"
    assert not any("earlier call's brief" in r for r in out["repairs"])


def test_direct_handler_calls_do_not_carry(fake_blender):
    # edit_scene and the tests call _handle_build_scene directly: no pending brief there
    sp._PENDING_BRIEF[:] = [BRIEF]
    out = sp._handle_build_scene({**CALLS[1], **QUIET})
    assert "no brief given" in out["verdict"]
