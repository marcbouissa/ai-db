"""Golden eval set: fixed specs (tests/fixtures/golden) that must keep solving cleanly.

Each fixture is ``{"spec", "expect", ["edit_steps", "expect_after_edit"]}``; ``expect`` holds
the instance count and per-category counts. Without Blender: validate -> solve -> check_layout
(no issues). With ``-m blender``: build_scene (and edit_scene) in a private headless Blender.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from ai_db_scene.edits import apply_steps
from ai_db_scene.layout import check_layout, solve_layout
from ai_db_scene.spec import validate_spec

GOLDEN = sorted((Path(__file__).parent / "fixtures" / "golden").glob("*.json"))
CASES = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in GOLDEN}


def _check(raw: dict[str, Any], expect: dict[str, Any]) -> None:
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    placements = solve_layout(spec)
    assert len(placements) == expect["instances"]
    cats = Counter(p.category for p in placements)
    assert {c: cats.get(c, 0) for c in expect["categories"]} == expect["categories"]
    assert check_layout(spec, placements) == []


def test_golden_set_covers_every_setting_and_feature():
    kinds = {c["spec"]["setting"]["kind"] for c in CASES.values()}
    assert {"indoor", "outdoor", "tabletop"} <= kinds
    assert any(o.get("grid") for c in CASES.values() for o in c["spec"]["objects"])
    assert any(r["rel"] == "above" for c in CASES.values() for r in c["spec"]["relations"])
    assert any("edit_steps" in c for c in CASES.values())


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_spec_solves_clean(name: str) -> None:
    case = CASES[name]
    _check(case["spec"], case["expect"])
    if "edit_steps" in case:
        edited, _done, errors = apply_steps(case["spec"], case["edit_steps"])
        assert edited is not None, errors
        _check(edited, case["expect_after_edit"])


@pytest.mark.blender
@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_spec_builds_in_blender(name: str, private_blender: int, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_db_scene.blender_client import BlenderClient
    from ai_db_scene.memory import SceneMemory
    from ai_db_scene.tools import scene_pipeline as sp
    from ai_db_scene.tools import scene_tools

    client = BlenderClient(port=private_blender, timeout=120)
    memory = SceneMemory("json_file", {"path": str(tmp_path / "mem")})
    memory.set_blender_client(client)
    monkeypatch.setattr(scene_tools, "_scene_memory", memory)
    names = "import bpy\nresult = sorted(o.name for o in bpy.context.scene.objects)"

    case = CASES[name]
    out = sp._handle_build_scene({"spec": case["spec"], "clear": True})
    assert out["status"] == "built", out
    assert out["objects"] == case["expect"]["instances"] and not out["failed_commands"]
    spec, _ = validate_spec(case["spec"])
    assert spec is not None
    ids = {p.id for p in solve_layout(spec)}
    assert ids <= set(client.run_code(names))
    if "edit_steps" in case:
        edited = sp._handle_edit_scene({"scene_id": out["scene_id"], "steps": case["edit_steps"]})
        assert edited["status"] == "built", edited
        assert edited["objects"] == case["expect_after_edit"]["instances"]
        removed = {s["id"] for s in case["edit_steps"] if s["op"] == "remove_object"}
        assert not removed & set(client.run_code(names))
