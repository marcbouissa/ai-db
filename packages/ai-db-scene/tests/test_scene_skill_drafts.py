"""draft_skill material from scene run records (ai_db_scene.skill_drafts)."""

from __future__ import annotations

from importlib.metadata import entry_points

from ai_db_scene.models import RunRecord
from ai_db_scene.skill_drafts import contribute, summarize_runs

SPEC = {"setting": {"kind": "office"}, "objects": [{"id": "d", "category": "desk", "count": 4},
                                                  {"id": "c", "category": "office_chair"}],
        "relations": [{"a": "c", "rel": "facing", "b": "d"}]}


def _runs():
    clean = RunRecord(brief="small office with four desks", spec=SPEC, outcome="success",
                      validation_reports=[{"issues_found": 0, "issues_left": [], "repairs": []}])
    messy = RunRecord(brief="open office", spec=SPEC, outcome="partial",
                      validation_reports=[{"issues_found": 3, "repairs": ["moved c: overlap with d"],
                                           "issues_left": [{"check": "overlap"}, {"check": "overlap"}]}])
    other = RunRecord(brief="forest clearing", spec={"setting": {"kind": "outdoor"}, "objects": []},
                      validation_reports=[{"issues_left": [{"check": "floating"}]}])
    return [clean, messy, other]


def test_summarize_runs_patterns_and_pitfalls():
    out = summarize_runs(_runs(), "office")
    pattern = "office: desk, office_chair; relations facing"
    assert out["workflow"] == [f"Spec pattern that built clean on the first try (1 run(s)): {pattern}"]
    assert out["pitfalls"][0] == "validator: overlap - seen 2 time(s) in 2 related run(s)"
    assert any("solver repair: moved c" in p for p in out["pitfalls"])
    assert not any("floating" in p for p in out["pitfalls"])  # unrelated run
    assert out["examples"] == ["Brief that worked first time: small office with four desks"]
    assert summarize_runs(_runs(), "spaceship") == {}


def test_category_topic_matches_object_categories():
    assert summarize_runs(_runs(), "office_chair")["workflow"]


def test_hook_registered_and_reads_store(tmp_path, monkeypatch):
    assert "scene" in {ep.name for ep in entry_points(group="ai_db.skill_drafts")}
    from ai_db_scene.memory import SceneMemory
    from ai_db_scene.tools import scene_tools

    mem = SceneMemory("json_file", {"path": str(tmp_path / "m")})
    monkeypatch.setattr(scene_tools, "_scene_memory", mem)
    for r in _runs():
        mem.record_run(r)
    assert contribute("office")["workflow"]
