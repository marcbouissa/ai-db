"""User feedback -> lessons that later builds check (ai_db_scene.feedback)."""

from __future__ import annotations

import pytest
from ai_db_scene import feedback as fb
from ai_db_scene.layout import Placement


def _pl(pid: str, cat: str, dims=(1.2, 0.5, 0.8)) -> Placement:
    return Placement(id=pid, category=cat, location=(0.0, 0.0, dims[2] / 2), rotation_deg_z=0.0, dims=dims)


def test_parse_reads_categories_and_issue():
    assert fb.parse("the bench back is floating") == {"categories": ["bench"], "issue": "floating"}
    assert fb.parse("The couch clips into the coffee table")["categories"] == ["coffee_table", "sofa"]
    assert fb.parse("clothes are too stiff")["issue"] == "stiff"


def test_add_reinforce_update_disable_delete():
    a = fb.add("the bench back is floating")
    assert a["status"] == "added" and a["lesson"]["categories"] == ["bench"]
    again = fb.add("The bench back is floating!")
    assert again["status"] == "reinforced" and again["lesson"]["times_reported"] == 2
    lid = a["lesson"]["id"]
    assert fb.update(lid, {"enabled": False})["enabled"] is False
    assert fb.list_lessons(include_disabled=False) == []
    with pytest.raises(ValueError, match="cannot change"):
        fb.update(lid, {"id": "x"})
    assert fb.delete(lid) and fb.list_lessons() == []
    note = fb.add("clothes are too stiff", subjects=["clothes"])
    assert note["lesson"]["issue"] == "stiff" and note["lesson"]["subjects"] == ["clothes"]


def test_checklist_and_apply_on_a_build(monkeypatch):
    fb.add("the bench back is floating")
    fb.add("the sofa faces the wall", severity="warning")
    fb.add("all armchairs look identical")
    assert any("bench" in line for line in fb.checklist({"bench", "rug"}))
    assert not any("sofa" in line for line in fb.checklist({"bench"}))
    monkeypatch.setattr(fb, "_floating_recipe_parts", lambda cat, pls: ["back"] if cat == "bench" else [])
    issues = [{"check": "facing", "severity": "warning", "objects": ["sofa"], "message": "faces the wall"}]
    pls = [_pl("bench", "bench"), _pl("sofa", "sofa"), _pl("armchair_1", "armchair")]
    added = fb.apply(issues, pls)
    assert issues[0]["lesson"]  # the validator's own facing issue is tied to the lesson
    checks = {i["check"] for i in added}
    assert checks == {"lesson_floating", "lesson_verify"}
    assert next(i for i in added if i["check"] == "lesson_floating")["severity"] == "blocking"
    trig = {x["text"]: x["times_triggered"] for x in fb.list_lessons()}
    assert trig["the bench back is floating"] == 1 and trig["the sofa faces the wall"] == 1


def test_optimize_merges_fixes_and_retires(monkeypatch):
    a = fb.add("bench back floating", categories=["bench"])["lesson"]
    b = fb.add("the back of the bench hovers", categories=["bench"], issue="floating")["lesson"]
    typo = fb.add("benhc legs sunk", categories=["benhc"])["lesson"]
    stale = fb.add("tv too small", categories=["tv"])["lesson"]
    fb.update(stale["id"], {})  # no-op edit keeps it valid
    lessons = fb._load()
    for x in lessons:
        if x["id"] == stale["id"]:
            x["builds_seen"] = fb.OUTDATED_AFTER_BUILDS
    fb._save(lessons)
    plan = fb.optimize()
    acts = {(p["action"], p.get("drop") or p.get("lesson")) for p in plan["proposals"]}
    assert ("merge", b["id"]) in acts and ("fix_category", typo["id"]) in acts and ("outdated", stale["id"]) in acts
    assert len(fb.list_lessons()) == 4  # proposals only
    fb.optimize(apply_changes=True)
    after = {x["id"]: x for x in fb.list_lessons()}
    assert b["id"] not in after and after[a["id"]]["times_reported"] == 2
    assert after[typo["id"]]["categories"] == ["bench"] and after[stale["id"]]["enabled"] is False


def test_schema_lists_lessons_for_the_brief():
    from ai_db_scene.tools import scene_pipeline as sp

    fb.add("the bench back is floating")
    out = sp._handle_get_scene_spec_schema({"brief": "a park with a bench and a tree"})
    assert any("bench back is floating" in line for line in out["lessons"])


def test_lessons_are_global_or_scoped_to_one_project():
    fb.add("the bench back is floating")                              # global
    fb.add("the sofa faces the wall", project="client_a", issue="facing")
    assert {x["project"] for x in fb.list_lessons()} == {"global", "client_a"}
    assert len(fb.list_lessons(project="client_b")) == 1               # global only
    assert len(fb.list_lessons(project="client_a")) == 2
    assert not fb.checklist({"sofa"}, project="client_b") and fb.checklist({"sofa"}, project="client_a")
    issues = [{"check": "facing", "severity": "warning", "objects": ["sofa"], "message": "x"}]
    fb.apply(issues, [_pl("sofa", "sofa")], project="client_b")
    assert "lesson" not in issues[0]                                    # another project's lesson is not used
    fb.apply(issues, [_pl("sofa", "sofa")], project="client_a")
    assert issues[0]["lesson"]
    # the same complaint in two projects stays two lessons; optimize never merges across them
    fb.add("the sofa faces the wall", project="client_b", issue="facing")
    assert not [p for p in fb.optimize()["proposals"] if p["action"] == "merge"]
