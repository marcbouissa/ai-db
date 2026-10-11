"""build_scene's automatic spec repair and fix loop."""

from __future__ import annotations

from ai_db_scene.layout import LayoutError
from ai_db_scene.repair import relax
from ai_db_scene.tools import scene_pipeline as sp

SPEC = {"setting": {"kind": "indoor", "bounds_m": [4.0, 3.0, 2.6]},
        "objects": [{"id": "sofa", "category": "sofa"}, {"id": "table", "category": "coffee_table"},
                    {"id": "chair", "category": "dining_chair", "count": 3}],
        "relations": [{"a": "sofa", "rel": "against_wall", "b": "wall_north"},
                      {"a": "table", "rel": "in_front_of", "b": "sofa"},
                      {"a": "table", "rel": "facing", "b": "sofa"}]}


def test_drops_the_named_positional_relation_before_walls():
    err = LayoutError("cannot place 'table'", ["table facing sofa", "sofa against_wall wall_north", "table in_front_of sofa (gap 0.4 m)"])
    data, note = relax(SPEC, err, [4.0, 3.0, 2.6])
    assert note == "dropped relation 'table facing sofa'"
    assert len(data["relations"]) == 2 and len(SPEC["relations"]) == 3  # input untouched


def test_enlarges_then_drops_copies_when_nothing_is_named():
    err = LayoutError("cannot place 'chair_3' (...)", ["sofa"])
    data, note = relax(SPEC, err, [4.0, 3.0, 2.6])
    assert data["setting"]["bounds_m"] == [4.8, 3.6, 2.6] and "enlarged" in note
    big = {**SPEC, "setting": {"kind": "indoor", "bounds_m": [7.2, 5.4, 2.6]}}
    data, note = relax(big, err, [4.0, 3.0, 2.6])  # already 1.8x: no more growth
    assert next(o for o in data["objects"] if o["id"] == "chair")["count"] == 2 and "fewer" in note


def test_build_scene_repairs_builds_and_applies_fixes(monkeypatch):
    sent: list[tuple[str, dict]] = []

    class Fake:
        def batch(self, cmds, timeout=None):
            return [{"ok": 1, "status": "success"} for _ in cmds]

        def request(self, handler, params, timeout=None):
            sent.append((handler, params))
            return {}

    reports = iter([
        {"status": "success", "result": {"issues": [
            {"check": "floating", "severity": "blocking", "message": "lamp floats",
             "fix": {"handler": "modify_object", "params": {"name": "lamp", "location": [0, 0, 0]}}}]}},
        {"status": "success", "result": {"issues": []}},
    ])
    monkeypatch.setattr(sp, "BlenderClient", Fake)
    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: next(reports))
    tiny = {**SPEC, "setting": {"kind": "indoor", "bounds_m": [2.0, 1.6, 2.6]}}  # too small at first
    out = sp._handle_build_scene({"spec": tiny, "snapshot": False})
    assert out["status"] == "built", out
    assert out["repairs"] and out["spec"] is not None
    assert sent == [("modify_object", {"name": "lamp", "location": [0, 0, 0]})]
    assert out["fixes_applied"] == ["floating: lamp floats"] and out["issues_left"] == []


def test_build_scene_reports_invalid_specs():
    out = sp._handle_build_scene({"spec": {"setting": {"kind": "nowhere"}}})
    assert out["status"] == "invalid" and out["errors"]


def test_edit_steps_apply_in_order_and_name_the_failing_step():
    from ai_db_scene.edits import apply_steps
    data, done, errors = apply_steps(SPEC, [
        {"op": "add_object", "object": {"id": "lamp", "category": "floor_lamp"},
         "relations": [{"a": "lamp", "rel": "beside", "b": "sofa"}]},
        {"op": "update_object", "id": "sofa", "set": {"material": "fabric:velvet"}},
        {"op": "remove_object", "id": "table"},
        {"op": "update_setting", "set": {"bounds_m": [5.0, 4.0, 2.6]}}])
    assert not errors and len(done) == 4
    assert {o["id"] for o in data["objects"]} == {"sofa", "chair", "lamp"}
    assert all("table" not in (r["a"], r["b"]) for r in data["relations"])
    assert next(o for o in data["objects"] if o["id"] == "sofa")["material"] == "fabric:velvet"
    _d, _done, errors = apply_steps(SPEC, [{"op": "remove_object", "id": "sofa"}, {"op": "update_object", "id": "x", "set": {"count": 2}}])
    assert errors == ["steps[1]: no object 'x'"]
    assert apply_steps(SPEC, [{"op": "explode"}])[2][0].startswith("steps[0].op")


def test_edit_scene_rebuilds_from_the_stored_spec_as_the_next_version(monkeypatch, tmp_path):
    from ai_db_scene.memory import SceneMemory
    from ai_db_scene.models import ObjectState, SceneState
    from ai_db_scene.tools import scene_tools

    def _sample_scene(sid):
        return SceneState(scene_id=sid, objects={f"{sid}-o": ObjectState(uuid=f"{sid}-o", name="Cube")})

    memory = SceneMemory(backend="json_file", config={"path": str(tmp_path / "m")})
    n = iter(range(100))
    memory.set_blender_capture(lambda: _sample_scene(f"v{next(n)}"))
    monkeypatch.setattr(scene_tools, "_scene_memory", memory)
    calls: list[str] = []

    class Fake:
        def batch(self, cmds, timeout=None):
            return [{"ok": 1, "status": "success"} for _ in cmds]

        def request(self, handler, params, timeout=None):
            calls.append(handler)
            return {}

    monkeypatch.setattr(sp, "BlenderClient", Fake)
    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})
    first = sp._handle_build_scene({"spec": SPEC, "prompt": "living room"})
    v1 = first["scene_id"]
    # the build was recorded as a successful run, so it is an exemplar for similar briefs
    assert first["run_id"]
    ex = memory.search_exemplars("cozy living room with a sofa", min_score=4.0, top_k=3)
    assert ex and ex[0].spec["objects"][0]["id"] == "sofa"
    out = sp._handle_edit_scene({"scene_id": v1, "steps": [
        {"op": "update_object", "id": "chair", "set": {"count": 2}}]})
    assert out["status"] == "built" and out["previous_scene_id"] == v1 and out["safety_snapshot"]
    assert "clear_scene" in calls  # a full rebuild
    chain = [s.scene_id for s in memory.get_scene_versions(out["scene_id"])]
    assert chain[:2] == [out["scene_id"], v1]
    stored = memory.restore_scene(out["scene_id"]).prompt_context.metadata["spec"]
    assert next(o for o in stored["objects"] if o["id"] == "chair")["count"] == 2
    bad = sp._handle_edit_scene({"scene_id": v1, "steps": [{"op": "remove_object", "id": "nope"}]})
    assert bad["status"] == "invalid"


def test_every_recipe_has_a_prior_a_default_material_and_unique_parts():
    from ai_db_scene.materials import CATEGORY_MATERIALS
    from ai_db_scene.priors import load_priors
    from ai_db_scene.recipes import recipe_categories, recipe_spec
    priors = load_priors()
    for cat in recipe_categories():
        assert cat in priors and cat in CATEGORY_MATERIALS, cat
        spec = recipe_spec(cat, f"x_{cat}", priors[cat].typical_m)
        ids = [p["id"] for p in spec["parts"]]
        assert len(ids) == len(set(ids)) and spec["bbox_m"] == list(priors[cat].typical_m), cat
