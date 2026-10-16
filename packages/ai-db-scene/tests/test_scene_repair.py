"""build_scene's automatic spec repair and fix loop."""

from __future__ import annotations

import pytest
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
    err = LayoutError("cannot place 'chair_3' (...); main blockers: 40x out of bounds", ["sofa"])
    data, note = relax(SPEC, err, [4.0, 3.0, 2.6])
    assert data["setting"]["bounds_m"] == [4.8, 3.6, 2.6] and "enlarged" in note
    big = {**SPEC, "setting": {"kind": "indoor", "bounds_m": [7.2, 5.4, 2.6]}}
    data, note = relax(big, err, [4.0, 3.0, 2.6])  # already 3.2x the area: no more growth
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
    assert [r for r in sent if r[0] != "run_code"] == [("modify_object", {"name": "lamp", "location": [0, 0, 0]})]
    assert any(r[0] == "run_code" and "factory_cube" in r[1]["code"] for r in sent)  # leftovers tidied first
    assert out["fixes_applied"] == ["floating: lamp floats"]
    assert [i for i in out["issues_left"] if i["check"] not in ("ergonomics", "realism")] == []  # advice may remain


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
    assert errors[0].startswith("steps[1]: no object 'x'") and len(errors) == 1
    assert apply_steps(SPEC, [{"op": "explode"}])[2][0].startswith("steps[0]: op 'explode'")


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


_NEW_OUTDOOR = ("playground_slide", "swing_set", "gazebo", "planter", "hedge", "signpost", "bollard",
                "parking_bay", "loading_dock", "shipping_container", "traffic_cone", "lamp_post_double")
_TOUCH = 0.006  # relative units: parts this close count as touching


def _part_box(part):
    """Axis-aligned bounds of a recipe part, as build_asset places it: size taken before
    rotation_deg (Euler XYZ), then the rotated bounds' bottom centre put at ``at``."""
    import itertools
    import math

    sx, sy, sz = part["size"]
    rx, ry, rz = (math.radians(a) for a in part.get("rotation_deg", (0, 0, 0)))

    def rot(p):
        x, y, z = p
        y, z = y * math.cos(rx) - z * math.sin(rx), y * math.sin(rx) + z * math.cos(rx)
        x, z = x * math.cos(ry) + z * math.sin(ry), -x * math.sin(ry) + z * math.cos(ry)
        x, y = x * math.cos(rz) - y * math.sin(rz), x * math.sin(rz) + y * math.cos(rz)
        return x, y, z

    pts = [rot(c) for c in itertools.product((-sx / 2, sx / 2), (-sy / 2, sy / 2), (-sz / 2, sz / 2))]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    ax, ay, az = part["at"]
    off = (ax - (lo[0] + hi[0]) / 2, ay - (lo[1] + hi[1]) / 2, az - lo[2])
    return [lo[i] + off[i] for i in range(3)], [hi[i] + off[i] for i in range(3)]


def _floating_parts(parts):
    """Parts not connected to the ground through a chain of touching parts."""
    boxes = {p["id"]: _part_box(p) for p in parts}
    ground = min(lo[2] for lo, _hi in boxes.values())
    reached = {pid for pid, (lo, _hi) in boxes.items() if lo[2] <= ground + _TOUCH}
    grew = True
    while grew:
        grew = False
        for pid, (lo, hi) in boxes.items():
            if pid in reached:
                continue
            if any(all(lo[i] <= boxes[q][1][i] + _TOUCH and boxes[q][0][i] <= hi[i] + _TOUCH for i in range(3))
                   for q in reached):
                reached.add(pid)
                grew = True
    return sorted(set(boxes) - reached)


def test_outdoor_recipes_have_no_floating_parts_and_plausible_priors():
    from ai_db_scene.materials import CATEGORY_MATERIALS
    from ai_db_scene.priors import load_priors
    from ai_db_scene.recipes import has_recipe, recipe_spec
    priors = load_priors()
    for cat in _NEW_OUTDOOR:
        assert has_recipe(cat) and cat in priors and cat in CATEGORY_MATERIALS, cat
        assert priors[cat].support == "ground", cat
        spec = recipe_spec(cat, cat, priors[cat].typical_m)
        assert _floating_parts(spec["parts"]) == [], cat
    assert priors["parking_bay"].typical_m[2] <= 0.03  # flat: other objects (cars) may stand on it
    assert priors["loading_dock"].can_support_others and priors["shipping_container"].can_support_others


def test_floating_part_check_catches_a_gap():
    parts = [{"id": "a", "shape": "box", "size": [1, 1, 0.5], "at": [0, 0, 0]},
             {"id": "b", "shape": "box", "size": [1, 1, 0.2], "at": [0, 0, 0.7]}]
    assert _floating_parts(parts) == ["b"]
    parts[1]["at"] = [0, 0, 0.5]
    assert _floating_parts(parts) == []
    tilted = {"id": "t", "shape": "box", "size": [0.1, 1.0, 0.1], "at": [0, 0, 0], "rotation_deg": [90, 0, 0]}
    lo, hi = _part_box(tilted)
    assert abs((hi[2] - lo[2]) - 1.0) < 1e-9 and abs(lo[2]) < 1e-9  # stood upright, base on the ground


_BBOX = """
import bpy
from mathutils import Vector
root = bpy.data.objects[{name!r}]
pts = [o.matrix_world @ Vector(c) for o in [root, *root.children_recursive] if o.type == "MESH" for c in o.bound_box]
result = {{"size": [max(p[i] for p in pts) - min(p[i] for p in pts) for i in range(3)],
           "min_z": min(p[2] for p in pts), "parts": sum(1 for o in root.children_recursive if o.type == "MESH")}}
"""


@pytest.mark.blender
def test_live_outdoor_recipes_build_with_exact_bbox(private_blender):
    from ai_db_scene.blender_client import BlenderClient
    from ai_db_scene.priors import load_priors
    from ai_db_scene.recipes import recipe_spec

    c = BlenderClient(port=private_blender, timeout=120)
    priors = load_priors()
    for cat in _NEW_OUTDOOR:
        dims = priors[cat].typical_m
        c.request("build_asset", {"spec": recipe_spec(cat, cat, dims)})
        got = c.run_code(_BBOX.format(name=cat))
        assert got["size"] == pytest.approx(list(dims), abs=2e-3), cat
        assert abs(got["min_z"]) < 1e-3 and got["parts"] >= 2, cat


def test_every_recipe_has_no_floating_parts():
    from ai_db_scene.priors import load_priors
    from ai_db_scene.recipes import recipe_categories, recipe_spec, variant_categories

    priors = load_priors()
    for cat in recipe_categories():
        dims = priors[cat].typical_m if cat in priors else (1.0, 1.0, 1.0)
        assert _floating_parts(recipe_spec(cat, cat, dims)["parts"]) == [], cat
        if cat in variant_categories():
            for seed in range(25):
                assert _floating_parts(recipe_spec(cat, cat, dims, seed=seed)["parts"]) == [], (cat, seed)


def _extent(parts):
    boxes = [_part_box(p) for p in parts]
    return [min(b[0][i] for b in boxes) for i in range(3)], [max(b[1][i] for b in boxes) for i in range(3)]


def test_seeded_recipe_variants_stay_connected_and_deterministic():
    from ai_db_scene.priors import load_priors
    from ai_db_scene.recipes import recipe_spec, variant_categories

    priors = load_priors()
    assert set(variant_categories()) >= {"dresser", "cabinet", "stool", "office_chair", "shelving_unit"}
    for cat in variant_categories():
        dims = priors[cat].typical_m
        assert recipe_spec(cat, "x", dims) == recipe_spec(cat, "x", dims, seed=None)  # unchanged default
        seen = set()
        for seed in range(25):
            spec = recipe_spec(cat, "x", dims, seed=seed)
            parts = spec["parts"]
            ids = [p["id"] for p in parts]
            assert len(ids) == len(set(ids)) and spec["bbox_m"] == list(dims), (cat, seed)
            assert _floating_parts(parts) == [], (cat, seed)
            lo, hi = _extent(parts)
            assert lo[2] >= -1e-9 and hi[2] <= 1.0 + 1e-6 and all(-0.52 <= v <= 0.52 for v in lo[:2] + hi[:2]), (cat, seed)
            assert recipe_spec(cat, "x", dims, seed=seed) == spec  # same seed, same parts
            seen.add(repr(parts))
        assert len(seen) >= 3, cat


@pytest.mark.blender
def test_live_every_bevelled_recipe_and_soft_shapes_build_exact(private_blender):
    """Rounded edges (soften_edges) and soft-goods shapes keep each asset at its exact bbox."""
    from ai_db_scene.blender_client import BlenderClient
    from ai_db_scene.priors import load_priors
    from ai_db_scene.recipes import recipe_categories, recipe_spec, soften_edges

    c = BlenderClient(port=private_blender, timeout=120)
    priors = load_priors()
    for cat in recipe_categories():
        dims = priors[cat].typical_m if cat in priors else (1.0, 1.0, 1.0)
        c.request("build_asset", {"spec": recipe_spec(cat, f"b_{cat}", dims)})
        got = c.run_code(_BBOX.format(name=f"b_{cat}"))
        assert got["size"] == pytest.approx(list(dims), abs=3e-3), cat
    soft = [{"id": "base", "shape": "rounded_slab", "size": [1.0, 1.0, 0.45], "at": [0, 0, 0], "corner_radius": 0.05,
             "material": "fabric:linen"},
            {"id": "seat", "shape": "cushion", "size": [0.96, 0.96, 0.2], "at": [0, 0, 0.45], "crown": 0.02,
             "material": "fabric:linen"},
            {"id": "back", "shape": "tufted_panel", "size": [1.0, 0.2, 0.35], "at": [0, 0.4, 0.65],
             "material": "velvet"}]
    dims = (0.9, 0.85, 1.0)
    c.request("build_asset", {"spec": {"asset": "soft", "bbox_m": list(dims), "parts": soften_edges(soft, dims)}})
    got = c.run_code(_BBOX.format(name="soft"))
    assert got["size"] == pytest.approx(list(dims), abs=3e-3) and got["parts"] >= 3


@pytest.mark.blender
def test_live_rebuild_without_clear_leaves_no_leftovers(private_blender, tmp_path, monkeypatch):
    """No factory cube, no second room shell, no stale copies after building twice without clear."""
    from ai_db_scene.blender_client import BlenderClient
    from ai_db_scene.memory import SceneMemory
    from ai_db_scene.tools import scene_pipeline as sp
    from ai_db_scene.tools import scene_tools

    client = BlenderClient(port=private_blender, timeout=120)
    memory = SceneMemory("json_file", {"path": str(tmp_path / "mem")})
    memory.set_blender_client(client)
    monkeypatch.setattr(scene_tools, "_scene_memory", memory)
    assert "Cube" in client.run_code("import bpy\nresult = [o.name for o in bpy.context.scene.objects]")  # factory scene
    for _ in range(2):
        out = sp._handle_build_scene({"spec": SPEC, "snapshot": False, "exposure_check": False})
        assert out["status"].startswith("built"), out
    names = client.run_code("import bpy\nresult = sorted(o.name for o in bpy.context.scene.objects)")
    assert "Cube" not in names and not any(n.endswith((".001", "_stale")) for n in names), names
    assert not [i for i in out["issues_left"] if i.get("check") == "interpenetration"]


@pytest.mark.blender
def test_live_new_build_replaces_previous_ai_db_scene_keeps_user_objects(private_blender, tmp_path, monkeypatch):
    from ai_db_scene.blender_client import BlenderClient
    from ai_db_scene.memory import SceneMemory
    from ai_db_scene.tools import scene_pipeline as sp
    from ai_db_scene.tools import scene_tools

    client = BlenderClient(port=private_blender, timeout=120)
    memory = SceneMemory("json_file", {"path": str(tmp_path / "mem")})
    memory.set_blender_client(client)
    monkeypatch.setattr(scene_tools, "_scene_memory", memory)
    client.run_code("import bpy\nbpy.ops.mesh.primitive_uv_sphere_add(location=(30, 30, 1))\n"
                    "bpy.context.object.name = 'my_sculpture'\nresult = 1")
    other = {"setting": {"kind": "indoor", "bounds_m": [6.0, 5.0, 2.7]},
             "objects": [{"id": "old_desk", "category": "desk"}], "relations": []}
    sp._handle_build_scene({"spec": other, "snapshot": False, "exposure_check": False})
    sp._handle_build_scene({"spec": SPEC, "snapshot": False, "exposure_check": False})
    names = client.run_code("import bpy\nresult = sorted(o.name for o in bpy.context.scene.objects)")
    assert "my_sculpture" in names and not any(n.startswith("old_desk") for n in names), names
