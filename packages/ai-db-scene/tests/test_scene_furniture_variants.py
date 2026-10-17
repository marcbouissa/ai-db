"""Design families for generated furniture: scenes differ, one room matches, edits keep it."""

from __future__ import annotations

import json

import pytest
from ai_db_scene import furniture_variants as fv
from ai_db_scene.priors import load_priors
from ai_db_scene.tools import scene_pipeline as sp

_LIVING = {
    "setting": {"kind": "indoor", "bounds_m": [6, 5, 2.8], "style": "modern"},
    "objects": [{"id": "sofa", "category": "sofa", "hero": True},
                {"id": "chair", "category": "armchair", "count": 2},
                {"id": "table", "category": "coffee_table"}],
    "relations": [{"a": "sofa", "rel": "against_wall", "b": "wall_north"},
                  {"a": "table", "rel": "in_front_of", "b": "sofa", "gap_m": 0.45},
                  {"a": "chair", "rel": "beside", "b": "table"}],
}
_CATS = ("sofa", "armchair", "coffee_table", "side_table", "dining_table", "dining_chair")


def _key(seed: int, style: str | None = "modern") -> list:
    return ["indoor", [6, 5, 2.8], style, [["chair", "armchair"], ["sofa", "sofa"]], fv.variation_marker(seed)]


def _look(p: dict) -> tuple:
    q = p["params"]
    return (q["arm_style"], q["back_style"], q["tufted"], q["leg_style"], q.get("bench_seat"), q["fabric"])


def test_same_key_same_piece_and_copies_match():
    dims = (2.1, 0.95, 0.85)
    assert fv.piece(_key(3), "sofa", dims, "modern") == fv.piece(_key(3), "sofa", dims, "modern")
    # sofa and armchairs of one room: one design family, one upholstery
    s, a = fv.piece(_key(3), "sofa", dims, "modern"), fv.piece(_key(3), "armchair", (0.85, 0.9, 0.9), "modern")
    assert s["design"] == a["design"] and s["params"]["fabric"] == a["params"]["fabric"]


def test_other_seeds_and_scenes_give_other_designs():
    dims = (2.1, 0.95, 0.85)
    looks = {_look(fv.piece(_key(seed, None), "sofa", dims)) for seed in range(20)}
    designs = {fv.piece(_key(seed, None), "sofa", dims)["design"] for seed in range(20)}
    assert len(designs) >= 5 and len(looks) >= 8
    arms = {fv.piece(_key(seed, None), "sofa", dims)["params"]["arm_style"] for seed in range(80)}
    assert arms >= {"track", "rolled", "slope", "metal_square", "skeletal"}
    # one style still varies across scenes (it names several families)
    assert len({fv.piece(_key(seed), "sofa", dims, "modern")["design"] for seed in range(20)}) >= 3


def test_style_maps_to_its_families_and_variant_wins():
    for seed in range(15):
        assert fv.family_for(_key(seed), "classic") in ("chesterfield", "lawson", "glam")
        assert fv.family_for(_key(seed), "Industrial loft") in ("bauhaus", "modern_track", "farmhouse")
        assert fv.family_for(_key(seed), "cozy") in ("lounge", "lawson", "scandinavian", "farmhouse")
        assert fv.family_for(_key(seed), "scandinavian", "chesterfield") == "chesterfield"
    assert set(fv.style_families("zzz")) == set(fv.families())  # an unknown style: any family
    assert fv.coerce_variant("chesterfield") == ("chesterfield", None)
    assert fv.coerce_variant("Mid-Century Modern")[0] == "mid_century"
    assert fv.coerce_variant("boho")[0] == "lounge"
    name, note = fv.coerce_variant("spaceship")
    assert name is None and "names no furniture family" in note


def test_rooms_pick_their_own_family():
    picks = {(fv.family_for(_key(s, None), None, room="a"), fv.family_for(_key(s, None), None, room="b"))
             for s in range(20)}
    assert any(a != b for a, b in picks)


def test_params_stay_inside_generator_ranges_for_every_family_and_prior_size():
    priors = load_priors()
    for name in fv.families():
        for cat in _CATS:
            pr = priors[cat]
            for dims in (pr.min_m, pr.typical_m, pr.max_m):
                for seed in range(4):
                    p = fv.piece(_key(seed), cat, dims, variant=name)
                    assert p["design"] == name and p["family"] == fv.CATEGORY_FAMILY[cat]
                    q = p["params"]
                    if p["family"] in ("sofa", "armchair"):
                        for k, (lo, hi) in fv._UPH_RANGES.items():
                            assert lo <= q[k] <= hi, (name, cat, dims, k, q[k])
                        assert q["arm_height"] <= q["back_height"] and q["seat_height"] + 0.3 <= q["back_height"]
                        n = q.get("seats", 1)
                        arm = 0 if q["arm_style"] == "none" else q["arm_width"]
                        assert n * 0.5 + 2 * arm <= q["width"] + 1e-6, (name, cat, dims, q)
                        if q["arm_style"] == "rolled":
                            assert q["arm_width"] >= 0.15
                    json.dumps(p)  # goes over the socket as is


def test_spec_variant_field_round_trips_and_coerces():
    data = json.loads(json.dumps(_LIVING))
    data["objects"][0]["variant"] = "Chesterfield"
    data["objects"][1]["style"] = "mid-century"
    data["objects"][2]["variant"] = "spaceship"
    spec, errors = sp._parse_spec(data)
    assert spec is not None, errors
    # the armchair pair (count 2) is split into chair_1 / chair_2 by the seating group: both keep it
    assert [o.variant for o in spec.objects] == ["chesterfield", "mid_century", "mid_century", None]
    assert any("spaceship" in n for n in spec.notes)
    out = spec.to_dict()["objects"]
    assert out[0]["variant"] == "chesterfield" and "variant" not in out[3]


def _gen(cmds):
    return {c["params"]["name"]: c["params"] for c in cmds if c["type"] == "generate_asset"}


def test_build_commands_use_families_only_for_marked_keys():
    spec, _ = sp._parse_spec(_LIVING)
    pls = sp.solve_layout(spec)
    legacy = sp.variation_key(spec)
    old, _ = sp.build_layout_commands(spec, pls, variation=legacy)
    assert _gen(old)["sofa"]["params"] == {"width": next(p.dims for p in pls if p.id == "sofa")[0],
                                           "depth": next(p.dims for p in pls if p.id == "sofa")[1]}
    assert not any("fitted" in c["params"].get("code", "") for c in old if c["type"] == "run_code")
    key = [*legacy, fv.variation_marker(0)]
    cmds, _ = sp.build_layout_commands(spec, pls, variation=key)
    gen = _gen(cmds)
    assert gen["chair_1"]["params"] == gen["chair_2"]["params"]  # copies are a matching pair
    assert gen["sofa"]["params"]["arm_style"] and gen["sofa"]["params"]["back_height"] == pytest.approx(
        min(1.05, next(p.dims for p in pls if p.id == "sofa")[2]))
    fit = next(c for c in cmds if c["type"] == "run_code" and "fitted" in c["params"]["code"])
    compile(fit["params"]["code"], "<fit>", "exec")
    assert all(f'"{n}"' in fit["params"]["code"] for n in ("sofa", "chair_1", "table"))
    fin = next(c for c in cmds if c["type"] == "run_code" and "ai_db_id" in c["params"]["code"])
    assert '"design": "' in fin["params"]["code"]
    # no material in the spec: the family's upholstery is kept (no reskin of the sofa)
    reskin = [c for c in cmds if c["type"] == "run_code" and "reskinned" in c["params"]["code"]]
    assert not reskin or '"sofa"' not in reskin[0]["params"]["code"]


def test_build_scene_stores_a_seeded_marked_key_and_edits_pass_it_back(monkeypatch):
    seen = []

    def fake_apply(args):
        seen.append(args["variation"])
        return {"status": "error", "message": "stop here"}

    monkeypatch.setattr(sp, "_handle_apply_layout", fake_apply)
    monkeypatch.setattr(sp, "BlenderClient", lambda *a, **k: (_ for _ in ()).throw(sp.BlenderError("no blender")))
    for seed in (0, 7):
        sp._build_scene({"spec": _LIVING, "seed": seed, "snapshot": False})
    assert fv.uses_families(seen[0]) and seen[0][-1]["seed"] == 0 and seen[1][-1]["seed"] == 7
    stored = seen[0]
    sp._build_scene({"spec": _LIVING, "seed": 0, "snapshot": False, "variation": stored})
    assert seen[-1] == stored  # an edit rebuilds with the stored key: same furniture
    old = sp.variation_key(sp._parse_spec(_LIVING)[0])
    sp._build_scene({"spec": _LIVING, "seed": 0, "snapshot": False, "variation": old})
    assert seen[-1] == old and not fv.uses_families(old)  # a version saved before keeps its furniture


_BBOX = """
import bpy
from mathutils import Vector
root = bpy.data.objects[{name!r}]
bpy.context.view_layer.update()
pts = [o.matrix_world @ Vector(c) for o in [root, *root.children_recursive] if o.type == "MESH" for c in o.bound_box]
result = {{"lo": [min(p[i] for p in pts) for i in range(3)], "hi": [max(p[i] for p in pts) for i in range(3)],
           "parts": sum(1 for o in root.children_recursive if o.type == "MESH")}}
"""


@pytest.mark.blender
def test_live_every_family_builds_at_the_exact_bbox(private_blender):
    from ai_db_scene.blender_client import BlenderClient

    c = BlenderClient(port=private_blender, timeout=300)
    priors = load_priors()
    cmds, checks = [], []
    for i, name in enumerate(fv.families()):
        for j, cat in enumerate(("sofa", "armchair", "coffee_table", "dining_chair")):
            sizes = [priors[cat].typical_m] + ([priors[cat].min_m, priors[cat].max_m] if cat == "sofa" else [])
            for k, dims in enumerate(sizes):
                p = fv.piece(_key(i), cat, dims, variant=name)
                obj = f"v_{name}_{cat}_{k}"
                at = [i * 4.0, j * 3.0 + k * 1.2, 0.0]
                cmds.append({"type": "generate_asset", "params": {
                    "family": p["family"], "params": p["params"], "seed": p["seed"], "name": obj, "location": at}})
                checks.append((obj, at, dims))
    cmds.append(fv.fit_command([{"name": o, "dims": list(d)} for o, _, d in checks]))
    results = c.batch(cmds)
    bad = [r for r in results if isinstance(r, dict) and r.get("status") == "error"]
    assert not bad, bad[:3]
    for obj, at, dims in checks:
        got = c.run_code(_BBOX.format(name=obj))
        size = [h - lo for lo, h in zip(got["lo"], got["hi"])]
        assert size == pytest.approx(list(dims), abs=3e-3), obj
        assert abs(got["lo"][2]) < 1e-3 and got["parts"] >= 3, obj
        centre = [(lo + h) / 2 for lo, h in zip(got["lo"][:2], got["hi"][:2])]
        assert centre == pytest.approx(at[:2], abs=2e-3), obj


@pytest.mark.blender
def test_live_build_scene_gives_one_room_one_family_at_placement_size(private_blender, tmp_path, monkeypatch):
    from ai_db_scene.blender_client import BlenderClient
    from ai_db_scene.memory import SceneMemory
    from ai_db_scene.tools import scene_tools

    client = BlenderClient(port=private_blender, timeout=300)
    memory = SceneMemory("json_file", {"path": str(tmp_path / "mem")})
    memory.set_blender_client(client)
    monkeypatch.setattr(scene_tools, "_scene_memory", memory)
    built = sp._handle_build_scene({"spec": _LIVING, "clear": True, "seed": 2, "exposure_check": False})
    assert built["status"].startswith("built"), built
    from ai_db_scene.constants import BUILD_MAX_REPAIRS
    from ai_db_scene.repair import solve_with_repairs

    fixed = solve_with_repairs(sp._parse_spec(_LIVING)[0].to_dict(), 2, BUILD_MAX_REPAIRS)  # as build_scene does
    pls = {p.id: p for p in fixed.placements or []}
    got = client.run_code(
        "import bpy\nfrom mathutils import Vector\nbpy.context.view_layer.update()\nout = {}\n"
        "for n in ('sofa', 'chair_1', 'chair_2', 'table'):\n"
        "    r = bpy.data.objects[n]\n"
        "    inv = r.matrix_world.inverted()\n"
        "    pts = [o.matrix_world @ Vector(c) for o in [r, *r.children_recursive] if o.type == 'MESH' for c in o.bound_box]\n"
        "    loc = [inv @ p for p in pts]\n"
        "    size = [(max(p[i] for p in loc) - min(p[i] for p in loc)) * r.scale[i] for i in range(3)]\n"
        "    out[n] = {'design': r.get('design'), 'size': size}\n"
        "result = out\n")
    assert got["sofa"]["design"] == got["chair_1"]["design"] == got["chair_2"]["design"] == got["table"]["design"]
    assert got["sofa"]["design"] in fv.style_families("modern")
    assert {"sofa", "chair_1", "table"} <= set(pls)
    for n in ("sofa", "chair_1", "table"):
        assert got[n]["size"] == pytest.approx(list(pls[n].dims), abs=3e-3), n
