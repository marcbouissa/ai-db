"""Outdoor terrain: sand/grass ground, a water body with a sloping shore, beach vegetation and
furniture (terrain.py, vegetation.py; bench brief beach_cove, 2026-10-17).

The weak model's own beach_cove specs (qwen3-14b aidb6 transcript) are the inputs here: an object
of category ``sandy_beach`` / ``ground`` / ``water``, ``floor_material: sand``, palms as ``tree``,
``at_edge_of water`` / ``connects beach`` relations.
"""

from __future__ import annotations

import itertools
import math

import pytest
from ai_db_scene import terrain
from ai_db_scene.layout import solve_layout
from ai_db_scene.materials import CATEGORY_MATERIALS, coerce_material, material_error
from ai_db_scene.priors import load_priors
from ai_db_scene.realism import check_brief
from ai_db_scene.recipes import has_recipe, recipe_spec
from ai_db_scene.room_kits import kits_for
from ai_db_scene.spec import SceneSpec, validate_spec
from ai_db_scene.tools import scene_pipeline as sp

BRIEF = ("A tropical beach cove outdoors: a sandy beach sloping into calm turquoise water, five palm trees, "
         "dense green bushes and ferns along the back, a wooden boardwalk from the bushes to the water, two sun "
         "loungers under a parasol, and a few rocks at the water's edge.")
# qwen3-14b-local-aidb6 beach_cove, its second build_scene call (verbatim spec)
QWEN_2 = {"setting": {"kind": "outdoor", "bounds_m": [20, 15, 5], "floor_material": "sand", "style": "tropical"},
          "objects": [{"id": "beach", "category": "ground", "material": "sand", "hero": True},
                      {"id": "water", "category": "water", "material": "turquoise", "hero": True},
                      {"id": "palm_trees", "category": "tree", "material": "palm_leaf", "count": 5},
                      {"id": "bushes", "category": "bush", "material": "leaf", "count": 3},
                      {"id": "boardwalk", "category": "wooden_boardwalk", "material": "wood", "count": 1},
                      {"id": "loungers", "category": "sun_lounger", "material": "fabric", "count": 2},
                      {"id": "parasol", "category": "parasol", "material": "canvas", "count": 1},
                      {"id": "rocks", "category": "rock", "material": "stone", "count": 4}],
          "relations": [{"a": "boardwalk", "rel": "connects", "b": "beach"},
                        {"a": "boardwalk", "rel": "leads_to", "b": "water"},
                        {"a": "loungers", "rel": "under", "b": "parasol"},
                        {"a": "bushes", "rel": "back_of", "b": "beach"},
                        {"a": "rocks", "rel": "at_edge_of", "b": "water"}]}
# its first call: category names it guessed
QWEN_1_OBJECTS = [{"id": "beach", "category": "sandy_beach", "material": "sand", "hero": True},
                  {"id": "water", "category": "turquoise_water", "material": "turquoise", "hero": True},
                  {"id": "palm1", "category": "palm_tree", "material": "wood", "count": 5},
                  {"id": "bushes", "category": "dense_green_bushes", "material": "leaf", "count": 3},
                  {"id": "loungers", "category": "sun_lounger", "material": "fabric", "count": 2},
                  {"id": "parasol", "category": "parasol", "material": "canvas", "count": 1}]
NEW = ("palm_tree", "fern", "tropical_bush", "boardwalk", "sun_lounger", "parasol")


def _spec(raw):
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    return spec


def _beach(bounds=(24.0, 18.0, 5.0)):
    return {"setting": {"kind": "outdoor", "bounds_m": list(bounds), "ground": "sand",
                        "water": {"kind": "sea", "side": "north"}},
            "objects": [{"id": "palms", "category": "palm_tree", "count": 3},
                        {"id": "bushes", "category": "tropical_bush", "count": 3},
                        {"id": "boardwalk", "category": "boardwalk"},
                        {"id": "loungers", "category": "sun_lounger", "count": 2},
                        {"id": "parasol", "category": "parasol"},
                        {"id": "rocks", "category": "rock", "count": 3}],
            "relations": [{"a": "boardwalk", "rel": "leads_to", "b": "water"},
                          {"a": "loungers", "rel": "under", "b": "parasol"},
                          {"a": "bushes", "rel": "back_of", "b": "water"},
                          {"a": "rocks", "rel": "at_edge_of", "b": "water"}]}


# --- what the model wrote ---------------------------------------------------------------------

def test_the_models_beach_spec_becomes_ground_water_and_water_relations():
    spec = _spec(QWEN_2)
    st = spec.setting
    assert st.ground == "sand" and st.floor_material is None
    assert st.water is not None and st.water.kind == "sea" and st.water.side == "north"
    cats = {o.id: o.category for o in spec.objects}
    assert "beach" not in cats and "water" not in cats  # the ground and the water are the setting
    assert cats["palm_trees"] == "palm_tree" and cats["bushes"] == "tropical_bush"  # a tropical style
    assert cats["boardwalk"] == "boardwalk"
    rels = {(r.a, r.rel, r.b) for r in spec.relations}
    assert {("boardwalk", "leads_to", "water"), ("bushes", "back_of", "water"),
            ("rocks", "at_edge_of", "water"), ("loungers", "under", "parasol")} <= rels
    assert sum(1 for r in spec.relations if r.a == "boardwalk") == 1  # connects + leads_to: one relation
    assert any("setting.water" in n for n in spec.notes) and any("ground" in n for n in spec.notes)


def test_guessed_category_names_validate():
    spec = _spec({"setting": {"kind": "outdoor", "bounds_m": [20, 15, 5]}, "objects": QWEN_1_OBJECTS,
                  "relations": [{"a": "loungers", "rel": "under", "b": "parasol"}]})
    assert spec.setting.ground == "sand" and spec.setting.water is not None
    assert {o.category for o in spec.objects} == {"palm_tree", "tropical_bush", "sun_lounger", "parasol"}


def test_a_relation_to_the_water_edge_adds_the_water_the_spec_forgot():
    # qwen aidb6 beach_cove (terrain build): floor_material sand, no water, 'rock1 at water_edge'
    spec = _spec({"setting": {"kind": "outdoor", "floor_material": "sand", "bounds_m": [20, 20, 5], "style": "tropical"},
                  "objects": [{"id": "boardwalk1", "category": "wooden_boardwalk"}, {"id": "rock1", "category": "rock", "count": 4}],
                  "relations": [{"a": "boardwalk1", "rel": "connects_to", "b": "water_edge"},
                                {"a": "rock1", "rel": "at", "b": "water_edge"}]})
    assert spec.setting.water is not None and spec.setting.ground == "sand"
    assert {(r.a, r.rel, r.b) for r in spec.relations} == {("boardwalk1", "leads_to", "water"),
                                                           ("rock1", "at_edge_of", "water")}


def test_a_boardwalk_with_no_relation_leads_to_the_water():
    raw = _beach()
    raw["relations"] = [r for r in raw["relations"] if r["a"] != "boardwalk"]
    spec = _spec(raw)
    assert terrain.Terrain(spec).rels["boardwalk"] == ["leads_to"]
    raw["relations"].append({"a": "boardwalk", "rel": "beside", "b": "bushes"})
    assert "boardwalk" not in terrain.Terrain(_spec(raw)).rels  # its own relation places it


def test_floor_material_sand_is_sand_ground_not_paint():
    spec = _spec({"setting": {"kind": "outdoor", "bounds_m": [10, 10, 4], "floor_material": "sand"},
                  "objects": [{"id": "r", "category": "rock"}]})
    assert spec.setting.ground == "sand" and spec.setting.floor_material is None
    lawn = _spec({"setting": {"kind": "outdoor", "bounds_m": [10, 10, 4], "floor_material": "lawn"},
                  "objects": [{"id": "r", "category": "rock"}]})
    assert lawn.setting.ground == "grass"
    indoor = _spec({"setting": {"kind": "indoor", "bounds_m": [5, 4, 3], "floor_material": "sand"},
                    "objects": [{"id": "r", "category": "rock"}]})
    assert indoor.setting.ground is None  # indoors a sand floor stays a floor material


def test_water_forms_and_refusals():
    assert terrain.parse_water("lagoon", (10, 10, 4), []) == terrain.Water("lagoon", "north", 0.3, None)
    assert terrain.parse_water("east", (10, 10, 4), []) == terrain.Water("sea", "east", 0.3, None)
    pond = terrain.parse_water({"kind": "pond"}, (20, 10, 4), [])
    assert pond is not None and pond.side is None and pond.region == (10.0, 5.0, 4.4, 2.2)
    errors: list[str] = []
    assert terrain.parse_water({"kind": "sea", "side": "up"}, (10, 10, 4), errors) is None and errors
    _s, errs = validate_spec({"setting": {"kind": "indoor", "bounds_m": [5, 4, 3], "water": "pond"},
                              "objects": [{"id": "r", "category": "rock"}]})
    assert any("only for setting.kind 'outdoor'" in e for e in errs)
    _s, errs = validate_spec({"setting": {"kind": "indoor", "bounds_m": [9, 9, 4]},
                              "objects": [{"id": "r", "category": "rock"}],
                              "relations": [{"a": "r", "rel": "at_edge_of", "b": "water"}]})
    assert any("needs setting.water" in e for e in errs)  # outdoors the water would be added
    _s, errs = validate_spec({"setting": {"kind": "outdoor", "bounds_m": [9, 9, 4]},
                              "objects": [{"id": "r", "category": "rock"}, {"id": "b", "category": "bench"}],
                              "relations": [{"a": "r", "rel": "back_of", "b": "b"}]})
    assert any("only relates an object to the water" in e for e in errs)


def test_round_trip_keeps_ground_and_water():
    spec = _spec(_beach())
    again = SceneSpec.from_dict(spec.to_dict())
    assert again.setting.ground == "sand" and again.setting.water == spec.setting.water
    assert {(r.a, r.rel, r.b) for r in again.relations} == {(r.a, r.rel, r.b) for r in spec.relations}


def test_schema_offers_ground_water_and_water_relations():
    from ai_db_scene.spec import spec_schema

    sch = spec_schema()
    st = sch["properties"]["setting"]["properties"]
    assert st["ground"]["enum"] == list(terrain.GROUND_KINDS) and "kind" in st["water"]["properties"]
    rels = sch["properties"]["relations"]["items"]["properties"]["rel"]["enum"]
    assert {"at_edge_of", "leads_to", "back_of", "near"} <= set(rels)
    cats = sch["properties"]["objects"]["items"]["properties"]["category"]["enum"]
    assert set(NEW) <= set(cats)


# --- the shore --------------------------------------------------------------------------------

def test_shore_profile_flat_land_slope_into_deepening_water_hills_beyond():
    ter = terrain.Terrain(_spec(_beach((20.0, 15.0, 5.0))))
    x = 10.0
    assert ter.signed(x, 0.5) < ter.dry() < 0 < ter.signed(x, 14.5)
    assert ter.height(x, 0.5) == pytest.approx(0.0)  # where the scene stands: flat
    ys = [i * 0.25 for i in range(61)]
    zs = [ter.height(x, y) for y in ys]
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(zs))  # never rises toward the water
    shore = next(y for y in ys if ter.signed(x, y) >= 0)
    assert ter.height(x, shore) == pytest.approx(ter.level, abs=0.03)  # the waterline is at water level
    assert ter.height(x, 14.5) < ter.level - 0.4  # deeper offshore
    assert ter.height(x, -15.0) > 0.5  # dunes / hills behind the beach
    # a cove: the water reaches further in at the middle than at the sides
    assert ter.signed(10.0, 10.0) > ter.signed(1.0, 10.0)
    nx, ny = ter.to_water(10.0, 5.0)
    assert ny > 0.95 and abs(nx) < 0.2


def test_pond_is_inland_and_grass_without_water_has_hills_only():
    spec = _spec({"setting": {"kind": "outdoor", "bounds_m": [20, 16, 4], "ground": "grass", "water": {"kind": "pond"}},
                  "objects": [{"id": "r", "category": "rock"}]})
    ter = terrain.Terrain(spec)
    assert ter.signed(10, 8) > 0 > ter.signed(1, 1) and ter.height(10, 8) < ter.level
    plain = terrain.Terrain(_spec({"setting": {"kind": "outdoor", "bounds_m": [20, 16, 4], "ground": "grass"},
                                   "objects": [{"id": "r", "category": "rock"}]}))
    assert plain.height(10, 8) == 0.0 and plain.height(-20, 8) > 0.5


def test_layout_keeps_land_dry_and_follows_the_water_relations():
    spec = _spec(_beach())
    ter = terrain.Terrain(spec)
    pls = {p.id: p for p in solve_layout(spec, seed=3)}
    for p in pls.values():
        x, y, _z = p.location
        yaw = math.radians(p.rotation_deg_z)
        f = (math.sin(yaw), -math.cos(yaw))
        n = ter.to_water(x, y)
        if p.category == "rock":
            assert terrain._EDGE_BAND_M[0] <= ter.signed(x, y) <= terrain._EDGE_BAND_M[1], p
        elif p.category == "boardwalk":
            front = (x + f[0] * p.dims[1] / 2, y + f[1] * p.dims[1] / 2)
            assert ter.signed(*front) > 0.5 and f[0] * n[0] + f[1] * n[1] > 0.9, p  # out over the water
        else:
            assert all(ter.signed(cx, cy) <= ter.dry() + 1e-6 for cx, cy in ter._corners(x, y, yaw, p.dims)), p
        if p.category == "sun_lounger":
            assert f[0] * n[0] + f[1] * n[1] > 0.7, p  # facing the water
            par = pls["parasol"]
            assert abs(x - par.location[0]) <= par.dims[0] / 2 and abs(y - par.location[1]) <= par.dims[1] / 2
    backs = [ter.signed(*p.location[:2]) for p in pls.values() if p.category == "tropical_bush"]
    palms = [ter.signed(*p.location[:2]) for p in pls.values() if p.category == "palm_tree"]
    assert max(backs) < sum(palms) / len(palms)  # the bushes are along the back


def test_settle_beds_rocks_on_the_slope_and_runs_the_boardwalk_back_to_the_bushes():
    spec = _spec(_beach())
    pls = solve_layout(spec, seed=3)
    out = {p.id: p for p in terrain.settle(spec, pls)}
    ter = terrain.Terrain(spec)
    for p in out.values():
        if p.category == "rock":
            zb = p.location[2] - p.dims[2] / 2
            ground = ter.ground_under(p.location[0], p.location[1], math.radians(p.rotation_deg_z), p.dims)
            assert zb == pytest.approx(ground - 0.15 * p.dims[2], abs=1e-4) and zb < 0
    before = next(p for p in pls if p.category == "boardwalk")
    after = out["boardwalk"]
    assert after.dims[1] >= before.dims[1]
    yaw = math.radians(after.rotation_deg_z)
    f = (math.sin(yaw), -math.cos(yaw))
    def tip(p):
        return (p.location[0] + f[0] * p.dims[1] / 2, p.location[1] + f[1] * p.dims[1] / 2)
    assert tip(after) == pytest.approx(tip(before), abs=1e-3)  # the water end stays; it grew landward
    back = (after.location[0] - f[0] * after.dims[1] / 2, after.location[1] - f[1] * after.dims[1] / 2)
    assert 0 <= back[0] <= 24 and 0 <= back[1] <= 18
    others = [p for p in out.values() if p.id != "boardwalk" and p.dims[2] > 0.05]
    assert not any(terrain._in_box(back[0], back[1], p, 0.0) for p in others)


# --- brief words, kits, build commands ----------------------------------------------------------

def test_brief_words_pick_ground_and_water():
    assert terrain.brief_terrain(BRIEF) == ("sand", {"kind": "lagoon", "side": "north"})
    assert terrain.brief_terrain("a beach with a bar") == ("sand", {"kind": "sea", "side": "north"})
    assert terrain.brief_terrain("a desert oasis") == ("sand", {"kind": "pond"})
    assert terrain.brief_terrain("koi pond garden") == ("grass", {"kind": "pond"})
    assert terrain.brief_terrain("a lake with a jetty") == ("grass", {"kind": "lake", "side": "north"})
    assert terrain.brief_terrain("a tropical garden") == ("grass", None)
    assert terrain.brief_terrain("a cozy living room") == (None, None)
    bare = _spec({"setting": {"kind": "outdoor", "bounds_m": [20, 15, 5]}, "objects": [{"id": "r", "category": "rock"}]})
    got, notes = terrain.apply_brief_defaults(bare, "a sunny beach")
    assert got.setting.ground == "sand" and got.setting.water is not None and len(notes) == 2
    indoor = _spec({"setting": {"kind": "indoor", "bounds_m": [5, 4, 3]}, "objects": [{"id": "r", "category": "rock"}]})
    assert terrain.apply_brief_defaults(indoor, "a beach house")[0] is indoor


def test_kits_for_beach_tropical_oasis_pond_carry_the_setting():
    kits = {k["room"]: k for k in kits_for(BRIEF)}
    assert {"beach", "tropical"} <= set(kits)
    assert kits["beach"]["setting"] == {"kind": "outdoor", "ground": "sand", "water": {"kind": "lagoon", "side": "north"}}
    beach_cats = {i["category"] for i in kits["beach"]["typical"]}
    assert {"palm_tree", "sun_lounger", "parasol", "boardwalk", "rock"} <= beach_cats
    assert kits_for("an oasis in the desert")[0]["setting"]["water"] == {"kind": "pond"}
    assert any(k["room"] == "pond" for k in kits_for("a lakeside jetty"))
    for k in kits_for(BRIEF) + kits_for("an oasis") + kits_for("a pond"):
        assert all(i["category"] in load_priors() for i in k["typical"]), k["room"]


def test_build_commands_terrain_plants_boardwalk_instead_of_flat_floor():
    spec = _spec(_beach())
    pls = solve_layout(spec, seed=3)
    cmds, names = sp.build_layout_commands(spec, pls)
    codes = [c["params"].get("code", "") for c in cmds if c["type"] == "run_code"]
    assert any("ai_db_terrain" in c and "sand_ground" in c and "sea_water" in c for c in codes)
    assert not any(c["params"].get("name") in ("floor", "ground_beyond") for c in cmds)
    assert names[:2] == ["sand_ground", "sea_water"]
    assert sum('"kind": "palm"' in c for c in codes) == 3 and sum('"kind": "tropical_bush"' in c for c in codes) == 3
    assert any("_BOARDWALK" in c or '"posts"' in c for c in codes)
    applied = {c["params"]["name"] for c in cmds if c["type"] == "apply_material"}
    assert {"palms_1_trunk", "palms_1_leaves", "boardwalk_deck", "boardwalk_posts"} <= applied
    rocks = [c["params"]["code"] for c in cmds if c["type"] == "run_code" and "convex_hull" in c["params"]["code"]]
    assert len(set(rocks)) == 3  # every rock its own shape (rocks.py: a faceted boulder, seeded by its id)
    # a plain outdoor setting keeps its flat floor and the ground to the horizon
    park = _spec({"setting": {"kind": "outdoor", "bounds_m": [10, 10, 4]}, "objects": [{"id": "r", "category": "rock"}]})
    cmds, _ = sp.build_layout_commands(park, solve_layout(park))
    assert {"floor", "ground_beyond"} <= {c["params"].get("name") for c in cmds}


def test_boardwalk_posts_reach_the_sea_floor():
    spec = _spec(_beach())
    pls = terrain.settle(spec, solve_layout(spec, seed=3))
    bw = next(p for p in pls if p.category == "boardwalk")
    cmd = terrain.boardwalk_commands(spec, bw, (-12.0, -9.0, 0.0), 1)[0]["params"]["code"]
    import ast
    import json
    import re

    payload = json.loads(ast.literal_eval(re.search(r"json\.loads\((.*)\)\n", cmd).group(1)))
    bottoms = [b for _y, b in payload["posts"]]
    assert min(bottoms) < -0.5 and max(bottoms) <= 0.0  # posts go deeper out over the water


# --- vocabulary --------------------------------------------------------------------------------

def test_new_categories_have_priors_materials_and_builders():
    priors = load_priors()
    from ai_db_scene.vegetation import has_plant

    for cat in NEW:
        assert cat in priors and priors[cat].support == "ground", cat
        assert material_error(CATEGORY_MATERIALS[cat]) is None, cat
        assert has_plant(cat) or has_recipe(cat) or cat in terrain.OVER_WATER, cat
    for cat in ("sun_lounger", "parasol"):
        spec = recipe_spec(cat, cat, priors[cat].typical_m)
        assert spec["bbox_m"] == list(priors[cat].typical_m)
        assert all(material_error(p["material"].split(":scale=")[0]) is None for p in spec["parts"]), cat
    assert priors["sun_lounger"].typical_m[2] < priors["parasol"].typical_m[2]  # fits under it
    assert 2 * priors["sun_lounger"].typical_m[0] < priors["parasol"].typical_m[0]  # two side by side


def test_material_and_colour_words_of_the_beach():
    assert coerce_material("palm_leaf")[0] == "foliage:#4f7d2e"
    assert coerce_material("leaf")[0] == "foliage"
    assert coerce_material("fabric:turquoise")[0] == "fabric:#2fb8b0"
    assert coerce_material("rattan")[0].startswith("wood:pine")
    assert coerce_material("teak")[0] == "wood:walnut"


def test_category_aliases_for_beach_objects():
    raw = {"setting": {"kind": "outdoor", "bounds_m": [20, 15, 5]},
           "objects": [{"id": "u", "category": "beach_umbrella"}, {"id": "l", "category": "deckchair"},
                       {"id": "j", "category": "jetty"}, {"id": "f", "category": "ferns"},
                       {"id": "p", "category": "coconut_palm"}]}
    assert [o.category for o in _spec(raw).objects] == ["parasol", "sun_lounger", "boardwalk", "fern", "palm_tree"]


def test_brief_check_counts_specific_kinds():
    from ai_db_scene.layout import Placement

    pls = [Placement(f"b_{i}", (0, 0, 0), 0, (1, 1, 1), "tropical_bush") for i in range(3)]
    pls += [Placement(f"p_{i}", (0, 0, 0), 0, (1, 1, 1), "palm_tree") for i in range(5)]
    msgs = [i["message"] for i in check_brief("five trees and dense bushes", pls)]
    assert not any("bush" in m or "tree" in m for m in msgs), msgs


# --- live ------------------------------------------------------------------------------------

_BBOX = """
import bpy
from mathutils import Vector
root = bpy.data.objects[{name!r}]
pts = [o.matrix_world @ Vector(c) for o in [root, *root.children_recursive] if o.type == "MESH" for c in o.bound_box]
result = {{"lo": [min(p[i] for p in pts) for i in range(3)], "hi": [max(p[i] for p in pts) for i in range(3)],
           "parts": sum(1 for o in root.children_recursive if o.type == "MESH")}}
"""


@pytest.mark.blender
def test_live_beach_assets_build_at_the_exact_bbox(private_blender):
    from ai_db_scene.blender_client import BlenderClient
    from ai_db_scene.layout import Placement
    from ai_db_scene.vegetation import plant_commands

    c = BlenderClient(port=private_blender, timeout=300)
    priors = load_priors()
    flat = _spec({"setting": {"kind": "outdoor", "bounds_m": [30, 30, 5]}, "objects": [{"id": "b", "category": "boardwalk"}]})
    checks = []
    x = 0.0
    for cat in ("palm_tree", "fern", "tropical_bush"):
        for k, dims in enumerate((priors[cat].typical_m, priors[cat].min_m)):
            name = f"t_{cat}_{k}"
            for cmd in plant_commands(cat, name, dims, [x, 0.0, 0.0], 30.0, k, None):
                c.request(cmd["type"], cmd["params"])
            checks.append((name, [x, 0.0], dims, 0.0))
            x += 12.0
    for cat in ("sun_lounger", "parasol", "rock"):
        dims = priors[cat].typical_m
        c.request("build_asset", {"spec": recipe_spec(cat, f"t_{cat}", dims, seed=1 if cat == "rock" else None)})
        c.request("modify_object", {"name": f"t_{cat}", "location": [x, 0.0, 0.0]})
        checks.append((f"t_{cat}", [x, 0.0], dims, 0.0))
        x += 6.0
    bw = Placement("t_boardwalk", (x, 0.0, 0.225), 0.0, (1.6, 8.0, 0.45), "boardwalk")
    for cmd in terrain.boardwalk_commands(flat, bw, (0.0, 0.0, 0.0), 4):
        c.request(cmd["type"], cmd["params"])
    checks.append(("t_boardwalk", [x, 0.0], (1.6, 8.0, 0.45), 0.01))
    sanity = c.run_code(
        "import bpy, bmesh\nout = {}\nfor o in bpy.context.scene.objects:\n    if o.type != 'MESH': continue\n"
        "    bm = bmesh.new(); bm.from_mesh(o.data)\n"
        "    bad = (sum(1 for e in bm.edges if len(e.link_faces) != 2), sum(1 for f in bm.faces if f.calc_area() < 1e-10))\n"
        "    bm.free()\n    if any(bad): out[o.name] = bad\nresult = out\n")
    plants = {k: v for k, v in sanity.items() if k.startswith(("t_palm", "t_fern", "t_tropical", "t_boardwalk"))}
    assert not plants, plants  # closed meshes: validate_scene's hole fill never runs on them
    for name, at, dims, slack in checks:
        got = c.run_code(_BBOX.format(name=name))
        if name.startswith(("t_palm", "t_fern", "t_tropical")):
            # turned 30 degrees: compare in the root's frame
            got = c.run_code(_BBOX.format(name=name).replace("o.matrix_world @", "root.matrix_world.inverted() @ o.matrix_world @"))
            at = [0.0, 0.0]
        size = [h - lo for lo, h in zip(got["lo"], got["hi"])]
        assert size == pytest.approx(list(dims), abs=3e-3 + slack), name
        assert abs(got["lo"][2]) < 2e-3 and got["parts"] >= 2, name
        centre = [(lo + h) / 2 for lo, h in zip(got["lo"][:2], got["hi"][:2])]
        assert centre == pytest.approx(at, abs=3e-3 + slack), name


@pytest.mark.blender
def test_live_beach_builds_terrain_water_and_frames_the_probe(private_blender, tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    from ai_db_scene.blender_client import BlenderClient
    from ai_db_scene.memory import SceneMemory
    from ai_db_scene.tools import scene_tools

    client = BlenderClient(port=private_blender, timeout=600)
    memory = SceneMemory("json_file", {"path": str(tmp_path / "mem")})
    memory.set_blender_client(client)
    monkeypatch.setattr(scene_tools, "_scene_memory", memory)
    built = sp._handle_build_scene({"spec": QWEN_2, "brief": BRIEF, "clear": True, "seed": 0, "exposure_check": False})
    assert built["status"].startswith("built"), built
    blocking = [i for i in built["issues_left"] if i.get("severity") == "blocking"]
    assert not blocking, blocking
    assert not any(i.get("check") == "mesh_sanity" for i in built["issues_left"]), built["issues_left"]
    got = client.run_code(
        "import bpy\nsc = bpy.context.scene\n"
        "t = [o for o in sc.objects if o.get('ai_db_terrain')]\n"
        "rocks = [o.location.z for o in sc.objects if o.parent is None and str(o.get('category')) == 'rock']\n"
        "result = {'terrain': sorted(o.name for o in t), 'mats': sorted(o.active_material.name for o in t),\n"
        "          'rocks': rocks, 'palms': sum(1 for o in sc.objects if o.name.endswith('_leaves') and 'palm' in o.name)}\n")
    assert got["terrain"] == ["lagoon_water", "sand_ground"] or got["terrain"] == ["sand_ground", "sea_water"], got
    assert any("ground_sand" in m for m in got["mats"]) and any("water" in m for m in got["mats"])
    assert got["palms"] == 5 and got["rocks"] and max(got["rocks"]) < 0  # rocks bedded on the shore slope
    # the bench probe frames the objects, not the ground to the horizon
    bench = Path(__file__).resolve().parents[1] / "eval" / "agent_bench"
    sys.path.insert(0, str(bench))
    try:
        import blender as bench_blender
    finally:
        sys.path.remove(str(bench))
    cam = client.run_code(bench_blender._PROBE_CAMERA.replace("VIEW", repr("interior"))
                          + "\nresult = {'loc': list(cam.location)}\n")
    assert all(abs(v) < 30 for v in cam["loc"][:2]), cam


def test_edit_scene_can_set_ground_and_water_and_relate_to_it():
    from ai_db_scene.edits import apply_steps

    base = _spec({"setting": {"kind": "outdoor", "bounds_m": [20, 15, 5]},
                  "objects": [{"id": "r", "category": "rock"}]}).to_dict()
    new, _notes, errors = apply_steps(base, [
        {"op": "update_setting", "set": {"ground": "sand", "water": {"kind": "lagoon", "side": "east"}}},
        {"op": "add_relation", "a": "r", "rel": "at_edge_of", "b": "water"}])
    assert not errors and new is not None, errors
    spec = _spec(new)
    assert spec.setting.ground == "sand" and spec.setting.water == terrain.Water("lagoon", "east", 0.3, None)
    assert ("r", "at_edge_of", "water") in {(r.a, r.rel, r.b) for r in spec.relations}
