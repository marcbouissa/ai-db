"""A bush stays a bush whatever its material, and ``boardwalk connects bushes`` makes the boardwalk
run from the bushes to the water (bench beach_cove, qwen3-14b-local-aidb7's first call).

qwen wrote ``{id: bush1, category: bush, material: green_fern}``: the material word made it a
fern, and the brief check then asked for the "dense green bushes" it had. Its
``boardwalk1 connects bush1`` was read as ``beside`` and the solver dropped it.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from ai_db_scene import terrain
from ai_db_scene.realism import check_brief
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.spec import validate_spec
from ai_db_scene.terrain import LINK_REL, WATER, apply_brief_defaults
from ai_db_scene.tools import scene_pipeline as sp

FX = json.loads((Path(__file__).parent / "fixtures" / "bench_specs_outdoor_aidb7.json").read_text())
FIRST = FX["first/qwen/beach_cove"]


def _spec():
    spec, errors = sp._spec_from_args({"spec": copy.deepcopy(FIRST["spec"])}, remembered=False)
    assert spec is not None, errors
    spec, _notes = apply_brief_defaults(spec, FIRST["brief"])
    return spec


def test_a_fern_material_does_not_make_a_bush_a_fern():
    spec = _spec()
    cats = {o.id: o.category for o in spec.objects}
    assert cats["bush1"] == "tropical_bush"  # a bush in a tropical setting, not a fern
    fixed = solve_with_repairs(spec.to_dict(), 1, 3)
    msgs = [i["message"] for i in check_brief(FIRST["brief"], fixed.placements)]
    assert not any("bush" in m for m in msgs), msgs
    assert any("fern" in m for m in msgs)  # the ferns it did not write are still asked for


def test_a_fern_id_still_makes_a_fern():
    data = {"setting": {"kind": "outdoor", "bounds_m": [20, 15, 3], "ground": "sand", "water": {"kind": "sea"}},
            "objects": [{"id": "fern1", "category": "bush"}, {"id": "shrub", "category": "bush", "material": "fern"}]}
    spec, errors = validate_spec(data)
    assert spec is not None, errors
    assert {o.id: o.category for o in spec.objects} == {"fern1": "fern", "shrub": "bush"}


def test_connects_links_the_boardwalk_to_the_bushes():
    spec = _spec()
    rels = {(r.a, r.rel, r.b) for r in spec.relations}
    assert ("boardwalk1", LINK_REL, "bush1") in rels
    assert ("boardwalk1", "leads_to", WATER) in rels
    assert any("runs from bush1 to the water" in n for n in spec.notes)
    assert ("bush1", "back_of", WATER) in rels  # the bushes still go along the back


def test_the_boardwalk_runs_from_the_bushes_to_the_water():
    spec = _spec()
    near = 0
    for seed in range(4):
        fixed = solve_with_repairs(spec.to_dict(), seed, 3)
        assert not any("boardwalk1" in r for r in fixed.repairs if "dropped" in r)
        pls = terrain.settle(fixed.spec, fixed.placements)
        ter = terrain.terrain_of(fixed.spec)
        (bw,) = [p for p in pls if p.category == "boardwalk"]
        yaw = math.radians(bw.rotation_deg_z)
        f = (math.sin(yaw), -math.cos(yaw))
        front = (bw.location[0] + f[0] * bw.dims[1] / 2, bw.location[1] + f[1] * bw.dims[1] / 2)
        back = (bw.location[0] - f[0] * bw.dims[1] / 2, bw.location[1] - f[1] * bw.dims[1] / 2)
        assert ter.signed(*front) > 0.5  # over the water
        bush = min(math.hypot(p.location[0] - back[0], p.location[1] - back[1])
                   for p in pls if p.id.startswith("bush1"))
        near += bush <= 1.6  # its land end at a bush
    assert near >= 3


def test_a_link_needs_a_boardwalk_and_water():
    data = {"setting": {"kind": "outdoor", "bounds_m": [20, 15, 3], "ground": "grass"},
            "objects": [{"id": "path", "category": "bench"}, {"id": "b", "category": "bush"}],
            "relations": [{"a": "path", "rel": LINK_REL, "b": "b"}]}
    spec, errors = validate_spec(data)
    assert spec is None and any(LINK_REL in e for e in errors)
