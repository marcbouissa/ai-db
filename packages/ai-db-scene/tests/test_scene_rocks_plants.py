"""Props that read as the real thing: faceted boulders (rocks.py) instead of squashed sphere domes,
and potted plants with a full crown."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ai_db_scene import rocks, vegetation
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.terrain import apply_brief_defaults
from ai_db_scene.tools import scene_pipeline as sp

FIX = Path(__file__).parent / "fixtures"
OUT = json.loads((FIX / "bench_specs_outdoor_aidb7.json").read_text())


def test_a_rock_is_a_faceted_hull_cut_by_cleavage_planes():
    cmds = rocks.rock_commands("rock_1", (0.8, 0.7, 0.5), [1.0, 2.0, 0.0], 30.0, 7)
    code = cmds[0]["params"]["code"]
    assert "convex_hull" in code and "bisect_plane" in code and "noise.noise" in code
    assert "use_smooth = False" in code  # facets, not a smooth dome
    assert '["asset"] = name' in code  # an asset root, checked as a whole like a recipe
    payload = json.loads(code.split("json.loads(", 1)[1].split(")\n", 1)[0].strip("'\""))
    assert payload["dims"] == [0.8, 0.7, 0.5] and payload["seed"] == 7 and 0 < payload["sink"] <= 0.2
    assert cmds[1] == {"type": "apply_material", "params": {"name": "rock_1_stone", "material": rocks.tone("rock_1", 7)}}
    assert len({rocks.tone(f"r{k}") for k in range(12)}) > 2  # rocks of one scene differ in tone


def test_the_build_makes_its_rocks_with_the_generator():
    e = copy.deepcopy(OUT["last/qwen/beach_cove"])
    spec, errors = sp._spec_from_args({"spec": e["spec"]}, remembered=False)
    assert spec is not None, errors
    spec, _notes = apply_brief_defaults(spec, e["brief"])
    fixed = solve_with_repairs(spec.to_dict(), 0, 3)
    cmds, _ = sp.build_layout_commands(fixed.spec, fixed.placements)
    rock_ids = [p.id for p in fixed.placements if p.category == "rock"]
    assert rock_ids
    made = [c for c in cmds if c["type"] == "run_code" and "convex_hull" in c["params"]["code"]]
    assert len(made) == len(rock_ids)
    assert not any(c["type"] == "build_asset" and c["params"]["spec"].get("category") == "rock" for c in cmds)
    seeds = {json.loads(c["params"]["code"].split("json.loads(", 1)[1].split(")\n", 1)[0].strip("'\""))["seed"]
             for c in made}
    assert len(seeds) == len(rock_ids)  # every rock its own shape


def test_a_potted_plant_has_a_full_crown():
    code = vegetation.plant_commands("potted_plant", "p", (0.6, 0.6, 1.2), [0, 0, 0], 0, 3, None)[0]["params"]["code"]
    assert "randint(28, 36)" in code and "low = k % 4 == 1" in code
