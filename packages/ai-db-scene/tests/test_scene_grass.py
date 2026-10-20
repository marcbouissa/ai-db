"""Grass blades (grass.py) on a grass ground, round the objects and in the camera's foreground."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ai_db_scene import grass
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.terrain import apply_brief_defaults
from ai_db_scene.tools import scene_pipeline as sp

FIX = Path(__file__).parent / "fixtures"
OUT = json.loads((FIX / "bench_specs_outdoor_aidb7.json").read_text())


def _solve(key):
    e = copy.deepcopy(OUT[key])
    spec, errors = sp._spec_from_args({"spec": e["spec"]}, remembered=False)
    assert spec is not None, errors
    spec, _notes = apply_brief_defaults(spec, e["brief"])
    fixed = solve_with_repairs(spec.to_dict(), 0, 3)
    return fixed.spec, fixed.placements


def test_a_park_grows_blades_round_its_objects_and_toward_the_camera():
    spec, pls = _solve("last/qwen/park_corner")
    assert spec.setting.ground == "grass"
    off = sp.frame_offset(spec)
    cam = {"location": [0.0, -8.0, 2.5], "rotation": [90.0, 0.0, 0.0]}  # looking along +Y
    cmds = grass.grass_commands(spec, pls, off, cam)
    assert [c["type"] for c in cmds] == ["run_code", "apply_material"]
    code = cmds[0]["params"]["code"]
    assert "BVHTree" in code and "plaza_" in code  # blades only on bare ground, not on the plaza
    assert f'"max": {grass.MAX_BLADES}' in code and '"ground": "grass_ground"' in code
    assert cmds[1]["params"]["name"] == grass.NAME


def test_no_blades_on_sand_or_indoors():
    spec, pls = _solve("last/qwen/beach_cove")
    assert grass.grass_commands(spec, pls, sp.frame_offset(spec), None) == []
    room = sp._spec_from_args({"spec": {"setting": {"kind": "indoor", "bounds_m": [5, 4, 3]},
                                        "objects": [{"id": "s", "category": "sofa"}]}}, remembered=False)[0]
    assert grass.grass_commands(room, [], (0.0, 0.0, 0.0), None) == []
