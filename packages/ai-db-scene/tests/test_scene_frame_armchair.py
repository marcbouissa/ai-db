"""The lobby's tube-frame armchairs (lead review of the qwen aidb7 round lobby).

The bauhaus family's square-tube frame showed as a huge mirror-chrome triangle on the armchair's
side. Cause: the model asked for ``poly_budget`` 50000 and the budget pass put one Decimate ratio
on every part of the armchair, so the 64-face tube loop collapsed into triangles spanning the
whole side. Now only dense parts take the cut, and the frame is brushed steel, not a mirror.
"""

from __future__ import annotations

import json

import pytest
from ai_db_scene import furniture_variants as fv
from ai_db_scene.constants import BUILD_DECIMATE_MIN_PART_FACES
from ai_db_scene.tools import scene_pipeline as sp


def _key(seed: int) -> list:
    return ["indoor", [10, 10, 4], "modern", [["armchair", "armchair"]], fv.variation_marker(seed)]


def test_tube_frame_is_never_mirror_chrome() -> None:
    for seed in range(40):
        for cat, dims in (("armchair", (0.85, 0.85, 0.8)), ("sofa", (2.0, 0.9, 0.8))):
            p = fv.piece(_key(seed), cat, dims, variant="bauhaus")
            assert p["params"]["arm_style"] == "metal_square"
            assert p["params"]["leg_material"] in ("metal:brushed", "black_metal"), p["params"]
            assert p["params"]["fabric"] not in ("chrome",)


def test_budget_pass_spares_low_poly_parts() -> None:
    code = sp._POLY_BUDGET.format(payload=json.dumps([1000, 0.1, BUILD_DECIMATE_MIN_PART_FACES]))
    compile(code, "poly_budget", "exec")
    assert "faces(p) >= min_part" in code
    assert BUILD_DECIMATE_MIN_PART_FACES >= 200  # a 64-face tube loop and 6-face rails stay whole


_POSE = """
import bpy
r = bpy.data.objects["chair"]
r["ai_db_id"] = "chair"
out = {}
for o in r.children_recursive:
    if o.type == "MESH":
        out[o.name] = [m.type for m in o.modifiers]
result = out
"""


@pytest.mark.blender
def test_live_budget_decimates_cushions_not_the_frame(private_blender) -> None:
    from ai_db_scene.blender_client import BlenderClient

    c = BlenderClient(port=private_blender, timeout=300)
    p = fv.piece(_key(3), "armchair", (0.85, 0.85, 0.8), variant="bauhaus")
    c.batch([{"type": "generate_asset", "params": {"family": p["family"], "params": p["params"], "seed": p["seed"],
                                                   "name": "chair", "location": [0, 0, 0]}}])
    c.run_code(_POSE)
    rep = c.run_code(sp._POLY_BUDGET.format(payload=json.dumps([2000, 0.1, BUILD_DECIMATE_MIN_PART_FACES])))
    assert rep["decimated"], rep
    mods = c.run_code(_POSE)
    frames = [n for n in mods if "arm_frame" in n or "rail" in n]
    assert frames, mods
    assert all("DECIMATE" not in mods[n] for n in frames), mods
    assert any("DECIMATE" in m for m in mods.values()), mods
