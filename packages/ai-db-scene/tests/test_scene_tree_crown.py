"""Broadleaf trees with a natural crown (lead review of the qwen aidb7 park: "lollipop blob-stacks").

The crown was a column of equal icosphere clumps on a long bare pole, every tree the same size.
Now the crown starts low (28-36 % of the height), is built of irregular, flattened, noise-displaced
lobes round the twig ends plus a few core lobes (one mass with a bumpy outline), spreading limbs
show under it, and every tree takes its own size inside its placement box.
"""

from __future__ import annotations

import json

import pytest
from ai_db_scene.tools import scene_pipeline as sp


def _code(seed: int, dims=(3.0, 3.0, 6.0), name: str = "tree") -> str:
    return sp._TREE.format(payload=json.dumps({"name": name, "dims": list(dims), "seed": seed,
                                               "location": [0.0, 0.0, 0.0], "rot_z": 0.0}))


def test_tree_code_compiles_with_the_new_crown():
    code = _code(1)
    compile(code, "tree", "exec")
    assert "noise.noise" in code and "shoots" in code
    assert "rng.uniform(0.82, 1.0)" in code  # its own size inside the box


_MEASURE = """
import bpy
from mathutils import Vector
out = {}
for n in NAMES:
    r = bpy.data.objects[n]
    bpy.context.view_layer.update()
    def pts(o):
        return [o.matrix_world @ Vector(v.co) for v in o.data.vertices]
    lv, wd = pts(bpy.data.objects[n + "_leaves"]), pts(bpy.data.objects[n + "_wood"])
    out[n] = {"top": max(p.z for p in lv + wd), "leaf_lo": min(p.z for p in lv),
              "leaf_w": max(p.x for p in lv) - min(p.x for p in lv),
              "leaf_d": max(p.y for p in lv) - min(p.y for p in lv),
              "wood_w": max(p.x for p in wd) - min(p.x for p in wd)}
result = out
"""


@pytest.mark.blender
def test_live_trees_have_low_broad_crowns_and_differ(private_blender):
    from ai_db_scene.blender_client import BlenderClient

    c = BlenderClient(port=private_blender, timeout=300)
    names = [f"t{s}" for s in range(4)]
    for s, n in enumerate(names):
        c.run_code(_code(s * 7 + 1, name=n))
    got = c.run_code("NAMES = " + json.dumps(names) + "\n" + _MEASURE)
    tops = [got[n]["top"] for n in names]
    assert all(4.8 <= t <= 6.0 + 1e-3 for t in tops), tops
    assert max(tops) - min(tops) > 0.15  # not clones
    for n in names:
        g = got[n]
        assert 0.25 * g["top"] < g["leaf_lo"] < 0.5 * g["top"], g  # crown low, limbs show below it
        assert g["leaf_w"] >= 0.75 * 3.0 * 0.85 and g["leaf_d"] >= 0.75 * 3.0 * 0.85, g  # a broad head
        assert g["leaf_w"] <= 3.0 + 1e-3 and g["leaf_d"] <= 3.0 + 1e-3, g  # inside its placement box
