"""Water seen from the beach (lead review of the qwen aidb7 beach_cove: grey-blue, no turquoise).

At eye level the transmissive water surface mostly showed the sky. The water now has an opaque
water-body layer: by the shore a lagoon's turquoise (a larger share), offshore a deeper blue,
and a lower specular level so the grazing-angle sky reflection does not wash it out.
"""

from __future__ import annotations

import colorsys
import copy
import json
from pathlib import Path

import pytest
from ai_db_scene import terrain
from ai_db_scene.tools import scene_pipeline as sp

REVIEW = json.loads((Path(__file__).parent / "fixtures" / "bench_qwen_review_aidb7.json").read_text())


def _payload(water: dict) -> dict:
    raw = copy.deepcopy(REVIEW["beach_cove"]["spec"])
    raw["setting"]["water"] = water
    spec, errors = sp._spec_from_args({"spec": raw}, remembered=False)
    assert spec is not None, errors
    return terrain.terrain_payload(spec, sp.frame_offset(spec))["water"]


def _hue_sat(rgb):
    h, s, _v = colorsys.rgb_to_hsv(*rgb)
    return h * 360, s


def test_lagoon_shallows_are_turquoise_and_the_sea_deeper_blue():
    lagoon_near, lagoon_far, share_near, share_far, spec_l, _band = terrain.WATER_LOOK["lagoon"]
    _sea_near, sea_far, *_rest = terrain.WATER_LOOK["sea"]
    h, s = _hue_sat(lagoon_near)
    assert 160 <= h <= 185 and s > 0.8  # turquoise
    assert share_near > share_far  # the shallows show more of the body colour
    assert spec_l < 0.5  # Principled default: the sky reflection at grazing angles is cut
    hs, _ = _hue_sat(sea_far)
    assert 205 <= hs <= 235 and sum(sea_far) < sum(lagoon_far)  # deeper blue offshore


@pytest.mark.parametrize("side,axis,sign", [("north", 1, 1), ("south", 1, -1), ("east", 0, 1), ("west", 0, -1)])
def test_shore_line_points_into_the_water(side, axis, sign):
    w = _payload({"kind": "lagoon", "side": side, "fraction": 0.3})
    sh = w["shore"]
    assert sh["axis"] == axis and sh["dir"] == sign
    assert w["look"][0] == list(terrain.WATER_LOOK["lagoon"][0])
    lo, hi = (w["box"][axis], w["box"][axis + 2])
    assert lo < sh["at"] < hi  # the waterline crosses the water box


def test_shader_code_builds_the_body_layer():
    code = terrain._TERRAIN_CODE
    assert "ShaderNodeMixShader" in code and "Specular IOR Level" in code
    compile(code, "terrain", "exec")
