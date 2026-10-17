"""A park fountain, not a birdbath (lead review of the qwen aidb7 park_corner).

The old recipe was a 2.4 m dish with a small bowl on a stalk. Now: a wide round basin with a
coping rim and water standing in it, a central column with two tiers of bowls (each with water)
and a finial; unsized, the fountain is 3.2 m across (basin 2.5-4 m).
"""

from __future__ import annotations

import pytest
from ai_db_scene.layout import solve_layout
from ai_db_scene.priors import load_priors
from ai_db_scene.recipes import recipe_spec
from ai_db_scene.spec import validate_spec
from ai_db_scene.tools import scene_pipeline as sp


def _parts(dims=(3.2, 3.2, 1.9)):
    return {p["id"]: p for p in recipe_spec("fountain", "fountain", dims)["parts"]}


def test_basin_with_coping_water_and_two_tiers():
    parts = _parts()
    basin = parts["basin"]
    assert basin["shape"] == "lathe" and basin["size"][0] == pytest.approx(1.0)  # the whole footprint
    rs = [r for r, _z in basin["profile"]]
    assert max(rs) > 0.48 and any(0.4 < r < 0.46 for r in rs)  # coping lip outside, inner wall
    for w in ("water_basin", "water_lower", "water_upper"):
        assert w in parts and parts[w]["material"].startswith("glass")
    assert parts["water_basin"]["size"][0] > 0.8  # water fills the basin
    lower, upper = parts["bowl_lower"], parts["bowl_upper"]
    assert lower["at"][2] < upper["at"][2] and lower["size"][0] > upper["size"][0]  # tiers narrow upwards
    top = max(p["at"][2] + p["size"][2] for p in parts.values())
    assert parts["finial"]["at"][2] + parts["finial"]["size"][2] == pytest.approx(top)


def test_unsized_fountain_is_park_sized():
    t = load_priors()["fountain"].typical_m
    assert 2.5 <= t[0] <= 4.0 and t[0] == t[1] and 1.4 <= t[2] <= 2.5
    raw = {"setting": {"kind": "outdoor", "bounds_m": [20, 20, 5]},
           "objects": [{"id": "fountain", "category": "fountain"}], "relations": []}
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    (pl,) = solve_layout(spec)
    assert 2.5 <= pl.dims[0] <= 4.0


def test_a_stone_fountain_keeps_its_water():
    assert "water" in sp._KEEP_PARTS["fountain"]
