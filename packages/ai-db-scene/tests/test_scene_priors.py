"""Priors loader: validation of every entry and the closed vocabulary."""

from __future__ import annotations

import copy
import json

import pytest
from ai_db_scene.priors import SUPPORTS, categories, load_priors, parse_priors

from ai_db.errors import AiDbConfigError


def test_bundled_priors_load_and_are_valid():
    priors = load_priors()
    assert len(priors) >= 60
    for name, p in priors.items():
        assert all(lo <= ty <= hi for lo, ty, hi in zip(p.min_m, p.typical_m, p.max_m, strict=True)), name
        assert p.support in SUPPORTS
        assert p.default_clearance_m >= 0


def test_required_categories_present():
    cats = set(categories())
    industrial = {"pallet", "pallet_rack", "shelving_unit", "conveyor", "forklift", "crate", "barrel", "workbench", "safety_barrier"}
    outdoor = {"tree", "bench", "rock"}
    arch = {"door", "window", "stairs", "wall_partition"}
    assert (industrial | outdoor | arch) <= cats
    assert categories() == sorted(cats)


def _doc():
    return json.loads(json.dumps({"categories": {"x": {
        "min_m": [1, 1, 1], "typical_m": [2, 2, 2], "max_m": [3, 3, 3],
        "support": "floor", "can_support_others": False, "default_clearance_m": 0.1}}}))


def test_parse_valid_minimal():
    assert parse_priors(_doc())["x"].typical_m == (2.0, 2.0, 2.0)


@pytest.mark.parametrize(
    ("mutate", "msg"),
    [
        (lambda d: d["categories"]["x"].update(min_m=[3, 1, 1]), "priors[x]: need min <= typical <= max on w, got 3.0 / 2.0 / 3.0"),
        (lambda d: d["categories"]["x"].update(typical_m=[2, 2, 9]), "priors[x]: need min <= typical <= max on h, got 1.0 / 9.0 / 3.0"),
        (lambda d: d["categories"]["x"].update(support="sky"), "priors[x].support: 'sky' not in"),
        (lambda d: d["categories"]["x"].update(max_m=[1, 2]), "priors[x].max_m: expected 3 positive numbers"),
        (lambda d: d["categories"]["x"].update(can_support_others="yes"), "priors[x].can_support_others: expected bool"),
        (lambda d: d["categories"]["x"].update(default_clearance_m=-1), "priors[x].default_clearance_m"),
        (lambda d: d.update(categories={}), "'categories' must be a non-empty object"),
    ],
)
def test_parse_invalid(mutate, msg):
    d = copy.deepcopy(_doc())
    mutate(d)
    with pytest.raises(AiDbConfigError) as ei:
        parse_priors(d)
    assert msg in str(ei.value)


def test_spec_vocabulary_is_priors():
    from ai_db_scene.spec import spec_schema

    enum = spec_schema()["properties"]["objects"]["items"]["properties"]["category"]["enum"]
    assert enum == categories()
