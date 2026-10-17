"""Outdoors the shore groups are placed first and palms frame the beach (bench beach_cove, 2026-10-17).

In the qwen aidb6 beach_cove demo five hero palms, placed first as the largest objects, took the
beach strip; the two loungers under their parasol ended at the back, ten metres from the water.
"""

from __future__ import annotations

import pytest
from ai_db_scene.layout import _Problem, check_layout, solve_layout
from ai_db_scene.priors import load_priors
from ai_db_scene.spec import validate_spec
from ai_db_scene.terrain import terrain_of

#: qwen3-14b-local-aidb6 beach_cove demo spec (its water as the brief names it, terrain_brief tests)
DEMO = {"setting": {"kind": "outdoor", "floor_material": "sand", "bounds_m": [20, 20, 5], "style": "tropical",
                    "water": {"kind": "lagoon"}},
        "objects": [{"id": "palm1", "category": "palm_tree", "count": 5, "material": "tropical_leaf", "hero": True},
                    {"id": "bush1", "category": "bush", "count": 3, "material": "dense_green"},
                    {"id": "boardwalk1", "category": "wooden_boardwalk", "material": "wood_plank", "hero": True},
                    {"id": "lounger1", "category": "sun_lounger", "count": 2, "material": "woven_rattan", "hero": True},
                    {"id": "parasol1", "category": "parasol", "material": "canvas_white", "hero": True},
                    {"id": "rock1", "category": "rock", "count": 4, "material": "weathered_stone"}],
        "relations": [{"a": "boardwalk1", "rel": "connects_to", "b": "water_edge"},
                      {"a": "lounger1", "rel": "under", "b": "parasol1"},
                      {"a": "rock1", "rel": "at", "b": "water_edge"}]}


def _spec():
    spec, errors = validate_spec(DEMO)
    assert spec is not None, errors
    return spec


def test_the_shore_group_is_placed_before_the_palms():
    spec = _spec()
    order = _Problem(spec, load_priors(), {}).order()
    first_palm = min(order.index(i) for i in order if i.startswith("palm1"))
    for oid in ("parasol1", "lounger1_1", "lounger1_2", "boardwalk1", "rock1_1"):
        assert order.index(oid) < first_palm


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_loungers_on_the_beach_strip_and_palms_framing_it(seed):
    spec = _spec()
    pls = solve_layout(spec, seed=seed)
    assert check_layout(spec, pls) == []
    ter = terrain_of(spec)
    assert ter is not None
    by = {p.id: p for p in pls}
    top = ter.dry()
    group = [ter.signed(*by[i].location[:2]) for i in ("parasol1", "lounger1_1", "lounger1_2")]
    assert all(top - 5.0 < s < top for s in group)  # a few metres up from the dry line, not at the back
    w = spec.setting.bounds_m[0]
    framing = 0
    for p in pls:
        if p.category != "palm_tree":
            continue
        x, y = p.location[:2]
        mid = abs(x - w / 2) / (w / 2) < 0.5
        s = ter.signed(x, y)
        assert not (mid and s > max(group) - 0.5), (p.id, x, y)  # never between the group and the water
        framing += not mid or s < min(group) - 1.0
    assert framing >= 4  # at the sides or behind the group
