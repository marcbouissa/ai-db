"""Room size vs content (aidb7 visual review, 2026-10-17).

Rooms came out far larger than their content: the repair loop grew a 10 x 8 m classroom to
12 x 9.6 m when the solver's sampler was merely unlucky, and every growth step was a full 20 %.
Now a spec without ``bounds_m`` is sized to its content plus walkways, the loop tries a second
seed before growing the room, and a growth step settles on the smallest in-between size that
solves. A size the model gave is never shrunk.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from ai_db_scene.normalize import table_shape_from_brief
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

FIX = json.loads((Path(__file__).parent / "fixtures" / "bench_specs_aidb7.json").read_text())


def _no_bounds(key):
    raw = copy.deepcopy(FIX[key]["spec"])
    raw.pop("brief", None)
    raw["setting"].pop("bounds_m")
    raw, _ = table_shape_from_brief(raw, FIX["briefs"][key.split("/")[-1]])
    spec, errors = sp._spec_from_args({"spec": raw})
    assert spec is not None, errors
    return spec


@pytest.mark.parametrize("key", ["raw/qwen/living_room", "raw/lightning/living_room", "raw/lightning/cafe",
                                 "raw/qwen/warehouse", "raw/lightning/round_lobby"])
def test_a_room_without_size_fits_its_content(key):
    spec = _no_bounds(key)
    assert any(n.startswith("setting.bounds_m: none given; sized to the content") for n in spec.notes)
    asked = FIX[key]["spec"]["setting"]["bounds_m"]
    w, d, _h = spec.setting.bounds_m
    fixed = solve_with_repairs(spec.to_dict(), 0, 6)
    assert fixed.error is None and fixed.placements and not fixed.dropped, (fixed.error, fixed.dropped)
    fw, fd, _ = fixed.data["setting"]["bounds_m"]
    assert fw * fd <= 1.45 * w * d  # at most one growth step
    if key.endswith(("warehouse", "lobby")):  # the models' 20 x 15 m / 8 x 8 m rooms were mostly empty floor
        assert fw * fd < 0.6 * asked[0] * asked[1]
    if spec.setting.shape == "ellipse":
        assert fw == fd


def test_growth_settles_between_steps_and_keeps_an_explicit_size():
    spec = _no_bounds("raw/lightning/cafe")
    fixed = solve_with_repairs(spec.to_dict(), 0, 6)
    assert fixed.placements and any(r.startswith("enlarged") for r in fixed.repairs)
    assert fixed.data["setting"]["bounds_m"][0] < 1.2 * spec.setting.bounds_m[0] - 0.05  # less than a full step
    spec, _ = sp._spec_from_args({"spec": FIX["raw/qwen/classroom"]["spec"]})
    fixed = solve_with_repairs(spec.to_dict(), 0, 6)  # seed 0 used to grow it to 12 x 9.6 m
    assert fixed.data["setting"]["bounds_m"][:2] == [10.0, 8.0], fixed.repairs
