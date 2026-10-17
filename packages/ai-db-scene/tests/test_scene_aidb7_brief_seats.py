"""The brief check counts the chairs built as seats (lightning aidb7 classroom and cafe).

nemotron-3.5-lightning-free-aidb7 wrote ``student_chair (12) in_front_of student_desk`` and
``tables with seats: 2``: the chairs were built as ``<desk>_seat_k`` dining chairs, but the brief
check looked for category ``chair`` only, said "the brief asks for chair but the scene has none",
and the model added 11-12 loose chairs on top of the seated ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ai_db_scene import realism
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

FIX = json.loads((Path(__file__).parent / "fixtures" / "bench_specs_aidb7.json").read_text())


@pytest.mark.parametrize("key", ["raw/lightning/classroom", "raw/lightning/cafe", "raw/qwen/classroom"])
def test_seat_chairs_count_as_chairs(key):
    run = FIX[key]
    spec, errors = sp._spec_from_args({"spec": run["spec"]})
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), 0, 6)
    assert fixed.placements, fixed.error
    seats = [p for p in fixed.placements if "_seat_" in p.id]
    assert seats
    issues = [i["message"] for i in realism.check_brief(run["brief"], fixed.placements)]
    assert not any("chair" in m for m in issues), issues


def test_a_brief_with_chairs_and_none_built_still_warns():
    from ai_db_scene.layout import Placement

    pl = [Placement("desk", (1.0, 1.0, 0.4), 0.0, (1.2, 0.6, 0.75), "desk")]
    issues = [i["message"] for i in realism.check_brief("a desk and a chair", pl)]
    assert any("chair but the scene has none" in m for m in issues), issues
