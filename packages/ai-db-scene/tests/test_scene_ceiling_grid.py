"""Ceiling objects never grow the room (nemotron-3.5-lightning-free-aidb5 warehouse, 2026-10-16).

The model's first warehouse spec hung four ceiling lights (light1..light4) each ``centered_in
room``: they cannot share the centre, the repair loop read "overlaps 'light1'" as lack of space,
grew the 50 x 30 m hall to 72 x 43.2 m and still failed with "cannot place 'light2'". The spec
below is that call's, verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_db_scene.layout import LayoutError
from ai_db_scene.repair import explain, relax, solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

WAREHOUSE = json.loads((Path(__file__).parent / "fixtures" / "bench_lightning_warehouse_aidb5.json").read_text())["args"]


def _lights(placements):
    return sorted((p for p in placements if p.category == "ceiling_light"), key=lambda p: p.id)


def test_lightning_warehouse_lights_go_on_a_ceiling_grid():
    spec, errors = sp._spec_from_args(WAREHOUSE)
    assert spec is not None, errors
    assert any("cannot share the centre" in n for n in spec.notes)
    fixed = solve_with_repairs(spec.to_dict(), 0, 6)
    assert fixed.error is None and fixed.placements is not None
    assert not any(r.startswith("enlarged") for r in fixed.repairs), fixed.repairs
    assert fixed.data["setting"]["bounds_m"] == [50, 30, 12]
    lights = _lights(fixed.placements)
    assert len(lights) == 4
    xy = sorted((round(p.location[0], 1), round(p.location[1], 1)) for p in lights)
    assert xy == [(12.5, 7.5), (12.5, 22.5), (37.5, 7.5), (37.5, 22.5)]  # a 2 x 2 grid over the hall
    assert all(abs(p.location[2] + p.dims[2] / 2 - 12) < 1e-6 for p in lights)  # on the ceiling


def test_a_ceiling_object_that_cannot_be_placed_never_grows_the_room():
    data = {"setting": {"kind": "indoor", "bounds_m": [50, 30, 12]},
            "objects": [{"id": "light1", "category": "ceiling_light"}, {"id": "light2", "category": "ceiling_light"}],
            "relations": [{"a": "light1", "rel": "centered_in", "b": "room"},
                          {"a": "light2", "rel": "centered_in", "b": "room"}]}
    err = LayoutError("cannot place 'light2' (ceiling_light, 0.5x0.5x0.1 m) in 50x30x12 m indoor setting after 2300 "
                      "candidates; main blockers: 2300x overlaps 'light1'. relations of 'light2': "
                      "['light2 centered_in room']", ["light1"])
    new, note = relax(data, err, [50, 30, 12])
    assert new["setting"]["bounds_m"] == [50, 30, 12]
    assert new["relations"] == [{"a": "light1", "rel": "centered_in", "b": "room"}]
    assert note.startswith("'light2' hangs on the ceiling grid")
    assert relax(new, err, [50, 30, 12]) is None  # nothing left to drop, count 1: give up, clearly
    assert "never the fix" in explain(err, grown=False)
