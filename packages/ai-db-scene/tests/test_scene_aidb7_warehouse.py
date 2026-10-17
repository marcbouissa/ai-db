"""qwen3-14b-local-aidb7 warehouse (visual review, 2026-10-17).

"Four pallet racks in rows, pallets with crates, a forklift": ``pallet_rack (4) in_row`` became
one back-to-back line of racks, the eight pallets lay scattered, the forklift stood at the end
of the line, and the model's plain "metal" turned racks, their loads, the forklift and the walls
into chrome. Now: two rows of racks facing each other across a forklift aisle, the forklift in
it, the pallets a staging grid with one crate per pallet, the racks/forklift keep their painted
finish and metal walls read as painted cladding.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

FIX = json.loads((Path(__file__).parent / "fixtures" / "bench_specs_aidb7.json").read_text())


def _solve(key, seed):
    spec, errors = sp._spec_from_args({"spec": FIX[key]["spec"]})
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), seed, 6)
    assert fixed.error is None and fixed.placements and not fixed.dropped, (fixed.error, fixed.dropped)
    return spec, fixed


@pytest.mark.parametrize("key", ["raw/qwen/warehouse", "last/qwen/warehouse"])
@pytest.mark.parametrize("seed", [0, 1])
def test_racks_in_rows_with_a_forklift_aisle(key, seed):
    _spec, fixed = _solve(key, seed)
    pl = {p.id: p for p in fixed.placements}
    racks = [p for p in fixed.placements if p.category == "pallet_rack"]
    assert len(racks) == 4
    a = [p for p in racks if p.id.startswith("pallet_rack_row_a")]
    b = [p for p in racks if p.id.startswith("pallet_rack_row_b")]
    assert len(a) == len(b) == 2
    # each row is a straight line, the rows parallel; between them a >= 3 m aisle
    yaw = math.radians(a[0].rotation_deg_z)
    across = (math.sin(yaw), -math.cos(yaw))  # the rack's front: across the aisle
    off = lambda p: p.location[0] * across[0] + p.location[1] * across[1]
    assert abs(off(a[0]) - off(a[1])) < 0.01 and abs(off(b[0]) - off(b[1])) < 0.01
    assert abs(a[0].rotation_deg_z - b[0].rotation_deg_z) % 360 == pytest.approx(180)
    aisle = abs(off(a[0]) - off(b[0])) - a[0].dims[1]
    assert aisle >= 3.0
    # the forklift stands in that aisle
    f = pl["forklift"]
    lo, hi = sorted((off(a[0]), off(b[0])))
    assert lo + a[0].dims[1] / 2 < off(f) < hi - a[0].dims[1] / 2
    # pallets on a floor grid, crates on distinct pallets
    pallets = [p for p in fixed.placements if p.category == "pallet"]
    assert len(pallets) == 8 and len({round(p.location[1], 2) for p in pallets}) <= 4
    crates = [p for p in fixed.placements if p.category == "crate"]
    under = [min(pallets, key=lambda q: math.dist(q.location[:2], c.location[:2])) for c in crates]
    assert len({q.id for q in under}) == len(crates)
    for c, q in zip(crates, under, strict=True):
        assert abs(c.location[0] - q.location[0]) < q.dims[0] / 2 and abs(c.location[1] - q.location[1]) < q.dims[1] / 2
        assert c.location[2] - c.dims[2] / 2 == pytest.approx(q.location[2] + q.dims[2] / 2, abs=0.01)


def test_plain_metal_keeps_the_painted_finish():
    spec, fixed = _solve("last/qwen/warehouse", 0)
    assert spec.setting.wall_material.startswith("paint:")
    cmds, _ = sp.build_layout_commands(fixed.spec, fixed.placements, variation=sp.variation_key(spec))
    code = " ".join(c["params"]["code"] for c in cmds if c["type"] == "run_code" and "reskinned" in c["params"]["code"])
    assert '"name": "ceiling_light_1"' in code  # other spec materials still apply
    for name in ("pallet_rack_row_a_1", "forklift", "pallet_1", "crate_1", "safety_barrier_1", "workbench"):
        assert f'"name": "{name}"' not in code  # no reskin: the recipe's own paint, wood, cardboard
