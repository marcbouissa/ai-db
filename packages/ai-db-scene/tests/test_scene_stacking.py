"""Stacking and shelf levels; dropped relations block "done".

Bench: ``pallet1 on rack1`` was refused ("pallet_rack cannot support other objects") and the
repair loop dropped it, after first growing a 50 x 30 m warehouse to 72 x 43 m (lightning aidb5,
run lost to a provider 504); ``crate on pallet`` was dropped in big-pickle aidb4. qwen aidb6's
cafe replied DONE over three dropped table relations (the verdict offered "if it does not
matter, reply DONE"). Fixtures: those specs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ai_db_scene.layout import SHELF_LEVELS, check_layout
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.spec import validate_spec
from ai_db_scene.tools import scene_pipeline as sp

SPECS = json.loads((Path(__file__).parent / "fixtures" / "bench_stacking_and_dropped.json").read_text())


def _solve(raw, seed=0):
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), seed, 6)
    assert fixed.error is None and fixed.placements and fixed.spec is not None, fixed.error
    return fixed, {p.id: p for p in fixed.placements}


def _local(p, q):
    """q's centre in p's frame (along p's width, along p's front)."""
    yaw = math.radians(p.rotation_deg_z)
    dx, dy = q.location[0] - p.location[0], q.location[1] - p.location[1]
    return dx * math.cos(yaw) + dy * math.sin(yaw), dx * math.sin(yaw) - dy * math.cos(yaw)


@pytest.mark.parametrize("seed", [0, 1])
def test_pallets_on_rack_levels_and_crates_on_pallets_are_kept(seed):
    fixed, pl = _solve(SPECS["lightning_aidb5_warehouse_provider504"], seed)
    assert not fixed.dropped and not any("enlarged" in r for r in fixed.repairs), fixed.repairs
    levels = [f * pl["rack1"].dims[2] for f in SHELF_LEVELS["pallet_rack"]]
    for pid, rack in (("pallet1_1", "rack1"), ("pallet1_2", "rack1"), ("pallet2_1", "rack2"), ("pallet2_2", "rack2")):
        p, r = pl[pid], pl[rack]
        base = p.location[2] - p.dims[2] / 2
        assert any(abs(base - lv) < 1e-6 for lv in levels), base
        along, depth = _local(r, p)
        assert abs(along) + max(p.dims[:2]) / 2 <= r.dims[0] / 2 + 1e-6 and abs(depth) < 1e-6  # within the bay
    for i in range(1, 5):  # crate1_i on pallet1_(i % 2): on its top
        c = pl[f"crate1_{i}"]
        under = pl["pallet1_1" if i % 2 else "pallet1_2"]
        assert c.location[2] - c.dims[2] / 2 == pytest.approx(under.location[2] + under.dims[2] / 2)
    assert not [i for i in check_layout(fixed.spec, fixed.placements) if i.startswith("hard")]


def test_crates_on_pallets_big_pickle():
    fixed, pl = _solve(SPECS["bigpickle_aidb4_warehouse"])
    assert not fixed.dropped
    assert all(pl[f"crate_{i}"].location[2] > pl["pallet_1"].location[2] for i in range(1, 9))


def test_crates_stack_when_the_pallet_top_is_full():
    raw = {"setting": {"kind": "indoor", "bounds_m": [6, 6, 4]},
           "objects": [{"id": "pallet", "category": "pallet"}, {"id": "crate", "category": "crate", "count": 6}],
           "relations": [{"a": "crate", "rel": "on", "b": "pallet"}]}
    fixed, pl = _solve(raw)
    assert not fixed.dropped and not fixed.repairs
    top = pl["pallet"].location[2] + pl["pallet"].dims[2] / 2
    crates = [p for k, p in pl.items() if k.startswith("crate")]
    stacked = [c for c in crates if c.location[2] - c.dims[2] / 2 > top + 1e-6]
    assert stacked  # 6 crates do not fit side by side on one pallet
    for c in stacked:  # each stands on a crate
        base = c.location[2] - c.dims[2] / 2
        assert any(abs(o.location[2] + o.dims[2] / 2 - base) < 1e-6 and math.dist(o.location[:2], c.location[:2]) < 1e-6
                   for o in crates if o is not c)
    assert not [i for i in check_layout(fixed.spec, fixed.placements) if i.startswith("hard")]


def test_the_rack_loses_its_built_in_stock_when_it_carries_objects():
    spec, errors = validate_spec(SPECS["lightning_aidb5_warehouse_provider504"])
    assert spec is not None, errors
    parts = [{"id": "post_lf"}, {"id": "load_pallet_0_0"}, {"id": "load_0_0"}, {"id": "beam_1f"}]
    cmds = [{"type": "build_asset", "params": {"spec": {"asset": aid, "parts": list(parts)}}} for aid in ("rack1", "rack3")]
    sp._strip_shelf_loads(spec, cmds)
    assert [q["id"] for q in cmds[0]["params"]["spec"]["parts"]] == ["post_lf", "beam_1f"]  # rack1 carries pallets
    assert len(cmds[1]["params"]["spec"]["parts"]) == 4  # rack3 carries nothing: keeps its stock


@pytest.fixture
def fake_blender(monkeypatch):
    class Fake:
        def batch(self, cmds, timeout=None):
            return [{"ok": 1, "status": "success"} for _ in cmds]

        def request(self, handler, params, timeout=None):
            return {}

        def run_code(self, code, timeout=None):
            return {}

    class Memory:
        def record_run(self, run):
            return "run-1"

        def snapshot_scene(self, **kw):
            return "scene-1"

    monkeypatch.setattr(sp, "BlenderClient", Fake)
    monkeypatch.setattr(sp, "_memory", lambda: Memory())
    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})


def test_dropped_relations_block_done_with_a_step_that_fits(fake_blender):
    out = sp._handle_build_scene({"spec": SPECS["qwen_aidb6_cafe"], "exposure_check": False,
                                  "brief": "A small cafe with four round tables with two chairs each, a counter "
                                           "with two bar stools, and three potted plants."})
    dropped = [i for i in out["issues_left"] if i["check"] == "dropped_relation"]
    assert dropped and all(i["severity"] == "blocking" for i in dropped)
    assert out["status"] == "built_with_issues" and not out["verdict"].startswith("done")
    assert "reply DONE" not in out["next"] and "not done" in out["next"]
    steps = [i["suggest"] for i in dropped if "suggest" in i]
    assert steps and all(s["op"] == "move" for s in steps)
    assert json.dumps(steps[0]) in out["next"]
    # the second table 'in_front_of counter' goes beside the table that kept that place
    front = next((i for i in dropped if " in_front_of counter" in i["message"]), None)
    if front is not None:
        assert front["suggest"]["to"]["beside"] in ("table1", "table3", "counter")
