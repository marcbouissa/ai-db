"""The room shell (shell.py): rooms are no empty boxes - a door, windows and trim, kept clear of
the furniture, unless the spec has its own openings or the brief wants no windows."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from ai_db_scene import shell
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

FIX = Path(__file__).parent / "fixtures"
IN = json.loads((FIX / "bench_specs_aidb7.json").read_text())


def _solve(key: str):
    spec, errors = sp._spec_from_args({"spec": copy.deepcopy(IN[key]["spec"])}, remembered=False)
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), 0, 3)
    assert fixed.placements is not None
    return fixed.spec, fixed.placements


def _top(p) -> float:
    return p.location[2] + p.dims[2] / 2


@pytest.mark.parametrize("key", ["last/qwen/living_room", "last/qwen/cafe", "last/qwen/classroom",
                                 "last/lightning/living_room"])
def test_a_rect_room_gets_a_door_windows_and_trim(key):
    spec, pls = _solve(key)
    sh = shell.plan(spec, pls, IN["briefs"][key.split("/")[-1]])
    assert sh is not None
    kinds = [o.kind for o in sh.openings]
    assert kinds.count("door") == 1 and kinds.count("window") >= 1
    assert sh.trim == ("skirting", "cove")
    assert "shell: the room got" in sh.note
    w, d, h = spec.setting.bounds_m
    for o in sh.openings:
        p = o.placement
        assert o.wall in ("wall_north", "wall_south", "wall_east", "wall_west")
        assert -0.01 <= p.location[0] <= w + 0.01 and -0.01 <= p.location[1] <= d + 0.01
        assert _top(p) <= h - 0.1
        if o.kind == "door":
            assert p.location[2] - p.dims[2] / 2 == pytest.approx(0.0, abs=1e-3)


def test_windows_keep_clear_of_tall_furniture_against_their_wall():
    """Nothing taller than the sill stands in front of a window (a sofa may stand under one)."""
    for key in ("last/qwen/living_room", "last/qwen/cafe", "last/qwen/classroom"):
        spec, pls = _solve(key)
        sh = shell.plan(spec, pls)
        for o in sh.openings:
            wl = next(x for x in shell._rect_walls(*spec.setting.bounds_m[:2]) if x.name == o.wall)
            p = o.placement
            if o.kind == "window":
                sill = p.location[2] - p.dims[2] / 2
                pt = (p.location[0] - wl.n[0] * p.dims[1] / 2, p.location[1] - wl.n[1] * p.dims[1] / 2)
                assert not shell._blocked(pls, pt, wl.t, wl.n, p.dims[0] / 2, 0.4, sill - 0.02, sill + p.dims[2]), \
                    (key, o.id)


def test_the_spec_s_own_windows_win():
    spec, pls = _solve("last/qwen/living_room")
    win = next(p for p in pls if p.category in ("sofa",))
    own = replace(spec, objects=(*spec.objects, replace(spec.objects[0], id="w1", category="window", count=1)))
    sh = shell.plan(own, [*pls, replace(win, id="w1", category="window")])
    assert sh is not None and sh.openings == () and sh.trim  # trim only
    assert "its own windows" in sh.note


@pytest.mark.parametrize("brief", ["a windowless basement den with a sofa", "a home cinema with a sofa"])
def test_a_windowless_brief_gets_a_door_but_no_window(brief):
    spec, pls = _solve("last/qwen/living_room")
    sh = shell.plan(spec, pls, brief)
    assert [o.kind for o in sh.openings] == ["door"]
    assert "windowless" in sh.note


def test_no_shell_outdoors_or_in_a_classic_room_s_own_mouldings():
    fx = json.loads((FIX / "bench_specs_outdoor_aidb7.json").read_text())
    spec, errors = sp._spec_from_args({"spec": copy.deepcopy(fx["last/qwen/park_corner"]["spec"])}, remembered=False)
    assert spec is not None, errors
    assert shell.plan(spec, []) is None
    room, pls = _solve("last/qwen/living_room")
    classic = replace(room, setting=replace(room.setting, style="classic"))
    sh = shell.plan(classic, pls)
    assert sh is not None and sh.trim == ()  # its crown and baseboard (setting.moulding_list) stay
    bare = replace(room, setting=replace(room.setting, mouldings=()))
    assert shell.plan(bare, pls).trim == ()


def test_a_round_room_gets_windows_round_its_wall():
    spec, pls = _solve("last/qwen/round_lobby")
    sh = shell.plan(spec, pls)
    assert spec.setting.shape == "ellipse"
    wins = [o for o in sh.openings if o.kind == "window"]
    assert len(wins) >= 3 and all(o.wall == "wall" for o in sh.openings)
    assert sum(o.kind == "door" for o in sh.openings) == 1
    assert all(o.placement.dims[2] > 1.6 for o in wins)  # a 5 m lobby: tall windows


def test_the_build_cuts_openings_and_sweeps_trim_but_frames_the_same_objects():
    spec, pls = _solve("last/qwen/living_room")
    sh = shell.plan(spec, pls)
    bare, _ = sp.build_layout_commands(spec, pls)
    cmds, _names = sp.build_layout_commands(spec, pls, shell=sh)
    room = next(c for c in cmds if c["type"] == "room_shell")["params"]
    assert {o["id"] for o in room["openings"]} == {o.id for o in sh.openings}
    assert {o["kind"] for o in room["openings"]} == {"window", "door"}
    assert any(c["type"] == "build_asset" and c["params"]["spec"].get("asset", "").startswith("shell_window")
               or c["type"] == "build_asset" and "shell_window" in json.dumps(c["params"]) for c in cmds)
    mould = [c for c in cmds if c["type"] == "run_code" and "Moulding_skirting" in c["params"]["code"]]
    assert mould and "Moulding_cove" in mould[0]["params"]["code"]
    cam = next(c for c in cmds if c["type"] == "set_camera")
    assert cam == next(c for c in bare if c["type"] == "set_camera")
