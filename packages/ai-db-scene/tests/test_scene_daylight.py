"""Daylight through the windows (daylight.py): the sun outside the main window wall, a sky glow in
every opening, a weak ceiling bounce, warm lamps; a windowless room keeps the old rig."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from ai_db_scene import daylight, shell
from ai_db_scene.constants import SCENE_FILL_W_PER_M2, SCENE_LAMP_W, SCENE_SUN_ENERGY
from ai_db_scene.layout import Placement
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

FIX = Path(__file__).parent / "fixtures"
IN = json.loads((FIX / "bench_specs_aidb7.json").read_text())


def _room():
    spec, errors = sp._spec_from_args({"spec": copy.deepcopy(IN["last/qwen/living_room"]["spec"])}, remembered=False)
    assert spec is not None, errors
    fixed = solve_with_repairs(spec.to_dict(), 0, 3)
    return fixed.spec, fixed.placements


def _rig(cmds):
    return next(c for c in cmds if c["type"] == "daylight_rig")["params"]


def _fill(cmds):
    return next(c for c in cmds if c["type"] == "add_light" and c["params"]["name"] == "ai_db_fill")["params"]


def test_the_sun_stands_outside_the_main_window_wall():
    # two windows in the north wall (front faces the room: yaw 0), one in the east wall
    wins = [Placement(f"w{k}", (x, 6.0, 1.7), 0.0, (1.5, 0.08, 1.6), "window") for k, x in enumerate((1.5, 4.5))]
    wins.append(Placement("w9", (6.0, 3.0, 1.7), 270.0, (1.5, 0.08, 1.6), "window"))
    az = daylight.sun_azimuth(wins)
    out = (math.cos(math.radians(az)), math.sin(math.radians(az)))
    assert out[1] > 0.7  # toward +Y: outside the north wall
    assert daylight.sun_azimuth([]) is None


def test_a_room_with_windows_is_lit_through_them():
    spec, pls = _room()
    sh = shell.plan(spec, pls)
    s2, p2 = shell.apply(spec, pls, sh)
    off = sp.frame_offset(s2)
    lit = sp.lighting_commands(s2, p2, off)
    bare = sp.lighting_commands(spec, pls, off)
    assert _rig(lit)["strength"] == daylight.SUN_ENERGY and _rig(bare)["strength"] == SCENE_SUN_ENERGY
    assert _rig(lit)["sun_azimuth"] == daylight.sun_azimuth(p2)
    assert _rig(lit)["exposure"] < _rig(bare)["exposure"]
    assert _fill(lit)["energy"] < _fill(bare)["energy"] / 3  # a bounce, not the main light
    glow = [c["params"] for c in lit if c["type"] == "add_light" and c["params"]["name"].endswith("_daylight")]
    assert len(glow) == sum(o.kind == "window" for o in sh.openings)
    assert any(c["type"] == "run_code" and "ai_db_window_glass" in c["params"]["code"] for c in lit)
    lamp = next(c["params"] for c in lit if c["type"] == "add_light" and c["params"]["name"] == "floor_lamp_light")
    assert lamp["energy"] == SCENE_LAMP_W["floor_lamp"] * daylight.LAMP_BOOST


def test_a_window_glow_faces_into_the_room():
    w = Placement("w", (0.04, 3.0, 1.7), 90.0, (1.5, 0.08, 1.6), "window")  # west wall: the room is +X
    (light, _shape) = daylight.window_light_commands([w], (0.0, 0.0, 0.0))[:2]
    rx, _ry, rz = (math.radians(v) for v in light["params"]["rotation"])
    # -Z turned by X 90 then Z rz: (-sin rz, cos rz) in plan
    d = (-math.sin(rz) * math.sin(rx), math.cos(rz) * math.sin(rx))
    assert d[0] > 0.99 and light["params"]["location"][0] > w.location[0]


def test_a_windowless_room_keeps_the_old_rig():
    spec, pls = _room()
    off = sp.frame_offset(spec)
    cmds = sp.lighting_commands(spec, pls, off)
    rig = _rig(cmds)
    assert (rig["sun_azimuth"], rig["strength"]) == (135.0, SCENE_SUN_ENERGY)
    area = sp.floor_area_m2(spec)
    assert _fill(cmds)["energy"] == round(min(max(SCENE_FILL_W_PER_M2 * area, 100.0), 2500.0), 1)
    assert not any(c["params"].get("name", "").endswith("_daylight") for c in cmds if c["type"] == "add_light")
