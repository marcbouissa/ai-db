"""A classroom is seen from the back, over the rows, looking where the class looks (aidb8 review:
"the desk rows don't face a board on a side wall"). In qwen and lightning aidb8 the rows did face
the board, but the build camera stood on a side wall (qwen: every back spot at eye height had the
teacher's desk hidden behind the rows) or in a back corner (lightning), so the board sat on a side
of the picture and the rows ran across it."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ai_db_scene import shell
from ai_db_scene.constants import SCENE_CAMERA_HEIGHT_M
from ai_db_scene.repair import solve_with_repairs
from ai_db_scene.tools import scene_pipeline as sp

CASES = json.loads((Path(__file__).parent / "fixtures" / "bench_aidb8" / "classrooms.json").read_text())
RUNS = [k for k in CASES if k != "source"]


def _built(run):
    case = CASES[run]
    fixed = solve_with_repairs(case["spec"], 0, 3)
    return shell.apply(fixed.spec, fixed.placements, shell.plan(fixed.spec, fixed.placements, case["brief"]))


@pytest.mark.parametrize("run", RUNS)
def test_the_first_camera_looks_over_the_class_at_the_board(run):
    spec, pls = _built(run)
    off = sp.frame_offset(spec)
    pose = sp.camera_poses(spec, pls, off)[0]
    board = next(p for p in pls if p.category == "whiteboard")
    desks = [p for p in pls if p.category == "desk" and "teacher" not in p.id]
    yaw = math.radians(pose["rotation"][2])
    view = (-math.sin(yaw), math.cos(yaw))  # rotation [90, 0, yaw] looks along +Y turned by yaw
    fy = sum(-math.cos(math.radians(p.rotation_deg_z)) for p in desks)
    fx = sum(math.sin(math.radians(p.rotation_deg_z)) for p in desks)
    n = math.hypot(fx, fy)
    assert (view[0] * fx + view[1] * fy) / n > math.cos(math.radians(20)), pose  # along the rows
    bx, by = board.location[0] + off[0], board.location[1] + off[1]
    cx, cy = pose["location"][0], pose["location"][1]
    to_board = ((bx - cx), (by - cy))
    assert (view[0] * to_board[0] + view[1] * to_board[1]) / math.hypot(*to_board) > 0.9, pose  # board ahead
    # behind every row of desks
    assert all(((p.location[0] + off[0] - cx) * view[0] + (p.location[1] + off[1] - cy) * view[1]) > 0 for p in desks)
    assert pose["location"][2] > SCENE_CAMERA_HEIGHT_M  # over the rows: the teacher's desk is not hidden
