"""Curved walls read as one surface (lead review of the qwen aidb7 round lobby: vertical bands).

The round room's shell had room_shell's default 64 facets, flat-shaded: on a 10 m lobby each
facet was ~0.5 m wide and caught the light on its own. Now the shell (and its moulding) has
128 facets and the room surfaces are smooth-shaded with sharp edges kept above 30 degrees.
"""

from __future__ import annotations

import math

import pytest
from ai_db_scene.layout import solve_layout
from ai_db_scene.spec import validate_spec
from ai_db_scene.tools import scene_pipeline as sp


def _cmds(shape: str, kind: str = "indoor", mouldings: list[str] | None = None) -> list[dict]:
    raw = {"setting": {"kind": kind, "bounds_m": [10, 10, 4], "shape": shape,
                       **({"mouldings": mouldings} if mouldings else {})},
           "objects": [{"id": "plant", "category": "potted_plant"}], "relations": []}
    spec, errors = validate_spec(raw)
    assert spec is not None, errors
    cmds, _ = sp.build_layout_commands(spec, solve_layout(spec))
    return cmds


@pytest.mark.parametrize("kind", ["indoor", "outdoor"])
def test_round_room_shell_has_fine_facets(kind: str) -> None:
    shell = next(c for c in _cmds("ellipse", kind) if c["type"] == "room_shell")
    segs = shell["params"]["segments"]
    assert segs >= 128
    facet = math.pi * 10 / segs  # facet width on a 10 m round room
    assert facet < 0.26


def test_room_surfaces_are_smooth_shaded_with_sharp_corners() -> None:
    code = next(c["params"]["code"] for c in _cmds("ellipse") if c["type"] == "run_code"
                and "room_surfaces" in c["params"]["code"])
    assert "shade_smooth()" in code and "set_sharp_from_angle" in code
    assert "math.radians(30.0)" in code
    compile(code, "room_material", "exec")


def test_rect_room_has_no_segments_param() -> None:
    shell = next(c for c in _cmds("rect") if c["type"] == "room_shell")
    assert "segments" not in shell["params"]


def test_moulding_follows_the_same_facets() -> None:
    assert sp._ROOM_ELLIPSE_SEGMENTS >= 128
    jobs = [c for c in _cmds("ellipse", mouldings=["baseboard"]) if c["type"] == "run_code"
            and "mouldings" in c["params"]["code"]]
    assert jobs
    payload = jobs[0]["params"]["code"]
    assert payload.count("], [") >= sp._ROOM_ELLIPSE_SEGMENTS - 2
