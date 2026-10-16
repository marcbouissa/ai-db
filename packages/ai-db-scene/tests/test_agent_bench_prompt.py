"""The bench prompt names tools as the arm's runner offers them (aidb6, 2026-10-17).

nemotron-3.5-lightning-free-aidb5 (OpenCode) called bare discover_servers (5x), discover_tools
(2x) and build_scene: OpenCode offers them as blender_discover_servers and aidb_build_scene,
so each call failed. aidb6 also asks the model to pass the brief to build_scene (no qwen aidb5
build did, so counts were never checked).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "agent_bench"))
import briefs


def test_opencode_arms_get_prefixed_tool_names():
    text = briefs.prompt("classroom", True, 6, "opencode")
    assert "blender_discover_servers / blender_discover_tools / blender_get_tool_schema / blender_invoke_tool" in text
    assert "START HERE: aidb_build_scene" in text and "aidb_edit_scene(scene_id, steps)" in text
    assert " build_scene" not in text.replace("aidb_build_scene", "")
    assert "brief: the request above, word for word" in text


def test_strata_and_claude_arms_get_bare_names():
    for runner in ("strata", "claude"):
        text = briefs.prompt("classroom", True, 6, runner)
        assert "through discover_servers / discover_tools" in text and "START HERE: build_scene" in text
        assert "aidb_" not in text and "blender_" not in text


def test_earlier_versions_keep_their_exact_text():
    for v in (1, 2, 5):
        assert briefs.prompt("cafe", True, v, "opencode") == briefs.prompt("cafe", True, v)
    assert briefs.prompt("cafe", False, 1, "opencode") == briefs.prompt("cafe", False)
