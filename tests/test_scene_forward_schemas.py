"""Forwarded scene tools must not advertise params the Blender addon rejects."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ai_db.mcp_tools import scene_tools

_DEFAULT = Path(__file__).resolve().parents[2] / "blender_mcp" / "tools.json"


def _tools_json() -> Path:
    return Path(os.environ.get("AI_DB_BLENDER_TOOLS_JSON") or _DEFAULT)


def test_forwarded_properties_subset_of_addon():
    path = _tools_json()
    if not path.exists():
        pytest.skip(f"{path} not present")
    addon = {t["name"]: set(t["inputSchema"].get("properties", {})) for t in json.loads(path.read_text())["tools"]}
    for name, _desc, schema in scene_tools._FORWARDED_TOOLS:
        op, drop = scene_tools._FORWARD_OVERRIDES.get(name, (name, ()))
        assert op in addon, f"{name}: handler {op!r} not in addon"
        extra = set(schema["properties"]) - set(drop) - addon[op]
        assert not extra, f"{name}: params unknown to addon handler {op}: {sorted(extra)}"
