"""Agent benchmark report (eval/agent_bench/report.py) from a synthetic bench folder."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "eval" / "agent_bench"


@pytest.fixture(scope="module")
def report() -> Any:
    sys.path.insert(0, str(BENCH_DIR))
    try:
        return importlib.import_module("report")
    finally:
        sys.path.remove(str(BENCH_DIR))


def _metrics(arm: str, brief: str, score: float, version: str | None) -> dict[str, Any]:
    tokens = {"input": 1000, "output": 200, "cache_write": 0, "cache_read": 0}
    return {"arm": arm, "brief": brief, "aidb_version": version, "harness_s": 10.0, "setup": None,
            "quality": {"score": score, "checklist_pct": 90.0, "blocking": 0},
            "agent": {"ok": True, "wall_s": 600.0, "model_s": 400.0, "tool_s": 200.0, "turns": 5,
                      "tokens": tokens, "cost_usd": 0.1, "tools": {}, "shell_commands": []}}


def _scene(out: Path, arm: str, brief: str, score: float, version: str | None, render: bool = True) -> None:
    d = out / arm / brief
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(_metrics(arm, brief, score, version)), encoding="utf-8")
    if render:
        (d / "probe_interior.png").write_bytes(b"\x89PNG")


def test_report_has_versions_thumbnails_and_aidb2_pairs(report, tmp_path):
    out = tmp_path / "bench"
    for brief in ("living_room", "cafe"):
        _scene(out, "m-raw", brief, 60.0, None)
        _scene(out, "m-aidb", brief, 70.0, "0123456789")
        _scene(out, "m-aidb2", brief, 85.0, "abcdef1234", render=brief != "cafe")
    (out / "notes-x").mkdir()  # not an arm: ignored
    text = Path(report.write(str(out))).read_text(encoding="utf-8")
    summary = text.split("## Summary per arm")[1].split("##")[0]
    rows = [ln for ln in summary.splitlines() if ln.startswith("| m-")]
    assert [r.split("|")[1].strip() for r in rows] == ["m-aidb", "m-aidb2", "m-raw"]
    assert "| m-aidb | 0123456 |" in summary and "| m-aidb2 | abcdef1 |" in summary and "| m-raw | – |" in summary
    paired = text.split("## With vs without ai-db, paired")[1].split("## Best scene")[0]
    assert "| m | aidb vs raw | 2 |" in paired and "| m | aidb2 vs raw | 2 |" in paired
    assert "| m | aidb2 vs aidb | 2 |" in paired
    assert "| 25.0 |" in paired and "| 15.0 |" in paired  # quality deltas
    every = text.split("## Every scene")[1].split("## Where")[0]
    assert '[<img src="m-aidb2/living_room/probe_interior.png" width="120"' in every
    assert "](m-aidb2/living_room/probe_interior.png)" in every
    cafe_v2 = next(ln for ln in every.splitlines() if ln.startswith("| m-aidb2 | cafe |"))
    assert "| abcdef1 | – |" in cafe_v2  # no render: no thumbnail
