"""The ai-db-scene wheel ships its skills, data files and entry points."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parents[1]


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
def test_wheel_contains_skills_and_data(tmp_path: Path) -> None:
    proc = subprocess.run(["uv", "build", str(PKG.relative_to(ROOT)), "-o", str(tmp_path), "-q"],
                          cwd=ROOT, capture_output=True, text=True, timeout=300, check=False)
    if proc.returncode != 0 and "network" in (proc.stderr or "").lower():
        pytest.skip("build backend not available offline")
    assert proc.returncode == 0, proc.stderr
    wheels = list(tmp_path.glob("ai_db_scene-*.whl"))
    assert len(wheels) == 1
    names = set(zipfile.ZipFile(wheels[0]).namelist())
    src = PKG / "ai_db_scene"
    expected = {f"ai_db_scene/{p.relative_to(src).as_posix()}" for p in src.glob("skills/*/SKILL.md")}
    expected |= {f"ai_db_scene/{p.relative_to(src).as_posix()}" for p in src.rglob("*.json")
                 if "__pycache__" not in p.parts}
    assert expected and "ai_db_scene/priors.json" in expected
    assert expected <= names, sorted(expected - names)
    with zipfile.ZipFile(wheels[0]) as z:
        meta_dir = next(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA"))
        meta = z.read(f"{meta_dir}/METADATA").decode()
        eps = z.read(f"{meta_dir}/entry_points.txt").decode()
    assert "Version: " in meta and "Requires-Dist: ai-db" in meta
    for group in ("ai_db.tools", "ai_db.skills", "ai_db.skill_drafts", "ai_db.skill_sources", "ai_db.chunkers"):
        assert f"[{group}]" in eps
