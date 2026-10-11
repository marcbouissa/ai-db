"""Hermetic environment for ai-db-scene tests: never touch the user's DBs or config."""

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from ai_db.config_template import build_template

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(build_template()), encoding="utf-8")
    monkeypatch.setenv("AI_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("AI_DB_CONFIG", str(config_file))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # scene store defaults under ~
    monkeypatch.setenv("AI_DB_SCENE_MEMORY", str(tmp_path / "home" / "scene_memory.db"))
    for var in ("AI_DB_DAEMON_URL", "AI_DB_API_KEY", "AI_DB_ADMIN_KEY"):  # never the live daemon
        monkeypatch.delenv(var, raising=False)
    return tmp_path
