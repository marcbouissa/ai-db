"""Hermetic environment for ai-db-scene tests: never touch the user's DBs or config."""

import json
import os
from pathlib import Path

import pytest

# tree-sitter-language-pack resolves its grammar cache from XDG_CACHE_HOME/HOME when a parser is
# first needed, and downloads any grammar it does not find there. isolated_env points
# XDG_CACHE_HOME (and HOME here) at each test's tmp dir, so the first test that parsed
# code (alphabetically tests/fixtures/callflow) downloaded the grammar over the network into a
# throwaway dir: slow, and failing under load or without network. Pin the cache for the whole
# session (and CLI subprocesses) to where it was before any test touched the environment.
os.environ.setdefault("TREE_SITTER_LANGUAGE_PACK_CACHE_DIR",
                      os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"))


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
