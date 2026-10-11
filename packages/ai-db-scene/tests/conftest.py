"""Hermetic environment for ai-db-scene tests: never touch the user's DBs or config."""

import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
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


# --- a private headless Blender with the addon, for @pytest.mark.blender tests ---------------
# Never the user's Blender: each test gets its own process, its own port (picked by the server)
# and an empty factory scene.
_BLENDER = Path.home() / "bin" / "blender"  # evaluated at import, before HOME is redirected
_ADDON = Path(os.environ.get("AI_DB_BLENDER_ADDON", "~/GitRepos/blender_mcp")).expanduser()
_SERVE = """
import os, sys, time
sys.path.insert(0, os.environ["ADDON_ROOT"])
from importlib import import_module
srv = import_module(os.environ["ADDON_PKG"] + ".socket_server").MCPServerState
port = srv.start(0)
with open(os.environ["READY"] + ".tmp", "w") as f:
    f.write(str(port))
os.replace(os.environ["READY"] + ".tmp", os.environ["READY"])  # atomic: never half a port number
stop, parent = os.environ["STOP"], int(os.environ["PARENT"])
while not os.path.exists(stop) and os.getppid() == parent:
    srv.itc.process_pending()
    time.sleep(0.01)
srv.stop()
os._exit(0)
"""
_START_S = 180.0  # upper bound for Blender to start listening (seconds under heavy load)


def _addon_package() -> str | None:
    found = [d.parent.name for d in _ADDON.glob("*/socket_server.py")] if _ADDON.exists() else []
    return found[0] if len(found) == 1 else None


def _blender_exe() -> str | None:
    return str(_BLENDER) if _BLENDER.exists() else shutil.which("blender")


@pytest.fixture
def private_blender(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    """Port of a fresh headless Blender serving the addon; AI_DB_BLENDER_PORT points at it."""
    exe, pkg = _blender_exe(), _addon_package()
    if exe is None or pkg is None:
        pytest.skip("Blender/addon not available")
    ready, stop, script = tmp_path / "blender_ready", tmp_path / "blender_stop", tmp_path / "serve.py"
    script.write_text(_SERVE, encoding="utf-8")
    env = {**os.environ, "ADDON_ROOT": str(_ADDON), "ADDON_PKG": pkg, "READY": str(ready),
           "STOP": str(stop), "PARENT": str(os.getpid())}
    proc = subprocess.Popen([exe, "-b", "--factory-startup", "--python", str(script)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + _START_S
        while not ready.exists():  # the server's own signal, not a guessed delay
            assert proc.poll() is None, f"Blender exited early (rc={proc.returncode})"
            assert time.monotonic() < deadline, "Blender did not start listening"
            time.sleep(0.05)
        port = int(ready.read_text(encoding="utf-8"))
        monkeypatch.setenv("AI_DB_BLENDER_PORT", str(port))
        yield port
    finally:
        stop.write_text("1", encoding="utf-8")
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
