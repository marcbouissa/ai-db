"""Reboot recovery: no state under /tmp, and a stale pid file never blocks a start."""

import json
import os
import subprocess
import sys
import time

from ai_db.config import DaemonConfig, load_config
from ai_db.constants import DEFAULT_DAEMON_PID_FILE
from ai_db.daemon.manager import DaemonManager, is_daemon_process


def test_state_and_runtime_defaults_never_point_into_tmp(tmp_path):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("XDG_", "AI_DB_"))}
    env["HOME"] = "/home/aidb-test"
    env["XDG_RUNTIME_DIR"] = "/run/user/4242"
    code = ("import json, ai_db.constants as c, ai_db.config as g;"
            "print(json.dumps([c.DEFAULT_DB_FILE, c.DEFAULT_CONFIG_FILE, c.DAEMON_LOG_FILE,"
            " c.DEFAULT_DAEMON_PID_FILE, c.RUNTIME_DIR, c.STATE_DIR, c.OUTPUT_STORE_DIR,"
            " g.DaemonConfig().pid_file, g.config_path(None)]))")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         check=True).stdout
    paths = json.loads(out)
    assert all(not p.startswith("/tmp") for p in paths), paths
    assert paths[3] == "/run/user/4242/ai-db/daemon.pid"


def test_pid_file_default_is_per_boot():
    assert DaemonConfig().pid_file == DEFAULT_DAEMON_PID_FILE


def _manager(tmp_path, pid: int) -> DaemonManager:
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text(str(pid))
    manager = DaemonManager(load_config())
    manager.pid_file = pid_file
    return manager


def test_a_reused_pid_is_not_a_running_daemon(tmp_path):
    # After a reboot the old pid can belong to anything, e.g. this pytest process.
    assert not is_daemon_process(os.getpid())
    assert not _manager(tmp_path, os.getpid()).is_running()


def test_a_live_daemon_process_is_running(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
                             "-m", "ai_db.cli", "daemon", "start"])
    try:
        deadline = time.monotonic() + 5
        while not is_daemon_process(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _manager(tmp_path, proc.pid).is_running()
    finally:
        proc.kill()
        proc.wait()
    assert not _manager(tmp_path, proc.pid).is_running()
