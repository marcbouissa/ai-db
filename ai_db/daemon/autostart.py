"""Start the local daemon on demand, and write an optional systemd user unit for it.

Every client (CLI command, `ai-db mcp`) reaches the daemon through ``DaemonClient``. When
a request cannot connect, ``ensure_daemon`` starts ``ai-db daemon start --foreground``
detached (its own session, log in DAEMON_LOG_FILE), waits for ``/health`` and the request
is retried once. On 14-16 Oct the systemd-managed daemon had exited cleanly (so
``Restart=on-failure`` did not restart it) and every CLI call and the MCP server failed
until someone started it by hand.

Only a daemon on this machine (loopback URL) is started. ``AI_DB_DAEMON_AUTOSTART=0``
turns it off; the error then says how to start the daemon.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from ai_db.constants import (
    DAEMON_AUTOSTART_WAIT_S,
    DAEMON_HEALTH_PROBE_S,
    DAEMON_LOG_FILE,
    RUNTIME_DIR,
)
from ai_db.errors import AiDbConfigError, AiDbDaemonUnavailableError

_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
_POLL_S = 0.25
AUTOSTART_ENV = "AI_DB_DAEMON_AUTOSTART"
SERVICE_NAME = "ai-db-daemon.service"

HOW_TO_START = ("start it with `ai-db daemon start` (or `systemctl --user start <unit>` if you "
                "installed one with `ai-db daemon install-service`)")


def autostart_enabled() -> bool:
    value = os.environ.get(AUTOSTART_ENV, "1").strip().lower()
    if value not in ("0", "1", "false", "true", "no", "yes", "off", "on"):
        raise AiDbConfigError(f"{AUTOSTART_ENV}={value!r}: use 1 (default) or 0")
    return value in ("1", "true", "yes", "on")


def health_ok(base_url: str, timeout: float = DAEMON_HEALTH_PROBE_S) -> bool:
    """One ``GET /health``; False when nothing answers (connection refused / timeout)."""
    try:
        with httpx.Client(timeout=timeout) as client:
            return client.get(f"{base_url}/health").status_code == 200
    except httpx.TransportError:
        return False  # not listening (yet): the caller decides what that means


@contextlib.contextmanager
def _start_lock() -> Iterator[None]:
    """One starter at a time: parallel CLI calls must not each spawn a daemon."""
    Path(RUNTIME_DIR).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(RUNTIME_DIR, "daemon-start.lock"), "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def daemon_command(host: str, port: int) -> list[str]:
    """The command an auto-start (and the systemd unit) runs."""
    return [sys.executable, "-m", "ai_db.cli", "daemon", "start", "--foreground",
            "--host", host, "--port", str(port)]


def ensure_daemon(base_url: str, reason: str) -> None:
    """Make the daemon at ``base_url`` answer ``/health``, starting it if it is local.

    ``reason`` is why the caller thinks it is down (shown if it cannot be started).
    Raises ``AiDbDaemonUnavailableError`` with how to start it by hand.
    """
    parts = urlsplit(base_url)
    host, port = parts.hostname or "", parts.port or 80
    down = f"daemon at {base_url} unreachable ({reason})"
    if host not in _LOOPBACK:
        raise AiDbDaemonUnavailableError(
            f"{down}; it is not on this machine, so it is not started automatically")
    if not autostart_enabled():
        raise AiDbDaemonUnavailableError(f"{down}; {AUTOSTART_ENV}=0, so {HOW_TO_START}")
    with _start_lock():
        if health_ok(base_url):
            return  # another client started it while this one waited for the lock
        _spawn_and_wait(base_url, host, port, down)


def _spawn_and_wait(base_url: str, host: str, port: int, down: str) -> None:
    log = Path(DAEMON_LOG_FILE)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ai-db] {down}: starting it (log: {log})", file=sys.stderr)
    with open(log, "ab") as out, open(os.devnull, "rb") as devnull:
        # Own session: the daemon outlives this CLI call / MCP server and its terminal.
        # The config is inherited through AI_DB_CONFIG (set by the CLI before any call).
        proc = subprocess.Popen(daemon_command(host, port), stdin=devnull, stdout=out,
                                stderr=out, cwd="/", start_new_session=True)
    deadline = time.monotonic() + DAEMON_AUTOSTART_WAIT_S
    while time.monotonic() < deadline:
        if health_ok(base_url):
            print(f"[ai-db] daemon started (pid {proc.pid})", file=sys.stderr)
            return
        code = proc.poll()
        if code is not None:
            raise AiDbDaemonUnavailableError(
                f"{down}; starting it failed (`{' '.join(daemon_command(host, port))}` exited "
                f"with {code}, see {log}); {HOW_TO_START}")
        time.sleep(_POLL_S)
    raise AiDbDaemonUnavailableError(
        f"{down}; started it (pid {proc.pid}) but /health did not answer within "
        f"{DAEMON_AUTOSTART_WAIT_S:g}s (see {log}); {HOW_TO_START}")


# --- systemd user unit ------------------------------------------------------------------


def service_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "systemd" / "user" / SERVICE_NAME


def service_unit(config_file: str, host: str, port: int) -> str:
    argv = daemon_command(host, port)
    exec_start = " ".join([argv[0], *argv[1:3], "--config", config_file, *argv[3:]])
    return f"""[Unit]
Description=ai-db daemon (written by `ai-db daemon install-service`)

[Service]
Type=simple
ExecStart={exec_start}
WorkingDirectory=/
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""


def install_service(config_file: str, host: str, port: int, force: bool = False) -> Path:
    """Write the unit file (never enables or starts it) and return its path."""
    path = service_path()
    if path.exists() and not force:
        raise AiDbConfigError(f"{path} already exists; pass --force to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(service_unit(config_file, host, port), encoding="utf-8")
    return path


def other_units(path: Path) -> list[Path]:
    """Other user units that already run an ai-db daemon (they would fight over the port)."""
    found = []
    for unit in sorted(path.parent.glob("*.service")):
        if unit == path:
            continue
        with contextlib.suppress(OSError):  # unreadable unit: not ours to judge
            text = unit.read_text(encoding="utf-8")
            if "daemon start" in text and ("ai-db" in text or "ai_db" in text):
                found.append(unit)
    return found


def print_service_install(host: str, port: int, force: bool) -> int:
    """`ai-db daemon install-service`: write the unit, print how to enable it."""
    from ai_db.config import config_path

    path = install_service(config_path(None), host, port, force=force)
    print(f"Wrote {path}")
    for other in other_units(path):
        print(f"Note: {other} also runs an ai-db daemon; enable only one of them "
              f"(both would bind {host}:{port}).")
    print("Enable it (starts now and at every login):")
    print(f"  systemctl --user daemon-reload && systemctl --user enable --now {SERVICE_NAME}")
    print(f"Logs: journalctl --user -u {SERVICE_NAME} -f")
    return 0
