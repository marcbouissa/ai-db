"""Daemon lifecycle management (start/stop/status/register/unregister)."""

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from ai_db.config import AppConfig, config_path, load_config
from ai_db.constants import DAEMON_CLIENT_TIMEOUT_S, DAEMON_LOG_FILE
from ai_db.daemon._json import as_dict
from ai_db.daemon.client import describe_http_error
from ai_db.errors import AiDbDaemonError

_HEALTH_POLL_S = 0.5


class DaemonManager:
    """Manages daemon lifecycle."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.daemon_config = self.config.daemon
        self.host = self.daemon_config.host
        self.port = self.daemon_config.port
        self.pid_file = Path(self.daemon_config.pid_file).expanduser()
        self.log_file = Path(DAEMON_LOG_FILE)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _config_file(self) -> str:
        """The config file this manager (and therefore the daemon) uses."""
        return self.config.source_path or config_path(None)

    @staticmethod
    def _admin_headers() -> dict[str, str]:
        key = os.environ.get("AI_DB_ADMIN_KEY")
        return {"Authorization": f"Bearer {key}"} if key else {}

    def get_pid(self) -> int | None:
        if not self.pid_file.exists():
            return None
        try:
            return int(self.pid_file.read_text().strip())
        except ValueError:
            return None

    def is_running(self) -> bool:
        pid = self.get_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)  # signal 0 only checks existence
            return True
        except OSError:
            return False

    async def _wait_for_health(self, timeout: float = 10.0) -> bool:
        start = time.time()
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() - start < timeout:
                with contextlib.suppress(httpx.TransportError):  # not listening yet
                    resp = await client.get(f"{self.base_url}/health")
                    if resp.status_code == 200:
                        return True
                await asyncio.sleep(_HEALTH_POLL_S)
        return False

    # --- start / stop ----------------------------------------------------------

    def start(self, foreground: bool = False, host: str | None = None, port: int | None = None) -> int:
        if self.is_running():
            print(f"Daemon already running (PID: {self.get_pid()})")
            return 1
        if host:
            self.host = host
        if port:
            self.port = port
        # The server reads the config through AI_DB_CONFIG (see server.lifespan).
        os.environ["AI_DB_CONFIG"] = self._config_file()
        return self._run_server() if foreground else self._daemonize()

    def _daemonize(self) -> int:
        """Double-fork; the daemon logs to DAEMON_LOG_FILE."""
        try:
            if os.fork() > 0:
                print(f"Daemon starting in background (log: {self.log_file})")
                return 0
        except OSError as e:
            print(f"Fork failed: {e}", file=sys.stderr)
            return 1

        # chdir("/") so the daemon never pins a directory; clients send their cwd with
        # each tool call, so relative paths still resolve where the user ran the command.
        os.chdir("/")
        os.setsid()
        os.umask(0o022)
        try:
            if os.fork() > 0:
                os._exit(0)
        except OSError as e:
            print(f"Second fork failed: {e}", file=sys.stderr)
            os._exit(1)

        sys.stdout.flush()
        sys.stderr.flush()
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(os.devnull, "rb") as devnull_in:
            os.dup2(devnull_in.fileno(), sys.stdin.fileno())
        with open(self.log_file, "ab") as log:
            os.dup2(log.fileno(), sys.stdout.fileno())
            os.dup2(log.fileno(), sys.stderr.fileno())
        return self._run_server()

    def _run_server(self) -> int:
        import uvicorn

        from ai_db.daemon.server import app

        logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))
        try:
            uvicorn.run(app, host=self.host, port=self.port, log_level="info")
        finally:
            self.pid_file.unlink(missing_ok=True)
        return 0

    def stop(self, force: bool = False) -> bool:
        pid = self.get_pid()
        if pid is None:
            print("Daemon not running")
            return False
        try:
            os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
            print(f"Daemon {'killed' if force else 'shutdown requested'} (PID: {pid})")
            timeout = self.daemon_config.shutdown_timeout
            start = time.time()
            while time.time() - start < timeout:
                try:
                    os.kill(pid, 0)
                    time.sleep(0.2)
                except ProcessLookupError:
                    break
            else:
                if not force:
                    print(f"Daemon did not stop gracefully within {timeout}s, forcing...")
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.5)
            self.pid_file.unlink(missing_ok=True)
            print("Daemon stopped")
            return True
        except ProcessLookupError:
            print("Daemon process not found")
            self.pid_file.unlink(missing_ok=True)
            return True
        except PermissionError:
            print(f"Permission denied stopping daemon (PID: {pid})", file=sys.stderr)
            return False

    def status(self) -> dict[str, Any]:
        running = self.is_running()
        result: dict[str, Any] = {
            "running": running,
            "pid": self.get_pid(),
            "url": self.base_url,
            "log_file": str(self.log_file),
            "config": self.daemon_config.to_dict(),
        }
        if not running:
            result["health"] = {"status": "stopped"}
            return result
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.base_url}/health")
            result["health"] = (resp.json() if resp.status_code == 200
                                else {"status": "unhealthy", "http_status": resp.status_code})
        except httpx.HTTPError as e:
            result["health"] = {"status": "unreachable", "error": describe_http_error(e)}
        except ValueError as e:
            result["health"] = {"status": "unhealthy", "error": f"/health is not JSON: {e}"}
        return result

    # --- projects --------------------------------------------------------------

    def _admin_call(self, method: str, path: str, what: str, **kw: Any) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=DAEMON_CLIENT_TIMEOUT_S, headers=self._admin_headers()) as client:
                resp = client.request(method, f"{self.base_url}{path}", **kw)
                resp.raise_for_status()
                return as_dict(resp.json())
        except httpx.HTTPStatusError as e:
            raise AiDbDaemonError(f"Failed to {what}: HTTP {e.response.status_code}: "
                                  f"{e.response.text.strip() or '(empty response body)'}") from e
        except httpx.HTTPError as e:
            raise AiDbDaemonError(f"Failed to {what}: {describe_http_error(e)}") from e
        except ValueError as e:
            raise AiDbDaemonError(f"Failed to {what}: response is not JSON: {e}") from e

    def _edit_config(self, edit: Callable[[dict[str, Any]], None]) -> None:
        """Apply ``edit`` to ``daemon.projects`` of the config file and write it back."""
        path = self._config_file()
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        edit(raw.setdefault("daemon", {}).setdefault("projects", {}))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
            f.write("\n")

    def register_project(self, name: str, db_path: str, config_path: str | None = None,
                         auto_start: bool = True) -> dict[str, Any]:
        """Register a project: in the running daemon, or in the config file if stopped."""
        entry = {"db_path": db_path, "config_path": config_path, "auto_start": auto_start}
        if self.is_running():
            return self._admin_call("POST", "/projects", "register project", json={"name": name, **entry})

        def add(projects: dict[str, Any]) -> None:
            projects[name] = entry

        self._edit_config(add)
        return {"status": "ok", "message": f"Project '{name}' registered in config (daemon not running)"}

    def unregister_project(self, name: str) -> dict[str, Any]:
        if self.is_running():
            return self._admin_call("DELETE", f"/projects/{name}", "unregister project")
        found: list[bool] = []

        def drop(projects: dict[str, Any]) -> None:
            found.append(projects.pop(name, None) is not None)

        self._edit_config(drop)
        if not found[0]:
            raise AiDbDaemonError(f"Project '{name}' not found in config")
        return {"status": "ok", "message": f"Project '{name}' unregistered from config"}

    def reload_config(self) -> dict[str, Any]:
        if not self.is_running():
            raise AiDbDaemonError("Daemon not running")
        return self._admin_call("POST", "/daemon/reload", "reload config")


def get_daemon_manager(config: AppConfig | None = None) -> DaemonManager:
    return DaemonManager(config)
