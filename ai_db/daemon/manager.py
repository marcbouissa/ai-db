"""Daemon lifecycle management (start/stop/status/register/unregister)."""

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from ai_db.config import AppConfig, load_config
from ai_db.daemon.config import DaemonConfig as InternalDaemonConfig, DaemonProjectConfig


class DaemonManager:
    """Manages daemon lifecycle."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        # Use internal mutable config for runtime modifications
        self.daemon_config = InternalDaemonConfig.from_dict(self.config.daemon.to_dict())
        self.pid_file = Path(self.daemon_config.pid_file).expanduser()
        self.base_url = f"http://{self.daemon_config.host}:{self.daemon_config.port}"

    def is_running(self) -> bool:
        """Check if daemon process is running."""
        if not self.pid_file.exists():
            return False
        try:
            pid = int(self.pid_file.read_text().strip())
            os.kill(pid, 0)  # Signal 0 just checks existence
            return True
        except (ValueError, OSError, ProcessLookupError):
            return False

    def get_pid(self) -> int | None:
        """Get daemon PID if running."""
        if not self.pid_file.exists():
            return None
        try:
            return int(self.pid_file.read_text().strip())
        except ValueError:
            return None

    async def _wait_for_health(self, timeout: float = 10.0) -> bool:
        """Wait for daemon to become healthy."""
        start = time.time()
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() - start < timeout:
                try:
                    resp = await client.get(f"{self.base_url}/health")
                    if resp.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        return False

    def start(self, foreground: bool = False, host: str | None = None, port: int | None = None) -> int:
        """Start the daemon."""
        if self.is_running():
            print(f"Daemon already running (PID: {self.get_pid()})")
            return 1

        # Update config if overrides provided
        if host:
            self.daemon_config.host = host
        if port:
            self.daemon_config.port = port
        self.base_url = f"http://{self.daemon_config.host}:{self.daemon_config.port}"

        if not foreground:
            # Double-fork daemonization
            return self._daemonize()
        else:
            # Run in foreground
            return self._run_server()

    def _daemonize(self) -> int:
        """Double-fork to daemonize."""
        # First fork
        try:
            pid = os.fork()
            if pid > 0:
                # Parent exits
                print(f"Daemon started in background (PID: {pid})")
                return 0
        except OSError as e:
            print(f"Fork failed: {e}", file=sys.stderr)
            return 1

        # Decouple from parent environment
        os.chdir("/")
        os.setsid()
        os.umask(0)

        # Second fork
        try:
            pid = os.fork()
            if pid > 0:
                # Parent exits
                sys.exit(0)
        except OSError as e:
            print(f"Second fork failed: {e}", file=sys.stderr)
            sys.exit(1)

        # Redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        with open(os.devnull, "rb") as devnull_in:
            os.dup2(devnull_in.fileno(), sys.stdin.fileno())
        with open(os.devnull, "ab") as devnull_out:
            os.dup2(devnull_out.fileno(), sys.stdout.fileno())
            os.dup2(devnull_out.fileno(), sys.stderr.fileno())

        # Write PID file
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))

        # Run server
        return self._run_server()

    def _run_server(self) -> int:
        """Run the FastAPI server."""
        import uvicorn

        # Ensure PID file exists
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))

        try:
            # Import here to avoid circular imports
            from ai_db.daemon.server import app

            uvicorn.run(
                app,
                host=self.daemon_config.host,
                port=self.daemon_config.port,
                log_level="info",
            )
        finally:
            # Cleanup PID file on exit
            if self.pid_file.exists():
                self.pid_file.unlink(missing_ok=True)
        return 0

    def stop(self, force: bool = False) -> bool:
        """Stop the daemon."""
        pid = self.get_pid()
        if pid is None:
            print("Daemon not running")
            return False

        try:
            if force:
                os.kill(pid, signal.SIGKILL)
                print(f"Daemon forcefully killed (PID: {pid})")
            else:
                os.kill(pid, signal.SIGTERM)
                print(f"Daemon shutdown requested (PID: {pid})")

            # Wait for process to exit
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

            # Cleanup PID file
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
        """Get daemon status."""
        running = self.is_running()
        pid = self.get_pid()

        result = {
            "running": running,
            "pid": pid,
            "url": self.base_url,
            "config": self.daemon_config.to_dict(),
        }

        if running:
            # Try to get health from daemon
            try:
                import httpx
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(f"{self.base_url}/health")
                    if resp.status_code == 200:
                        result["health"] = resp.json()
                    else:
                        result["health"] = {"status": "unhealthy", "http_status": resp.status_code}
            except Exception as e:
                result["health"] = {"status": "unreachable", "error": str(e)}
        else:
            result["health"] = {"status": "stopped"}

        return result

    def register_project(
        self,
        name: str,
        db_path: str,
        config_path: str | None = None,
        auto_start: bool = True,
    ) -> dict[str, Any]:
        """Register a project with the daemon."""
        if not self.is_running():
            # Update config file directly
            return self._register_project_config(name, db_path, config_path, auto_start)

        try:
            with httpx.Client(timeout=10.0) as client:
                payload = {
                    "name": name,
                    "db_path": db_path,
                    "config_path": config_path,
                    "auto_start": auto_start,
                }
                resp = client.post(f"{self.base_url}/projects", json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to register project: {e}")

    def _register_project_config(
        self,
        name: str,
        db_path: str,
        config_path: str | None,
        auto_start: bool,
    ) -> dict[str, Any]:
        """Register project by updating config file."""
        config_path_resolved = self.config.source_path or self.config_path()
        with open(config_path_resolved, "r") as f:
            raw = json.load(f)

        if "daemon" not in raw:
            raw["daemon"] = {}
        if "projects" not in raw["daemon"]:
            raw["daemon"]["projects"] = {}

        raw["daemon"]["projects"][name] = {
            "db_path": db_path,
            "config_path": config_path,
            "auto_start": auto_start,
        }

        with open(config_path_resolved, "w") as f:
            json.dump(raw, f, indent=2)
            f.write("\n")

        return {"status": "ok", "message": f"Project '{name}' registered in config (daemon not running)"}

    def config_path(self) -> str:
        """Get config file path."""
        from ai_db.config import config_path
        return config_path(None)

    def unregister_project(self, name: str) -> dict[str, Any]:
        """Unregister a project."""
        if not self.is_running():
            return self._unregister_project_config(name)

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.delete(f"{self.base_url}/projects/{name}")
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to unregister project: {e}")

    def _unregister_project_config(self, name: str) -> dict[str, Any]:
        """Unregister project by updating config file."""
        config_path = self.config_path()
        with open(config_path, "r") as f:
            raw = json.load(f)

        if "daemon" in raw and "projects" in raw["daemon"] and name in raw["daemon"]["projects"]:
            del raw["daemon"]["projects"][name]
            with open(config_path, "w") as f:
                json.dump(raw, f, indent=2)
                f.write("\n")
            return {"status": "ok", "message": f"Project '{name}' unregistered from config"}
        else:
            raise RuntimeError(f"Project '{name}' not found in config")

    def reload_config(self) -> dict[str, Any]:
        """Reload daemon configuration."""
        if not self.is_running():
            raise RuntimeError("Daemon not running")

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(f"{self.base_url}/daemon/reload")
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to reload config: {e}")


# Convenience function
def get_daemon_manager(config: AppConfig | None = None) -> DaemonManager:
    return DaemonManager(config)