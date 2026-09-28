"""Daemon configuration models (internal, separate from main config.py)."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DaemonProjectConfig:
    db_path: str
    config_path: str | None = None
    auto_start: bool = True


@dataclass
class DaemonConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    pid_file: str = "~/.config/ai-db/daemon.pid"
    projects: dict[str, DaemonProjectConfig] = field(default_factory=dict)
    auto_discover: bool = True
    shutdown_timeout: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "host": self.host,
            "port": self.port,
            "pid_file": self.pid_file,
            "auto_discover": self.auto_discover,
            "shutdown_timeout": self.shutdown_timeout,
            "projects": {
                name: {"db_path": p.db_path, "config_path": p.config_path, "auto_start": p.auto_start}
                for name, p in self.projects.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DaemonConfig":
        projects = {}
        for name, p in data.get("projects", {}).items():
            projects[name] = DaemonProjectConfig(
                db_path=p["db_path"],
                config_path=p.get("config_path"),
                auto_start=p.get("auto_start", True),
            )
        return cls(
            enabled=data.get("enabled", False),
            host=data.get("host", "127.0.0.1"),
            port=data.get("port", 8080),
            pid_file=data.get("pid_file", "~/.config/ai-db/daemon.pid"),
            projects=projects,
            auto_discover=data.get("auto_discover", True),
            shutdown_timeout=data.get("shutdown_timeout", 30.0),
        )