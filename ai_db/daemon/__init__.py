"""ai-db Daemon Module.

Provides a long-running HTTP server that holds VectorDB instances with loaded
embedding/rerank models, serving requests from CLI and MCP clients.
Daemon settings live in ``ai_db.config.DaemonConfig`` (the ``daemon`` config key).

Exports resolve lazily: importing ``ai_db.daemon.client`` (which every CLI command does)
must not import the FastAPI server, which alone costs ~280 ms.
"""

from typing import Any

_EXPORTS = {
    "DaemonClient": "ai_db.daemon.client",
    "create_daemon_client": "ai_db.daemon.client",
    "DaemonManager": "ai_db.daemon.manager",
    "get_daemon_manager": "ai_db.daemon.manager",
    "ProjectPool": "ai_db.daemon.server",
    "app": "ai_db.daemon.server",
    "get_pool": "ai_db.daemon.server",
}

__all__ = ["DaemonClient", "DaemonManager", "ProjectPool", "app", "create_daemon_client", "get_daemon_manager", "get_pool"]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'ai_db.daemon' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(_EXPORTS[name]), name)
