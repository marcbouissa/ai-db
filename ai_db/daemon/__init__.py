"""ai-db Daemon Module.

Provides a long-running HTTP server that holds VectorDB instances with loaded
embedding/rerank models, serving requests from CLI and MCP clients.
"""

from ai_db.daemon.client import DaemonClient, create_daemon_client
from ai_db.daemon.config import DaemonConfig, DaemonProjectConfig
from ai_db.daemon.manager import DaemonManager, get_daemon_manager
from ai_db.daemon.server import ProjectPool, app, get_pool

__all__ = [
    "DaemonClient",
    "DaemonConfig",
    "DaemonManager",
    "DaemonProjectConfig",
    "ProjectPool",
    "app",
    "create_daemon_client",
    "get_daemon_manager",
    "get_pool",
]