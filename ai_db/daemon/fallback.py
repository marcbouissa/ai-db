"""Daemon-less CLI path for commands that only need the local SQLite DB.

With ``AI_DB_DAEMON_URL`` set, `sync` and `context`/`remember` go to the daemon. When it
does not answer, ``DaemonClient`` first tries to start it (``ai_db.daemon.autostart``).
Only when that fails too do these commands run here, on the local DB, through the same
``ServiceDispatcher`` handlers the daemon itself runs (``ProjectPool.run_tool`` calls
``ServiceDispatcher.execute``): one implementation, two transports. The project and
database resolve the same way as in the daemon (``storage.options.path`` of the config, or
the default DB), and a notice on stderr says that the daemon was bypassed.

Other commands (query, investigate, ...) do not fall back: without the daemon they would
load the embedding/rerank models in the CLI process, which is what the daemon is for; they
fail with the error that says how to start it.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from ai_db.errors import AiDbDaemonUnavailableError

LOCAL_WHEN_DAEMON_DOWN = frozenset({"sync", "context", "ctx", "remember"})


def local_if_daemon_down(command: str, client: Any, daemon_url: str | None,
                         make_local: Callable[[], Any]) -> tuple[Any, str | None]:
    """``(dispatcher, daemon_url)`` to use: the daemon client, or local and ``None``."""
    if daemon_url is None or command not in LOCAL_WHEN_DAEMON_DOWN:
        return client, daemon_url
    try:
        client.health()  # starts the local daemon if it is down
    except AiDbDaemonUnavailableError as exc:
        print(f"[ai-db] {exc}", file=sys.stderr)
        print(f"[ai-db] running `{command}` without the daemon, directly on the local "
              "database", file=sys.stderr)
        client.close()
        return make_local(), None
    return client, daemon_url
