#!/usr/bin/env python3
"""
ai-db MCP Server (stdio JSON-RPC 2.0)
=====================================
Exposes code intelligence and vector indexing capabilities to agents via Model Context Protocol:
Dynamically reflects all registered tools from ServiceDispatcher and routes execution uniformly.
"""

import argparse
import json
import os
import sys
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ai_db import __version__
from ai_db.constants import DEFAULT_DB_FILE
from ai_db.daemon.client import DaemonClient, create_daemon_client
from ai_db.dispatcher import ServiceDispatcher
from ai_db.mcp_tools.scene_tools import register_scene_tools
from ai_db.mcp_tools.video_tools import register_video_tools
from ai_db.video.factory import load_video_processors


def parse_args():
    parser = argparse.ArgumentParser(
        prog="ai-db mcp",
        description="Run stdio JSON-RPC MCP server for ai-db"
    )
    parser.add_argument("db_file", nargs="?", default=DEFAULT_DB_FILE, help="SQLite database file (default: %(default)s)")
    parser.add_argument("--daemon-url", default=None, help="Connect to ai-db daemon (http://host:port)")
    parser.add_argument("--project", default=None, help="Project name (required with --daemon-url)")
    return parser.parse_args()


class StdioMCPServer:
    """Stdio JSON-RPC 2.0 server wrapping ServiceDispatcher."""

    def __init__(self, db_path: str = DEFAULT_DB_FILE, dispatcher: ServiceDispatcher | DaemonClient | None = None,
                 stdout: Any = None):
        self.db_path = db_path
        self.dispatcher = dispatcher if dispatcher is not None else ServiceDispatcher(db_path=db_path)
        self.db = getattr(self.dispatcher, "db", None)
        # Where notifications/progress frames go. Overridable so tests can use
        # a fake stream instead of the real stdout.
        self.stdout = stdout if stdout is not None else sys.stdout

    def _progress_callback(self, meta: dict[str, Any]) -> Any:
        """Build a progress callback if the client supplied a progressToken.

        Returns None otherwise, so a client that does not ask for progress
        pays nothing and never sees notifications.
        """
        token = meta.get("progressToken")
        if token is None:
            return None
        out = self.stdout

        def report(done: int, total: int, message: str = "") -> None:
            frame = {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {
                    "progressToken": token,
                    "progress": done,
                    "total": total,
                    "message": message,
                },
            }
            # Written straight to the stream and flushed: a long sync must not
            # look like a hang to the client.
            out.write(json.dumps(frame) + "\n")
            out.flush()

        return report

    @staticmethod
    def _tool_error(req_id: Any, message: str) -> dict[str, Any]:
        """Build a JSON-RPC tool-error result so failures reach the client as data.

        Handlers must not raise: an exception unwinds run_stdio, the process
        exits, and the client reports only "EOF" with no cause attached.
        """
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": message}],
                "isError": True,
            },
        }

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any] | None:
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False}
                    },
                    "serverInfo": {
                        "name": "ai-db",
                        "version": __version__
                    }
                }
            }

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            try:
                raw_tools = self.dispatcher.list_tools()
            except Exception as err:  # noqa: BLE001 - JSON-RPC boundary: report to client
                # An unreachable daemon raises here (httpx.ConnectError). Letting
                # it escape kills the process and the client sees a bare EOF.
                return self._tool_error(req_id, f"Error listing tools: {err!s}")
            tools = []
            for t in raw_tools:
                td = dict(t)
                if "inputSchema" not in td and "parameters_schema" in td:
                    td["inputSchema"] = td["parameters_schema"]
                tools.append(td)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": tools
                }
            }

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            progress = self._progress_callback(params.get("_meta") or {})
            try:
                res = self.dispatcher.execute(tool_name, arguments, progress=progress)
                text = res if isinstance(res, str) else json.dumps(res, indent=2)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": text
                            }
                        ],
                        "isError": False
                    }
                }
            except KeyError:
                return self._tool_error(req_id, f"Unknown tool: {tool_name}")
            except Exception as err:  # noqa: BLE001 - JSON-RPC boundary: report to client
                return self._tool_error(
                    req_id, f"Error executing tool '{tool_name}': {err!s}"
                )

        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found"
                }
            }
        return None


def run_stdio(db_path: str = DEFAULT_DB_FILE, dispatcher: ServiceDispatcher | DaemonClient | None = None,
              config: Any | None = None, daemon_url: str | None = None, project: str | None = None):
    if dispatcher is None:
        if daemon_url:
            dispatcher = create_daemon_client(daemon_url, project=project)
        else:
            from ai_db.config import load_config
            cfg = config if config is not None else load_config()
            dispatcher = ServiceDispatcher(db_path=db_path, config=cfg)
            dispatcher._get_db()  # fail fast on invalid storage/provider config

    # Register scene memory and video tools.
    # Only for the local dispatcher: a DaemonClient proxies every tool call to the
    # daemon and serves its tool list from the daemon, so local registration is
    # both impossible (register_tool raises) and wrong (it would shadow the daemon's
    # tool set).
    if isinstance(dispatcher, ServiceDispatcher):
        register_scene_tools(dispatcher)
        register_video_tools(dispatcher)

    # Load video processors from entry points
    load_video_processors()

    server = StdioMCPServer(db_path=db_path, dispatcher=dispatcher)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                         "error": {"code": -32700, "message": f"Parse error: {exc}"}}) + "\n")
            sys.stdout.flush()
            continue

        try:
            resp = server.handle_request(req)
        except Exception as err:  # noqa: BLE001 - last-resort guard for the transport
            # Without this, any escaping exception (a handler that forgot to
            # catch, a bug in schema normalization) terminates the process and
            # the client reports only "EOF". Keep the session alive and say why.
            sys.stderr.write(
                f"[ai-db] unhandled error in {req.get('method')!r}: {err!r}\n"
            )
            sys.stderr.flush()
            if req.get("id") is None:
                continue
            resp = {
                "jsonrpc": "2.0",
                "id": req["id"],
                "error": {"code": -32603, "message": f"Internal error: {err!s}"},
            }
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    args = parse_args()
    run_stdio(args.db_file, daemon_url=args.daemon_url, project=args.project)
