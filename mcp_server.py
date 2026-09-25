#!/usr/bin/env python3
"""
ai-db MCP Server (stdio JSON-RPC 2.0)
=====================================
Exposes code intelligence and vector indexing capabilities to agents via Model Context Protocol:
Dynamically reflects all registered tools from ServiceDispatcher and routes execution uniformly.
"""

import json
import os
import sys
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ai_db import __version__
from ai_db.constants import DEFAULT_DB_FILE
from ai_db.dispatcher import ServiceDispatcher


class StdioMCPServer:
    """Stdio JSON-RPC 2.0 server wrapping ServiceDispatcher."""

    def __init__(self, db_path: str = DEFAULT_DB_FILE, dispatcher: ServiceDispatcher | None = None,
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
            raw_tools = self.dispatcher.list_tools()
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
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Unknown tool: {tool_name}"
                            }
                        ],
                        "isError": True
                    }
                }
            except Exception as err:  # noqa: BLE001 - JSON-RPC boundary: report to client
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error executing tool '{tool_name}': {err!s}"
                            }
                        ],
                        "isError": True
                    }
                }

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


def run_stdio(db_path: str = DEFAULT_DB_FILE, dispatcher: ServiceDispatcher | None = None,
              config: Any | None = None):
    if dispatcher is None:
        from ai_db.config import load_config
        cfg = config if config is not None else load_config()
        dispatcher = ServiceDispatcher(db_path=db_path, config=cfg)
        dispatcher._get_db()  # fail fast on invalid storage/provider config
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

        resp = server.handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    db_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB_FILE
    run_stdio(db_file)
