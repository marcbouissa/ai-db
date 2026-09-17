"""
ai_db.server.http_server
Lightweight HTTP/JSON API server for ai-db.
Zero new dependencies — uses stdlib http.server only.

Usage:
    ai-db serve --port 8765

API:
    POST /          { "tool": "<name>", "args": { ... } }
    GET  /status    Returns DB status JSON
    GET  /health    Returns { "ok": true }

Supported tools mirror MCP: analyze, expand, locate, context_save,
context_recall, optimize, diff, callers, todos
"""
import json
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict


class _AiDbHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt: str, *args: Any):
        # Only log errors; suppress per-request noise
        pass

    def _send_json(self, status: int, data: Any):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        from ai_db import VectorDB
        db = VectorDB(self.server.db_path)
        if self.path in ("/health", "/health/"):
            self._send_json(200, {"ok": True})
        elif self.path in ("/status", "/status/"):
            self._send_json(200, db.status())
        else:
            self._send_json(404, {"error": "Not found. Use POST / with {tool, args}."})
        db.close()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body) if body else {}
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return

        tool = req.get("tool", "")
        args = req.get("args", {})

        from ai_db import VectorDB
        db = VectorDB(self.server.db_path)
        try:
            result = _dispatch(db, tool, args)
            self._send_json(200, result)
        except Exception as e:
            self._send_json(500, {"error": str(e)})
        finally:
            db.close()


def _dispatch(db: Any, tool: str, args: Dict[str, Any]) -> Any:
    """Routes tool name to the appropriate VectorDB method."""
    import os

    if tool == "locate":
        return db.locate_targets(
            q=args.get("query", ""),
            scope=args.get("scope", "."),
            k=int(args.get("k", 5))
        )
    elif tool == "analyze":
        filepath = args.get("filepath") or args.get("path", "")
        return db.analyze_file(
            filepath,
            depth=args.get("depth", "structure"),
            span=tuple(args["span"][:2]) if args.get("span") else None,
            focus=args.get("focus"),
            q=args.get("q"),
            since=args.get("since"),
        )
    elif tool == "expand":
        return db.expand_ref(args.get("ref", ""),
                             depth=args.get("depth", "full"),
                             span=tuple(args["span"][:2]) if args.get("span") else None)
    elif tool == "diff":
        filepath = args.get("filepath") or args.get("path", "")
        return db.diff_file(filepath, since=args.get("since", "last"))
    elif tool == "callers":
        return db.query_callers(args.get("name", ""), relative_to=os.getcwd())
    elif tool == "todos":
        return db.query_annotations(
            kind=args.get("kind"),
            filepath=args.get("filepath"),
        )
    elif tool == "optimize":
        return db.optimize()
    elif tool == "context_save":
        return db.save_context(
            session_id=args.get("session_id", "default"),
            summary=args.get("summary", ""),
            title=args.get("title"),
            active_files=args.get("active_files"),
            open_tasks=args.get("open_tasks"),
            full_notes=args.get("full_notes"),
        )
    elif tool == "context_recall":
        return db.get_context(
            session_id=args.get("session_id"),
        ) or {}
    elif tool == "status":
        return db.status()
    else:
        return {"error": f"Unknown tool: '{tool}'. Available: locate, analyze, expand, diff, callers, todos, optimize, context_save, context_recall, status"}


def start_http_server(db_path: str, host: str = "127.0.0.1", port: int = 8765):
    """Starts the HTTP server (blocking). Call from CLI."""
    server = HTTPServer((host, port), _AiDbHandler)
    server.db_path = db_path
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ai-db serve] Stopped.")
        server.server_close()
