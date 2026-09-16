#!/usr/bin/env python3
"""
ai-db MCP Server (stdio JSON-RPC 2.0)
=====================================
Exposes tokenopt-analyzer v2 capabilities to agents via Model Context Protocol:
- analyze: multi-depth code AST analysis with token budgeting and cursor continuation
- expand: progressive disclosure handle expansion (pay-per-section)
- locate: relevance-ranked target file/symbol locator using BM25
"""

import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from vectordb import VectorDB, DEFAULT_DB_FILE

TOOLS_REGISTRY = [
    {
        "name": "analyze",
        "description": "Analyze code AST with depth control (summary/structure/targeted/full), range targeting, or BM25 filtering. Returns symbols, opaque refs, and token metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths or glob patterns (max 32)"
                },
                "depth": {
                    "type": "string",
                    "enum": ["summary", "structure", "targeted", "full"],
                    "default": "structure",
                    "description": "Analysis depth (summary=sigs only, structure=AST outline, targeted=matching bodies, full=all)"
                },
                "q": {
                    "type": "string",
                    "description": "Question or concept filter to extract only relevant nodes"
                },
                "focus": {
                    "type": "string",
                    "description": "Symbol name or substring filter"
                },
                "span": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "[start_line, end_line] to extract range with context"
                },
                "ctx_lines": {
                    "type": "integer",
                    "default": 10,
                    "description": "Context lines around span"
                },
                "since": {
                    "type": "string",
                    "description": "Diff mode: inspect changed line spans since git hash or timestamp"
                },
                "max_out": {
                    "type": "integer",
                    "description": "Token budget ceiling; truncates by relevance and returns continuation cursor"
                },
                "cursor": {
                    "type": "string",
                    "description": "Continuation cursor handle from previous truncated batch"
                },
                "format": {
                    "type": "string",
                    "enum": ["stub", "sexp", "json"],
                    "default": "stub",
                    "description": "Output format: 'stub' (native code skeleton, best for LLM reasoning), 'sexp' (minimum tokens), or 'json'"
                }
            },
            "required": ["targets"]
        }
    },
    {
        "name": "expand",
        "description": "Expand an opaque progressive-disclosure handle (ref:hash) returned by analyze to inspect full or targeted source body.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Opaque ref handle (e.g. 'ref:b3b64697')"
                },
                "depth": {
                    "type": "string",
                    "enum": ["targeted", "full"],
                    "default": "full",
                    "description": "Expansion depth"
                },
                "span": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional sub-span [start_line, end_line] within the ref"
                }
            },
            "required": ["ref"]
        }
    },
    {
        "name": "locate",
        "description": "Locate top-k matching source files or symbol snippets by natural question or concept using BM25 ranking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query or concept to locate"
                },
                "scope": {
                    "type": "string",
                    "default": ".",
                    "description": "Directory or repository path scope"
                },
                "k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Max results to return"
                },
                "format": {
                    "type": "string",
                    "enum": ["stub", "sexp", "json"],
                    "default": "stub",
                    "description": "Output format: 'stub' (compact text lines), 'sexp' (S-expression), or 'json'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "context_save",
        "description": "Save current conversation memory, active files, pending tasks, and architectural decisions to database.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "default": "main",
                    "description": "Session identifier / checkpoint name"
                },
                "summary": {
                    "type": "string",
                    "description": "Concise summary of progress, goals, and architectural decisions"
                },
                "title": {
                    "type": "string",
                    "description": "Descriptive title for this session state"
                },
                "active_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Active files relevant to current session"
                },
                "open_tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Pending task list"
                },
                "project": {
                    "type": "string",
                    "description": "Project scope (default: auto-detected)"
                }
            },
            "required": ["summary"]
        }
    },
    {
        "name": "context_recall",
        "description": "Recall conversation memory, decisions, active files, and tasks to expand agent context limit across sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID to recall (default: latest in project)"
                },
                "query": {
                    "type": "string",
                    "description": "Optional search term to search across past memories with BM25"
                },
                "project": {
                    "type": "string",
                    "description": "Project scope (default: auto-detected)"
                }
            }
        }
    },
    {
        "name": "optimize",
        "description": "Defragment database, merge full-text FTS5 index B-trees, update query planner stats, and reclaim disk space.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prune_missing": {
                    "type": "boolean",
                    "default": True,
                    "description": "Prune records for deleted files before defragmentation"
                },
                "default_format": {
                    "type": "string",
                    "enum": ["stub", "sexp", "json"],
                    "description": "Configure and persist default output format for future queries"
                }
            }
        }
    }
]


class StdioMCPServer:
    def __init__(self, db_path: str = DEFAULT_DB_FILE):
        self.db_path = db_path
        self.db = VectorDB(db_path)

    def handle_request(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
                        "version": "2.0.0"
                    }
                }
            }

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": TOOLS_REGISTRY
                }
            }

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            try:
                result_content = self.dispatch_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result_content, indent=2) if not isinstance(result_content, str) else result_content
                            }
                        ],
                        "isError": False
                    }
                }
            except Exception as err:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error executing tool '{tool_name}': {str(err)}"
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

    def dispatch_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        if tool_name == "analyze":
            raw_targets = args.get("targets", [])
            if isinstance(raw_targets, str):
                raw_targets = [raw_targets]

            depth = args.get("depth", "structure")
            q = args.get("q")
            focus = args.get("focus")
            raw_span = args.get("span")
            span_tuple = tuple(raw_span[:2]) if raw_span and len(raw_span) >= 2 else None
            since = args.get("since")
            ctx_lines = args.get("ctx_lines", 10)
            max_out = args.get("max_out")
            cursor = args.get("cursor")

            fmt = args.get("format", "stub")

            if len(raw_targets) == 1 and not any(c in raw_targets[0] for c in ["*", "?", "["]) and os.path.isfile(os.path.expanduser(raw_targets[0])):
                res = self.db.analyze_file(
                    raw_targets[0],
                    depth=depth,
                    span=span_tuple,
                    focus=focus,
                    q=q,
                    since=since,
                    ctx_lines=ctx_lines
                )
                if max_out and res["meta"]["tokens_out"] > max_out:
                    res["meta"]["truncated"] = True
                    res["symbols"] = res["symbols"][:max(1, len(res["symbols"]) // 2)]
                out_data = res
            else:
                out_data = self.db.analyze_batch(
                    targets=raw_targets,
                    depth=depth,
                    q=q,
                    focus=focus,
                    span=span_tuple,
                    since=since,
                    max_out=max_out,
                    cursor=cursor,
                    ctx_lines=ctx_lines
                )

            if fmt == "stub":
                return VectorDB.format_as_stub(out_data)
            elif fmt == "sexp":
                return VectorDB.format_as_sexp(out_data)
            return out_data

        elif tool_name == "expand":
            ref = args.get("ref", "")
            depth = args.get("depth", "full")
            raw_span = args.get("span")
            span_tuple = tuple(raw_span[:2]) if raw_span and len(raw_span) >= 2 else None
            exp = self.db.expand_ref(ref, depth=depth, span=span_tuple)
            if not exp:
                raise ValueError(f"Ref handle '{ref}' not found or expired.")
            return exp

        elif tool_name == "locate":
            query = args.get("query", "")
            scope = args.get("scope", ".")
            k = args.get("k", 5)
            fmt = args.get("format", "stub")
            hits = self.db.locate_targets(query, scope=scope, k=k)
            if fmt == "stub":
                lines = []
                for h in hits:
                    span_str = f"L{h['span'][0]}-{h['span'][1]}" if h.get("span") else ""
                    lines.append(f"# {h['file']}:{span_str} (score:{h.get('score', 0)})")
                    snippet = h.get("snippet", "").strip()
                    if snippet:
                        for s_line in snippet.splitlines()[:3]:
                            lines.append(f"  {s_line}")
                    lines.append("")
                return "\n".join(lines).strip()
            elif fmt == "sexp":
                def sexp_esc(val):
                    return f'"{str(val).replace(chr(34), chr(92)+chr(34)).replace(chr(10), " ")}"'
                hit_sexps = []
                for h in hits:
                    span_s = f"({h['span'][0]} {h['span'][1]})" if h.get("span") else "nil"
                    hit_sexps.append(f"(:hit :file {sexp_esc(h['file'])} :name {sexp_esc(h.get('name'))} :span {span_s} :score {h.get('score', 0)})")
                return f"(:locate :query {sexp_esc(query)} :hits ({' '.join(hit_sexps)}))"
            return hits

        elif tool_name == "context_save":
            proj = args.get("project")
            res = self.db.save_context(
                session_id=args.get("session_id", "main"),
                summary=args.get("summary", ""),
                title=args.get("title"),
                active_files=args.get("active_files", []),
                open_tasks=args.get("open_tasks", []),
                project=proj
            )
            return {
                "saved": True,
                "session_id": res["session_id"],
                "project": res["project"],
                "timestamp": res["timestamp"]
            }

        elif tool_name == "context_recall":
            proj = args.get("project")
            sid = args.get("session_id")
            query = args.get("query")
            if query:
                return self.db.query_contexts(query, project=proj, top_k=3)
            ctx = self.db.get_context(session_id=sid, project=proj)
            if not ctx:
                return {"found": False, "message": "No saved context found"}
            # Return token-dense markdown ready for context injection
            return {
                "found": True,
                "session_id": ctx["session_id"],
                "project": ctx["project"],
                "title": ctx["title"],
                "summary": ctx["summary"],
                "active_files": ctx["active_files"],
                "open_tasks": ctx["open_tasks"],
                "notes": ctx["full_notes"]
            }

        elif tool_name == "optimize":
            prune_missing = args.get("prune_missing", True)
            default_format = args.get("default_format")
            return self.db.optimize(prune_missing=prune_missing, default_format=default_format)

        else:
            raise ValueError(f"Unknown tool: {tool_name}")


def run_stdio(db_path: str = DEFAULT_DB_FILE):
    server = StdioMCPServer(db_path)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue

        resp = server.handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    db_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB_FILE
    run_stdio(db_file)
