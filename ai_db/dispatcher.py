"""
ai_db/dispatcher.py
Unified ServiceDispatcher and Tool Registry for ai-db.
Provides centralized execution dispatching for CLI, stdio MCP, and HTTP REST transports.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolDefinition:
    """Represents a registered tool definition with schema and execution handler."""
    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    category: str = "general"

    def to_mcp_dict(self) -> dict[str, Any]:
        """Returns MCP-compliant tool representation with inputSchema and parameters_schema."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters_schema,
            "parameters_schema": self.parameters_schema,
        }


class ServiceDispatcher:
    """
    Central tool registry and execution dispatcher.
    Enforces uniform parameter validation, error handling, and JSON Schema definitions
    across CLI, MCP, and HTTP transports.
    """

    def __init__(self, db: Any | None = None, db_path: str | None = None,
                 config: Any | None = None):
        self.db = db
        self.config = config
        self.db_path = db_path or getattr(db, "db_path", None)
        self._lazy_db: Any | None = None
        self._tools: dict[str, ToolDefinition] = {}
        # Set for the duration of a tools/call that carries a progressToken.
        self._progress: Any = None
        self._register_default_tools()

    def register_tool(
        self,
        name: str | ToolDefinition,
        description: str = "",
        parameters_schema: dict[str, Any] | None = None,
        handler: Callable[[dict[str, Any]], Any] | None = None,
        *,
        input_schema: dict[str, Any] | None = None,
        category: str = "general",
    ) -> None:
        """
        Registers a tool definition.
        Accepts either a ToolDefinition instance or separate arguments.
        """
        if isinstance(name, ToolDefinition):
            self._tools[name.name] = name
            return

        schema = parameters_schema or input_schema or {"type": "object", "properties": {}}
        if handler is None:
            raise ValueError(f"Handler callable must be provided for tool '{name}'")

        tool = ToolDefinition(
            name=name,
            description=description,
            parameters_schema=schema,
            handler=handler,
            category=category,
        )
        self._tools[name] = tool

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Retrieves a registered tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """Returns list of all registered tools with schemas."""
        return [tool.to_mcp_dict() for tool in self._tools.values()]

    def execute(self, tool_name: str, arguments: dict[str, Any] | None = None,
                progress: Any = None) -> Any:
        """
        Validates arguments against schema and executes the tool handler.
        Raises KeyError if tool is unregistered.
        Raises ValueError if required arguments are missing.

        ``progress`` is an optional ``progress(done, total, message)`` callback.
        It is published on the dispatcher for the duration of the call so deep
        code (indexer, investigator) can report without knowing about MCP.
        Other transports simply pass None.
        """
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise TypeError(f"Arguments must be a dictionary, got {type(arguments).__name__}")

        tool = self._tools.get(tool_name)
        if not tool:
            raise KeyError(f"Unknown tool: '{tool_name}'. Available: {list(self._tools.keys())}")

        previous = self._progress
        self._progress = progress
        try:
            return self._execute_validated(tool_name, tool, arguments)
        finally:
            self._progress = previous
            # Detach the sink so a later direct call (CLI, tests) cannot report.
            for candidate in (self.db, self._lazy_db):
                indexer = getattr(candidate, "indexer", None) if candidate else None
                if indexer is not None and hasattr(indexer, "progress_sink"):
                    indexer.progress_sink = None

    def report_progress(self, done: int, total: int, message: str = "") -> None:
        """Emit a progress update if a transport supplied a callback."""
        cb = getattr(self, "_progress", None)
        if cb is not None:
            cb(int(done), int(total), message)

    def _execute_validated(self, tool_name: str, tool: Any,
                           arguments: dict[str, Any]) -> Any:
        args = dict(arguments)

        # Legacy parameter normalization
        if tool_name == "analyze" and "targets" not in args and ("filepath" in args or "path" in args):
            single = args.get("filepath") or args.get("path")
            args["targets"] = [single] if single else []
        elif tool_name == "locate" and "query" not in args and "q" in args:
            args["query"] = args["q"]
        elif tool_name == "diff" and "path" not in args and "filepath" in args:
            args["path"] = args["filepath"]

        # Validate required schema fields
        schema = tool.parameters_schema or {}
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in args or args[field] is None:
                raise ValueError(f"Missing required parameter '{field}' for tool '{tool_name}'")

        # Apply schema defaults
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for prop_name, prop_spec in properties.items():
                if isinstance(prop_spec, dict) and "default" in prop_spec and prop_name not in args:
                    args[prop_name] = prop_spec["default"]

        return tool.handler(args)

    def _get_db(self, args: dict[str, Any] | None = None) -> Any:
        """Resolves active VectorDB instance."""
        db = self._resolve_db(args)
        # Bridge progress to the indexer, which lives under VectorDB and has no
        # knowledge of MCP (TODO 14.1).
        if db is not None and getattr(self, "_progress", None) is not None:
            indexer = getattr(db, "indexer", None)
            if indexer is not None:
                indexer.progress_sink = self.report_progress
        return db

    def _resolve_db(self, args: dict[str, Any] | None = None) -> Any:
        if self.db is not None:
            return self.db
        target_path = (args.get("db") if args else None) or self.db_path
        if target_path and self._lazy_db is not None and getattr(self._lazy_db, "db_path", None) != target_path:
            self._lazy_db.close()
            from ai_db import VectorDB
            self._lazy_db = VectorDB(target_path, config=self.config)
            return self._lazy_db
        if self._lazy_db is None:
            from ai_db import VectorDB
            self._lazy_db = VectorDB(target_path, config=self.config)
        return self._lazy_db

    def close(self) -> None:
        """Closes lazily initialized database connection if open."""
        if self._lazy_db is not None:
            self._lazy_db.close()
            self._lazy_db = None

    # =========================================================================
    # Default Tool Handlers Registration
    # =========================================================================

    def _register_default_tools(self) -> None:
        # 1. locate
        self.register_tool(
            name="locate",
            description="Locate candidate files and snippet spans for a question using BM25 ranking.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query or concept to locate"},
                    "scope": {"type": "string", "default": ".", "description": "Directory or repository path scope"},
                    "k": {"type": "integer", "default": 5, "description": "Maximum number of results to return"},
                    "format": {"type": "string", "enum": ["stub", "sexp", "json"], "default": "json", "description": "Output serialization format"},
                },
                "required": ["query"],
            },
            handler=self._handle_locate,
            category="search",
        )

        # 2. outline
        self.register_tool(
            name="outline",
            description="Extract top-level class and function signatures / outline from a file without requiring database indexing.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to extract outline from"},
                },
                "required": ["path"],
            },
            handler=self._handle_outline,
            category="inspection",
        )

        # 3. symbol
        self.register_tool(
            name="symbol",
            description="Exact code symbol lookup (classes, functions, methods) with file and line definition.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol name to look up"},
                    "project": {"type": "string", "description": "Project scope (default: auto-detected)"},
                    "allow_project": {"type": "array", "items": {"type": "string"}, "description": "Allowed projects for read-only access"},
                },
                "required": ["name"],
            },
            handler=self._handle_symbol,
            category="search",
        )

        # 4. analyze
        self.register_tool(
            name="analyze",
            description="Token-optimized AST code inspection with progressive depth control (summary/structure/targeted/full), range targeting, question filtering, and diff inspection.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "targets": {"type": "array", "items": {"type": "string"}, "description": "File paths or glob patterns to analyze"},
                    "depth": {"type": "string", "enum": ["summary", "structure", "targeted", "full"], "default": "structure"},
                    "q": {"type": "string", "description": "Question or concept filter"},
                    "focus": {"type": "string", "description": "Symbol name or substring filter"},
                    "span": {"type": "array", "items": {"type": "integer"}, "description": "Line range [start, end]"},
                    "ctx_lines": {"type": "integer", "default": 10, "description": "Context lines around span"},
                    "since": {"type": "string", "description": "Diff mode: inspect changed line spans"},
                    "max_out": {"type": "integer", "description": "Token budget ceiling"},
                    "cursor": {"type": "string", "description": "Continuation cursor handle"},
                    "format": {"type": "string", "enum": ["stub", "sexp", "json", "outline", "prose"], "default": "json"},
                    "no_cache": {"type": "boolean", "default": False, "description": "Bypass semantic cache"},
                },
                "required": ["targets"],
            },
            handler=self._handle_analyze,
            category="analysis",
        )

        # 5. expand
        self.register_tool(
            name="expand",
            description="Expand an opaque progressive-disclosure handle (ref:hash) returned by analyze to inspect full or targeted source body.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Opaque ref handle (e.g. 'ref:b3b64697')"},
                    "depth": {"type": "string", "enum": ["targeted", "full"], "default": "full"},
                    "span": {"type": "array", "items": {"type": "integer"}, "description": "Optional sub-span [start_line, end_line]"},
                },
                "required": ["ref"],
            },
            handler=self._handle_expand,
            category="analysis",
        )

        # 6. sync
        self.register_tool(
            name="sync",
            description="Scan directory or file, update AST symbols, chunks, and FTS5 full-text index.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": ".", "description": "Directory or file to sync"},
                    "project": {"type": "string", "description": "Project scope (default: auto-detected)"},
                    "verbose": {"type": "boolean", "default": False, "description": "Verbose logging"},
                },
            },
            handler=self._handle_sync,
            category="indexing",
        )

        # 7. check
        self.register_tool(
            name="check",
            description="Validate Python AST syntax and report syntax errors with line/column/message.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path or file to check"},
                    "project": {"type": "string", "description": "Project scope"},
                    "allow_project": {"type": "array", "items": {"type": "string"}, "description": "Allowed projects"},
                },
            },
            handler=self._handle_check,
            category="quality",
        )

        # 8. status
        self.register_tool(
            name="status",
            description="Retrieve database health metrics, entity counts (files, chunks, symbols, syntax errors), and storage size.",
            parameters_schema={
                "type": "object",
                "properties": {},
            },
            handler=self._handle_status,
            category="storage",
        )

        # 9. diff
        self.register_tool(
            name="diff",
            description="Show changed line spans in a file since last indexed version or a git ref.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to diff"},
                    "since": {"type": "string", "default": "last", "description": "Git commit/branch/tag, or 'last' for DB snapshot"},
                },
                "required": ["path"],
            },
            handler=self._handle_diff,
            category="analysis",
        )

        # 10. callers
        self.register_tool(
            name="callers",
            description="Find all call sites and references to a symbol.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol name to find callers of"},
                    "project": {"type": "string", "description": "Project scope"},
                    "allow_project": {"type": "array", "items": {"type": "string"}, "description": "Allowed projects"},
                },
                "required": ["name"],
            },
            handler=self._handle_callers,
            category="search",
        )

        # 11. trace
        self.register_tool(
            name="trace",
            description="Trace call flow from an entry point (symbol, file:line, or script) showing chronological execution order with await/spawn/callback/deferred markers.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "entry": {"type": "string", "description": "Entry point: symbol name, file:line, or script path"},
                    "direction": {"type": "string", "enum": ["down", "up"], "default": "down", "description": "Trace direction: down=callees, up=callers"},
                    "depth": {"type": "integer", "default": 20, "description": "Maximum depth to trace"},
                    "max_nodes": {"type": "integer", "default": 200, "description": "Maximum total nodes to visit"},
                    "include_tests": {"type": "boolean", "default": False, "description": "Include test callers when tracing up"},
                    "format": {"type": "string", "enum": ["tree", "json", "mermaid"], "default": "tree", "description": "Output format"},
                    "with_code": {"type": "boolean", "default": False, "description": "Include source code context at call sites"},
                    "project": {"type": "string", "description": "Project scope"},
                    "allow_project": {"type": "array", "items": {"type": "string"}, "description": "Allowed projects for read-only access"},
                },
                "required": ["entry"],
            },
            handler=self._handle_trace,
            category="analysis",
        )

        # 12. todos
        self.register_tool(
            name="todos",
            description="List TODO, FIXME, HACK, and NOTE code annotations across indexed files.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["todo", "fixme", "hack", "note", "xxx"]},
                    "filepath": {"type": "string", "description": "Filter by file path"},
                    "project": {"type": "string", "description": "Project scope"},
                },
            },
            handler=self._handle_todos,
            category="quality",
        )

        # 12. context_save
        self.register_tool(
            name="context_save",
            description="Persist conversation memory, active files, pending tasks, and architectural decisions.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "default": "main", "description": "Session ID / conversation name"},
                    "summary": {"type": "string", "description": "Summary of session state, goals, and decisions"},
                    "title": {"type": "string", "description": "Descriptive title for session state"},
                    "active_files": {"type": "array", "items": {"type": "string"}, "description": "Active files in session"},
                    "open_tasks": {"type": "array", "items": {"type": "string"}, "description": "Pending task list"},
                    "notes": {"type": "string", "description": "Detailed notes/decisions"},
                    "project": {"type": "string", "description": "Project scope"},
                },
                "required": ["summary"],
            },
            handler=self._handle_context_save,
            category="memory",
        )

        # 13. context_recall
        self.register_tool(
            name="context_recall",
            description="Recall session context, decisions, active files, and tasks across past sessions.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Optional session ID to recall"},
                    "query": {"type": "string", "description": "Optional search term across past memories"},
                    "project": {"type": "string", "description": "Project scope"},
                },
            },
            handler=self._handle_context_recall,
            category="memory",
        )

        # 14. optimize
        self.register_tool(
            name="optimize",
            description="Defragment database, merge full-text FTS5 index B-trees, update query planner stats, and reclaim disk space.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "prune_missing": {"type": "boolean", "default": True, "description": "Prune records for deleted files"},
                    "default_format": {"type": "string", "enum": ["stub", "sexp", "json", "outline", "prose"]},
                },
            },
            handler=self._handle_optimize,
            category="storage",
        )

        # 15. telemetry
        self.register_tool(
            name="telemetry",
            description="Retrieve performance metrics, query latency percentiles, token compression savings across serializations, cache hit rate, and codebase weak points.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "reset": {"type": "boolean", "default": False, "description": "Reset accumulated telemetry metrics"},
                    "detail": {"type": "boolean", "default": False, "description": "Include detailed breakdown of weak points"},
                },
            },
            handler=self._handle_telemetry,
            category="telemetry",
        )

        # investigate
        self.register_tool(
            name="investigate",
            description=(
                "Call this INSTEAD of multiple grep/read/search calls. Returns a complete, ranked "
                "evidence pack for the question: entry points with reasons, full bodies of the most "
                "relevant code, stubs + ref handles (use 'expand') for related code, call graph, "
                "covering tests, recent commits and unresolved symbols, all within a token budget. "
                "mode: 'locate' (where is X), 'explain' (how does X work: adds callees, classes, "
                "tests), 'impact' (what breaks if X changes: adds transitive callers, tests), "
                "'flow' (call path through the graph), 'diff' (what this change touches: seeds "
                "from `git diff <since>`, plus the callers of every changed symbol, so the pack is "
                "the review surface of the change)."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question or concept. Optional in diff mode, where it only breaks ties."},
                    "mode": {"type": "string", "enum": ["locate", "explain", "impact", "flow", "diff"], "default": "explain"},
                    "format": {"type": "string", "enum": ["json", "compact", "stub", "sexp"], "default": "json",
                               "description": "Output style. 'json' is indented and parseable; 'compact' is the same data on one line (~20% smaller); 'stub' drops bodies and scaffolding for an agent (~5x smaller), with bodies still one 'expand' away by ref; 'sexp' is a navigable tree."},
                    "since": {"type": "string", "description": "diff mode only: git ref to diff against, e.g. 'HEAD~1', 'main'"},
                    "root": {"type": "string", "description": "diff mode only: repository root to diff in (default '.')"},
                    "budget_tokens": {"type": "integer", "default": 8000, "description": "Max tokens of the returned pack (>= 500)"},
                    "project": {"type": "string", "description": "Project scope"},
                    "allow_project": {"type": "array", "items": {"type": "string"}, "description": "Allowed projects"},
                    "languages": {"type": "array", "items": {"type": "string"}},
                    "chunk_types": {"type": "array", "items": {"type": "string"}},
                    "modified_since": {"type": "number"},
                },
            },
            handler=self._handle_investigate,
            category="search",
        )

        # 16. query
        self.register_tool(
            name="query",
            description="Search code/doc chunks (retriever configured by retrieval.mode: lexical BM25 or hybrid BM25+vector).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"},
                    "top": {"type": "integer", "default": 5, "description": "Number of chunks to retrieve"},
                    "project": {"type": "string", "description": "Project scope"},
                    "allow_project": {"type": "array", "items": {"type": "string"}, "description": "Allowed projects"},
                    "languages": {"type": "array", "items": {"type": "string"}, "description": "Only these languages (python, typescript, go, markdown, text, ...)"},
                    "chunk_types": {"type": "array", "items": {"type": "string"}, "description": "Only these chunk types (code, class_header, module, md, txt, lib_meta)"},
                    "modified_since": {"type": "number", "description": "Only files modified at/after this unix timestamp"},
                },
                "required": ["query"],
            },
            handler=self._handle_query,
            category="search",
        )

        # 17. prune
        self.register_tool(
            name="prune",
            description="Prune database records for deleted or missing files.",
            parameters_schema={
                "type": "object",
                "properties": {},
            },
            handler=self._handle_prune,
            category="storage",
        )

        # 18. route_skill
        self.register_tool(
            name="route_skill",
            description="Dynamically route user prompt to matching skills based on lexical & vector similarity.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "User prompt to route"},
                    "top": {"type": "integer", "default": 3, "description": "Max skills to return"},
                    "min_confidence": {"type": "number", "default": 0.0, "description": "Confidence threshold"},
                    "project": {"type": "string", "description": "Project scope"},
                },
                "required": ["prompt"],
            },
            handler=self._handle_route_skill,
            category="routing",
        )

        # 19. sync_skills
        self.register_tool(
            name="sync_skills",
            description="Scan and index skill definitions from directories.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "skill_dirs": {"type": "array", "items": {"type": "string"}, "description": "Skill directories"},
                    "project": {"type": "string", "description": "Project scope"},
                },
            },
            handler=self._handle_sync_skills,
            category="routing",
        )

    # =========================================================================
    # Handlers Implementation
    # =========================================================================

    def _handle_locate(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        query = args.get("query", "")
        scope = args.get("scope", ".")
        path_prefix = args.get("path_prefix")
        k = int(args.get("k", 5))
        fmt = args.get("format")
        hits = db.locate_targets(q=query, scope=scope, k=k, path_prefix=path_prefix)
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
            def sexp_esc(val: Any) -> str:
                return f'"{str(val).replace(chr(34), chr(92)+chr(34)).replace(chr(10), " ")}"'
            hit_sexps = []
            for h in hits:
                span_s = f"({h['span'][0]} {h['span'][1]})" if h.get("span") else "nil"
                hit_sexps.append(f"(:hit :file {sexp_esc(h['file'])} :name {sexp_esc(h.get('name'))} :span {span_s} :score {h.get('score', 0)})")
            return f"(:locate :query {sexp_esc(query)} :hits ({' '.join(hit_sexps)}))"
        return hits

    def _handle_outline(self, args: dict[str, Any]) -> Any:
        raw_path = args.get("path", "")
        target_path = os.path.abspath(os.path.expanduser(raw_path))
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"File not found: {raw_path}")
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        from ai_db.parser.ast_visitor import extract_file_outline
        outline = extract_file_outline(target_path, content)
        total_lines = len(content.splitlines())
        return {
            "file": target_path,
            "lines": total_lines,
            "outline": [{"line": lineno, "label": label} for lineno, label in outline]
        }

    def _handle_symbol(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        name = args.get("name", "")
        project = args.get("project")
        allow_projects = args.get("allow_project") or args.get("allowed_projects")
        return db.query_symbol(name, relative_to=os.getcwd(), project=project, allowed_projects=allow_projects)

    def _handle_analyze(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        raw_targets = args.get("targets")
        if not raw_targets:
            single_path = args.get("filepath") or args.get("path")
            if single_path:
                raw_targets = [single_path]
            else:
                raw_targets = ["."]
        elif isinstance(raw_targets, str):
            raw_targets = [raw_targets]

        depth = args.get("depth", "structure")
        q = args.get("q") or args.get("question")
        focus = args.get("focus")
        raw_span = args.get("span")
        if isinstance(raw_span, str) and ":" in raw_span:
            parts = raw_span.split(":")
            span_tuple = (int(parts[0]), int(parts[1])) if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() else None
        elif isinstance(raw_span, (list, tuple)) and len(raw_span) >= 2:
            span_tuple = (int(raw_span[0]), int(raw_span[1]))
        else:
            span_tuple = None

        since = args.get("since")
        ctx_lines = int(args.get("ctx_lines") or args.get("ctx") or 10)
        max_out = args.get("max_out")
        cursor = args.get("cursor")
        no_cache = bool(args.get("no_cache", False))
        fmt = args.get("format")

        if len(raw_targets) == 1 and not any(c in raw_targets[0] for c in ["*", "?", "["]) and os.path.isfile(os.path.expanduser(raw_targets[0])):
            res = db.analyze_file(
                raw_targets[0],
                depth=depth,
                span=span_tuple,
                focus=focus,
                q=q,
                since=since,
                ctx_lines=ctx_lines,
                no_cache=no_cache
            )
            if max_out and res.get("meta", {}).get("tokens_out", 0) > max_out:
                res["meta"]["truncated"] = True
                res["symbols"] = res["symbols"][:max(1, len(res["symbols"]) // 2)]
            out_data = res
        else:
            out_data = db.analyze_batch(
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
            from ai_db import VectorDB
            return VectorDB.format_as_stub(out_data)
        elif fmt == "sexp":
            from ai_db import VectorDB
            return VectorDB.format_as_sexp(out_data)
        return out_data

    def _handle_expand(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        ref = args.get("ref", "")
        depth = args.get("depth", "full")
        raw_span = args.get("span")
        if isinstance(raw_span, str) and ":" in raw_span:
            parts = raw_span.split(":")
            span_tuple = (int(parts[0]), int(parts[1])) if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() else None
        elif isinstance(raw_span, (list, tuple)) and len(raw_span) >= 2:
            span_tuple = (int(raw_span[0]), int(raw_span[1]))
        else:
            span_tuple = None

        exp = db.expand_ref(ref, depth=depth, span=span_tuple)
        if not exp:
            raise ValueError(f"Ref handle '{ref}' not found or expired.")
        return exp

    def _handle_sync(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        target_path = os.path.abspath(os.path.expanduser(args.get("path", ".")))
        from ai_db.utils import detect_project_name
        proj_name = args.get("project") or detect_project_name(target_path)
        verbose = bool(args.get("verbose", False))

        if os.path.isdir(target_path):
            return db.sync(target_path, project=proj_name, verbose=verbose)
        elif os.path.isfile(target_path):
            from ai_db.utils import compute_sha256
            current_sha = compute_sha256(target_path)
            db.prune_file(target_path)
            db._index_file(target_path, current_sha, project=proj_name or "global")
            if hasattr(db, "conn") and db.conn:
                db.conn.commit()
            return {"added": 1, "updated": 0, "pruned": 0, "skipped": 0}
        else:
            return db.sync(target_path, project=proj_name, verbose=verbose)

    def _handle_check(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        path = args.get("path")
        project = args.get("project")
        allow_projects = args.get("allow_project") or args.get("allowed_projects")
        if path:
            target_path = os.path.abspath(os.path.expanduser(path))
            if os.path.exists(target_path):
                if os.path.isdir(target_path):
                    db.sync(target_path, project=project, verbose=False)
                elif os.path.isfile(target_path):
                    from ai_db.utils import compute_sha256
                    db.prune_file(target_path)
                    db._index_file(target_path, compute_sha256(target_path), project=project or "global")
                    if hasattr(db, "conn") and db.conn:
                        db.conn.commit()
        return db.check_syntax(path, relative_to=os.getcwd(), project=project, allowed_projects=allow_projects)

    def _handle_status(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        return db.status()

    def _handle_diff(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        raw_path = args.get("path") or args.get("filepath", "")
        since = args.get("since", "last")
        return db.diff_file(raw_path, since=since)

    def _handle_callers(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        name = args.get("name", "")
        project = args.get("project")
        allow_projects = args.get("allow_project") or args.get("allowed_projects")
        return db.query_callers(name, relative_to=os.getcwd(), project=project, allowed_projects=allow_projects)

    def _handle_trace(self, args: dict[str, Any]) -> Any:
        from ai_db.analysis.trace import TraceEngine
        from ai_db.analysis.trace_format import format_json, format_mermaid, format_tree

        db = self._get_db(args)
        entry = args.get("entry", "")
        direction = args.get("direction", "down")
        depth = int(args.get("depth", 20))
        max_nodes = int(args.get("max_nodes", 200))
        include_tests = bool(args.get("include_tests", False))
        fmt = args.get("format", "tree")
        with_code = bool(args.get("with_code", False))

        project = args.get("project")
        allow_projects = args.get("allow_project") or args.get("allowed_projects")
        allowed_projects = [project] + allow_projects if project and allow_projects else ([project] if project else allow_projects)

        trace_engine = TraceEngine(
            db.backend,
            allowed_projects=allowed_projects,
            trace_wait_patterns=self.config.trace_wait_patterns if self.config else None,
            trace_wait_patterns_extend=self.config.trace_wait_patterns_extend if self.config else None,
        )
        result = trace_engine.trace(
            entry=entry,
            depth=depth,
            max_nodes=max_nodes,
            direction=direction,
            include_tests=include_tests,
        )

        if fmt == "json":
            return format_json(result)
        elif fmt == "mermaid":
            return format_mermaid(result)
        else:
            return format_tree(result, with_code=with_code, context_lines=3)

    def _handle_todos(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        kind = args.get("kind")
        filepath = args.get("filepath") or args.get("file")
        if filepath:
            filepath = os.path.abspath(os.path.expanduser(filepath))
        project = args.get("project")
        allow_projects = args.get("allow_project") or args.get("allowed_projects")
        return db.query_annotations(kind=kind, filepath=filepath, project=project, allowed_projects=allow_projects)

    def _handle_context_save(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        session_id = args.get("session_id", "main")
        summary = args.get("summary", "")
        title = args.get("title")
        active_files = args.get("active_files") or args.get("files", [])
        open_tasks = args.get("open_tasks") or args.get("tasks", [])
        notes = args.get("notes") or args.get("full_notes")
        project = args.get("project")
        return db.save_context(
            session_id=session_id,
            summary=summary,
            project=project,
            title=title,
            active_files=active_files,
            open_tasks=open_tasks,
            full_notes=notes,
        )

    def _handle_context_recall(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        session_id = args.get("session_id")
        query = args.get("query")
        project = args.get("project")
        allow_projects = args.get("allow_project") or args.get("allowed_projects")
        if query:
            return db.query_contexts(query, project=project, allowed_projects=allow_projects, top_k=3)
        if args.get("list"):
            return db.list_contexts(project=project, allowed_projects=allow_projects)
        ctx = db.get_context(session_id=session_id, project=project, allowed_projects=allow_projects)
        return ctx or {}

    def _handle_optimize(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        prune_missing = args.get("prune_missing", True)
        if "no_prune" in args:
            prune_missing = not args["no_prune"]
        default_format = args.get("default_format")
        return db.optimize(prune_missing=prune_missing, default_format=default_format)

    def _handle_telemetry(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        reset = bool(args.get("reset", False))
        tracker = getattr(db, "telemetry_tracker", None)
        if tracker is None and hasattr(db, "conn") and db.conn:
            from ai_db.telemetry.tracker import TelemetryTracker
            tracker = TelemetryTracker(db.conn, db_path=getattr(db, "db_path", None))
            db.telemetry_tracker = tracker

        if tracker is not None:
            if reset:
                tracker.reset()
                return {"status": "reset", "message": "Telemetry metrics reset."}
            summary = tracker.get_summary()
            backend = getattr(db, "backend", None)
            if backend is not None and hasattr(backend, "get_query_log"):
                from ai_db.telemetry.stages import stage_stats
                summary["stages"] = stage_stats(backend.get_query_log(limit=1000))
            return summary

        backend_name = getattr(getattr(db, "backend", None), "backend_name", "sqlite")
        return {
            "token_savings": {
                "cumulative_raw_tokens": 0,
                "emitted_tokens": 0,
                "net_tokens_saved": 0,
                "reduction_pct": 0.0,
                "estimated_cost_saved_usd": 0.0,
            },
            "latency": {
                "backend": backend_name,
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            },
            "cache": {
                "lookups": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate_pct": 0.0,
            },
            "weak_points": {
                "syntax_errors": 0,
                "complexity_hotspots": [],
                "unindexed_files": 0,
            },
        }

    def _handle_query(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        search_str = args.get("query") or args.get("search", "")
        top_k = int(args.get("top", 5))
        project = args.get("project")
        allow_projects = args.get("allow_project") or args.get("allowed_projects")
        return db.query(search_str, top=top_k, project=project, allowed_projects=allow_projects,
                        languages=args.get("languages"), chunk_types=args.get("chunk_types"),
                        modified_since=args.get("modified_since"))

    def _handle_investigate(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        pack = db.investigate(
            args["query"], budget_tokens=int(args.get("budget_tokens", 8000)),
            mode=args.get("mode", "explain"), project=args.get("project"),
            allowed_projects=args.get("allow_project") or args.get("allowed_projects"),
            languages=args.get("languages"), chunk_types=args.get("chunk_types"),
            modified_since=args.get("modified_since"),
            since=args.get("since"), root=args.get("root"),
        )
        # Default stays a dict so every existing caller keeps parsing JSON.
        # A non-default style returns a rendered string, which the MCP and HTTP
        # transports already pass through verbatim, so one renderer serves all
        # three instead of being reimplemented per transport.
        style = args.get("format") or "json"
        if style == "json":
            return pack
        from ai_db.analyzer.formatters import format_pack
        return format_pack(pack, style)

    def _handle_prune(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        backend = getattr(db, "backend", getattr(db, "db", db))
        if hasattr(backend, "get_all_filepaths"):
            all_paths = backend.get_all_filepaths()
        elif hasattr(db, "get_all_filepaths"):
            all_paths = db.get_all_filepaths()
        else:
            all_paths = []
        pruned_count = 0
        for p in all_paths:
            if not os.path.exists(p):
                if hasattr(backend, "delete_file"):
                    backend.delete_file(p)
                elif hasattr(db, "delete_file"):
                    db.delete_file(p)
                elif hasattr(db, "prune_file"):
                    db.prune_file(p)
                pruned_count += 1
        return {"pruned": pruned_count}

    def _handle_route_skill(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        prompt = args.get("prompt", "")
        top_k = int(args.get("top", 3))
        min_conf = float(args.get("min_confidence", 0.0))
        project = args.get("project")
        allow_projects = args.get("allow_project") or args.get("allowed_projects")
        return db.route_skills(prompt, top_k=top_k, min_confidence=min_conf, project=project, allowed_projects=allow_projects)

    def _handle_sync_skills(self, args: dict[str, Any]) -> Any:
        db = self._get_db(args)
        skill_dirs = args.get("skill_dirs") or args.get("dir")
        project = args.get("project")
        return db.sync_skills(skill_dirs=skill_dirs, project=project)
