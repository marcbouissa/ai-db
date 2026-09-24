import os
import sys
import json
import time
import argparse
import subprocess
from typing import List, Dict, Any, Optional

from ai_db import (
    DEFAULT_DB_FILE,
    DEFAULT_CONFIG_FILE,
    DEFAULT_SKILL_DIRS,
    VectorDB,
    detect_project_name,
    run_watch,
    __version__,
)
from ai_db.dispatcher import ServiceDispatcher
from ai_db.config import AppConfig, load_config, config_path, masked_dict, CONFIG_VERSION
from ai_db.errors import AiDbConfigError


def _run_eval(args: argparse.Namespace, cfg: AppConfig) -> int:
    import tempfile
    from ai_db.eval.harness import run, run_pack, compare_to_baseline, materialize_tracked

    with tempfile.TemporaryDirectory(prefix="ai_db_eval_") as tmp:
        root = os.path.abspath(args.root)
        if not args.all_files:
            root = materialize_tracked(root, os.path.join(tmp, "corpus"))
        db = VectorDB(os.path.join(tmp, "eval.db"), config=cfg)
        try:
            if args.pack:
                result = run_pack(args.golden, root, db, budget_tokens=args.budget)
            else:
                result = run(args.golden, root, db, k=args.k)
        finally:
            db.close()
    summary = {k: v for k, v in result.items() if k != "per_query"}
    print(json.dumps(summary, indent=2))
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    if args.baseline:
        if args.pack:
            raise AiDbConfigError("--baseline gates recall@k; it cannot be combined with --pack")
        with open(args.baseline, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        err = compare_to_baseline(result, baseline)
        if err:
            print(f"[ai-db eval] FAIL: {err}", file=sys.stderr)
            return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    try:
        return _main(argv)
    except AiDbConfigError as exc:
        print(f"[ai-db] configuration error: {exc}", file=sys.stderr)
        return 2


def _cmd_init(args: argparse.Namespace) -> int:
    from ai_db.config_template import build_template, migrate_legacy

    target = config_path(args.config)
    if args.migrate:
        if not os.path.isfile(target):
            raise AiDbConfigError(f"--migrate: no config at {target}")
        with open(target, "r", encoding="utf-8") as f:
            old = json.load(f)
        try:
            data = migrate_legacy(old)
        except ValueError as exc:
            raise AiDbConfigError(f"--migrate: {exc}") from exc
    else:
        if os.path.exists(target) and not args.force:
            raise AiDbConfigError(f"config already exists at {target}; use --force to overwrite")
        try:
            data = build_template(storage=args.storage, embedding=args.embedding,
                                  rerank=args.rerank, mode=args.mode)
        except ValueError as exc:
            raise AiDbConfigError(str(exc)) from exc
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"[ai-db init] wrote {target}")
    print("Next: ai-db config check")
    return 0


def _cmd_reindex(args: argparse.Namespace, cfg: AppConfig) -> int:
    from ai_db.storage.factory import StorageBackendFactory

    if cfg.retrieval_mode != "hybrid":
        raise AiDbConfigError("reindex --embeddings needs retrieval.mode 'hybrid'")
    backend = StorageBackendFactory.from_config(cfg, db_path=args.db)
    backend.initialize()
    backend.drop_vector_index()  # must happen before VectorDB checks the model guard
    db = VectorDB(backend, config=cfg)
    try:
        count = db.embed_missing()
    finally:
        db.close()
    print(f"[ai-db reindex] embedded {count} chunks with {db.embedder.model_id}")
    return 0


def _cmd_config(args: argparse.Namespace, cfg: AppConfig) -> int:
    if args.config_action == "show":
        print(json.dumps({"path": cfg.source_path, **masked_dict(cfg)}, indent=2))
        return 0
    if args.config_action == "check":
        from ai_db.health import check_config
        report = check_config(cfg)
        for line in report.lines:
            print(line)
        return 0 if report.ok else 1
    raise AiDbConfigError("usage: ai-db config {show,check}")


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-db",
        description="ai-db: Token-Optimized Vector DB & Code Index"
    )
    parser.add_argument("--version", action="version", version=f"ai-db {__version__}")
    parser.add_argument("--config", default=None, help="Config file path (default: $AI_DB_CONFIG or ~/.config/ai-db/config.json)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init
    init_p = subparsers.add_parser("init", help="Write an explicit config file (required before other commands)")
    init_p.add_argument("--storage", default="sqlite", help="Storage provider entry point (default: sqlite)")
    init_p.add_argument("--embedding", choices=["none", "sentence_transformers", "openai_compatible", "voyage"], default="none")
    init_p.add_argument("--rerank", choices=["none", "sentence_transformers", "voyage", "cohere"], default="none")
    init_p.add_argument("--mode", choices=["lexical", "hybrid"], default=None, help="Retrieval mode (default: derived from --embedding)")
    init_p.add_argument("--force", action="store_true", help="Overwrite an existing config")
    init_p.add_argument("--migrate", action="store_true", help="Rewrite a legacy (unversioned) config into version 1")

    # reindex
    reindex_p = subparsers.add_parser("reindex", help="Rebuild derived indexes")
    reindex_p.add_argument("--embeddings", action="store_true", required=True,
                           help="Drop all vectors and re-embed every chunk with the configured model")
    reindex_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # config
    config_p = subparsers.add_parser("config", help="Inspect or verify the active config")
    config_p.add_argument("config_action", choices=["show", "check"])

    # sync
    sync_p = subparsers.add_parser("sync", help="Sync files in a directory")
    sync_p.add_argument("path", nargs="?", default=".", help="Target path")
    sync_p.add_argument("--project", default=None, help="Target project scope (default: auto-detected)")
    sync_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # query
    query_p = subparsers.add_parser("query", help="Query knowledge chunks (BM25)")
    query_p.add_argument("search", help="Search query")
    query_p.add_argument("--top", type=int, default=5)
    query_p.add_argument("--project", default=None, help="Active project scope (default: auto-detected)")
    query_p.add_argument("--allow-project", action="append", default=[], help="Allowed project for read-only access (repeatable)")
    query_p.add_argument("--lang", action="append", default=None, help="Only this language (repeatable)")
    query_p.add_argument("--type", dest="chunk_type", action="append", default=None, help="Only this chunk type (repeatable)")
    query_p.add_argument("--since", type=float, default=None, help="Only files modified at/after this unix timestamp")
    query_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # check / lint
    check_p = subparsers.add_parser("check", aliases=["lint"], help="Check AST code validation and syntax errors")
    check_p.add_argument("path", nargs="?", default=None, help="Target path or file to check")
    check_p.add_argument("--project", default=None, help="Active project scope (default: auto-detected)")
    check_p.add_argument("--allow-project", action="append", default=[], help="Allowed project for read-only access (repeatable)")
    check_p.add_argument("--watch", action="store_true", help="Continuously re-check on file changes (F11)")
    check_p.add_argument("--interval", type=float, default=2.0, help="Poll interval in seconds for --watch mode (default: 2.0)")
    check_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # symbol
    symbol_p = subparsers.add_parser("symbol", help="Exact symbol resolution (class, def, interface)")
    symbol_p.add_argument("name", help="Symbol name to find")
    symbol_p.add_argument("--project", default=None, help="Active project scope (default: auto-detected)")
    symbol_p.add_argument("--allow-project", action="append", default=[], help="Allowed project for read-only access (repeatable)")
    symbol_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # outline
    outline_p = subparsers.add_parser("outline", help="File outline / skeleton extraction")
    outline_p.add_argument("path", help="Filepath to extract outline from")
    outline_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # watch
    watch_p = subparsers.add_parser("watch", help="Watch directory and auto-sync on changes")
    watch_p.add_argument("path", nargs="?", default=".", help="Target directory to watch")
    watch_p.add_argument("--daemon", action="store_true", help="Run watcher in background daemon mode")
    watch_p.add_argument("--interval", type=float, default=2.0, help="Polling/check interval in seconds")
    watch_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # sync-all
    syncall_p = subparsers.add_parser("sync-all", help="Sync all repositories registered in config.json")
    syncall_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # route-skill
    route_p = subparsers.add_parser("route-skill", aliases=["suggest-skills", "route"], help="Analyze prompt and route to best matching skill(s)")
    route_p.add_argument("prompt", help="User prompt or task description to match against skills")
    route_p.add_argument("--top", type=int, default=3, help="Maximum number of skills to return")
    route_p.add_argument("--format", choices=["dense", "json", "path"], default="dense", help="Output format")
    route_p.add_argument("--min-confidence", type=float, default=None, help="Minimum confidence (0.0–1.0) to include a skill result (default: 0.15)")
    route_p.add_argument("--project", default=None, help="Active project scope (default: auto-detected)")
    route_p.add_argument("--allow-project", action="append", default=[], help="Allowed project for read-only access (repeatable)")
    route_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # sync-skills
    syncskills_p = subparsers.add_parser("sync-skills", help="Index all installed skills into the knowledge base")
    syncskills_p.add_argument("--project", default="global", help="Project scope for skills (default: global)")
    syncskills_p.add_argument("--dir", action="append", default=None, help="Skill directory to index")
    syncskills_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # context (chat / session memory)
    ctx_p = subparsers.add_parser("context", aliases=["ctx"], help="Manage session context and chat memory")
    ctx_sub = ctx_p.add_subparsers(dest="ctx_action", help="Context action")

    ctx_save = ctx_sub.add_parser("save", help="Save or update current chat context")
    ctx_save.add_argument("session_id", nargs="?", default="main", help="Session ID / conversation name")
    ctx_save.add_argument("--summary", "-s", required=True, help="Summary of session state, goals, and decisions")
    ctx_save.add_argument("--title", "-t", default=None, help="Title for this session context")
    ctx_save.add_argument("--files", "-f", action="append", default=[], help="Active files in session (repeatable)")
    ctx_save.add_argument("--tasks", action="append", default=[], help="Open pending tasks (repeatable)")
    ctx_save.add_argument("--notes", default=None, help="Detailed markdown notes/decisions (defaults to summary)")
    ctx_save.add_argument("--project", default=None, help="Project scope (default: auto-detected)")
    ctx_save.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    ctx_get = ctx_sub.add_parser("get", help="Retrieve latest or specified session context")
    ctx_get.add_argument("session_id", nargs="?", default=None, help="Session ID (default: most recent)")
    ctx_get.add_argument("--format", choices=["dense", "json", "markdown"], default="markdown", help="Output format")
    ctx_get.add_argument("--project", default=None, help="Project scope (default: auto-detected)")
    ctx_get.add_argument("--allow-project", action="append", default=[], help="Allowed project (repeatable)")
    ctx_get.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    ctx_list = ctx_sub.add_parser("list", help="List stored session contexts")
    ctx_list.add_argument("--project", default=None, help="Project scope (default: auto-detected)")
    ctx_list.add_argument("--allow-project", action="append", default=[], help="Allowed project (repeatable)")
    ctx_list.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    ctx_query = ctx_sub.add_parser("query", help="Search session contexts with BM25")
    ctx_query.add_argument("search", help="Search query")
    ctx_query.add_argument("--top", type=int, default=3, help="Max results")
    ctx_query.add_argument("--project", default=None, help="Project scope (default: auto-detected)")
    ctx_query.add_argument("--allow-project", action="append", default=[], help="Allowed project (repeatable)")
    ctx_query.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # remember shortcut command
    remember_p = subparsers.add_parser("remember", help="Recall current project context memory")
    remember_p.add_argument("session_id", nargs="?", default=None, help="Optional session ID")
    remember_p.add_argument("--project", default=None, help="Project scope (default: auto-detected)")
    remember_p.add_argument("--allow-project", action="append", default=[], help="Allowed project (repeatable)")
    remember_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # analyze (RFC: tokenopt-analyzer v2)
    analyze_p = subparsers.add_parser("analyze", help="Token-optimized code analysis (F1-F15)")
    analyze_p.add_argument("targets", nargs="*", default=[], help="File(s) or glob pattern to analyze")
    analyze_p.add_argument("--depth", choices=["summary", "structure", "targeted", "full"], default="structure", help="Analysis depth (default: structure)")
    analyze_p.add_argument("--span", default=None, help="Range target START:END (e.g. 50:120)")
    analyze_p.add_argument("--ctx", type=int, default=10, help="Context lines around span (default: 10)")
    analyze_p.add_argument("--focus", default=None, help="Symbol or token focus filter")
    analyze_p.add_argument("-q", "--question", default=None, help="Question-driven filter query")
    analyze_p.add_argument("--since", default=None, help="Diff mode: inspect changes since hash or timestamp")
    analyze_p.add_argument("--max-out", type=int, default=None, help="Token budget ceiling for response")
    analyze_p.add_argument("--cursor", default=None, help="Continuation cursor handle")
    analyze_p.add_argument("--format", "--fmt", dest="format", choices=["json", "stub", "sexp", "outline", "prose"], default=None, help="Output format (default: active DB default, initially 'stub')")
    analyze_p.add_argument("--no-cache", action="store_true", help="Bypass semantic cache")
    analyze_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # expand (F7 progressive disclosure)
    expand_p = subparsers.add_parser("expand", help="Expand progressive disclosure ref handle (pay-per-section)")
    expand_p.add_argument("ref", help="Opaque ref handle (e.g. ref:a8f9c1)")
    expand_p.add_argument("--depth", choices=["targeted", "full"], default="full", help="Expansion depth")
    expand_p.add_argument("--span", default=None, help="Sub-span START:END within ref")
    expand_p.add_argument("--format", choices=["json", "raw"], default="json", help="Output format")
    expand_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # investigate
    inv_p = subparsers.add_parser("investigate", aliases=["inv"], help="One-call evidence pack for a question (replaces grep/read loops)")
    inv_p.add_argument("query", help="Question or concept")
    inv_p.add_argument("--mode", choices=["locate", "explain", "impact"], default="explain")
    inv_p.add_argument("--budget", type=int, default=8000, help="Token budget for the pack (default: 8000)")
    inv_p.add_argument("--project", default=None, help="Active project scope (default: auto-detected)")
    inv_p.add_argument("--allow-project", action="append", default=[], help="Allowed project (repeatable)")
    inv_p.add_argument("--lang", action="append", default=None, help="Only this language (repeatable)")
    inv_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # locate
    locate_p = subparsers.add_parser("locate", help="Locate files and snippet spans for a question")
    locate_p.add_argument("query", help="Question or concept to search for")
    locate_p.add_argument("--scope", default=".", help="Directory scope to restrict search (default: .)")
    locate_p.add_argument("-k", type=int, default=5, help="Number of target candidates (default: 5)")
    locate_p.add_argument("--format", "--fmt", dest="format", choices=["json", "stub", "sexp"], default=None, help="Output format")
    locate_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # mcp (JSON-RPC MCP Server)
    mcp_p = subparsers.add_parser("mcp", help="Run stdio JSON-RPC MCP server for Claude/Cursor/Antigravity")
    mcp_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # status
    status_p = subparsers.add_parser("status", help="Show database statistics and index health")
    status_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # prune
    prune_p = subparsers.add_parser("prune", help="Remove dead/deleted files from the index")
    prune_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # optimize / vacuum
    opt_p = subparsers.add_parser("optimize", aliases=["vacuum"], help="Defragment database and optimize FTS index")
    opt_p.add_argument("--no-prune", action="store_true", help="Skip pruning missing files before vacuum")
    opt_p.add_argument("--default-format", choices=["stub", "sexp", "json", "outline", "prose"], default=None, help="Configure and persist default output format for future queries")
    opt_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # diff
    diff_p = subparsers.add_parser("diff", help="Show changed line spans in a file since last index or git ref")
    diff_p.add_argument("path", help="Path to file to diff")
    diff_p.add_argument("--since", default="last", help="Git ref/hash/timestamp to compare against, or 'last' for DB snapshot (default: last)")
    diff_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    diff_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # callers
    callers_p = subparsers.add_parser("callers", help="Find all callers/references to a symbol")
    callers_p.add_argument("name", help="Symbol name to find callers of")
    callers_p.add_argument("--project", default=None, help="Project scope (default: auto-detected)")
    callers_p.add_argument("--allow-project", action="append", default=[], help="Allowed project for read-only access (repeatable)")
    callers_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # todos
    todos_p = subparsers.add_parser("todos", help="List TODO/FIXME/HACK annotations")
    todos_p.add_argument("--kind", choices=["todo", "fixme", "hack", "note", "xxx"], default=None, help="Filter by annotation kind")
    todos_p.add_argument("--file", default=None, help="Filter by specific file path")
    todos_p.add_argument("--project", default=None, help="Project scope (default: auto-detected)")
    todos_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    todos_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # serve (HTTP JSON API server)
    serve_p = subparsers.add_parser("serve", help="Run HTTP JSON API server")
    serve_p.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    serve_p.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    serve_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # telemetry
    telemetry_p = subparsers.add_parser("telemetry", help="View performance and token compression telemetry")
    telemetry_p.add_argument("--reset", action="store_true", help="Reset accumulated telemetry metrics")
    telemetry_p.add_argument("--format", "--fmt", dest="format", choices=["dense", "json"], default="dense", help="Output format")
    telemetry_p.add_argument("--db", default=None, help="SQLite path override (default: storage.options.path)")

    # eval (retrieval quality harness)
    eval_p = subparsers.add_parser("eval", help="Evaluate retrieval quality against a golden query set")
    eval_p.add_argument("--golden", required=True, help="Path to golden JSONL file")
    eval_p.add_argument("--root", default=".", help="Repository root the golden paths are relative to")
    eval_p.add_argument("-k", type=int, default=10, help="Cutoff for recall/nDCG (default: 10)")
    eval_p.add_argument("--baseline", default=None, help="Baseline JSON; fail if recall@k drops > 0.02")
    eval_p.add_argument("--save", default=None, help="Write the result JSON to this path")
    eval_p.add_argument("--pack", action="store_true",
                        help="Evaluate investigate packs (pack recall; golden kind = mode)")
    eval_p.add_argument("--budget", type=int, default=8000, help="Pack token budget for --pack")
    eval_p.add_argument("--all-files", action="store_true",
                        help="Index every file under --root (default: only git-tracked files)")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "init":
        return _cmd_init(args)

    cfg = load_config(args.config)
    # Every downstream component resolves the same file.
    os.environ["AI_DB_CONFIG"] = cfg.source_path or config_path(args.config)

    if args.command == "config":
        return _cmd_config(args, cfg)

    if args.command == "eval":
        return _run_eval(args, cfg)

    if args.command == "reindex":
        return _cmd_reindex(args, cfg)

    # Transport and long-running server/daemon commands
    if args.command == "mcp":
        import mcp_server
        mcp_server.run_stdio(args.db, config=cfg)
        return 0

    if args.command == "serve":
        from ai_db.server.http_server import start_http_server
        print(f"[ai-db serve] Listening on http://{args.host}:{args.port} | DB: {args.db}")
        start_http_server(args.db, args.host, args.port, config=cfg)
        return 0

    if args.command == "watch":
        if getattr(args, "daemon", False):
            if os.fork() > 0:
                print(f"[ai-db watch] Daemon started in background for '{args.path}'")
                sys.exit(0)
            os.setsid()
            if os.fork() > 0:
                sys.exit(0)
            sys.stdout.flush()
            sys.stderr.flush()
            devnull = open(os.devnull, "wb+")
            os.dup2(devnull.fileno(), sys.stdin.fileno())
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            devnull.close()

        run_watch(args.db, args.path, interval=args.interval)
        return 0

    # Initialize unified ServiceDispatcher for all domain service commands
    dispatcher = ServiceDispatcher(db_path=args.db, config=cfg)

    # Handle sync-all
    if args.command == "sync-all":
        paths = cfg.auto_sync_paths
        if not paths:
            print(f"No auto_sync_paths configured in {cfg.source_path}. Example format:")
            print(json.dumps({"auto_sync_paths": ["~/projects/my-project"]}, indent=2))
            dispatcher.close()
            return 0

        total_added = total_updated = total_pruned = total_skipped = 0
        for p in paths:
            expanded = os.path.abspath(os.path.expanduser(p))
            if not os.path.exists(expanded):
                print(f"Skipping non-existent path: {p}")
                continue
            print(f"Syncing: {p}...")
            res = dispatcher.execute("sync", {"path": expanded, "verbose": True})
            total_added += res["added"]
            total_updated += res["updated"]
            total_pruned += res["pruned"]
            total_skipped += res["skipped"]

        dispatcher.close()
        print(f"Total: +{total_added} ~{total_updated} -{total_pruned} ={total_skipped}")
        return 0

    # Auto-detect project scope from CWD if not provided
    active_proj = getattr(args, "project", None) or detect_project_name(os.getcwd())
    allowed_projs = getattr(args, "allow_project", []) or []

    if args.command == "sync":
        target_path = os.path.abspath(args.path)
        proj_name = args.project or detect_project_name(target_path)
        res = dispatcher.execute("sync", {"path": target_path, "project": proj_name, "verbose": True})
        print(f"+{res['added']} ~{res['updated']} -{res['pruned']} ={res['skipped']}")

    elif args.command in ("investigate", "inv"):
        pack = dispatcher.execute("investigate", {
            "query": args.query, "mode": args.mode, "budget_tokens": args.budget,
            "project": active_proj, "allow_project": allowed_projs, "languages": args.lang,
        })
        print(json.dumps(pack, indent=2, ensure_ascii=False))

    elif args.command == "query":
        hits = dispatcher.execute("query", {
            "query": args.search,
            "top": args.top,
            "project": active_proj,
            "allow_project": allowed_projs,
            "languages": args.lang,
            "chunk_types": args.chunk_type,
            "modified_since": args.since,
        })
        if not hits:
            print("NO_HITS")
        else:
            for h in hits:
                print(f"@{h['file']}:{h['lines']} ({h['name']}) [{h['project']}] [{h['score']}]")
                clean_snippet = "\n".join([line for line in h['snippet'].splitlines() if line.strip()])
                print(f"  {clean_snippet}")

    elif args.command in ("check", "lint"):
        if getattr(args, "watch", False):
            print(f"[ai-db check --watch] Monitoring '{args.path or '.'}' every {args.interval}s. Ctrl-C to stop.")
            prev_error_keys = set()
            while True:
                try:
                    errors = dispatcher.execute("check", {
                        "path": args.path,
                        "project": active_proj,
                        "allow_project": allowed_projs
                    })
                    current_keys = {f"{e['file']}:{e['line']}:{e['col']}" for e in errors}
                    new_keys = current_keys - prev_error_keys
                    resolved_keys = prev_error_keys - current_keys
                    for err in errors:
                        k = f"{err['file']}:{err['line']}:{err['col']}"
                        if k in new_keys:
                            print(f"[NEW] SYNTAX_ERROR: {err['file']}:{err['line']}:{err['col']} {err['message']}")
                    for k in resolved_keys:
                        print(f"[RESOLVED] {k}")
                    prev_error_keys = current_keys
                    time.sleep(args.interval)
                except KeyboardInterrupt:
                    print("\n[ai-db check --watch] Stopped.")
                    break
        else:
            errors = dispatcher.execute("check", {
                "path": args.path,
                "project": active_proj,
                "allow_project": allowed_projs
            })
            if not errors:
                print("No syntax errors detected.")
                dispatcher.close()
                return 0
            else:
                for err in errors:
                    print(f"SYNTAX_ERROR: {err['file']}:{err['line']}:{err['col']} [{err['project']}] {err['message']}")
                print(f"Total errors: {len(errors)}")
                dispatcher.close()
                sys.exit(1)

    elif args.command == "symbol":
        symbols = dispatcher.execute("symbol", {
            "name": args.name,
            "project": active_proj,
            "allow_project": allowed_projs
        })
        if not symbols:
            print(f"NO_SYMBOLS_FOUND: {args.name}")
        else:
            for s in symbols:
                print(f"@{s['file']}:L{s['line']} ({s['symbol_type']} {s['name']}) [{s['project']}]")

    elif args.command == "outline":
        target_path = os.path.abspath(args.path)
        if not os.path.exists(target_path):
            print(f"Error: File not found: {args.path}", file=sys.stderr)
            dispatcher.close()
            sys.exit(1)
        out = dispatcher.execute("outline", {"path": target_path})
        total_lines = out.get("lines", 0)
        print(f"=== {out['file']} ({total_lines} lines) ===")
        for item in out.get("outline", []):
            print(f"  L{item['line']:<4} {item['label']}")

    elif args.command in ("route-skill", "suggest-skills", "route"):
        min_conf = getattr(args, "min_confidence", None)
        matches = dispatcher.execute("route_skill", {
            "prompt": args.prompt,
            "top": args.top,
            "min_confidence": min_conf if min_conf is not None else 0.0,
            "project": active_proj,
            "allow_project": allowed_projs
        })
        if not matches:
            if args.format == "json":
                print("[]")
            else:
                print("NO_SKILL_MATCH")
        else:
            if args.format == "json":
                print(json.dumps(matches, indent=2))
            elif args.format == "path":
                for m in matches:
                    print(m["filepath"])
            else:
                for idx, m in enumerate(matches, 1):
                    prefix = "PRIMARY" if idx == 1 else f"SECONDARY #{idx}"
                    rel_p = os.path.relpath(m["filepath"], os.getcwd())
                    reasons_str = "; ".join(m.get("reasons", []))
                    print(f"[{prefix}] {m['name']} (conf: {m['confidence']}) [{m.get('project', 'global')}] -> {rel_p}")
                    print(f"  Reason: {reasons_str}")
                    if m.get("description"):
                        print(f"  Desc: {m['description'][:140]}...")

    elif args.command == "sync-skills":
        res = dispatcher.execute("sync_skills", {"skill_dirs": args.dir, "project": args.project})
        print(f"+{res['added']} ~{res['updated']} -{res['pruned']} ={res['skipped']}")

    elif args.command in ("context", "ctx"):
        if args.ctx_action == "save":
            files_list = []
            for f in args.files:
                files_list.extend([x.strip() for x in f.split(",") if x.strip()])
            tasks_list = []
            for t in args.tasks:
                tasks_list.extend([x.strip() for x in t.split(",") if x.strip()])

            res = dispatcher.execute("context_save", {
                "session_id": args.session_id,
                "summary": args.summary,
                "project": active_proj,
                "title": args.title,
                "active_files": files_list,
                "open_tasks": tasks_list,
                "notes": args.notes
            })
            print(f"[ai-db context] Saved session '{res['session_id']}' for project '{res['project']}'")

        elif args.ctx_action == "get":
            ctx = dispatcher.execute("context_recall", {
                "session_id": args.session_id,
                "project": active_proj,
                "allow_project": allowed_projs
            })
            if not ctx or not ctx.get("summary"):
                print("NO_CONTEXT_FOUND")
            else:
                if args.format == "json":
                    print(json.dumps(ctx, indent=2))
                elif args.format == "dense":
                    print(f"[{ctx['project']}:{ctx['session_id']}] {ctx['title']} | Files: {len(ctx.get('active_files', []))} | Tasks: {len(ctx.get('open_tasks', []))}")
                    print(f"Summary: {ctx['summary']}")
                else:
                    print(f"### [Context Memory: {ctx['project']} / {ctx['session_id']}]")
                    print(f"**Title**: {ctx['title']}")
                    print(f"**Summary**: {ctx['summary']}")
                    if ctx.get('active_files'):
                        print("\n**Active Files**:")
                        for f in ctx['active_files']:
                            print(f"- {f}")
                    if ctx.get('open_tasks'):
                        print("\n**Pending Tasks**:")
                        for idx, t in enumerate(ctx['open_tasks'], 1):
                            print(f"{idx}. {t}")
                    if ctx.get('full_notes') and ctx['full_notes'] != ctx['summary']:
                        print(f"\n**Notes & Decisions**:\n{ctx['full_notes']}")

        elif args.ctx_action == "list":
            contexts = dispatcher.execute("context_recall", {
                "list": True,
                "project": active_proj,
                "allow_project": allowed_projs
            })
            if not contexts:
                print("NO_SAVED_CONTEXTS")
            else:
                for c in contexts:
                    date_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["timestamp"]))
                    print(f"[{c['project']}] {c['session_id']} ({date_str}) - {c['title']} ({c['active_files_count']} files, {c['open_tasks_count']} tasks)")

        elif args.ctx_action == "query":
            hits = dispatcher.execute("context_recall", {
                "query": args.search,
                "project": active_proj,
                "allow_project": allowed_projs
            })
            if not hits:
                print("NO_HITS")
            else:
                for h in hits:
                    print(f"[{h['project']}:{h['session_id']}] {h['title']}")
                    print(f"  {h['summary']}")
        else:
            ctx_p.print_help()

    elif args.command == "remember":
        ctx = dispatcher.execute("context_recall", {
            "session_id": args.session_id,
            "project": active_proj,
            "allow_project": allowed_projs
        })
        if not ctx or not ctx.get("summary"):
            print("NO_CONTEXT_FOUND")
        else:
            print(f"### [ai-db Context Memory: {ctx['project']} / {ctx['session_id']}]")
            print(f"**Title**: {ctx['title']}")
            print(f"**Summary**: {ctx['summary']}")
            if ctx.get('active_files'):
                print("\n**Active Files**:")
                for f in ctx['active_files']:
                    print(f"- {f}")
            if ctx.get('open_tasks'):
                print("\n**Pending Tasks**:")
                for idx, t in enumerate(ctx['open_tasks'], 1):
                    print(f"{idx}. {t}")
            if ctx.get('full_notes') and ctx['full_notes'] != ctx['summary']:
                print(f"\n**Notes & Decisions**:\n{ctx['full_notes']}")

    elif args.command == "analyze":
        parsed_span = None
        if args.span:
            parts = args.span.split(":")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                parsed_span = (int(parts[0]), int(parts[1]))
            else:
                print(f"Error: Invalid --span format '{args.span}'. Expected START:END (e.g. 50:120)", file=sys.stderr)
                dispatcher.close()
                sys.exit(1)

        targets = args.targets
        if not targets:
            db_inst = dispatcher._get_db({"db": args.db})
            last_analysis = db_inst.get_session_state("last_analysis") if hasattr(db_inst, "get_session_state") else None
            if last_analysis and "targets" in last_analysis:
                targets = last_analysis["targets"]
            else:
                targets = ["."]

        output_data = dispatcher.execute("analyze", {
            "targets": targets,
            "depth": args.depth,
            "span": parsed_span,
            "focus": args.focus,
            "q": args.question,
            "since": args.since,
            "ctx_lines": args.ctx,
            "max_out": args.max_out,
            "cursor": args.cursor,
            "no_cache": args.no_cache,
            "format": args.format
        })

        db_inst = dispatcher._get_db({"db": args.db})
        default_fmt = db_inst.get_session_state("default_format") if hasattr(db_inst, "get_session_state") else None
        effective_fmt = args.format or default_fmt or "stub"

        if isinstance(output_data, str):
            print(output_data)
        elif effective_fmt == "json":
            print(json.dumps(output_data, indent=2))
        elif effective_fmt == "stub":
            print(VectorDB.format_as_stub(output_data))
        elif effective_fmt == "sexp":
            print(VectorDB.format_as_sexp(output_data))
        elif effective_fmt == "outline":
            items = output_data.get("symbols", []) if "symbols" in output_data else []
            if not items and "results" in output_data:
                for fpath, fres in output_data["results"].items():
                    print(f"=== {fpath} ({fres['meta']['tokens_out']} tokens) ===")
                    for s in fres.get("symbols", []):
                        sig = s.get("sig", s["name"])
                        span_str = f"L{s['span'][0]}-L{s['span'][1]}" if "span" in s else ""
                        print(f"  [{s.get('ref', '')}] {span_str:10} {sig}")
            else:
                for s in items:
                    sig = s.get("sig", s["name"])
                    span_str = f"L{s['span'][0]}-L{s['span'][1]}" if "span" in s else ""
                    print(f"[{s.get('ref', '')}] {span_str:10} {sig}")
        else:
            meta = output_data.get("meta", {})
            print(f"### Analysis Result ({meta.get('tokens_out', 0)} tokens, cached={meta.get('cached', False)})")
            if "symbols" in output_data:
                for s in output_data["symbols"]:
                    print(f"- **{s['name']}** ({s['kind']}) `{s.get('ref', '')}`: {s.get('sig', '')}")
                    if s.get("body"):
                        print(f"```\n{s['body']}\n```")
            elif "results" in output_data:
                for fpath, fres in output_data["results"].items():
                    print(f"\n#### {fpath}")
                    for s in fres.get("symbols", []):
                        print(f"- **{s['name']}** `{s.get('ref', '')}`: {s.get('sig', '')}")
                        if s.get("body"):
                            print(f"```\n{s['body']}\n```")

    elif args.command == "expand":
        parsed_span = None
        if args.span:
            parts = args.span.split(":")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                parsed_span = (int(parts[0]), int(parts[1]))
            else:
                print(f"Error: Invalid --span format '{args.span}'. Expected START:END", file=sys.stderr)
                dispatcher.close()
                sys.exit(1)

        try:
            exp = dispatcher.execute("expand", {
                "ref": args.ref,
                "depth": args.depth,
                "span": parsed_span
            })
        except Exception as e:
            print(f"Error: Ref '{args.ref}' not found or expired: {e}", file=sys.stderr)
            dispatcher.close()
            sys.exit(1)

        if args.format == "json":
            print(json.dumps(exp, indent=2))
        else:
            print(exp.get("body", ""))

    elif args.command == "locate":
        hits = dispatcher.execute("locate", {
            "query": args.query,
            "scope": args.scope,
            "k": args.k,
            "format": args.format
        })
        if not hits:
            print("NO_LOCATE_HITS")
        elif isinstance(hits, str):
            print(hits)
        else:
            db_inst = dispatcher._get_db({"db": args.db})
            default_fmt = db_inst.get_session_state("default_format") if hasattr(db_inst, "get_session_state") else None
            effective_fmt = args.format or default_fmt or "stub"
            if effective_fmt == "stub":
                for h in hits:
                    span_str = f"L{h['span'][0]}-{h['span'][1]}" if h.get("span") else ""
                    print(f"# {h['file']}:{span_str} (score:{h.get('score', 0)})")
                    snippet = h.get("snippet", "").strip()
                    if snippet:
                        for s_line in snippet.splitlines()[:4]:
                            print(f"  {s_line}")
                    print()
            elif effective_fmt == "sexp":
                def sexp_esc(val):
                    return f'"{str(val).replace(chr(34), chr(92)+chr(34)).replace(chr(10), " ")}"'
                hit_sexps = []
                for h in hits:
                    span_s = f"({h['span'][0]} {h['span'][1]})" if h.get("span") else "nil"
                    hit_sexps.append(f"(:hit :file {sexp_esc(h['file'])} :name {sexp_esc(h.get('name'))} :span {span_s} :score {h.get('score', 0)})")
                print(f"(:locate :query {sexp_esc(args.query)} :hits ({' '.join(hit_sexps)}))")
            else:
                print(json.dumps(hits, indent=2))

    elif args.command == "status":
        s = dispatcher.execute("status", {})
        db_inst = dispatcher._get_db({"db": args.db})
        active_fmt = db_inst.get_session_state("default_format") if hasattr(db_inst, "get_session_state") else "stub"
        skills_str = f" | skills:{s.get('skills', 0)}" if "skills" in s else ""
        contexts_str = f" | contexts:{s.get('contexts', 0)}" if "contexts" in s else ""
        print(f"db:{s['db']} | files:{s['files']} | chunks:{s['chunks']} | symbols:{s['symbols']} | syntax_errors:{s['syntax_errors']}{skills_str}{contexts_str} | size:{s['kb']}KB | default_format:{active_fmt}")

    elif args.command == "prune":
        res = dispatcher.execute("prune", {})
        print(f"pruned:{res['pruned']}")

    elif args.command in ("optimize", "vacuum"):
        prune_flag = not getattr(args, "no_prune", False)
        def_fmt = getattr(args, "default_format", None)
        res = dispatcher.execute("optimize", {"prune_missing": prune_flag, "default_format": def_fmt})
        print(f"[ai-db optimize] Merged FTS indexes, updated planner stats & vacuumed.")
        print(f"Size: {res['initial_kb']}KB -> {res['final_kb']}KB (reclaimed: {res['reclaimed_kb']}KB, pruned_files: {res['pruned_files']})")
        print(f"Active default output format: '{res['default_format']}'")

    elif args.command == "diff":
        abs_path = os.path.abspath(args.path)
        if not os.path.exists(abs_path):
            print(f"Error: File not found: {args.path}", file=sys.stderr)
            dispatcher.close()
            sys.exit(1)
        diff = dispatcher.execute("diff", {"path": abs_path, "since": args.since})
        if args.format == "json":
            print(json.dumps(diff, indent=2))
        else:
            added = diff.get("added", [])
            changed = diff.get("changed", [])
            removed = diff.get("removed", [])
            if not added and not changed and not removed:
                print("No changes detected since last indexed version.")
            else:
                for span in added:
                    print(f"+ L{span[0]}-{span[1]}")
                for span in changed:
                    print(f"~ L{span[0]}-{span[1]}")
                for span in removed:
                    print(f"- L{span[0]}-{span[1]}")

    elif args.command == "callers":
        hits = dispatcher.execute("callers", {
            "name": args.name,
            "project": active_proj,
            "allow_project": allowed_projs
        })
        if not hits:
            print(f"NO_CALLERS_FOUND: {args.name}")
        else:
            for h in hits:
                print(f"@{h['file']}:L{h['line']} ({h['ref_type']} in {h['caller']}) [{h['project']}]")

    elif args.command == "todos":
        hits = dispatcher.execute("todos", {
            "kind": args.kind,
            "filepath": os.path.abspath(args.file) if args.file else None,
            "project": active_proj
        })
        if not hits:
            print("NO_ANNOTATIONS_FOUND")
        elif args.format == "json":
            print(json.dumps(hits, indent=2))
        else:
            for h in hits:
                rel = os.path.relpath(h["filepath"], os.getcwd())
                sym_str = f" [{h['symbol']}]" if h.get("symbol") else ""
                print(f"{h['kind'].upper()}: {rel}:L{h['line']}{sym_str} — {h['content']}")

    elif args.command == "telemetry":
        res = dispatcher.execute("telemetry", {"reset": getattr(args, "reset", False)})
        if getattr(args, "reset", False):
            print("Telemetry metrics reset.")
        elif getattr(args, "format", "dense") == "json":
            print(json.dumps(res, indent=2))
        else:
            print("=== Telemetry Dashboard ===")
            tokens = res.get("tokens", res.get("token_savings", {}))
            latency = res.get("latency", {})
            cache = res.get("cache", {})
            weak = res.get("weak_points", {})
            print(f"Token Savings: {tokens.get('net_saved_tokens', tokens.get('net_tokens_saved', 0))} tokens saved ({tokens.get('savings_pct', tokens.get('reduction_pct', 0.0))}%)")
            print(f"Query Latency: avg={latency.get('avg_ms', latency.get('avg_latency_ms', 0.0))}ms, p50={latency.get('p50_ms', 0.0)}ms, p95={latency.get('p95_ms', 0.0)}ms")
            print(f"Cache Performance: {cache.get('hits', 0)} hits / {cache.get('lookups', 0)} lookups ({cache.get('hit_rate_pct', 0.0)}%)")
            if weak:
                print(f"Weak Points: {weak.get('syntax_errors', 0)} syntax errors, {len(weak.get('complexity_hotspots', []))} hotspots")

    dispatcher.close()
    return 0


if __name__ == "__main__":
    main()
