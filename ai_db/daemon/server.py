"""FastAPI daemon server with per-project VectorDB pool."""

import asyncio
import json
import os
import time
import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from ai_db import VectorDB, load_config
from ai_db.config import AppConfig
from ai_db.constants import DEFAULT_DB_FILE
from ai_db.daemon.config import DaemonConfig, DaemonProjectConfig
from ai_db.daemon.models import (
    BatchToolCallRequest,
    BatchToolCallResponse,
    DaemonHealth,
    ListToolsResponse,
    MetricsResponse,
    ProjectMetricsResponse,
    ProjectResponse,
    ProjectStatus,
    ProjectTemplate,
    RegisterProjectRequest,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)
from ai_db.dispatcher import ServiceDispatcher


@dataclass
class ProjectMetrics:
    """Metrics for a single project."""
    requests_total: int = 0
    requests_errors: int = 0
    latency_sum_ms: float = 0.0
    last_access: float = 0.0
    model_load_time_ms: float = 0.0


class ConfigWatcher:
    """Watches config file for changes and triggers reload."""
    
    def __init__(self, config_path: str, reload_callback, interval: float = 2.0):
        self.config_path = os.path.expanduser(config_path)
        self.reload_callback = reload_callback
        self.interval = interval
        self._last_hash: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
    
    def _compute_hash(self) -> Optional[str]:
        try:
            with open(self.config_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return None
    
    async def start(self):
        self._running = True
        self._last_hash = self._compute_hash()
        self._task = asyncio.create_task(self._watch())
    
    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _watch(self):
        while self._running:
            await asyncio.sleep(self.interval)
            if not self._running:
                break
            current_hash = self._compute_hash()
            if current_hash and current_hash != self._last_hash:
                self._last_hash = current_hash
                try:
                    await self.reload_callback()
                except Exception as e:
                    print(f"[daemon] Hot reload failed: {e}")


class ProjectPool:
    """Manages lazy-loaded VectorDB instances per project."""

    def __init__(self, global_config: AppConfig, daemon_config: DaemonConfig):
        self.global_config = global_config
        self.daemon_config = daemon_config
        self.projects: dict[str, VectorDB] = {}
        self.project_configs: dict[str, AppConfig] = {}
        self.dispatchers: dict[str, ServiceDispatcher] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._start_time = time.time()
        # Progress callbacks: token -> callback
        self._progress_callbacks: dict[str, asyncio.Queue] = {}
        # Metrics per project
        self._metrics: dict[str, ProjectMetrics] = {}
        # Model warming status
        self._warmed: set[str] = set()

    def _get_lock(self, name: str) -> asyncio.Lock:
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    def register_progress_callback(self, token: str) -> asyncio.Queue:
        """Register a progress callback for a token, returns a queue to consume progress."""
        queue: asyncio.Queue = asyncio.Queue()
        self._progress_callbacks[token] = queue
        return queue

    def unregister_progress_callback(self, token: str) -> None:
        """Unregister a progress callback."""
        self._progress_callbacks.pop(token, None)

    def get_progress_callback(self, token: str):
        """Get a progress callback function for a token."""
        queue = self._progress_callbacks.get(token)
        if queue is None:
            return None

        async def report(done: int, total: int, message: str = "") -> None:
            await queue.put({"progressToken": token, "progress": done, "total": total, "message": message})

        return report

    async def get_or_create(self, name: str) -> tuple[VectorDB, ServiceDispatcher]:
        """Get or create VectorDB and dispatcher for a project."""
        lock = self._get_lock(name)
        async with lock:
            if name in self.projects:
                # Update last access for idle TTL
                if name in self._metrics:
                    self._metrics[name].last_access = time.time()
                return self.projects[name], self.dispatchers[name]

            # Determine project config
            if name in self.project_configs:
                cfg = self.project_configs[name]
            elif name in self.daemon_config.projects:
                proj_cfg = self.daemon_config.projects[name]
                if proj_cfg.config_path:
                    cfg = load_config(proj_cfg.config_path)
                else:
                    cfg = self.global_config
            else:
                cfg = self.global_config

            # Determine db_path
            if name in self.daemon_config.projects:
                db_path = os.path.expanduser(self.daemon_config.projects[name].db_path)
            else:
                # Auto-discovered project - use default DB path
                raw_path = cfg.storage.options.get("path")
                db_path = os.path.expanduser(raw_path) if raw_path else DEFAULT_DB_FILE

            # Create VectorDB and dispatcher
            vectordb = VectorDB(db_path, config=cfg)
            dispatcher = ServiceDispatcher(db=vectordb, config=cfg)

            self.projects[name] = vectordb
            self.project_configs[name] = cfg
            self.dispatchers[name] = dispatcher
            # Initialize metrics for new project
            if name not in self._metrics:
                self._metrics[name] = ProjectMetrics()
            self._metrics[name].last_access = time.time()

            return vectordb, dispatcher

    async def remove(self, name: str) -> None:
        """Remove a project from the pool."""
        lock = self._get_lock(name)
        async with lock:
            if name in self.projects:
                self.projects[name].close()
                del self.projects[name]
            if name in self.project_configs:
                del self.project_configs[name]
            if name in self.dispatchers:
                del self.dispatchers[name]
            if name in self._locks:
                del self._locks[name]

    async def list_projects(self) -> dict[str, ProjectStatus]:
        """Get status of all projects (loaded and registered)."""
        result = {}
        # First, add loaded projects
        for name, db in self.projects.items():
            try:
                status = db.status()
                result[name] = ProjectStatus(
                    name=name,
                    db_path=db.db_path,
                    chunks=status.get("chunks", 0),
                    files=status.get("files", 0),
                    symbols=status.get("symbols", 0),
                    syntax_errors=status.get("syntax_errors", 0),
                    size_kb=status.get("kb", 0),
                    loaded=True,
                )
            except Exception:
                result[name] = ProjectStatus(
                    name=name,
                    db_path=db.db_path,
                    loaded=False,
                )
        # Then, add registered but unloaded projects
        for name, proj_cfg in self.daemon_config.projects.items():
            if name not in result:
                result[name] = ProjectStatus(
                    name=name,
                    db_path=os.path.expanduser(proj_cfg.db_path),
                    loaded=False,
                )
        return result

    async def shutdown(self) -> None:
        """Close all projects."""
        for name in list(self.projects.keys()):
            await self.remove(name)

    async def warm_project(self, name: str) -> None:
        """Pre-load embedding/rerank models for a project."""
        if name in self._warmed:
            return
        lock = self._get_lock(name)
        async with lock:
            if name in self._warmed:
                return
            if name not in self.projects:
                await self.get_or_create(name)
            db = self.projects[name]
            # Access embedder and reranker to trigger loading
            start = time.time()
            _ = db.embedder
            _ = db.reranker
            self._metrics[name] = self._metrics.get(name, ProjectMetrics())
            self._metrics[name].model_load_time_ms = (time.time() - start) * 1000
            self._warmed.add(name)
            print(f"[daemon] Warmed project '{name}' in {self._metrics[name].model_load_time_ms:.0f}ms")

    async def cleanup_idle(self, ttl_seconds: int = 300) -> None:
        """Unload projects idle for more than ttl_seconds."""
        now = time.time()
        for name in list(self.projects.keys()):
            metrics = self._metrics.get(name)
            if metrics and (now - metrics.last_access) > ttl_seconds:
                print(f"[daemon] Unloading idle project '{name}'")
                await self.remove(name)
                self._metrics.pop(name, None)
                self._warmed.discard(name)


# Global pool instance
_pool: ProjectPool | None = None


def get_pool() -> ProjectPool:
    if _pool is None:
        raise RuntimeError("Daemon not initialized")
    return _pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool, _config_watcher, _idle_ttl_task
    # Startup
    from ai_db.config import load_config as load_main_config
    from ai_db.daemon.config import DaemonConfig as InternalDaemonConfig

    config = load_main_config()
    daemon_cfg = InternalDaemonConfig.from_dict(config.daemon.to_dict())
    _pool = ProjectPool(config, daemon_cfg)

    # Config file for hot reload
    config_path = config.source_path or "~/.config/ai-db/config.json"

    # Start config watcher
    _config_watcher = ConfigWatcher(
        config_path=config_path,
        reload_callback=_reload_config,
        interval=2.0,
    )
    await _config_watcher.start()

    # Start idle TTL task
    _idle_ttl_task = asyncio.create_task(_idle_ttl_cleanup())

    # Start WAL checkpoint task
    _wal_checkpoint_task = asyncio.create_task(_wal_checkpoint_loop())

    # Auto-start configured projects with model warming
    for name, proj_cfg in daemon_cfg.projects.items():
        if proj_cfg.auto_start:
            try:
                await _pool.get_or_create(name)
                await _pool.warm_project(name)
            except Exception as e:
                print(f"[daemon] Failed to auto-start project '{name}': {e}")

    # Auto-discover projects from auto_sync_paths
    if daemon_cfg.auto_discover:
        for path in config.auto_sync_paths:
            expanded = os.path.abspath(os.path.expanduser(path))
            if os.path.exists(expanded):
                proj_name = os.path.basename(expanded)
                if proj_name not in _pool.projects and proj_name not in daemon_cfg.projects:
                    try:
                        await _pool.get_or_create(proj_name)
                        await _pool.warm_project(proj_name)
                    except Exception as e:
                        print(f"[daemon] Failed to auto-discover project '{proj_name}': {e}")

    yield

    # Shutdown
    if _config_watcher:
        await _config_watcher.stop()
    if _idle_ttl_task:
        _idle_ttl_task.cancel()
        try:
            await _idle_ttl_task
        except asyncio.CancelledError:
            pass
    if _wal_checkpoint_task:
        _wal_checkpoint_task.cancel()
        try:
            await _wal_checkpoint_task
        except asyncio.CancelledError:
            pass
    if _pool:
        await _pool.shutdown()
        _pool = None


_config_watcher: Optional[ConfigWatcher] = None
_idle_ttl_task: Optional[asyncio.Task] = None
_wal_checkpoint_task: Optional[asyncio.Task] = None


async def _reload_config():
    """Reload configuration and re-register projects."""
    global _pool
    if _pool is None:
        return
    
    from ai_db.config import load_config as load_main_config
    from ai_db.daemon.config import DaemonConfig as InternalDaemonConfig

    new_config = load_main_config()
    old_pool = _pool

    # Create new pool
    daemon_cfg = InternalDaemonConfig.from_dict(new_config.daemon.to_dict())
    _pool = ProjectPool(new_config, daemon_cfg)

    # Migrate running projects with warming
    for name in old_pool.projects:
        if name in new_config.daemon.projects:
            try:
                await _pool.get_or_create(name)
                await _pool.warm_project(name)
            except Exception as e:
                print(f"[daemon] Failed to migrate project '{name}': {e}")

    # Shutdown old pool
    await old_pool.shutdown()


async def _idle_ttl_cleanup():
    """Periodically unload idle projects."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        if _pool is None:
            break
        await _pool.cleanup_idle(ttl_seconds=300)  # 5 min default TTL


async def _idle_ttl_cleanup():
    """Periodically unload idle projects."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        if _pool is None:
            break
        await _pool.cleanup_idle(ttl_seconds=300)  # 5 min default TTL


async def _wal_checkpoint_loop():
    """Periodically checkpoint SQLite WAL files."""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        if _pool is None:
            break
        for name, db in _pool.projects.items():
            try:
                # Use storage backend's wal_checkpoint method
                if hasattr(db, 'backend') and hasattr(db.backend, 'wal_checkpoint'):
                    db.backend.wal_checkpoint("TRUNCATE")
                    print(f"[daemon] WAL checkpoint completed for project '{name}'")
            except Exception as e:
                print(f"[daemon] WAL checkpoint failed for '{name}': {e}")


# Auth configuration
API_KEYS: dict[str, str] = {}  # project_name -> api_key
RATE_LIMITS: dict[str, tuple[int, float]] = {}  # project_name -> (requests, window_seconds)

# Load API keys from environment
def _load_api_keys():
    import os
    keys = os.environ.get("AI_DB_API_KEYS", "")
    for pair in keys.split(","):
        if "=" in pair:
            project, key = pair.split("=", 1)
            API_KEYS[project.strip()] = key.strip()


# Auth middleware
async def verify_api_key(request: Request, call_next):
    """Verify API key for protected endpoints."""
    # Skip auth for health, metrics, templates
    if request.url.path in ("/health", "/metrics", "/templates"):
        return await call_next(request)
    
    # Extract project name from URL path
    # Path format: /projects/{name}/...
    path_parts = request.url.path.strip("/").split("/")
    project = None
    if len(path_parts) >= 2 and path_parts[0] == "projects":
        project = path_parts[1]
    
    if not project:
        return await call_next(request)
    
    # Check if project has API key configured
    if project in API_KEYS:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Missing or invalid Authorization header")
        token = auth_header[7:]
        if token != API_KEYS[project]:
            raise HTTPException(403, "Invalid API key")
    
    return await call_next(request)


# Rate limiting middleware
class RateLimiter:
    def __init__(self):
        self._buckets: dict[str, list[float]] = {}
    
    def is_allowed(self, key: str, limit: int, window: float) -> bool:
        now = time.time()
        if key not in self._buckets:
            self._buckets[key] = []
        # Remove old entries
        self._buckets[key] = [t for t in self._buckets[key] if now - t < window]
        if len(self._buckets[key]) >= limit:
            return False
        self._buckets[key].append(now)
        return True


rate_limiter = RateLimiter()


async def rate_limit(request: Request, call_next):
    """Apply rate limiting per project."""
    project = request.path_params.get("name", "global")
    limit, window = RATE_LIMITS.get(project, (100, 60.0))  # Default: 100 req/min
    
    client_ip = request.client.host if request.client else "unknown"
    key = f"{project}:{client_ip}"
    
    if not rate_limiter.is_allowed(key, limit, window):
        raise HTTPException(429, f"Rate limit exceeded for project '{project}'")
    
    return await call_next(request)


# Request logging middleware
async def log_requests(request: Request, call_next):
    """Log all requests in JSON format."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    # Get project name from path
    project = request.path_params.get("name", "unknown")
    
    response = await call_next(request)
    
    duration_ms = (time.time() - start_time) * 1000
    
    log_entry = {
        "timestamp": time.time(),
        "method": request.method,
        "path": request.url.path,
        "project": project,
        "client_ip": client_ip,
        "status_code": response.status_code,
        "duration_ms": round(duration_ms, 2),
    }
    
    # Log to stdout in JSON format
    print(json.dumps(log_entry))
    
    return response


# Load API keys on startup
_load_api_keys()

app = FastAPI(title="ai-db Daemon", lifespan=lifespan)

# Add middleware
app.middleware("http")(verify_api_key)
app.middleware("http")(rate_limit)
app.middleware("http")(log_requests)


@app.get("/health", response_model=DaemonHealth)
async def health():
    pool = get_pool()
    projects = await pool.list_projects()
    return DaemonHealth(
        status="ok",
        projects=projects,
        uptime_seconds=time.time() - pool._start_time,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    pool = get_pool()
    global_requests = 0
    global_errors = 0
    total_latency = 0.0
    project_metrics = {}
    
    for name, m in pool._metrics.items():
        avg_latency = m.latency_sum_ms / m.requests_total if m.requests_total > 0 else 0.0
        project_metrics[name] = ProjectMetricsResponse(
            requests_total=m.requests_total,
            requests_errors=m.requests_errors,
            avg_latency_ms=avg_latency,
            model_load_time_ms=m.model_load_time_ms,
            last_access=m.last_access,
            warmed=name in pool._warmed,
        )
        global_requests += m.requests_total
        global_errors += m.requests_errors
        total_latency += m.latency_sum_ms
    
    global_avg = total_latency / global_requests if global_requests > 0 else 0.0
    
    return MetricsResponse(
        global_requests=global_requests,
        global_errors=global_errors,
        global_avg_latency_ms=global_avg,
        projects=project_metrics,
        uptime_seconds=time.time() - pool._start_time,
    )


@app.get("/templates", response_model=list[ProjectTemplate])
async def list_templates():
    """List available project templates."""
    templates = [
        ProjectTemplate(
            name="python",
            description="Python project with code-optimized embeddings",
            config={
                "storage": {"provider": "sqlite", "options": {"path": "~/.local/share/ai-db/{name}.db"}},
                "retrieval": {"mode": "hybrid", "doc_weight": 0.5},
                "embedding": {"provider": "sentence_transformers", "model": "nomic-ai/nomic-embed-code", "device": "cuda", "batch_size": 32},
                "rerank": {"provider": "sentence_transformers", "model": "BAAI/bge-reranker-base", "device": "cuda"},
                "index": {"doc_weight": 0.5, "ignore": [".agents/**"]},
                "access": {"cross_project": {}},
                "auto_sync_paths": [],
                "trace": {"wait_patterns": None, "wait_patterns_extend": None},
                "daemon": {"enabled": True, "host": "127.0.0.1", "port": 8080, "auto_discover": True, "projects": {}},
            },
        ),
        ProjectTemplate(
            name="javascript",
            description="JavaScript/TypeScript project with code intelligence",
            config={
                "storage": {"provider": "sqlite", "options": {"path": "~/.local/share/ai-db/{name}.db"}},
                "retrieval": {"mode": "hybrid", "doc_weight": 0.5},
                "embedding": {"provider": "sentence_transformers", "model": "jina-embeddings-v2-base-code", "device": "cuda", "batch_size": 32},
                "rerank": {"provider": "sentence_transformers", "model": "BAAI/bge-reranker-base", "device": "cuda"},
                "index": {"doc_weight": 0.5, "ignore": [".agents/**", "node_modules/**", "dist/**", "build/**"]},
                "access": {"cross_project": {}},
                "auto_sync_paths": [],
                "trace": {"wait_patterns": None, "wait_patterns_extend": None},
                "daemon": {"enabled": True, "host": "127.0.0.1", "port": 8080, "auto_discover": True, "projects": {}},
            },
        ),
        ProjectTemplate(
            name="rust",
            description="Rust project with code intelligence",
            config={
                "storage": {"provider": "sqlite", "options": {"path": "~/.local/share/ai-db/{name}.db"}},
                "retrieval": {"mode": "hybrid", "doc_weight": 0.5},
                "embedding": {"provider": "sentence_transformers", "model": "BAAI/bge-code-v1", "device": "cuda", "batch_size": 32},
                "rerank": {"provider": "sentence_transformers", "model": "BAAI/bge-reranker-base", "device": "cuda"},
                "index": {"doc_weight": 0.5, "ignore": [".agents/**", "target/**", "Cargo.lock"]},
                "access": {"cross_project": {}},
                "auto_sync_paths": [],
                "trace": {"wait_patterns": None, "wait_patterns_extend": None},
                "daemon": {"enabled": True, "host": "127.0.0.1", "port": 8080, "auto_discover": True, "projects": {}},
            },
        ),
        ProjectTemplate(
            name="go",
            description="Go project with code intelligence",
            config={
                "storage": {"provider": "sqlite", "options": {"path": "~/.local/share/ai-db/{name}.db"}},
                "retrieval": {"mode": "hybrid", "doc_weight": 0.5},
                "embedding": {"provider": "sentence_transformers", "model": "nomic-ai/nomic-embed-code", "device": "cuda", "batch_size": 32},
                "rerank": {"provider": "sentence_transformers", "model": "BAAI/bge-reranker-base", "device": "cuda"},
                "index": {"doc_weight": 0.5, "ignore": [".agents/**", "vendor/**", "go.sum"]},
                "access": {"cross_project": {}},
                "auto_sync_paths": [],
                "trace": {"wait_patterns": None, "wait_patterns_extend": None},
                "daemon": {"enabled": True, "host": "127.0.0.1", "port": 8080, "auto_discover": True, "projects": {}},
            },
        ),
    ]
    return templates


@app.get("/projects", response_model=list[ProjectResponse])
async def list_projects():
    pool = get_pool()
    result = []
    for name in set(list(pool.projects.keys()) + list(pool.daemon_config.projects.keys())):
        proj_cfg = pool.daemon_config.projects.get(name)
        if proj_cfg:
            result.append(ProjectResponse(
                name=name,
                db_path=proj_cfg.db_path,
                config_path=proj_cfg.config_path,
                auto_start=proj_cfg.auto_start,
            ))
        else:
            # Auto-discovered
            result.append(ProjectResponse(
                name=name,
                db_path=pool.projects[name].db_path,
                config_path=None,
                auto_start=True,
            ))
    return result


@app.post("/projects", response_model=ProjectResponse)
async def register_project(req: RegisterProjectRequest):
    pool = get_pool()

    # Validate project name
    if not req.name or not req.name.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "Project name must be alphanumeric with dashes/underscores")

    # Check if already registered
    if req.name in pool.projects or req.name in pool.daemon_config.projects:
        raise HTTPException(409, f"Project '{req.name}' already registered")

    # Load config if provided
    cfg = pool.global_config
    if req.config_path:
        cfg = load_config(req.config_path)

    # Create project config
    proj_cfg = DaemonProjectConfig(
        db_path=req.db_path,
        config_path=req.config_path,
        auto_start=req.auto_start,
    )
    pool.daemon_config.projects[req.name] = proj_cfg
    pool.project_configs[req.name] = cfg

    # Create if auto_start
    if req.auto_start:
        await pool.get_or_create(req.name)

    return ProjectResponse(
        name=req.name,
        db_path=req.db_path,
        config_path=req.config_path,
        auto_start=req.auto_start,
    )


@app.delete("/projects/{name}")
async def unregister_project(name: str):
    pool = get_pool()

    if name not in pool.projects and name not in pool.daemon_config.projects:
        raise HTTPException(404, f"Project '{name}' not found")

    await pool.remove(name)

    if name in pool.daemon_config.projects:
        del pool.daemon_config.projects[name]
    if name in pool.project_configs:
        del pool.project_configs[name]

    return {"status": "ok", "message": f"Project '{name}' unregistered"}


@app.post("/query/cross-project", response_model=ToolCallResponse)
async def cross_project_query(req: ToolCallRequest):
    """Execute query across multiple projects."""
    pool = get_pool()
    
    # Get projects to query
    projects_arg = req.arguments.get("projects")
    if not projects_arg:
        # Default to all loaded projects
        projects = list(pool.projects.keys())
    else:
        projects = projects_arg if isinstance(projects_arg, list) else [projects_arg]
    
    if not projects:
        return ToolCallResponse(
            content=[{"type": "text", "text": "[]"}],
            isError=False,
        )
    
    # Extract query
    query = req.arguments.get("query", "")
    top = req.arguments.get("top", 5)
    
    all_results = []
    for proj_name in projects:
        if proj_name not in pool.projects:
            continue
        try:
            _, dispatcher = await pool.get_or_create(proj_name)
            result = dispatcher.execute("query", {
                "query": query,
                "top": top,
                "project": proj_name,
            })
            # Add project name to results
            if isinstance(result, list):
                for r in result:
                    r["source_project"] = proj_name
                all_results.extend(result)
        except Exception as e:
            print(f"[daemon] Cross-project query error for '{proj_name}': {e}")
    
    # Sort by score descending
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Format result as text content
    import json
    text = json.dumps(all_results, indent=2)
    
    return ToolCallResponse(
        content=[{"type": "text", "text": text}],
        isError=False,
    )


@app.get("/projects/{name}/status", response_model=ProjectStatus)
async def project_status(name: str):
    pool = get_pool()
    projects = await pool.list_projects()
    if name not in projects:
        raise HTTPException(404, f"Project '{name}' not found")
    return projects[name]


@app.post("/projects/{name}/tools/list", response_model=ListToolsResponse)
async def list_tools(name: str):
    pool = get_pool()
    _, dispatcher = await pool.get_or_create(name)
    raw_tools = dispatcher.list_tools()
    tools = []
    for t in raw_tools:
        td = dict(t)
        if "inputSchema" not in td and "parameters_schema" in td:
            td["inputSchema"] = td["parameters_schema"]
        tools.append(ToolDefinition(**td))
    return ListToolsResponse(tools=tools)


@app.post("/projects/{name}/tools/call", response_model=ToolCallResponse)
async def call_tool(name: str, req: ToolCallRequest):
    pool = get_pool()
    _, dispatcher = await pool.get_or_create(name)

    start_time = time.time()
    try:
        # Add project to arguments if not present
        arguments = dict(req.arguments)
        if "project" not in arguments:
            arguments["project"] = name

        # Execute tool with progress support
        progress = None
        if req._meta and "progressToken" in req._meta:
            token = req._meta["progressToken"]
            progress = pool.get_progress_callback(token)

        result = dispatcher.execute(req.name, arguments, progress=progress)

        # Format result as text content
        if isinstance(result, str):
            text = result
        else:
            import json
            text = json.dumps(result, indent=2)

        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        if name in pool._metrics:
            m = pool._metrics[name]
            m.requests_total += 1
            m.latency_sum_ms += latency_ms
            m.last_access = time.time()

        return ToolCallResponse(
            content=[{"type": "text", "text": text}],
            isError=False,
        )
    except KeyError as e:
        if name in pool._metrics:
            pool._metrics[name].requests_errors += 1
        return ToolCallResponse(
            content=[{"type": "text", "text": f"Unknown tool: {e}"}],
            isError=True,
        )
    except Exception as e:
        if name in pool._metrics:
            pool._metrics[name].requests_errors += 1
        return ToolCallResponse(
            content=[{"type": "text", "text": f"Error executing tool '{req.name}': {e!s}"}],
            isError=True,
        )


@app.post("/projects/{name}/tools/call/batch", response_model=BatchToolCallResponse)
async def call_tool_batch(name: str, req: BatchToolCallRequest):
    """Execute multiple tool calls in a single request."""
    pool = get_pool()
    _, dispatcher = await pool.get_or_create(name)

    results = []
    for call in req.calls:
        try:
            # Add project to arguments if not present
            arguments = dict(call.arguments)
            if "project" not in arguments:
                arguments["project"] = name

            # Execute tool with progress support
            progress = None
            if call._meta and "progressToken" in call._meta:
                token = call._meta["progressToken"]
                progress = pool.get_progress_callback(token)

            result = dispatcher.execute(call.name, arguments, progress=progress)

            # Format result as text content
            if isinstance(result, str):
                text = result
            else:
                import json
                text = json.dumps(result, indent=2)

            results.append(ToolCallResponse(
                content=[{"type": "text", "text": text}],
                isError=False,
            ))
        except KeyError as e:
            results.append(ToolCallResponse(
                content=[{"type": "text", "text": f"Unknown tool: {e}"}],
                isError=True,
            ))
        except Exception as e:
            results.append(ToolCallResponse(
                content=[{"type": "text", "text": f"Error executing tool '{call.name}': {e!s}"}],
                isError=True,
            ))

    return BatchToolCallResponse(results=results)


@app.get("/projects/{name}/tools/call/progress")
async def tool_progress(name: str, token: str):
    """Server-Sent Events endpoint for tool progress updates."""
    pool = get_pool()

    # Ensure project exists
    await pool.get_or_create(name)

    # Register progress callback
    queue = pool.register_progress_callback(token)

    async def event_generator():
        try:
            # Send initial connected event
            yield f"data: {json.dumps({'type': 'connected', 'token': token})}\n\n"

            while True:
                # Wait for progress update
                try:
                    progress_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps({'type': 'progress', **progress_data})}\n\n"
                    # If progress is complete (done == total), break
                    if progress_data.get("progress") == progress_data.get("total") and progress_data.get("total", 0) > 0:
                        break
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            pool.unregister_progress_callback(token)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.websocket("/projects/{name}/tools/call/progress/ws")
async def tool_progress_ws(websocket: WebSocket, name: str, token: str):
    """WebSocket endpoint for tool progress updates (bidirectional)."""
    pool = get_pool()
    await websocket.accept()

    # Ensure project exists
    await pool.get_or_create(name)

    # Register progress callback
    queue = pool.register_progress_callback(token)

    try:
        # Send initial connected event
        await websocket.send_json({"type": "connected", "token": token})

        while True:
            # Wait for progress update
            try:
                progress_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json({"type": "progress", **progress_data})
                # If progress is complete (done == total), break
                if progress_data.get("progress") == progress_data.get("total") and progress_data.get("total", 0) > 0:
                    break
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})

    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        pool.unregister_progress_callback(token)
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/daemon/reload")
async def reload_config():
    """Reload configuration and re-register projects."""
    global _pool
    if _pool is None:
        raise HTTPException(500, "Daemon not initialized")

    # Reload main config
    new_config = load_config()
    old_pool = _pool

    # Create new pool
    _pool = ProjectPool(new_config, new_config.daemon)

    # Migrate running projects
    for name in old_pool.projects:
        if name in new_config.daemon.projects:
            try:
                await _pool.get_or_create(name)
            except Exception as e:
                print(f"[daemon] Failed to migrate project '{name}': {e}")

    # Shutdown old pool
    await old_pool.shutdown()

    return {"status": "ok", "message": "Configuration reloaded"}


@app.post("/daemon/shutdown")
async def shutdown():
    """Graceful shutdown."""
    import os
    import signal

    # Schedule shutdown after response
    async def do_shutdown():
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(do_shutdown())
    return {"status": "ok", "message": "Shutting down..."}


# Exception handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc!s}"},
    )