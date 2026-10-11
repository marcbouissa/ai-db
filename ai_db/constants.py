import os

_xdg_data = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
DEFAULT_DB_FILE = os.environ.get(
    "AI_DB_PATH",
    os.path.join(_xdg_data, "ai-db", "codebase_knowledge.db")
)
DEFAULT_CONFIG_FILE = os.path.expanduser("~/.config/ai-db/config.json")

# --- compute dtype selection (see ai_db.device.resolve_dtype) ---
# Candidate dtypes for CPU inference, best-first. float16 is deliberately
# absent: most CPUs have no hardware fp16 accumulate and pay for subnormal
# handling. Measured 0.11 chunks/s vs 0.50 for float32 on an i7-11800H.
CPU_DTYPES = ("float32", "bfloat16")
# Matmul size for the dtype probe: large enough to be compute-bound rather than
# dominated by per-op dispatch overhead on any current CPU.
DTYPE_PROBE_SIZE = 512
DTYPE_PROBE_ITERS = 5
# Best-of-N, so a scheduler hiccup on a loaded machine cannot decide the dtype
# for the rest of the process.
DTYPE_PROBE_REPEATS = 3
# How much faster the model's native dtype must be before we keep it. A margin
# above 1.0 keeps us from thrashing on measurement noise; 1.25 sits well clear
# of the run-to-run spread seen on an otherwise idle machine.
BF16_PROBE_MARGIN = 1.25
_custom_skill_dirs = os.environ.get("AI_DB_SKILL_DIRS")
if _custom_skill_dirs:
    DEFAULT_SKILL_DIRS = [os.path.expanduser(p.strip()) for p in _custom_skill_dirs.split(":") if p.strip()]
else:
    DEFAULT_SKILL_DIRS = [
        os.path.join(_xdg_data, "ai-db", "skills"),
        os.path.expanduser("~/.config/ai-db/skills"),
        os.path.expanduser("~/.gemini/config/skills"),
        os.path.expanduser("~/.gemini/antigravity/builtin/skills"),
    ]

INDEXABLE_EXTENSIONS = {
    ".py", ".pyi", ".js", ".ts", ".d.ts", ".jsx", ".tsx", ".html", ".css", ".scss",
    ".json", ".md", ".yaml", ".yml", ".toml", ".sh", ".bash",
    ".c", ".cpp", ".cc", ".h", ".hpp", ".rs", ".go", ".java", ".cs", ".sql",
    ".mjs", ".cjs", ".pkl",
    ".txt", ".rst"
}

# Directories completely ignored (raw binary or runtime garbage)
HARD_IGNORE_DIRS = {
    # Python
    "__pycache__", ".pytest_cache", ".tox", ".mypy_cache", ".ruff_cache",
    ".hypothesis", "htmlcov", "coverage", ".eggs", "eggs",
    # JS/TS
    "node_modules", ".turbo", ".next", ".nuxt", ".svelte-kit",
    # Build outputs
    "dist", "build", "out", "target",
    # VCS
    ".git", ".svn", ".hg",
    # Vendor
    "vendor",
    # VMs / envs
    ".venv", "venv",
}

# Subpaths to filter out inside node_modules and .venv to prevent noise & bloat
VENDOR_NOISE_EXTENSIONS = {
    ".map", ".min.js", ".min.css", ".bundle.js", ".chunk.js", ".wasm", ".lock"
}

# Syncs with at least this many changed files parse in a process pool.
PARALLEL_PARSE_MIN_FILES = 32

# Chunking
MAX_CHUNK_TOKENS = 512  # tiktoken o200k_base
TEXT_WINDOW_LINES = 60
TEXT_WINDOW_OVERLAP = 10
# The extension -> language map lives in ai_db/parser/languages.py.

# Lexical search
CODE_STOPWORDS = frozenset({
    "self", "return", "def", "the", "a", "an", "of", "to", "in", "is", "are", "be",
    "and", "or", "for", "on", "by", "with", "from", "this", "that", "it", "its",
    "where", "what", "how", "which", "does", "do", "done", "when", "why", "who",
    "can", "should", "would", "there", "into", "as", "at",
})

# Retrieval
CANDIDATE_POOL = 50  # candidates per retriever list before fusion/rerank
RRF_K = 60
QUERY_LOG_KEEP = 10000  # query_log rows kept (newest); telemetry covers this window
DEFAULT_DOC_WEIGHT = 0.5  # index.doc_weight: score factor for md/txt chunks (docs rank below code)
DEFAULT_RERANK_TOP_N = 10  # rerank.top_n: fused candidates the reranker re-scores

# Final ranking (tune only with `ai-db eval`)
RERANK_TOP = 30
RANK_W_RERANK, RANK_W_FUSED, RANK_W_GRAPH, RANK_W_EXACT = 0.70, 0.15, 0.10, 0.05
NORERANK_W_FUSED, NORERANK_W_GRAPH, NORERANK_W_EXACT = 0.70, 0.25, 0.05  # eval-tuned
DIVERSITY_WINDOW = 10
DIVERSITY_MAX_PER_FILE = 3

# investigate
SEED_K = {"locate": 12, "explain": 8, "impact": 5, "flow": 5, "diff": 8}
IMPACT_DEPTH = 3
MAX_EXPANDED_PER_SEED = 10
INVESTIGATE_MIN_BUDGET = 500
MAX_TESTS = 10
INVESTIGATE_BODY_SHARE = 0.6  # fraction of the budget available for full bodies
MAX_OMITTED = 15
EXPLAIN_CALLER_SEEDS = 3
MODULE_SEED_FACTOR = 0.3
COMPOUND_MAX_DEFINITIONS = 2
DIFF_QUERY_BOOST = 25.0  # a full query match must outrank raw diff density, or the mode is unusable
                        # for a targeted "just the X" review; 25 is the density ratio
                        # beyond which density wins.
DIFF_SEEDS_PER_FILE = 3  # diversity device only: a diff must not spend every seed slot on one file
# --- investigate pack rendering (see ai_db.analysis.formatters) ---
# Cap on how many items a rendered pack lists per section. A pack holds up to
# the budget's worth of evidence; printing all of it defeats the point of
# rendering it compactly, and the rest is reachable by ref.
PACK_STUB_MAX_ITEMS = 20
# Truncate a `why` provenance string. These are one short clause in practice;
# the cap only stops a pathological one.
PACK_STUB_WHY_CHARS = 160
DIFF_DEFAULT_SINCE = "HEAD"  # diff mode compares the working tree against this unless told otherwise
GIT_TIMEOUT_S = 10  # git invocations from investigate/trace must not stall a request
LINTER_TIMEOUT_S = 15  # an external linter (bash -n, yaml) taking longer is reported as an error
EVAL_GIT_TIMEOUT_S = 120  # `ai-db eval` worktree checkouts (a whole tree, not one request)
STALE_RESYNC_MAX_FILES = 20  # stale result files re-synced inline per read; above this they are flagged
DELEGATE_MAX_TOKENS = 120  # seeds this small also get their callees' callees
REF_ID_HEX_LEN = 12  # analysis ref ids `ref:<sha1 prefix>`: 48 bits (32 collided at ~1% over 10k refs)

# Skill router
SKILL_W_TRIGGER = 8.0  # weight for exact trigger phrase match in prompt
SKILL_MIN_CONFIDENCE = 0.15  # route_skills drops matches below this confidence by default
SKILL_W_BM25 = 5.0  # BM25 contribution after min-max normalisation to [0, 1]
SKILL_W_BM25_FLAT = 2.5  # BM25 contribution when every hit scored the same
SKILL_BM25_CANDIDATES = 20  # skills fetched from FTS before fusion
SKILL_CONTENT_MAX_CHARS = 4000  # skill body stored for FTS

# trace
TRACE_DEPTH = 20
TRACE_MAX_NODES = 200
TRACE_SNAP_LINES = 3  # trace file:line on a blank/comment/decorator line takes the definition starting within this many lines
TRACE_CONTEXT_LINES = 3
TRACE_WAIT_PATTERNS = (
    "wait_*",
    "*_until*",
    "join",
    "result",
    "gather",
    "wait_for",
    "await",
    "sleep",
    "barrier",
    "event",
    "condition",
    "semaphore",
    "lock",
    "acquire",
    "release",
)

# Daemon (ai_db/daemon/)
DAEMON_IDLE_TTL_S = 300  # unload a project's VectorDB after this long without a request
DAEMON_IDLE_CHECK_S = 60  # how often idle projects are looked for
DAEMON_WAL_CHECKPOINT_S = 300  # how often each project's WAL is checkpointed
DAEMON_CONFIG_POLL_S = 2.0  # config-file hot-reload poll interval
DAEMON_RATE_LIMIT = 100  # requests per DAEMON_RATE_WINDOW_S, per project and client IP
DAEMON_RATE_WINDOW_S = 60.0
DAEMON_PROGRESS_KEEPALIVE_S = 30.0  # progress stream keepalive when no event arrives
DAEMON_CLIENT_TIMEOUT_S = 30.0  # HTTP timeout of DaemonClient / DaemonManager requests
DAEMON_LOG_FILE = os.path.join(
    os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state")), "ai-db", "daemon.log")
INDEX_LANGUAGES_STATE_KEY = "index_languages"  # session_state key: grammars the index was built with

# Recall extensions (ai_db/ext/recall.py, built-in "vectors" in ai_db/ext/vector_recall.py)
RECALL_VECTORS_STATE_KEY = "recall_vectors_meta"  # session_state: model the skill/context vectors came from
CPU_THREADS_MAX = 8  # torch intra-op threads for CPU inference; override with AI_DB_CPU_THREADS
EMBED_BULK_THRESHOLD = 256  # pending chunks at or above which embedding.bulk_device is used
RECALL_EMBED_BATCH = 64  # texts per embed_documents call when backfilling
SKILL_W_RECALL = 5.0  # weight of a recall provider's similarity in route_skills (same scale as BM25)
SKILL_RECALL_MIN_SIM = 0.3  # similarities at or below this add nothing (unrelated skills)
CONTEXT_RECALL_CANDIDATES = 20  # contexts per ranking (BM25, each provider) before RRF fusion

# Command-output filter (ai_db/output_filter.py): `ai-db run` / `ai-db filter` / MCP filter_output
OUTPUT_BUDGET_TOKENS = 1200  # default size of the filtered result
OUTPUT_MAX_BLOCK_LINES = 60  # longer paragraphs are split into blocks of this size
OUTPUT_COLLAPSE_MIN = 3  # this many consecutive same-shape lines collapse into one marker
OUTPUT_TAIL_LINES = 6  # the last lines (usually the summary) are always candidates
OUTPUT_W_NEED = 3.0  # weight of relevance to the agent's stated need (BM25, normalised)
OUTPUT_W_SIGNAL = 1.5  # weight of error/failure lines in a block
OUTPUT_W_SUMMARY = 1.0  # weight of summary lines (pytest totals, "test result:", ...)
OUTPUT_W_TAIL = 0.75  # bonus for the final block
OUTPUT_STORE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "ai-db", "outputs")
OUTPUT_STORE_KEEP = 50  # stored raw outputs kept for re-filtering (oldest pruned)
OUTPUT_RERANK_MAX_BLOCKS = 200  # blocks scored by the semantic model (best by keyword/signal first)
OUTPUT_W_SEMANTIC = 0.7  # share of need-relevance from the semantic score when --rerank is on
OUTPUT_RERANK_BLOCK_CHARS = 2000  # text per block sent to the model
TODO_LINE_CHARS = 160  # `ai-db todos` prints each hit on one line, cut to this width
