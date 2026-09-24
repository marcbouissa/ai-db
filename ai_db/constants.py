import os

_xdg_data = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
DEFAULT_DB_FILE = os.environ.get(
    "AI_DB_PATH",
    os.path.join(_xdg_data, "ai-db", "codebase_knowledge.db")
)
DEFAULT_CONFIG_FILE = os.path.expanduser("~/.config/ai-db/config.json")
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
    ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".java", ".sql",
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
EXT_TO_LANG = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".java": "java",
}

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

# Final ranking (tune only with `ai-db eval`)
RERANK_TOP = 30
RANK_W_RERANK, RANK_W_FUSED, RANK_W_GRAPH, RANK_W_EXACT = 0.70, 0.15, 0.10, 0.05
NORERANK_W_FUSED, NORERANK_W_GRAPH, NORERANK_W_EXACT = 0.70, 0.25, 0.05  # eval-tuned
DIVERSITY_WINDOW = 10
DIVERSITY_MAX_PER_FILE = 3

# investigate
SEED_K = {"locate": 12, "explain": 8, "impact": 5}
IMPACT_DEPTH = 3
MAX_EXPANDED_PER_SEED = 10
INVESTIGATE_MIN_BUDGET = 500
MAX_TESTS = 10
INVESTIGATE_BODY_SHARE = 0.6  # fraction of the budget available for full bodies
MAX_OMITTED = 15
EXPLAIN_CALLER_SEEDS = 3
MODULE_SEED_FACTOR = 0.3
COMPOUND_MAX_DEFINITIONS = 2
DELEGATE_MAX_TOKENS = 120  # seeds this small also get their callees' callees
