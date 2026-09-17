import os

_xdg_data = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
DEFAULT_DB_FILE = os.environ.get(
    "AI_DB_PATH",
    os.path.join(_xdg_data, "ai-db", "codebase_knowledge.db")
)
DEFAULT_CONFIG_FILE = os.path.expanduser("~/.config/ai-db/config.json")
DEFAULT_SKILL_DIRS = [
    os.path.expanduser("~/.gemini/config/skills"),
    os.path.expanduser("~/.gemini/antigravity/builtin/skills")
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
