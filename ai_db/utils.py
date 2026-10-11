import contextvars
import hashlib
import os
import re

from ai_db.constants import HARD_IGNORE_DIRS, INDEXABLE_EXTENSIONS, VENDOR_NOISE_EXTENSIONS

_PROJECT_MARKERS = (".git", "package.json", "pyproject.toml", "Cargo.toml", "go.mod")


def detect_project_root(path: str) -> str:
    """Nearest ancestor directory of ``path`` holding a project marker (``.git``,
    ``pyproject.toml``...); the directory of ``path`` itself when there is none."""
    curr = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(curr):
        curr = os.path.dirname(curr)
    check_dir = curr
    home_dir = os.path.expanduser("~")
    while check_dir and check_dir != "/" and check_dir != home_dir:
        if any(os.path.exists(os.path.join(check_dir, marker)) for marker in _PROJECT_MARKERS):
            return check_dir
        parent = os.path.dirname(check_dir)
        if parent == check_dir:
            break
        check_dir = parent
    return curr


def detect_project_name(path: str) -> str:
    root = detect_project_root(path)
    return os.path.basename(root) if root != "/" else "global"

def get_allowed_projects(current_project: str, explicit_allowed: list[str] | None = None,
                         cross_project: dict[str, list[str]] | None = None) -> list[str]:
    """Projects readable from ``current_project``: itself, 'global', explicit ones and
    those granted in ``access.cross_project`` of the config."""
    allowed = {"global"}
    if current_project:
        allowed.add(current_project)
    if explicit_allowed:
        for p in explicit_allowed:
            if p:
                for sub in p.split(","):
                    sub = sub.strip()
                    if sub:
                        allowed.add(sub)
    if cross_project and current_project in cross_project:
        allowed.update(cross_project[current_project])
    return sorted(allowed)

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def tokenize(text: str) -> list[str]:
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    tokens = re.findall(r"\w{2,}", s1.lower())
    return tokens

def strip_code_bloat(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    compact: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                compact.append(line)
            prev_blank = True
        else:
            compact.append(line)
            prev_blank = False
    return "\n".join(compact)

def should_index_path(rel_path: str, filename: str) -> bool:
    parts = rel_path.split(os.sep)
    ext = os.path.splitext(filename)[1].lower()
    if any(p in HARD_IGNORE_DIRS for p in parts):
        return False
    if ".git" in parts:
        return filename in ("config", "HEAD", "description")
    if "node_modules" in parts:
        if any(filename.endswith(ne) for ne in VENDOR_NOISE_EXTENSIONS):
            return False
        if filename == "package.json":
            return True
        if filename.endswith(".d.ts") or ext in (".ts", ".pyi"):
            return True
        return filename in ("index.js", "main.js", "README.md")
    if any(p in (".venv", "venv") for p in parts):
        if any(filename.endswith(ne) for ne in VENDOR_NOISE_EXTENSIONS):
            return False
        if filename in ("METADATA", "RECORD", "py.typed", "pyproject.toml"):
            return True
        if filename.endswith(".pyi"):
            return True
        return filename == "__init__.py"
    if any(p in (".vscode", ".idea") for p in parts):
        return ext in (".json", ".xml", ".yaml", ".yml")
    if ext in INDEXABLE_EXTENSIONS:
        return not any(filename.endswith(ne) for ne in VENDOR_NOISE_EXTENSIONS)
    return False


# The working directory of the *client* that issued the current tool call. The CLI and
# MCP server run in the caller's directory, but the daemon is a long-lived process
# (chdir'd to "/" when daemonized), so it receives the client's cwd with each call.
_CLIENT_CWD: contextvars.ContextVar[str | None] = contextvars.ContextVar("ai_db_client_cwd", default=None)


def client_cwd() -> str:
    """Directory that relative paths and display paths of the current call refer to."""
    return _CLIENT_CWD.get() or os.getcwd()


def resolve_client_path(path: str) -> str:
    """``path`` (``~`` and relative allowed) as an absolute path, relative to client_cwd()."""
    return os.path.normpath(os.path.join(client_cwd(), os.path.expanduser(path)))
