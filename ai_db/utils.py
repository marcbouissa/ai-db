import hashlib
import os
import re

from ai_db.constants import HARD_IGNORE_DIRS, INDEXABLE_EXTENSIONS, VENDOR_NOISE_EXTENSIONS


def detect_project_name(path: str) -> str:
    curr = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(curr):
        curr = os.path.dirname(curr)
    check_dir = curr
    home_dir = os.path.expanduser("~")
    while check_dir and check_dir != "/" and check_dir != home_dir:
        if any(os.path.exists(os.path.join(check_dir, marker)) for marker in [".git", "package.json", "pyproject.toml", "Cargo.toml", "go.mod"]):
            return os.path.basename(check_dir)
        parent = os.path.dirname(check_dir)
        if parent == check_dir:
            break
        check_dir = parent
    return os.path.basename(curr) if curr != "/" else "global"

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
    compact = []
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
        if filename in ("index.js", "main.js", "README.md"):
            return True
        return False
    if any(p in (".venv", "venv") for p in parts):
        if any(filename.endswith(ne) for ne in VENDOR_NOISE_EXTENSIONS):
            return False
        if filename in ("METADATA", "RECORD", "py.typed", "pyproject.toml"):
            return True
        if filename.endswith(".pyi"):
            return True
        if filename == "__init__.py":
            return True
        return False
    if any(p in (".vscode", ".idea") for p in parts):
        return ext in (".json", ".xml", ".yaml", ".yml")
    if ext in INDEXABLE_EXTENSIONS:
        if any(filename.endswith(ne) for ne in VENDOR_NOISE_EXTENSIONS):
            return False
        return True
    return False
