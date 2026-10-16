"""Read-only git access for history mining: commit lists, classification, hunks, blame.

Everything here shells out to ``git`` with explicit argument lists (no shell) and only
uses read commands (``log``, ``show``, ``blame``, ``ls-tree``, ``rev-parse``), so it is
safe to point at a checkout someone else is working in: it never touches the index,
the working tree or refs.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

# Code we mine by default. Pathspecs, relative to the repo root.
DEFAULT_INCLUDE: tuple[str, ...] = (
    "*.cs", "*.py", "*.ts", "*.tsx", "*.js", "*.jsx", "*.mjs", "*.cjs",
    "*.c", "*.cc", "*.cpp", "*.h", "*.hpp", "*.go", "*.rs", "*.java", "*.kt",
    "*.pkl", "*.bzl", "*.sh",
)
# Vendored, generated and build-output trees. ``:(glob)`` so ``**`` spans directories.
DEFAULT_EXCLUDE: tuple[str, ...] = (
    "third_party/**", "third_party_*/**", "**/third_party/**", "external/**", "bazel-*/**",
    "logs/**", "**/node_modules/**", "**/dist/**", "**/build/**", "**/bin/**", "**/obj/**",
    "**/vendor/**", "**/.venv/**", "**/*.g.cs", "**/*.Designer.cs", "**/*.min.js",
    "**/*.generated.*", "**/Generated/**", "**/*_pb2.py",
)

_FIX_CONVENTIONAL = re.compile(r"^(fix|hotfix|bugfix)(\([^)]*\))?!?:", re.IGNORECASE)
_FIX_WORDS = re.compile(
    r"\b(fix(es|ed|ing)?|hotfix(es)?|bug(s|fix)?|crash(es|ed|ing)?|regression|broken|"
    r"wrong|incorrect|null ?ref(erence)?|race condition|leak(s|ing)?|typo)\b",
    re.IGNORECASE,
)
_REVERT_SUBJECT = re.compile(r'^revert\b', re.IGNORECASE)
_REVERTS_SHA = re.compile(r"This reverts commit ([0-9a-f]{7,40})", re.IGNORECASE)
_CONVENTIONAL = re.compile(r"^([a-z]+)(\([^)]*\))?!?:", re.IGNORECASE)
_KIND_WORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("refactor", re.compile(r"\b(refactor|cleanup|clean up|rename|restructure)\b", re.IGNORECASE)),
    ("test", re.compile(r"\btests?\b", re.IGNORECASE)),
    ("docs", re.compile(r"\b(docs?|documentation|readme)\b", re.IGNORECASE)),
    ("feat", re.compile(r"\b(add(s|ed)?|implement(s|ed)?|introduce|support|feature)\b", re.IGNORECASE)),
    ("chore", re.compile(r"\b(bump|update|upgrade|chore|release|version)\b", re.IGNORECASE)),
)


@dataclass
class Commit:
    sha: str
    parents: list[str]
    date: str
    author: str
    subject: str
    body: str = ""
    files: list[str] = field(default_factory=list)
    kind: str = "other"
    reverts: str | None = None

    def to_dict(self) -> dict:
        return {"sha": self.sha, "parents": self.parents, "date": self.date,
                "author": self.author, "subject": self.subject, "kind": self.kind,
                "reverts": self.reverts, "files": self.files}


@dataclass
class Hunk:
    file: str
    old_start: int
    new_start: int
    removed: list[tuple[int, str]] = field(default_factory=list)  # (old line no, text)
    added: list[tuple[int, str]] = field(default_factory=list)  # (new line no, text)
    header: str = ""

    def to_dict(self) -> dict:
        return {"file": self.file, "old_start": self.old_start, "new_start": self.new_start,
                "header": self.header,
                "removed": [[n, t] for n, t in self.removed],
                "added": [[n, t] for n, t in self.added]}


class GitError(RuntimeError):
    pass


def run_git(repo: str, args: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", repo, *args], capture_output=True, check=False)
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args[:3])}: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return proc.stdout.decode("utf-8", errors="replace")


def pathspecs(include: tuple[str, ...] | list[str] = DEFAULT_INCLUDE,
              exclude: tuple[str, ...] | list[str] = DEFAULT_EXCLUDE) -> list[str]:
    """Git pathspecs: every include anywhere in the tree, minus the excludes."""
    specs = [f":(glob)**/{p}" if "/" not in p else f":(glob){p}" for p in include]
    specs += [f":(glob,exclude){p}" for p in exclude]
    return specs


def classify(subject: str, body: str = "") -> tuple[str, str | None]:
    """(kind, reverted sha). Kind is fix | revert | merge | feat | refactor | test | docs |
    chore | other. Conventional-commit prefixes win; otherwise keywords decide."""
    reverted = None
    m = _REVERTS_SHA.search(body or "")
    if m:
        reverted = m.group(1)
    if _REVERT_SUBJECT.match(subject) or reverted:
        return "revert", reverted
    if subject.lower().startswith("merge "):
        return "merge", None
    if _FIX_CONVENTIONAL.match(subject):
        return "fix", None
    conv = _CONVENTIONAL.match(subject)
    if conv:
        t = conv.group(1).lower()
        aliases = {"feature": "feat", "enhance": "feat", "perf": "refactor", "style": "refactor",
                   "build": "chore", "ci": "chore", "deps": "chore"}
        t = aliases.get(t, t)
        if t in {"feat", "refactor", "test", "docs", "chore"}:
            # A conventional "feat:" that says "fix" in its subject is still a feature.
            return t, None
    if _FIX_WORDS.search(subject):
        return "fix", None
    for kind, rx in _KIND_WORDS:
        if rx.search(subject):
            return kind, None
    return "other", None


def list_commits(repo: str, rev: str = "HEAD", specs: list[str] | None = None,
                 no_merges: bool = True) -> list[Commit]:
    """Commits reachable from ``rev`` touching ``specs``, oldest first."""
    fmt = "%x1e%H%x1f%P%x1f%aI%x1f%an%x1f%s%x1f%b%x1f"
    args = ["log", "--reverse", f"--format={fmt}", "--name-only", "--no-renames"]
    if no_merges:
        args.append("--no-merges")
    args.append(rev)
    if specs:
        args += ["--", *specs]
    out = run_git(repo, args)
    commits: list[Commit] = []
    for rec in out.split("\x1e"):
        if not rec.strip():
            continue
        parts = rec.split("\x1f")
        if len(parts) < 7:
            continue
        sha, parents, date, author, subject, body, tail = parts[:7]
        files = [f for f in tail.strip().splitlines() if f.strip()]
        kind, reverted = classify(subject, body)
        commits.append(Commit(sha=sha, parents=parents.split(), date=date, author=author,
                              subject=subject, body=body.strip()[:2000], files=files,
                              kind=kind, reverts=reverted))
    return commits


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def parse_patch(patch: str) -> list[Hunk]:
    """Unified diff -> hunks with old/new line numbers per removed/added line."""
    hunks: list[Hunk] = []
    cur: Hunk | None = None
    path: str | None = None
    old_no = new_no = 0
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            cur, path = None, None
            continue
        if line.startswith("+++ "):
            target = line[4:].strip()
            path = None if target == "/dev/null" else re.sub(r"^b/", "", target)
            continue
        if line.startswith("--- "):
            source = line[4:].strip()
            if source != "/dev/null":
                path = re.sub(r"^a/", "", source)
            continue
        m = _HUNK_RE.match(line)
        if m and path is not None:
            old_no, new_no = int(m.group(1)), int(m.group(3))
            cur = Hunk(file=path, old_start=old_no, new_start=new_no, header=m.group(5).strip())
            hunks.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("-"):
            cur.removed.append((old_no, line[1:]))
            old_no += 1
        elif line.startswith("+"):
            cur.added.append((new_no, line[1:]))
            new_no += 1
        elif line.startswith("\\"):
            continue
        else:
            old_no += 1
            new_no += 1
    return hunks


def commit_hunks(repo: str, sha: str, specs: list[str] | None = None,
                 context: int = 0) -> list[Hunk]:
    args = ["show", "--format=", f"--unified={context}", "--no-color", "--no-renames",
            "--no-ext-diff", sha]
    if specs:
        args += ["--", *specs]
    return parse_patch(run_git(repo, args, check=False))


def blame_lines(repo: str, rev: str, path: str,
                lines: list[int]) -> dict[int, tuple[str, int, str]]:
    """line number at ``rev`` -> (sha that introduced it, its line number in that commit,
    its path in that commit). ``-w``: whitespace-only changes do not count."""
    if not lines:
        return {}
    ranges: list[tuple[int, int]] = []
    for n in sorted(set(lines)):
        if ranges and n == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], n)
        else:
            ranges.append((n, n))
    args = ["blame", "-w", "--porcelain"]
    for a, b in ranges:
        args += ["-L", f"{a},{b}"]
    args += [rev, "--", path]
    out = run_git(repo, args, check=False)
    result: dict[int, tuple[str, int, str]] = {}
    filenames: dict[str, str] = {}
    pending: list[tuple[str, int, int]] = []
    for line in out.splitlines():
        parts = line.split(" ")
        if (len(parts) >= 3 and len(parts[0]) == 40
                and all(c in "0123456789abcdef" for c in parts[0])):
            try:
                pending.append((parts[0], int(parts[1]), int(parts[2])))
            except ValueError:
                continue
        elif line.startswith("filename ") and pending:
            filenames[pending[-1][0]] = line[len("filename "):]
    for sha, orig, final in pending:
        result[final] = (sha, orig, filenames.get(sha, path))
    return result


def read_blob(repo: str, rev: str, path: str) -> str | None:
    proc = subprocess.run(["git", "-C", repo, "show", f"{rev}:{path}"], capture_output=True,
                          check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def glob_regex(pattern: str) -> re.Pattern[str]:
    """Glob -> regex over a repo-relative POSIX path. ``**/`` spans directories (or none),
    ``*`` stays inside one path segment. A pattern without ``/`` matches the basename."""
    if "/" not in pattern:
        pattern = "**/" + pattern
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + r"\Z")


def path_filter(include: tuple[str, ...] | list[str] = DEFAULT_INCLUDE,
                exclude: tuple[str, ...] | list[str] = DEFAULT_EXCLUDE):
    inc = [glob_regex(p) for p in include]
    exc = [glob_regex(p) for p in exclude]

    def selected(path: str) -> bool:
        return any(r.match(path) for r in inc) and not any(r.match(path) for r in exc)
    return selected


def ls_files(repo: str, rev: str) -> list[str]:
    """Regular files at ``rev`` (symlinks and submodules left out: their blobs are paths)."""
    files = []
    for line in run_git(repo, ["ls-tree", "-r", rev]).splitlines():
        meta, _, path = line.partition("\t")
        mode, kind = meta.split(" ")[:2]
        if kind == "blob" and mode in ("100644", "100755") and path:
            files.append(path)
    return files


def read_blobs(repo: str, rev: str, paths: list[str]):
    """Yield (path, text or None) for each path at ``rev`` through one ``git cat-file``."""
    proc = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert proc.stdin is not None and proc.stdout is not None
    try:
        for path in paths:
            proc.stdin.write(f"{rev}:{path}\n".encode())
            proc.stdin.flush()
            header = proc.stdout.readline().decode("utf-8", "replace").split()
            if len(header) < 3 or header[1] == "missing":
                yield path, None
                continue
            size = int(header[2])
            data = proc.stdout.read(size)
            proc.stdout.read(1)  # trailing newline
            yield path, data.decode("utf-8", errors="replace")
    finally:
        proc.stdin.close()
        proc.wait()


def rev_parse(repo: str, rev: str) -> str:
    return run_git(repo, ["rev-parse", "--verify", f"{rev}^{{commit}}"]).strip()
