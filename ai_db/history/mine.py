"""Mine a repository's git history into a JSON report for rule generation and backtesting.

The report splits history (first-parent order is not assumed; ``git log --no-merges``,
oldest first) at a cutoff into a *train* range (to learn from) and a *test* range (to
backtest on). For each range it records:

- ``commits``: sha, date, subject, kind (fix/revert/feat/...), touched code files;
- ``hunks``: zero-context hunks per commit (removed/added lines with line numbers),
  skipping bulk commits (vendor drops, mass renames) that would drown the statistics;
- ``introduced``: SZZ-lite -- for each fix commit, the commits that introduced the lines
  it removed (``git blame -w`` at the fix's parent);
- ``cochange``: file pairs that change together, with support and confidence;
- ``hotspots``: files with the most fix commits;
- ``reverts``: revert commits and what they reverted.

Only read-only git commands are used; the repository is never modified.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from ai_db.history.gitlog import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    Commit,
    blame_lines,
    commit_hunks,
    list_commits,
    path_filter,
    pathspecs,
    rev_parse,
    run_git,
)

REPORT_VERSION = 1


@dataclass
class MineOptions:
    rev: str = "HEAD"
    cutoff: str | None = None  # sha/ref: last commit of the train range
    cutoff_ratio: float = 0.8  # used when ``cutoff`` is None
    include: tuple[str, ...] = DEFAULT_INCLUDE
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE
    max_files: int = 40  # commits touching more code files are "bulk": no hunks
    max_changed_lines: int = 2000
    max_line_chars: int = 240
    szz: bool = True
    szz_max_lines: int = 40  # removed lines blamed per fix commit
    cochange_min_support: int = 4
    cochange_min_confidence: float = 0.6
    top: int = 40


def _hunks_for(repo: str, commit: Commit, specs: list[str], opts: MineOptions,
               selected) -> list[dict]:
    hunks = [h for h in commit_hunks(repo, commit.sha, specs) if selected(h.file)]
    changed = sum(len(h.removed) + len(h.added) for h in hunks)
    if changed > opts.max_changed_lines:
        return []
    out = []
    for h in hunks:
        d = h.to_dict()
        d["removed"] = [[n, t[:opts.max_line_chars]] for n, t in h.removed]
        d["added"] = [[n, t[:opts.max_line_chars]] for n, t in h.added]
        out.append(d)
    return out


def _szz(repo: str, commit: Commit, hunks: list[dict], opts: MineOptions) -> list[dict]:
    """Which commits introduced the lines this fix removed."""
    if not commit.parents:
        return []
    parent = commit.parents[0]
    by_file: dict[str, list[tuple[int, str]]] = defaultdict(list)
    budget = opts.szz_max_lines
    for h in hunks:
        for n, text in h["removed"]:
            if budget <= 0:
                break
            if len(text.strip()) < 3:  # braces and blank lines blame to anything
                continue
            by_file[h["file"]].append((n, text))
            budget -= 1
    found = []
    for path, lines in by_file.items():
        blamed = blame_lines(repo, parent, path, [n for n, _ in lines])
        for n, text in lines:
            hit = blamed.get(n)
            if hit is None:
                continue
            sha, orig_line, orig_path = hit
            found.append({"fix": commit.sha, "intro": sha, "file": orig_path,
                          "line": orig_line, "fix_file": path, "fix_line": n, "text": text})
    return found


def _cochange(commits: list[Commit], opts: MineOptions) -> list[dict]:
    file_count: Counter[str] = Counter()
    pair_count: Counter[tuple[str, str]] = Counter()
    for c in commits:
        files = sorted(set(c.files))
        if not files or len(files) > 20:
            continue
        file_count.update(files)
        pair_count.update(itertools.combinations(files, 2))
    pairs: list[dict[str, Any]] = []
    for (a, b), n in pair_count.items():
        if n < opts.cochange_min_support:
            continue
        conf_ab = n / file_count[a]
        conf_ba = n / file_count[b]
        if max(conf_ab, conf_ba) < opts.cochange_min_confidence:
            continue
        pairs.append({"a": a, "b": b, "support": n, "conf_a_to_b": round(conf_ab, 3),
                      "conf_b_to_a": round(conf_ba, 3), "count_a": file_count[a],
                      "count_b": file_count[b]})
    pairs.sort(key=lambda p: (-int(p["support"]),
                              -max(float(p["conf_a_to_b"]), float(p["conf_b_to_a"]))))
    return pairs[:opts.top * 2]


def _hotspots(commits: list[Commit], opts: MineOptions) -> list[dict]:
    total: Counter[str] = Counter()
    fixes: Counter[str] = Counter()
    for c in commits:
        for f in set(c.files):
            total[f] += 1
            if c.kind == "fix":
                fixes[f] += 1
    ranked = sorted(((n, n / total[f], f) for f, n in fixes.items() if n >= 2),
                    key=lambda x: (-x[0], -x[1]))
    spots = [{"file": f, "fix_commits": n, "commits": total[f], "fix_ratio": round(r, 3)}
             for n, r, f in ranked]
    return spots[:opts.top]


def _reverts(repo: str, commits: list[Commit], by_sha: dict[str, Commit]) -> list[dict]:
    out = []
    for c in commits:
        if c.kind != "revert":
            continue
        target = None
        if c.reverts:
            target = next((x for s, x in by_sha.items() if s.startswith(c.reverts)), None)
        subject = target.subject if target else None
        if subject is None and c.reverts:
            subject = run_git(repo, ["log", "-1", "--format=%s", c.reverts],
                              check=False).strip() or None
        out.append({"sha": c.sha, "subject": c.subject, "reverted": c.reverts,
                    "reverted_subject": subject,
                    "files": c.files})
    return out


def _range_report(repo: str, commits: list[Commit], specs: list[str], opts: MineOptions,
                  selected, by_sha: dict[str, Commit], with_hunks: bool = True) -> dict:
    out_commits = []
    hunks: dict[str, list[dict]] = {}
    introduced: list[dict] = []
    for c in commits:
        d = c.to_dict()
        d["bulk"] = len(c.files) > opts.max_files
        out_commits.append(d)
        if not with_hunks or d["bulk"]:
            continue
        hs = _hunks_for(repo, c, specs, opts, selected)
        if not hs:
            continue
        hunks[c.sha] = hs
        if opts.szz and c.kind == "fix":
            introduced.extend(_szz(repo, c, hs, opts))
    kinds = Counter(c.kind for c in commits)
    return {
        "commits": out_commits,
        "hunks": hunks,
        "introduced": introduced,
        "cochange": _cochange(commits, opts),
        "hotspots": _hotspots(commits, opts),
        "reverts": _reverts(repo, commits, by_sha),
        "kinds": dict(kinds),
    }


def mine(repo: str, opts: MineOptions | None = None) -> dict:
    """Mine ``repo`` (read-only) and return the JSON-serialisable report."""
    opts = opts or MineOptions()
    started = time.time()
    head = rev_parse(repo, opts.rev)
    specs = pathspecs(opts.include, opts.exclude)
    selected = path_filter(opts.include, opts.exclude)
    commits = list_commits(repo, head, specs)
    for c in commits:
        c.files = [f for f in c.files if selected(f)]
    commits = [c for c in commits if c.files]
    by_sha = {c.sha: c for c in commits}

    if opts.cutoff:
        cut_sha = rev_parse(repo, opts.cutoff)
        idx = next((i for i, c in enumerate(commits) if c.sha == cut_sha), None)
        if idx is None:
            # The cutoff commit touched no code: split at the last code commit before it.
            ancestors = set(list_commits_shas(repo, cut_sha))
            idx = max((i for i, c in enumerate(commits) if c.sha in ancestors), default=-1)
        split = idx + 1
    else:
        split = round(len(commits) * opts.cutoff_ratio)
    train, test = commits[:split], commits[split:]

    report = {
        "version": REPORT_VERSION,
        "tool": "ai-db history mine",
        "repo": repo,
        "rev": head,
        "include": list(opts.include),
        "exclude": list(opts.exclude),
        "cutoff": {
            "sha": train[-1].sha if train else None,
            "date": train[-1].date if train else None,
            "train_commits": len(train),
            "test_commits": len(test),
        },
        "train": _range_report(repo, train, specs, opts, selected, by_sha),
        "test": _range_report(repo, test, specs, opts, selected, by_sha),
    }
    report["elapsed_s"] = round(time.time() - started, 1)
    return report


def list_commits_shas(repo: str, rev: str) -> list[str]:
    return run_git(repo, ["rev-list", rev]).split()
