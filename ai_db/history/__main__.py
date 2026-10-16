"""``python -m ai_db.history {mine,conventions} REPO ...`` -- JSON reports on stdout or --out.

Standalone entry point so the miner can be used before it is wired into ``ai-db``'s CLI,
dispatcher and MCP server (see ``ai_db/history/__init__.py`` for the wiring notes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from ai_db.history.conventions import ConventionOptions, conventions
from ai_db.history.gitlog import DEFAULT_EXCLUDE, DEFAULT_INCLUDE
from ai_db.history.mine import MineOptions, mine


def _globs(values: list[str] | None, default: tuple[str, ...], extra: list[str] | None
           ) -> tuple[str, ...]:
    base = tuple(values) if values else default
    return base + tuple(extra or ())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ai_db.history",
                                description="Mine git history and code conventions (read-only)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("mine", "conventions"):
        sp = sub.add_parser(name)
        sp.add_argument("repo", help="Path to a git repository (never modified)")
        sp.add_argument("--rev", default="HEAD", help="Revision to read (default: HEAD)")
        sp.add_argument("--include", action="append",
                        help="Glob of files to include (repeatable; replaces the defaults)")
        sp.add_argument("--exclude", action="append",
                        help="Glob of files to exclude (repeatable; replaces the defaults)")
        sp.add_argument("--also-exclude", action="append",
                        help="Glob to exclude in addition to the defaults (repeatable)")
        sp.add_argument("--out", help="Write the JSON report here instead of stdout")
        if name == "mine":
            sp.add_argument("--cutoff", help="Last commit (sha/ref) of the train range")
            sp.add_argument("--cutoff-ratio", type=float, default=0.8,
                            help="Train fraction of code commits when --cutoff is not given")
            sp.add_argument("--no-szz", action="store_true",
                            help="Skip blaming the lines fix commits removed")
            sp.add_argument("--max-files", type=int, default=40,
                            help="Commits touching more code files are bulk (no hunks)")
        else:
            sp.add_argument("--max-files", type=int, default=4000,
                            help="Maximum files to parse")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = os.path.abspath(args.repo)
    include = _globs(args.include, DEFAULT_INCLUDE, None)
    exclude = _globs(args.exclude, DEFAULT_EXCLUDE, args.also_exclude)
    if args.cmd == "mine":
        report = mine(repo, MineOptions(rev=args.rev, cutoff=args.cutoff,
                                        cutoff_ratio=args.cutoff_ratio, include=include,
                                        exclude=exclude, szz=not args.no_szz,
                                        max_files=args.max_files))
    else:
        report = conventions(repo, ConventionOptions(rev=args.rev, include=include,
                                                     exclude=exclude, max_files=args.max_files))
    text = json.dumps(report, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out} ({len(text) // 1024} KiB)", file=sys.stderr)
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
