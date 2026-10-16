"""Naming and structure conventions of a codebase, measured with ai-db's tree-sitter parser.

Reads files straight from a git revision (``git show rev:path``), so the conventions can
be measured as of a past cutoff without checking anything out, and nothing is indexed.

For each (language, symbol kind) it counts the identifier styles (PascalCase, camelCase,
snake_case, ...) and a few language-specific checks (C# interfaces start with ``I``, C#
async methods end in ``Async``, Python test functions start with ``test_``). A convention
with high adherence over many symbols is a candidate review rule; the report keeps
examples and counter-examples (file:line) as evidence. Syntax errors found while parsing
are reported per file as lint findings.
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from ai_db.history.gitlog import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    ls_files,
    path_filter,
    read_blobs,
    rev_parse,
)
from ai_db.parser.languages import language_for

STYLES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("UPPER_CASE", re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)+$|^[A-Z]{2,}[0-9]*$")),
    ("PascalCase", re.compile(r"^[A-Z][a-zA-Z0-9]*$")),
    ("camelCase", re.compile(r"^[a-z][a-zA-Z0-9]*$")),
    ("snake_case", re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")),
    ("_camelCase", re.compile(r"^_[a-z][a-zA-Z0-9]*$")),
    ("_snake_case", re.compile(r"^_[a-z][a-z0-9_]*$")),
    ("dunder", re.compile(r"^__\w+__$")),
)


def name_style(name: str) -> str:
    # camelCase and snake_case overlap on a single lowercase word ("run"): call it "lower".
    if re.fullmatch(r"[a-z][a-z0-9]*", name):
        return "lower"
    if re.fullmatch(r"_[a-z][a-z0-9]*", name):
        return "_lower"
    for style, rx in STYLES:
        if rx.match(name):
            return style
    return "other"


@dataclass
class ConventionOptions:
    rev: str = "HEAD"
    include: tuple[str, ...] = DEFAULT_INCLUDE
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE
    max_files: int = 4000
    max_bytes: int = 300_000
    examples: int = 3


def _compatible(style: str, dominant: str) -> bool:
    # A one-word lowercase name fits camelCase and snake_case alike.
    return (style == dominant
            or (style == "lower" and dominant in ("camelCase", "snake_case"))
            or (style == "_lower" and dominant in ("_camelCase", "_snake_case")))


def conventions(repo: str, opts: ConventionOptions | None = None) -> dict:
    from ai_db.parser.ts_graph import extract_graph

    opts = opts or ConventionOptions()
    started = time.time()
    rev = rev_parse(repo, opts.rev)
    selected = path_filter(opts.include, opts.exclude)
    files = [f for f in ls_files(repo, rev) if selected(f) and language_for(f)]
    files = files[:opts.max_files]

    styles: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    samples: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    checks: dict[str, Counter[str]] = defaultdict(Counter)
    check_samples: dict[tuple[str, str], list[str]] = defaultdict(list)
    syntax: list[dict] = []
    per_lang_files: Counter[str] = Counter()
    parsed = 0

    def note(check: str, ok: bool, where: str) -> None:
        key = "ok" if ok else "violations"
        checks[check][key] += 1
        if len(check_samples[(check, key)]) < opts.examples:
            check_samples[(check, key)].append(where)

    for path, content in read_blobs(repo, rev, files):
        if content is None or len(content) > opts.max_bytes or "\0" in content:
            continue
        lang = language_for(path) or "?"
        try:
            symbols, _, errors = extract_graph(path, content)
        except Exception:  # noqa: BLE001 - a grammar failure must not stop the survey
            syntax.append({"file": path, "errors": -1, "first": {"line": 0,
                                                                 "message": "parser failed"}})
            continue
        parsed += 1
        per_lang_files[lang] += 1
        if errors:
            syntax.append({"file": path, "errors": len(errors),
                           "first": {"line": errors[0][0], "message": errors[0][2][:120]}})
        for s in symbols:
            if s.get("local"):
                continue
            name, kind = str(s.get("name", "")), str(s.get("symbol_type", ""))
            if not name or "." in name:
                continue
            where = f"{path}:{s.get('line')}"
            st = name_style(name)
            styles[(lang, kind)][st] += 1
            if len(samples[(lang, kind, st)]) < opts.examples:
                samples[(lang, kind, st)].append(f"{where} {name}")
            sig = str(s.get("signature") or "")
            if lang == "csharp" and kind == "interface":
                note("csharp.interface_prefix_I", bool(re.match(r"^I[A-Z]", name)),
                     f"{where} {name}")
            if lang == "csharp" and kind == "method" and re.search(r"\basync\b", sig) \
                    and re.search(r"\b(Task|ValueTask)\b", sig):
                note("csharp.async_suffix", name.endswith("Async") or name == "Main",
                     f"{where} {name}")
            if lang == "csharp" and kind == "method" and re.search(r"\basync\s+void\b", sig):
                note("csharp.no_async_void", False, f"{where} {name}")
            if lang == "python" and kind == "function" and "/test" in f"/{path}" \
                    and path.rsplit("/", 1)[-1].startswith("test_") \
                    and not name.startswith("_"):
                note("python.test_prefix", name.startswith("test_") or
                     name in ("setUp", "tearDown", "setup_method", "teardown_method"),
                     f"{where} {name}")

    conv: list[dict] = []
    for (lang, kind), counter in styles.items():
        total = sum(counter.values())
        if total == 0:
            continue
        dominant, _ = max(((s, n) for s, n in counter.items() if s not in ("lower", "_lower")),
                          key=lambda x: x[1], default=("lower", 0))
        ok = sum(n for s, n in counter.items() if _compatible(s, dominant))
        conv.append({
            "language": lang, "kind": kind, "total": total, "dominant": dominant,
            "adherence": round(ok / total, 4),
            "styles": dict(counter.most_common()),
            "examples": samples[(lang, kind, dominant)],
            "counter_examples": [e for s in counter if not _compatible(s, dominant)
                                 for e in samples[(lang, kind, s)]][:opts.examples],
        })
    conv.sort(key=lambda c: -int(c["total"]))

    check_list = []
    for name, counter in checks.items():
        total = counter["ok"] + counter["violations"]
        check_list.append({"check": name, "total": total, "ok": counter["ok"],
                           "violations": counter["violations"],
                           "adherence": round(counter["ok"] / total, 4) if total else 0.0,
                           "examples": check_samples[(name, "ok")],
                           "counter_examples": check_samples[(name, "violations")]})

    return {
        "tool": "ai-db history conventions",
        "repo": repo,
        "rev": rev,
        "files_considered": len(files),
        "files_parsed": parsed,
        "languages": dict(per_lang_files),
        "naming": conv,
        "checks": check_list,
        "syntax_errors": syntax[:200],
        "elapsed_s": round(time.time() - started, 1),
    }
