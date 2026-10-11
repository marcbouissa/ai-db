"""Scene material for ``draft_skill`` (core ``ai_db.skill_drafts`` entry point).

Run records are the scene plugin's experience: specs that built clean on the first try
become Workflow lines, the validator checks and repairs that fire most often become
Pitfalls. Deterministic: counting and formatting, no model call.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

#: runs read from the store per draft
DRAFT_RUN_LIMIT = 500
DRAFT_TOP_PATTERNS = 5
DRAFT_TOP_ERRORS = 6
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if len(t) > 2}


def _run_dict(run: Any) -> dict[str, Any]:
    return run.to_dict() if hasattr(run, "to_dict") else dict(run)


def _run_words(run: dict[str, Any]) -> set[str]:
    spec = run.get("spec") or {}
    words = _tokens(str(run.get("brief") or ""))
    setting = spec.get("setting") or {}
    words |= _tokens(f"{setting.get('kind', '')} {setting.get('style') or ''}")
    for o in spec.get("objects") or []:
        if isinstance(o, dict):
            words |= _tokens(str(o.get("category", "")).replace("_", " ")) | {str(o.get("category", ""))}
    return words


def _first_try(run: dict[str, Any]) -> bool:
    if run.get("outcome") != "success":
        return False
    reports = run.get("validation_reports") or []
    first = reports[0] if reports else {}
    return isinstance(first, dict) and not first.get("issues_found") and not first.get("repairs")


def _pattern(spec: dict[str, Any]) -> str:
    setting = spec.get("setting") or {}
    cats = Counter(str(o.get("category")) for o in spec.get("objects") or [] if isinstance(o, dict))
    rels = Counter(str(r.get("rel")) for r in spec.get("relations") or [] if isinstance(r, dict))
    objs = ", ".join(f"{c} x{n}" if n > 1 else c for c, n in cats.most_common(6))
    rel = ", ".join(r for r, _ in rels.most_common(4))
    return f"{setting.get('kind', '?')}: {objs or 'no objects'}" + (f"; relations {rel}" if rel else "")


def _errors(run: dict[str, Any]) -> list[str]:
    out = []
    for rep in run.get("validation_reports") or []:
        if not isinstance(rep, dict):
            continue
        for issue in rep.get("issues_left") or []:
            if isinstance(issue, dict) and issue.get("check"):
                out.append(f"validator: {issue['check']}")
        for r in rep.get("repairs") or []:
            out.append(f"solver repair: {str(r).split(':')[0].strip()[:60]}")
    return out


def summarize_runs(runs: list[Any], topic: str) -> dict[str, list[str]]:
    """Workflow lines from first-try specs and Pitfalls from frequent errors of the runs
    related to ``topic`` (brief, setting or object-category overlap)."""
    q = _tokens(topic.replace("_", " ")) | {topic.strip().lower()}
    related = [d for d in map(_run_dict, runs) if q & _run_words(d)]
    if not related:
        return {}
    clean = [d for d in related if _first_try(d)]
    patterns = Counter(_pattern(d.get("spec") or {}) for d in clean)
    errors = Counter(e for d in related for e in _errors(d))
    workflow = [f"Spec pattern that built clean on the first try ({n} run(s)): {p}"
                for p, n in patterns.most_common(DRAFT_TOP_PATTERNS)]
    pitfalls = [f"{e} - seen {n} time(s) in {len(related)} related run(s)"
                for e, n in errors.most_common(DRAFT_TOP_ERRORS)]
    examples = [f"Brief that worked first time: {str(d.get('brief'))[:160]}"
                for d in clean[:2] if d.get("brief")]
    out: dict[str, list[str]] = {"workflow": workflow, "pitfalls": pitfalls, "examples": examples,
                                 "sources": [f"run:{d.get('run_id')}" for d in clean[:DRAFT_TOP_PATTERNS]]}
    return {k: v for k, v in out.items() if v}


def contribute(topic: str, config: Any = None) -> dict[str, list[str]]:
    """``ai_db.skill_drafts`` hook: summarise this machine's scene run records."""
    from ai_db_scene.tools.scene_tools import get_scene_memory

    runs = get_scene_memory().backend.list_runs(limit=DRAFT_RUN_LIMIT)
    return summarize_runs(runs, topic)
