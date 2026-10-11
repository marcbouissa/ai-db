"""User feedback that ai-db learns from: "the bench back is floating", "the clothes are too stiff".

Each piece of feedback becomes a *lesson*: the objects it is about (categories), the kind of
problem (an issue key below) and, when one exists, the automatic check that catches it. Later
builds of scenes holding those objects run the lesson: an automatic check is made blocking for
them, and a problem no check can see is put in front of the agent as an explicit "verify this"
item (in get_scene_spec_schema before it writes the spec, and in build_scene's critique).

Lessons can be listed, edited, disabled, re-enabled and deleted. ``optimize`` reviews them:
duplicates to merge, mistyped categories (closest known one), and outdated ones (their category
no longer exists, or they have not matched anything in many builds).

Scope: a lesson is ``global`` (every project) unless recorded with ``project``; a build in
project X checks the global lessons and X's own, never another project's (the same rule as
ai-db's skills). The scene tools take ``project`` the same way (default "default").

Stored as JSON next to the scene memory (``AI_DB_SCENE_LESSONS`` overrides the path).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import threading
import time
import uuid
from typing import Any

from ai_db_scene.constants import SCENE_MEMORY_PATH

#: issue key -> (words that name it, validator checks that see it, hint for the agent)
ISSUES: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    "floating": (("floating", "float", "hovering", "hover", "detached", "not touching", "gap", "loose"),
                 ("floating",), "every part must touch what holds it; check the named parts' contact"),
    "sunk": (("sunk", "sinking", "buried", "through the floor", "below the floor"), ("sunk",),
             "the object must stand on the floor, not in it"),
    "intersecting": (("intersect", "intersecting", "clipping", "clips", "overlap", "overlapping", "inside each other"),
                     ("interpenetration", "collision", "mesh_sanity"), "objects must not pass through each other"),
    "scale": (("too big", "too small", "huge", "tiny", "oversized", "undersized", "proportion", "scale"),
              ("scale",), "compare its size with the category's typical size"),
    "facing": (("facing", "backwards", "turned away", "wrong way", "faces the wall"), ("facing", "faces_wall"),
               "check which way it faces"),
    "stiff": (("stiff", "rigid", "not draping", "doesn't drape", "does not drape", "hard looking", "like cardboard"),
              (), "soft goods must drape: cloth simulation or more subdivisions, folds, no straight edges"),
    "material": (("fake", "plastic look", "flat colour", "flat color", "untextured", "no texture", "cg look"),
                 ("materials",), "give it a material kind with variation, not a flat colour"),
    "identical": (("identical", "all the same", "repetitive", "same every time", "look the same"), (),
                  "vary repeated objects (seed, preset, material)"),
    "lighting": (("too dark", "black render", "too bright", "blown out", "washed out"), ("lighting",),
                 "check exposure and lights"),
    "camera": (("can't see", "cannot see", "not visible", "camera"), ("camera",), "the camera must show it"),
}
SEVERITIES = ("warning", "blocking")
OUTDATED_AFTER_BUILDS = 50  # a lesson that matched nothing in this many builds is flagged as stale
SIMILAR = 0.82  # text similarity above which two lessons with the same issue are duplicates

#: words people use for a category (the category's own name, its plural and spaced form match too)
ALIASES = {"couch": "sofa", "settee": "sofa", "loveseat": "sofa", "telly": "tv", "television": "tv",
           "carpet": "rug", "lamp": "floor_lamp", "bookcase": "bookshelf", "shelf": "shelving_unit",
           "fridge": "refrigerator", "tub": "bathtub", "cupboard": "cabinet", "chest of drawers": "dresser",
           "plant": "potted_plant", "board": "whiteboard"}

_LOCK = threading.Lock()


def lessons_path() -> str:
    env = os.environ.get("AI_DB_SCENE_LESSONS")
    if env:
        return os.path.expanduser(env)
    mem = os.path.expanduser(os.environ.get("AI_DB_SCENE_MEMORY") or SCENE_MEMORY_PATH)
    return os.path.join(os.path.dirname(mem) or ".", "scene_lessons.json")


def _load() -> list[dict[str, Any]]:
    try:
        with open(lessons_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save(lessons: list[dict[str, Any]]) -> None:
    path = lessons_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(lessons, f, indent=1)
    os.replace(tmp, path)


def known_categories() -> set[str]:
    from ai_db_scene.priors import load_priors

    return set(load_priors())


def parse(text: str, categories: set[str] | None = None) -> dict[str, Any]:
    """Categories, issue and unmatched subject words of a free-text complaint."""
    cats = categories if categories is not None else known_categories()
    low = " " + re.sub(r"[^a-z0-9 ]+", " ", text.lower()) + " "
    found: list[str] = []
    for c in sorted(cats, key=len, reverse=True):  # "coffee table" before "table"
        for form in {c, c.replace("_", " "), c.replace("_", " ") + "s", c + "s"}:
            if f" {form} " in low and c not in found:
                found.append(c)
    for word, c in ALIASES.items():
        if re.search(rf"\b{re.escape(word)}s?\b", low) and c in cats and c not in found:
            found.append(c)
    issue = next((k for k, (words, _, _) in ISSUES.items() if any(w in low for w in words)), "other")
    return {"categories": found, "issue": issue}


GLOBAL = "global"


def _in_scope(lesson: dict[str, Any], project: str | None) -> bool:
    scope = lesson.get("project", GLOBAL)
    return scope == GLOBAL or project is None or scope == project


def add(text: str, categories: list[str] | None = None, issue: str | None = None,
        severity: str = "blocking", subjects: list[str] | None = None,
        project: str = GLOBAL) -> dict[str, Any]:
    """Store a lesson (or strengthen an equal one: times_reported + 1). Explicit categories/issue
    override what the text parse finds."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text: describe the problem, e.g. 'the bench back is floating'")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {list(SEVERITIES)}")
    if issue is not None and issue not in ISSUES and issue != "other":
        raise ValueError(f"issue must be one of {sorted(ISSUES)} or 'other'")
    p = parse(text)
    cats = list(categories) if categories else p["categories"]
    lesson = {"id": uuid.uuid4().hex[:8], "text": text.strip()[:500], "categories": cats,
              "subjects": list(subjects or []), "issue": issue or p["issue"], "severity": severity,
              "project": project or GLOBAL,
              "enabled": True, "created": time.time(), "updated": time.time(), "times_reported": 1,
              "times_triggered": 0, "builds_seen": 0, "last_triggered": None}
    with _LOCK:
        lessons = _load()
        for old in lessons:
            if old["issue"] == lesson["issue"] and set(old["categories"]) == set(cats) and \
                    old.get("project", GLOBAL) == lesson["project"] and \
                    _similar(str(old["text"]), str(lesson["text"])) >= SIMILAR:
                old["times_reported"] += 1
                old["updated"] = time.time()
                old["enabled"] = True
                _save(lessons)
                return {"status": "reinforced", "lesson": old}
        lessons.append(lesson)
        _save(lessons)
    out: dict[str, Any] = {"status": "added", "lesson": lesson}
    if not cats:
        out["note"] = ("no known object category found in the text: the lesson applies to every scene; "
                       "pass categories=[...] to scope it")
    return out


def list_lessons(include_disabled: bool = True, project: str | None = None) -> list[dict[str, Any]]:
    """All lessons (project None), or the ones a build in ``project`` uses (global + its own)."""
    return [x for x in _load() if (include_disabled or x.get("enabled")) and _in_scope(x, project)]


def update(lesson_id: str, changes: dict[str, Any]) -> dict[str, Any]:
    allowed = {"text", "categories", "subjects", "issue", "severity", "enabled", "project"}
    bad = sorted(set(changes) - allowed)
    if bad:
        raise ValueError(f"cannot change {bad}; allowed {sorted(allowed)}")
    if "severity" in changes and changes["severity"] not in SEVERITIES:
        raise ValueError(f"severity must be one of {list(SEVERITIES)}")
    if "issue" in changes and changes["issue"] not in ISSUES and changes["issue"] != "other":
        raise ValueError(f"issue must be one of {sorted(ISSUES)} or 'other'")
    with _LOCK:
        lessons = _load()
        for x in lessons:
            if x["id"] == lesson_id:
                x.update(changes)
                x["updated"] = time.time()
                _save(lessons)
                return x
    raise ValueError(f"no lesson {lesson_id!r}")


def delete(lesson_id: str) -> bool:
    with _LOCK:
        lessons = _load()
        keep = [x for x in lessons if x["id"] != lesson_id]
        if len(keep) == len(lessons):
            return False
        _save(keep)
        return True


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def relevant(categories: set[str], brief: str = "", project: str | None = None) -> list[dict[str, Any]]:
    """Enabled lessons about any of ``categories`` (or naming a subject word in the brief), plus
    unscoped ones."""
    low = brief.lower()
    return [x for x in _load() if x.get("enabled") and _in_scope(x, project) and (
        not x["categories"] and not x.get("subjects") or set(x["categories"]) & categories
        or any(s.lower() in low for s in x.get("subjects") or []))]


def checklist(categories: set[str], brief: str = "", project: str | None = None) -> list[str]:
    """What the agent must verify for this scene, one line per lesson (for get_scene_spec_schema)."""
    return [f"[{x['severity']}] {', '.join(x['categories']) or 'any object'}: {x['text']} "
            f"-> {ISSUES.get(x['issue'], ((), (), 'verify it before finishing'))[2]}"
            for x in relevant(categories, brief, project)]


def apply(issues: list[dict[str, Any]], placements: list[Any], project: str | None = None) -> list[dict[str, Any]]:
    """Run the lessons on a finished build: escalate validator issues a lesson names, add a
    floating-parts check for recipe objects, and a 'verify' item for what no check can see.
    Returns the lesson issues to append; updates trigger counts."""
    by_obj = {pl.id: pl.category for pl in placements}
    present = set(by_obj.values())
    with _LOCK:
        lessons = _load()
        hits = [x for x in lessons if x.get("enabled") and _in_scope(x, project)
                and (not x["categories"] or set(x["categories"]) & present)]
        for x in lessons:
            if x.get("enabled") and _in_scope(x, project):
                x["builds_seen"] = x.get("builds_seen", 0) + 1
        added: list[dict[str, Any]] = []
        for x in hits:
            checks = ISSUES.get(x["issue"], ((), (), ""))[1]
            scoped = set(x["categories"]) or present
            triggered = False
            for i in issues:
                if i.get("check") in checks and _issue_categories(i, by_obj) & (scoped | {""}):
                    i["severity"] = x["severity"] if x["severity"] == "blocking" else i.get("severity", "warning")
                    i["lesson"] = x["id"]
                    triggered = True
            if x["issue"] == "floating":
                for cat in sorted(scoped & present):
                    loose = _floating_recipe_parts(cat, placements)
                    if loose:
                        triggered = True
                        added.append({"check": "lesson_floating", "severity": x["severity"], "lesson": x["id"],
                                      "message": f"{cat}: parts {loose} do not touch their support "
                                                 f"(user reported: {x['text']})"})
            if not checks and x["issue"] != "floating":
                objs = sorted(o for o, c in by_obj.items() if c in scoped) or ["the scene"]
                added.append({"check": "lesson_verify", "severity": "warning", "lesson": x["id"],
                              "message": f"user reported: {x['text']} - verify {', '.join(objs[:6])}: "
                                         f"{ISSUES.get(x['issue'], ((), (), 'look for it'))[2]}"})
            if triggered:
                x["times_triggered"] = x.get("times_triggered", 0) + 1
                x["last_triggered"] = time.time()
        _save(lessons)
    return added


def _issue_categories(issue: dict[str, Any], by_obj: dict[str, str]) -> set[str]:
    """Categories of the objects an issue names (parts are named after their asset root);
    {""} when it names none (a scene-wide issue)."""
    names = issue.get("objects") or ([issue["object"]] if issue.get("object") else [])
    cats = {c for n in names for o, c in by_obj.items() if str(n) == o or str(n).startswith(o + "_")
            or str(n).startswith(o + ".")}
    return cats or {""}


def _floating_recipe_parts(category: str, placements: list[Any]) -> list[str]:
    from ai_db_scene.recipes import has_recipe, recipe_spec
    from ai_db_scene.user_recipes import floating_parts

    if not has_recipe(category):
        return []
    pl = next(p for p in placements if p.category == category)
    return floating_parts(recipe_spec(category, pl.id, pl.dims)["parts"])


def optimize(apply_changes: bool = False) -> dict[str, Any]:
    """Review the lessons: duplicates (same issue and categories, similar text) to merge,
    mistyped categories (closest known category), unknown issues, and outdated lessons (category
    gone, or nothing matched in OUTDATED_AFTER_BUILDS builds). With apply_changes the proposals
    are carried out (merge: keep the oldest, add up reports; fix: replace the category;
    outdated: disable, never delete)."""
    cats = known_categories()
    with _LOCK:
        lessons = _load()
        proposals: list[dict[str, Any]] = []
        merged: set[str] = set()
        for i, a in enumerate(lessons):
            if a["id"] in merged:
                continue
            for b in lessons[i + 1:]:
                if b["id"] in merged or a["issue"] != b["issue"] or \
                        a.get("project", GLOBAL) != b.get("project", GLOBAL):
                    continue
                same_scope = set(a["categories"]) == set(b["categories"])
                if same_scope and _similar(str(a["text"]), str(b["text"])) >= 0.6 or \
                        same_scope and a["issue"] != "other":
                    proposals.append({"action": "merge", "keep": a["id"], "drop": b["id"],
                                      "why": f"same issue '{a['issue']}' on {a['categories'] or 'any object'}"})
                    merged.add(b["id"])
            bad = [c for c in a["categories"] if c not in cats]
            for c in bad:
                close = difflib.get_close_matches(c, sorted(cats), n=1, cutoff=0.6)
                proposals.append({"action": "fix_category" if close else "outdated", "lesson": a["id"],
                                  "category": c, "to": close[0] if close else None,
                                  "why": f"'{c}' is not a known category" + (f"; did you mean '{close[0]}'?"
                                                                              if close else "")})
            if a["issue"] not in ISSUES and a["issue"] != "other":
                proposals.append({"action": "fix_issue", "lesson": a["id"], "to": parse(a["text"], cats)["issue"],
                                  "why": f"unknown issue '{a['issue']}'"})
            if a.get("enabled") and a.get("builds_seen", 0) >= OUTDATED_AFTER_BUILDS and not a.get("times_triggered"):
                proposals.append({"action": "outdated", "lesson": a["id"],
                                  "why": f"matched nothing in {a['builds_seen']} builds (likely fixed)"})
        if apply_changes and proposals:
            idx = {x["id"]: x for x in lessons}
            for p in proposals:
                if p["action"] == "merge" and p["keep"] in idx and p["drop"] in idx:
                    idx[p["keep"]]["times_reported"] += idx[p["drop"]].get("times_reported", 1)
                    idx[p["keep"]]["times_triggered"] += idx[p["drop"]].get("times_triggered", 0)
                    if idx[p["drop"]]["severity"] == "blocking":
                        idx[p["keep"]]["severity"] = "blocking"
                    del idx[p["drop"]]
                elif p["action"] == "fix_category" and p["lesson"] in idx:
                    x = idx[p["lesson"]]
                    x["categories"] = [p["to"] if c == p["category"] else c for c in x["categories"]]
                elif p["action"] == "fix_issue" and p["lesson"] in idx:
                    idx[p["lesson"]]["issue"] = p["to"]
                elif p["action"] == "outdated" and p["lesson"] in idx:
                    idx[p["lesson"]]["enabled"] = False
            lessons = list(idx.values())
            _save(lessons)
    return {"proposals": proposals, "applied": bool(apply_changes and proposals), "lessons": len(lessons)}
