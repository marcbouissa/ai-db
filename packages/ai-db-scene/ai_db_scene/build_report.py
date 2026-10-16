"""The build_scene / edit_scene result as a weak model reads it.

The Strata agent shows a tool result's first 2,500 characters (qwen3-14b-local-aidb5). The
classroom build's result opened with 27 created names and 28 material notes, each repeating the
whole material vocabulary: verdict, next and scene_id were cut off, so the model invented
``scene_001`` for edit_scene and rebuilt the scene seven times until its 32k context ran out.

``present_build`` puts what the model acts on first (verdict, next, status, scene_id), folds
repeated notes into one line per distinct coercion, shortens numbered name runs and drops the
echoed spec once the version is saved (edit_scene starts from the stored one).
"""

from __future__ import annotations

import re
from typing import Any

#: keys in the order a model should read them; the rest follows in its own order
_FIRST = ("verdict", "next", "status", "scene_id", "objects", "issues_left", "created", "repairs")
#: ``objects[3].material: <message>``, ``setting.wall_material: <message>``
_NOTE = re.compile(r"^([A-Za-z_]\w*(?:\[\d+\])?(?:\.[A-Za-z_]\w*)*): (.+)$", re.DOTALL)
_INDEXED = re.compile(r"^(\w+)\[(\d+)\](.*)$")
#: the vocabulary tail of a material that fell back to the category default
_KINDS = re.compile(r" \((kinds: \[[^\]]*\], e\.g\. [^)]*)\)")
_NUMBERED = re.compile(r"^(.*?)(\d+)$")
#: indices listed per folded note before "+N more"
_MAX_INDICES = 6
#: a created list longer than this has its numbered runs shortened
_NAMES_KEEP = 12


def _paths(paths: list[str]) -> str:
    """``objects[0].material``, ``objects[4].material`` -> ``objects[0,4].material``."""
    groups: dict[tuple[str, str], list[str]] = {}
    plain: list[str] = []
    for p in dict.fromkeys(paths):
        m = _INDEXED.match(p)
        if m:
            groups.setdefault((m.group(1), m.group(3)), []).append(m.group(2))
        else:
            plain.append(p)
    out = list(plain)
    for (head, tail), idx in groups.items():
        more = f" +{len(idx) - _MAX_INDICES} more" if len(idx) > _MAX_INDICES else ""
        out.append(f"{head}[{','.join(idx[:_MAX_INDICES])}{more}]{tail}")
    return ", ".join(out)


def compact_notes(notes: list[Any]) -> list[Any]:
    """One line per distinct note; the material vocabulary at most once, at the end."""
    groups: dict[str, list[str]] = {}
    kinds: str | None = None
    for n in notes:
        if not isinstance(n, str):
            groups.setdefault(repr(n), [])
            continue
        m = _NOTE.match(n)
        path, msg = (m.group(1), m.group(2)) if m else ("", n)
        k = _KINDS.search(msg)
        if k:
            kinds = kinds or k.group(1)
            msg = msg[:k.start()] + msg[k.end():]
        groups.setdefault(msg, []).append(path)
    out = []
    for msg, paths in groups.items():
        where = _paths([p for p in paths if p])
        out.append(f"{where}: {msg}" if where else msg)
    if kinds:
        out.append(f"material {kinds}")
    return out


def compact_names(names: list[Any]) -> list[Any]:
    """A long created list with numbered runs shortened: ``student_desk_001 .. student_desk_012 (12)``."""
    if len(names) <= _NAMES_KEEP:
        return names
    runs: dict[str, list[str]] = {}
    order: list[str] = []
    for n in names:
        m = _NUMBERED.match(n) if isinstance(n, str) else None
        key = m.group(1) if m else f"\0{n}"
        if key not in runs:
            order.append(key)
        runs.setdefault(key, []).append(n)
    out: list[Any] = []
    for key in order:
        run = runs[key]
        if key.startswith("\0") or len(run) < 3:
            out += run
        else:
            out.append(f"{run[0]} .. {run[-1]} ({len(run)})")
    return out


def _issue(i: Any) -> Any:
    """An issue without its empty fields (fix: null, objects: []) and with a short objects list."""
    if not isinstance(i, dict):
        return i
    out = {k: v for k, v in i.items() if v not in (None, [], {}, "")}
    objs = out.get("objects")
    if isinstance(objs, list) and len(objs) > _MAX_INDICES:
        out["objects"] = [*objs[:_MAX_INDICES], f"+{len(objs) - _MAX_INDICES} more"]
    return out


def present_build(out: dict[str, Any]) -> dict[str, Any]:
    """The build/edit result, compact and in reading order (idempotent)."""
    res = dict(out)
    if isinstance(res.get("repairs"), list):
        res["repairs"] = compact_notes(res["repairs"])
    if isinstance(res.get("created"), list):
        res["created"] = compact_names(res["created"])
    if isinstance(res.get("issues_left"), list):
        res["issues_left"] = [_issue(i) for i in res["issues_left"]]
    if res.get("scene_id"):
        res.pop("spec", None)  # stored with the version: edit_scene starts from it
    first = {k: res[k] for k in _FIRST if k in res}
    return {**first, **{k: v for k, v in res.items() if k not in first}}
