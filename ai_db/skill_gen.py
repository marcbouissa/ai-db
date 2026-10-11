"""Skill generation: agents write their own SKILL.md files.

``draft_skill(topic)`` gathers what ai-db already knows (matching skills, saved contexts,
and contributions of installed plugins) into a skeleton SKILL.md, deterministically (no
LLM call). The calling agent edits it and saves it with ``create_skill``, which validates
it, writes ``<generated_dir>/<name>/SKILL.md`` and re-syncs skills so ``route_skill``
finds it immediately. ``list_generated_skills`` / ``delete_skill`` only ever see
``generated_dir``: packaged and plugin skills are read-only from here.

``generated_dir`` is ``skills.generated_dir`` in the config, else ``generated/`` under the
first default skill dir (``~/.local/share/ai-db/skills``, or the first AI_DB_SKILL_DIRS
entry). It is always part of a default ``sync_skills``.

Plugins add draft material through the ``ai_db.skill_drafts`` entry-point group. The value
is a function ``(topic: str, config: AppConfig | None) -> dict[str, list[str]]`` whose keys
are section names (``when_to_use``, ``workflow``, ``pitfalls``, ``examples``, ``sources``);
e.g. in a plugin's pyproject.toml::

    [project.entry-points."ai_db.skill_drafts"]
    scene = "ai_db_scene.skill_drafts:contribute"
"""

from __future__ import annotations

import json
import os
import re
import shutil
from importlib.metadata import entry_points
from typing import Any

from ai_db import constants
from ai_db.errors import AiDbConfigError
from ai_db.logger import _logger

SKILL_DRAFT_GROUP = "ai_db.skill_drafts"
SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SKILL_NAME_MAX = 64
SKILL_DESC_MIN = 20
SKILL_DESC_MAX = 1024
SKILL_BODY_MAX = 40_000
SKILL_TRIGGERS_MAX = 40
SKILL_TRIGGER_MAX_CHARS = 80
SKILL_DRAFT_TOP = 5  # existing skills / contexts shown by draft_skill
SKILL_DUPLICATE_CONFIDENCE = 0.5  # draft_skill warns at or above this route confidence
_HEADING = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_BAD_TRIGGER = re.compile(r"[\n:#|,\"]")
_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
SECTIONS = (("when_to_use", "When to use"), ("workflow", "Workflow"),
            ("pitfalls", "Pitfalls"), ("examples", "Examples"))
#: rendered only when a source fills them (e.g. an {"asset": ...} source)
OPTIONAL_SECTIONS = (("asset_build", "How the asset is built"),)
_SOURCE_KEYS = ("url", "text", "video", "asset", "path", "title", "license")


def generated_skills_dir(config: Any = None) -> str:
    """Absolute folder that create_skill writes to (see module docstring)."""
    explicit = getattr(config, "skills_generated_dir", None)
    if explicit:
        return os.path.abspath(os.path.expanduser(str(explicit)))
    dirs = constants.DEFAULT_SKILL_DIRS
    base = dirs[0] if dirs else os.path.join(
        os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")), "ai-db", "skills")
    return os.path.abspath(os.path.join(os.path.expanduser(base), "generated"))


def _skill_path(gen_dir: str, name: str) -> str:
    if not isinstance(name, str) or len(name) > SKILL_NAME_MAX or not SKILL_NAME_RE.match(name):
        raise ValueError(f"skill name must be kebab-case [a-z0-9-], at most {SKILL_NAME_MAX} chars, got {name!r}")
    path = os.path.realpath(os.path.join(gen_dir, name, "SKILL.md"))
    if os.path.commonpath([path, os.path.realpath(gen_dir)]) != os.path.realpath(gen_dir):
        raise ValueError(f"skill path escapes the generated skills folder: {name!r}")
    return path


def _str_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [v for v in value.split(",")]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{field} must be a list of strings")
    return [v.strip() for v in value if v.strip()]


def render_skill(name: str, description: str, body: str, triggers: list[str] | None = None,
                 related_skills: list[str] | None = None, domain: str | None = None,
                 sources: list[dict[str, str]] | None = None) -> str:
    """SKILL.md text in the frontmatter format of the packaged skills."""
    meta = ["metadata:", "  author: ai-db create_skill", '  version: "1.0.0"']
    if domain:
        meta.append(f"  domain: {domain}")
    meta.append("  role: generated")
    if triggers:
        meta.append(f"  triggers: {', '.join(triggers)}")
    if related_skills:
        meta.append(f"  related-skills: {', '.join(related_skills)}")
    if sources:
        # one JSON object per line: valid YAML flow mappings, safe for any URL or title
        meta.append("  sources:")
        meta += [f"    - {json.dumps(src, ensure_ascii=False)}" for src in sources]
    return "\n".join(["---", f"name: {name}", f"description: {description}", *meta, "---", "",
                      body.strip(), ""])


def validate_sources(sources: Any) -> list[dict[str, str]]:
    """``metadata.sources`` entries: {url|text|video|asset|path, title?, license?} strings."""
    if sources is None:
        return []
    if not isinstance(sources, list) or len(sources) > 20:
        raise ValueError("sources must be a list of at most 20 objects")
    out = []
    for s in sources:
        if not isinstance(s, dict) or not s or set(s) - set(_SOURCE_KEYS) \
                or not all(isinstance(v, str) and "\n" not in v and len(v) <= 500 for v in s.values()):
            raise ValueError(f"source {s!r}: keys from {list(_SOURCE_KEYS)}, one-line strings")
        out.append(dict(s))
    return out


def validate_skill(name: str, description: str, body: str, triggers: Any = None,
                   related_skills: Any = None, domain: Any = None) -> tuple[list[str], list[str]]:
    """Raise ValueError on anything create_skill must refuse; returns (triggers, related)."""
    if not isinstance(name, str) or len(name) > SKILL_NAME_MAX or not SKILL_NAME_RE.match(name):
        raise ValueError(f"skill name must be kebab-case [a-z0-9-], at most {SKILL_NAME_MAX} chars, got {name!r}")
    if not isinstance(description, str):
        raise TypeError("description must be a string")
    desc = description.strip()
    if not SKILL_DESC_MIN <= len(desc) <= SKILL_DESC_MAX:
        raise ValueError(f"description must be {SKILL_DESC_MIN}-{SKILL_DESC_MAX} chars, got {len(desc)}")
    if "\n" in desc or '"' in desc or desc.startswith((">", "|", "-")):
        raise ValueError("description must be one line, without double quotes or a YAML block marker")
    if not isinstance(body, str) or not body.strip():
        raise ValueError("body is empty")
    if len(body) > SKILL_BODY_MAX:
        raise ValueError(f"body is {len(body)} chars, max {SKILL_BODY_MAX}")
    if body.lstrip().startswith("---"):
        raise ValueError("body must not carry frontmatter; pass name/description/triggers as arguments")
    if not _HEADING.search(body):
        raise ValueError("body needs at least one markdown heading (e.g. '# Title')")
    trigs = _str_list(triggers, "triggers")
    if len(trigs) > SKILL_TRIGGERS_MAX:
        raise ValueError(f"at most {SKILL_TRIGGERS_MAX} triggers, got {len(trigs)}")
    for t in trigs:
        if len(t) > SKILL_TRIGGER_MAX_CHARS or _BAD_TRIGGER.search(t):
            raise ValueError(f"trigger {t!r}: at most {SKILL_TRIGGER_MAX_CHARS} chars, no newline : # | , or quote")
    related = _str_list(related_skills, "related_skills")
    for r in related:
        if not SKILL_NAME_RE.match(r):
            raise ValueError(f"related skill {r!r} is not a kebab-case skill name")
    if domain is not None and (not isinstance(domain, str) or not _DOMAIN_RE.match(domain)):
        raise ValueError(f"domain must be a short lowercase word, got {domain!r}")
    return trigs, related


def _under(path: str, folder: str) -> bool:
    path, folder = os.path.realpath(path), os.path.realpath(folder)
    return os.path.commonpath([path, folder]) == folder


def create_skill(db: Any, gen_dir: str, name: str, description: str, body: str,
                 triggers: Any = None, related_skills: Any = None, domain: str | None = None,
                 overwrite: bool = False, sources: Any = None) -> dict[str, Any]:
    trigs, related = validate_skill(name, description, body, triggers, related_skills, domain)
    srcs = validate_sources(sources)
    path = _skill_path(gen_dir, name)
    # a skill of the same name elsewhere (packaged, plugin, user dir) is not ours to shadow
    for sk in db.backend.get_skills():
        if sk.name == name and not _under(sk.filepath, gen_dir):
            raise ValueError(f"skill {name!r} already exists outside the generated folder "
                             f"({sk.filepath}); pick another name")
    existed = os.path.exists(path)
    if existed and not overwrite:
        raise ValueError(f"generated skill {name!r} exists; pass overwrite=true to replace it")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = render_skill(name, description.strip(), body, trigs, related, domain, srcs)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    sync = db.sync_skills(verbose=False)
    return {"name": name, "path": path, "status": "updated" if existed else "created", "sync": sync}


def list_generated_skills(gen_dir: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not os.path.isdir(gen_dir):
        return out
    for entry in sorted(os.listdir(gen_dir)):
        p = os.path.join(gen_dir, entry, "SKILL.md")
        if not SKILL_NAME_RE.match(entry) or not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            head = f.read(4096)
        m = re.search(r"^description:\s*(.+)$", head, re.MULTILINE)
        out.append({"name": entry, "path": p, "description": m.group(1).strip() if m else "",
                    "modified": os.path.getmtime(p)})
    return out


def delete_skill(db: Any, gen_dir: str, name: str) -> dict[str, Any]:
    path = _skill_path(gen_dir, name)
    if not os.path.isfile(path):
        raise ValueError(f"no generated skill {name!r} in {gen_dir}")
    shutil.rmtree(os.path.dirname(path))
    return {"name": name, "deleted": path, "sync": db.sync_skills(verbose=False)}


def _slug(topic: str) -> str:
    words = re.findall(r"[a-z0-9]+", topic.lower())
    return "-".join(words)[:SKILL_NAME_MAX].strip("-") or "new-skill"


def plugin_contributions(topic: str, config: Any) -> dict[str, dict[str, Any]]:
    """Run every ``ai_db.skill_drafts`` hook; a failing hook reports its error, never raises."""
    out: dict[str, dict[str, Any]] = {}
    for ep in sorted(entry_points(group=SKILL_DRAFT_GROUP), key=lambda e: e.name):
        try:
            hook = ep.load()
        except ImportError as e:
            raise AiDbConfigError(f"skill-draft plugin '{ep.name}' ({ep.value}) failed to import: {e}") from e
        try:
            res = hook(topic, config)
        except Exception as e:  # noqa: BLE001 - a plugin bug must not break draft_skill
            _logger.debug(f"skill-draft plugin {ep.name} failed: {e!r}")
            out[ep.name] = {"error": str(e)}
            continue
        if isinstance(res, dict):
            out[ep.name] = {k: [str(x) for x in v] for k, v in res.items() if isinstance(v, list)}
    return out


def draft_skill(db: Any, topic: str, config: Any = None, sources: Any = None,
                fetch: Any = None) -> dict[str, Any]:
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic is required")
    topic = topic.strip()
    from ai_db.skill_sources import read_sources

    read = read_sources(sources, config, fetch)
    existing = db.route_skills(topic, top_k=SKILL_DRAFT_TOP, min_confidence=0.0)
    try:
        contexts = db.query_contexts(topic, top_k=SKILL_DRAFT_TOP) or []
    except Exception as e:  # noqa: BLE001 - recall is optional material for a draft
        _logger.debug(f"draft_skill: context recall failed: {e!r}")
        contexts = []
    plugins = plugin_contributions(topic, config)

    sections: dict[str, list[str]] = {k: [] for k, _ in SECTIONS + OPTIONAL_SECTIONS}
    for key, lines in read["sections"].items():
        if key in sections:
            sections[key].extend(lines)
    recorded: list[dict[str, str]] = read["sources"]
    sources = []
    for c in contexts:
        title = c.get("title") or c.get("session_id") or "context"
        summary = " ".join(str(c.get("summary") or "").split())[:300]
        if summary:
            sections["workflow"].append(f"(from saved context '{title}') {summary}")
            sources.append(f"context:{c.get('session_id')}")
    for pname, contrib in plugins.items():
        for key, lines in contrib.items():
            if key in sections:
                sections[key].extend(lines)
            elif key == "sources":
                sources.extend(f"{pname}:{s}" for s in lines)

    placeholders = {"when_to_use": f"TODO: the requests about {topic} this skill should answer.",
                    "workflow": "TODO: numbered steps, tool calls with their key arguments.",
                    "pitfalls": "TODO: what goes wrong and how to avoid it.",
                    "examples": "TODO: one worked example (input -> calls -> result)."}
    lines = [f"# {topic[:1].upper() + topic[1:]}", ""]
    for key, title in SECTIONS + OPTIONAL_SECTIONS:
        if key not in placeholders and not sections[key]:
            continue
        lines += [f"## {title}", ""]
        # a code block or other multi-line item stands on its own, not as a bullet
        lines += [x if "\n" in x else f"- {x}" for x in sections[key]] or [f"- {placeholders[key]}"]
        lines.append("")
    body = "\n".join(lines).rstrip() + "\n"
    name = _slug(topic)
    triggers = [t for t in dict.fromkeys([" ".join(re.findall(r"[a-z0-9]+", topic.lower()))[:SKILL_TRIGGER_MAX_CHARS]]) if t]
    description = f"TODO: one line saying what this skill does for {topic} and when to use it."
    dupes = [s for s in existing if s.get("confidence", 0) >= SKILL_DUPLICATE_CONFIDENCE]
    return {
        "name": name,
        "description": description,
        "triggers": triggers,
        "body": body,
        "skill_md": render_skill(name, description, body, triggers, sources=recorded),
        "existing_skills": existing,
        "possible_duplicates": [s["name"] for s in dupes],
        "contexts": [{"session_id": c.get("session_id"), "title": c.get("title")} for c in contexts],
        "plugins": plugins,
        "sources": sources,
        "metadata_sources": recorded,
        "warnings": read["warnings"],
        "next": "Extend an existing skill if one already covers this; otherwise replace the TODOs "
                "and call create_skill(name, description, body, triggers, sources=metadata_sources).",
    }


def register_skill_tools(dispatcher: Any) -> None:
    """Add create_skill / draft_skill / list_generated_skills / delete_skill."""

    def _ctx(args: dict[str, Any]) -> tuple[Any, str]:
        db = dispatcher._get_db(args)
        return db, generated_skills_dir(getattr(db, "config", None) or dispatcher.config)

    def _create(args: dict[str, Any]) -> Any:
        db, gen = _ctx(args)
        return create_skill(db, gen, args.get("name", ""), args.get("description", ""), args.get("body", ""),
                            args.get("triggers"), args.get("related_skills"), args.get("domain"),
                            bool(args.get("overwrite", False)), args.get("sources"))

    def _draft(args: dict[str, Any]) -> Any:
        db, _ = _ctx(args)
        return draft_skill(db, args.get("topic", ""), getattr(db, "config", None) or dispatcher.config,
                           args.get("sources"))

    def _list(args: dict[str, Any]) -> Any:
        return list_generated_skills(_ctx(args)[1])

    def _delete(args: dict[str, Any]) -> Any:
        db, gen = _ctx(args)
        return delete_skill(db, gen, args.get("name", ""))

    str_or_list = {"oneOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}]}
    dispatcher.register_tool(
        name="create_skill",
        description="Save a new agent skill (SKILL.md) to the generated-skills folder and index it so "
                    "route_skill finds it at once. Start from draft_skill. Packaged skills cannot be replaced.",
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "kebab-case skill name"},
                "description": {"type": "string", "description": f"One line, {SKILL_DESC_MIN}-{SKILL_DESC_MAX} chars"},
                "body": {"type": "string", "description": "Markdown body (no frontmatter), with a heading"},
                "triggers": {**str_or_list, "description": "Phrases that should route to this skill"},
                "related_skills": {**str_or_list, "description": "Names of related skills"},
                "domain": {"type": "string", "description": "Short domain word, e.g. blender"},
                "overwrite": {"type": "boolean", "default": False, "description": "Replace an existing generated skill"},
                "sources": {"type": "array", "items": {"type": "object"},
                            "description": "metadata.sources: [{url|text|video|asset, title?, license?}]"},
            },
            "required": ["name", "description", "body"],
        },
        handler=_create, category="routing")
    dispatcher.register_tool(
        name="draft_skill",
        description="Draft a SKILL.md for a topic from what ai-db already knows (existing skills, saved "
                    "contexts, plugin run history) and optional sources: tutorial text or URL (fetched as "
                    "text), a local video (subtitles/keyframes), a .gltf/.glb/.blend asset or Blender object. "
                    "No LLM call; edit the result and save with create_skill.",
        parameters_schema={"type": "object", "properties": {
            "topic": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "object"},
                        "description": "[{text, title?, url?, license?} | {url} | {video: path} | {asset: path or object}]"}},
            "required": ["topic"]},
        handler=_draft, category="routing")
    dispatcher.register_tool(
        name="list_generated_skills",
        description="List skills saved with create_skill.",
        parameters_schema={"type": "object", "properties": {}},
        handler=_list, category="routing")
    dispatcher.register_tool(
        name="delete_skill",
        description="Delete a skill saved with create_skill (generated skills only) and re-index.",
        parameters_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        handler=_delete, category="routing")
