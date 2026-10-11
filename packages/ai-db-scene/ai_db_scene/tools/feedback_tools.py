"""record_feedback / list_feedback / update_feedback / delete_feedback / optimize_feedback:
user complaints about built scenes become lessons later builds check (see ai_db_scene.feedback)."""

from __future__ import annotations

from typing import Any


def register_feedback_tools(dispatcher: Any) -> None:
    from ai_db_scene import feedback as fb

    issues = sorted(fb.ISSUES) + ["other"]
    for name, desc, schema, handler in (
        ("record_feedback",
         ("Call this whenever the user complains about a built scene or asset ('the bench back is "
          "floating', 'the clothes are too stiff', 'all sofas look the same'). It becomes a lesson: "
          "later builds with those objects check it automatically (or list it to verify), and "
          "get_scene_spec_schema shows it before you write a spec. Categories and issue are read from "
          "the text; pass them to be exact. Repeating an existing lesson reinforces it."),
         {"type": "object", "properties": {
             "text": {"type": "string", "description": "the user's complaint, in their words"},
             "categories": {"type": "array", "items": {"type": "string"},
                            "description": "object categories it is about (default: found in the text)"},
             "subjects": {"type": "array", "items": {"type": "string"},
                          "description": "other words that scope it (e.g. 'clothes', 'curtain') when no category fits"},
             "issue": {"type": "string", "enum": issues}, "severity": {"type": "string", "enum": list(fb.SEVERITIES)},
             "project": {"type": "string", "description": "scope it to one project (default: global, every project)"}},
          "required": ["text"]},
         lambda a: fb.add(a.get("text", ""), a.get("categories"), a.get("issue"),
                          a.get("severity", "blocking"), a.get("subjects"), a.get("project") or fb.GLOBAL)),
        ("list_feedback", "List the lessons learned from user feedback (with how often each was reported and matched).",
         {"type": "object", "properties": {"include_disabled": {"type": "boolean", "default": True},
                                           "project": {"type": "string",
                                                       "description": "only what a build in this project uses"}}},
         lambda a: fb.list_lessons(bool(a.get("include_disabled", True)), a.get("project"))),
        ("update_feedback",
         "Edit, disable (enabled=false) or re-enable a lesson: set any of text, categories, subjects, issue, severity, enabled.",
         {"type": "object", "properties": {"id": {"type": "string"}, "set": {"type": "object"}},
          "required": ["id", "set"]},
         lambda a: fb.update(a.get("id", ""), a.get("set") or {})),
        ("delete_feedback", "Delete a lesson for good (disable it with update_feedback to keep it).",
         {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
         lambda a: {"deleted": fb.delete(a.get("id", ""))}),
        ("optimize_feedback",
         ("Review the lessons: duplicates to merge, mistyped categories (with the closest known one), "
          "unknown issues and outdated lessons (category gone, or matched nothing in many builds). "
          "Returns proposals; apply=true carries them out (outdated ones are disabled, not deleted). "
          "Show the user the proposals before applying."),
         {"type": "object", "properties": {"apply": {"type": "boolean", "default": False}}},
         lambda a: fb.optimize(bool(a.get("apply", False)))),
    ):
        dispatcher.register_tool(name=name, description=desc, parameters_schema=schema, handler=handler,
                                 category="scene")
