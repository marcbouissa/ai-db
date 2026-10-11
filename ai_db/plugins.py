"""Tool plugins: packages that add tools to every ServiceDispatcher.

A plugin declares an entry point in the ``ai_db.tools`` group whose value is a
``register(dispatcher) -> None`` function, e.g. in its pyproject.toml::

    [project.entry-points."ai_db.tools"]
    scene = "ai_db_scene.tools:register"

Plugins are registered by ``ServiceDispatcher`` itself, so the CLI, the MCP server and
the daemon always expose the same tool set. ``register`` must be cheap: import heavy
dependencies (torch, model weights) inside the tool handlers, not at registration.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from ai_db.errors import AiDbConfigError

TOOL_PLUGIN_GROUP = "ai_db.tools"
#: Entry points whose value is a ``() -> str`` returning a directory of ``<name>/SKILL.md``
#: skills shipped with the package (indexed by ``sync_skills`` with the default dirs).
SKILL_PLUGIN_GROUP = "ai_db.skills"


def register_tool_plugins(dispatcher: Any) -> list[str]:
    """Register every installed tool plugin with ``dispatcher``; returns their names."""
    names = []
    for ep in sorted(entry_points(group=TOOL_PLUGIN_GROUP), key=lambda e: e.name):
        try:
            register = ep.load()
        except ImportError as e:
            raise AiDbConfigError(
                f"tool plugin '{ep.name}' ({ep.value}) failed to import: {e}. "
                f"Fix or uninstall the package that provides it.") from e
        register(dispatcher)
        names.append(ep.name)
    return names


def plugin_skill_dirs() -> list[str]:
    """Skill directories contributed by installed plugins (``ai_db.skills`` entry points)."""
    dirs = []
    for ep in sorted(entry_points(group=SKILL_PLUGIN_GROUP), key=lambda e: e.name):
        try:
            dirs.append(str(ep.load()()))
        except ImportError as e:
            raise AiDbConfigError(
                f"skill plugin '{ep.name}' ({ep.value}) failed to import: {e}. "
                f"Fix or uninstall the package that provides it.") from e
    return dirs
