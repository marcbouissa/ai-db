"""Which file extensions are parsed with which tree-sitter grammar.

The single source of truth for the extension -> language map (previously duplicated in
``constants.EXT_TO_LANG`` and ``ts_graph.SUPPORTED_LANGUAGES``).

Grammars are optional per language: ``index.languages`` in the config selects which ones
are used. A file whose language is not selected is still indexed and searchable, but as
plain text: no structural chunks, symbols, call graph or syntax check, and its grammar is
never loaded (or downloaded -- tree-sitter-language-pack fetches each grammar on first use
into its cache, see ``ai-db grammars``).
"""

from __future__ import annotations

import os
from collections.abc import Collection, Iterable
from typing import Any

from ai_db.errors import AiDbConfigError

EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".java": "java",
    ".cs": "csharp",
    ".pkl": "pkl",
}

# Every language ai-db has a symbol/graph query for (ai_db/parser/queries/<lang>.scm).
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(EXT_TO_LANG.values())


def language_for(filepath: str) -> str | None:
    """The grammar ai-db *would* use for ``filepath``, regardless of selection."""
    return EXT_TO_LANG.get(os.path.splitext(filepath)[1].lower())


def enabled_language_for(filepath: str, languages: Collection[str]) -> str | None:
    """The grammar to parse ``filepath`` with, or None when its language is not selected."""
    lang = language_for(filepath)
    return lang if lang is not None and lang in languages else None


def validate_languages(languages: Iterable[str]) -> frozenset[str]:
    """Check a user selection against the supported languages."""
    chosen = frozenset(languages)
    unknown = chosen - SUPPORTED_LANGUAGES
    if unknown:
        raise AiDbConfigError(
            f"unknown languages {sorted(unknown)} in 'index.languages'; "
            f"supported: {sorted(SUPPORTED_LANGUAGES)}")
    return chosen


def extensions_of(languages: Collection[str]) -> frozenset[str]:
    """File extensions whose parsing depends on ``languages``."""
    return frozenset(ext for ext, lang in EXT_TO_LANG.items() if lang in languages)


# Grammars that name a declaration with a plain child node instead of a ``name`` field
# (Pkl: ``clazz``/``typeAlias``/``classProperty`` -> first ``identifier`` child,
# ``classMethod`` -> ``methodHeader`` -> ``identifier``).
_NAMED_BY_FIRST_IDENTIFIER = frozenset({"clazz", "typeAlias", "classProperty"})
_NAMED_BY_HEADER = {"classMethod": "methodHeader"}


def declared_name(node: Any) -> Any | None:
    """The node holding a declaration's name, for grammars with or without a ``name`` field."""
    found = node.child_by_field_name("name")
    if found is not None:
        return found
    if node.type in _NAMED_BY_HEADER:
        node = next((c for c in node.named_children if c.type == _NAMED_BY_HEADER[node.type]), None)
        if node is None:
            return None
    elif node.type not in _NAMED_BY_FIRST_IDENTIFIER:
        return None
    return next((c for c in node.named_children if c.type == "identifier"), None)
