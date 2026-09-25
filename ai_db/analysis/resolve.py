"""Shared symbol resolver for investigate and trace.

Resolution order (deterministic, no guessing):
1. Same file definitions
2. Imported module definitions (via import refs)
3. self./class member definitions (same class)
4. Unique project-wide qualified name

Unresolvable names go to an `unresolved` set; the resolver never guesses.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ai_db.search.query_builder import split_identifier
from ai_db.search.ranking import last_component
from ai_db.storage.models import ChunkRecord

_BUILTINS = frozenset(dir(__builtins__)) | frozenset({
    "append", "extend", "get", "items", "keys", "values", "join", "split", "strip", "format",
    "startswith", "endswith", "replace", "lower", "upper", "update", "pop", "add", "execute",
    "fetchall", "fetchone", "cursor", "commit", "encode", "decode", "read", "write", "close",
})

COMPOUND_MAX_DEFINITIONS = 2


@dataclass
class ResolveResult:
    """Result of resolving a set of callee names."""
    resolved: dict[str, list[ChunkRecord]]  # callee_name -> [chunk]
    unresolved: set[str]


def scope_of(qualified_name: str) -> str:
    """Chunk qualified name -> cross-ref caller scope (``module.Class.method``)."""
    return f"module.{qualified_name}"


def qualified_of(scope: str) -> str | None:
    """Cross-ref caller scope -> chunk qualified name (None for module-level code)."""
    if scope == "module":
        return None
    return scope.removeprefix("module.")


class Resolver:
    """Deterministic symbol resolver shared by investigate and trace."""

    def __init__(self, backend: Any):
        self.db = backend
        self._import_cache: dict[str, set[str]] = {}

    def _imports(self, filepath: str) -> set[str]:
        """Dotted-name parts imported anywhere in ``filepath`` (cached per call)."""
        cached = self._import_cache.get(filepath)
        if cached is None:
            cached = set()
            for ref in self.db.get_refs_from(filepath, None, ("import",)):
                cached.update(ref.callee_name.split("."))
            self._import_cache[filepath] = cached
        return cached

    def _linked(self, user_file: str, definition: ChunkRecord) -> bool:
        """``user_file`` can reach ``definition``: same file, or it imports the definition's
        module (file stem) or its top-level name."""
        if user_file == definition.filepath:
            return True
        stem = os.path.splitext(os.path.basename(definition.filepath))[0]
        top = definition.qualified_name.split(".")[0]
        return bool(self._imports(user_file) & {stem, top})

    def _resolve(
        self,
        names: set[str],
        near: ChunkRecord,
        found: dict[str, list[ChunkRecord]],
    ) -> dict[str, list[ChunkRecord]]:
        """Resolve names near a given chunk. Returns dict of name -> [ChunkRecord]."""
        out: dict[str, list[ChunkRecord]] = {}
        for name in names:
            cands = [c for c in found.get(name, []) if c.id != near.id]
            same = [c for c in cands if c.filepath == near.filepath]
            linked = [c for c in cands if self._linked(near.filepath, c)]
            if same or linked:
                out[name] = sorted(same or linked, key=lambda c: c.id or 0)[:1]
            elif len(cands) == 1:
                out[name] = cands
            elif cands and len(split_identifier(name)) >= 2:
                # compound name, several definitions (interface + implementation): keep both
                out[name] = sorted(cands, key=lambda c: c.id or 0)[:COMPOUND_MAX_DEFINITIONS]
        return out

    def _caller_links_to(self, ref: Any, target: ChunkRecord) -> bool:
        """A caller counts if the called name is a compound identifier, or if its file
        can reach the target (same file or import). Single-word names need the link."""
        name = last_component(target.qualified_name)
        return len(split_identifier(name)) >= 2 or self._linked(ref.caller_filepath, target)

    def resolve_callees(
        self,
        names: set[str],
        near: ChunkRecord,
        allowed_projects: list[str] | None,
    ) -> ResolveResult:
        """Resolve a set of callee names near a given chunk."""
        found = self.db.find_chunks_by_symbol(sorted(names), allowed_projects)
        resolved = self._resolve(names, near, found)
        unresolved = set(names) - set(resolved)
        return ResolveResult(resolved=resolved, unresolved=unresolved)

    def resolve_callers(
        self,
        name: str,
        target: ChunkRecord,
        allowed_projects: list[str] | None,
        limit: int = 100,
    ) -> list[ChunkRecord]:
        """Resolve callers of a given symbol name."""
        refs = self.db.query_symbol_callers(
            callee_name=name, allowed_projects=allowed_projects, limit=limit
        )
        callers: list[ChunkRecord] = []
        for ref in refs:
            if ref.ref_type not in ("call", "inherit"):
                continue
            qn = qualified_of(ref.caller_name)
            if qn is None:
                continue
            if not self._caller_links_to(ref, target):
                continue
            caller = self.db.get_chunk_by_qualified_name(ref.caller_filepath, qn)
            if caller is not None:
                callers.append(caller)
        return callers

    def find_test_callers(
        self,
        name: str,
        target: ChunkRecord,
        allowed_projects: list[str] | None,
        limit: int = 100,
    ) -> list[ChunkRecord]:
        """Resolve test callers of a given symbol name."""
        refs = self.db.query_symbol_callers(
            callee_name=name, allowed_projects=allowed_projects, limit=limit
        )
        callers: list[ChunkRecord] = []
        for ref in refs:
            if ref.ref_type != "call":
                continue
            qn = qualified_of(ref.caller_name)
            if qn is None:
                continue
            from ai_db.analysis.investigate import is_test_path
            if not is_test_path(ref.caller_filepath):
                continue
            if not self._caller_links_to(ref, target):
                continue
            caller = self.db.get_chunk_by_qualified_name(ref.caller_filepath, qn)
            if caller is not None:
                callers.append(caller)
        return callers