"""Call-flow trace engine for chronological execution order analysis.

Given an entry point (symbol, file:line, or script), performs a depth-first walk
in call order (seq) and produces a trace with await/spawn/callback/deferred markers
and a readiness summary for async operations.
"""
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from typing import Any

from ai_db.analysis.resolve import Resolver
from ai_db.constants import (
    TRACE_DEPTH,
    TRACE_MAX_NODES,
    TRACE_SNAP_LINES,
    TRACE_WAIT_PATTERNS,
)
from ai_db.storage.models import ChunkRecord
from ai_db.utils import resolve_client_path


@dataclass
class TraceNode:
    """A node in the call trace."""
    qualified_name: str
    filepath: str
    start_line: int
    end_line: int
    call_line: int | None = None
    await_kind: str | None = None
    guard: str | None = None
    receiver: str | None = None
    depth: int = 0
    is_recursive: bool = False
    is_revisit: bool = False
    children: list[TraceNode] = field(default_factory=list)


@dataclass
class TraceResult:
    """Result of a trace operation."""
    root: TraceNode | None
    nodes: list[TraceNode]
    edges: list[tuple[str, str]]
    readiness_summary: list[str]
    unresolved: list[str]


def _plausible_targets(ref: Any, candidates: list[ChunkRecord]) -> list[ChunkRecord]:
    """Drop name-only matches a call's receiver rules out.

    ``self.x()`` / ``Cls.x()`` resolved to a class (``callee_resolved``) keeps that
    class's method; a call on some other object whose type is unknown
    (``_CLIENT_CWD.reset()``) cannot be any class's method we can name, so it keeps only
    module-level functions instead of linking to whichever class has a ``reset``.
    """
    if ref.callee_resolved:
        exact = [c for c in candidates if c.qualified_name == ref.callee_resolved
                 or c.qualified_name.endswith("." + ref.callee_resolved)]
        return exact or candidates
    if ref.callee_qualifier and ref.callee_qualifier not in ("self", "cls"):
        return [c for c in candidates if "." not in c.qualified_name]
    return candidates


class TraceEngine:
    """Traces call flow from an entry point."""

    def __init__(
        self,
        backend: Any,
        allowed_projects: list[str] | None = None,
        trace_wait_patterns: list[str] | None = None,
        trace_wait_patterns_extend: list[str] | None = None,
    ):
        self.db = backend
        self.allowed_projects = allowed_projects
        self.resolver = Resolver(backend)
        self._seen_nodes: set[int] = set()
        self._call_stack: list[int] = []

        # Build wait patterns: defaults + extend + override
        self._wait_patterns = list(TRACE_WAIT_PATTERNS)
        if trace_wait_patterns_extend:
            self._wait_patterns.extend(trace_wait_patterns_extend)
        if trace_wait_patterns:
            self._wait_patterns = trace_wait_patterns

    def trace(
        self,
        entry: str,
        depth: int = TRACE_DEPTH,
        max_nodes: int = TRACE_MAX_NODES,
        direction: str = "down",
        include_tests: bool = False,
    ) -> TraceResult:
        """Trace call flow from an entry point.

        Args:
            entry: Symbol name, "file:line", or script filepath
            depth: Maximum depth to trace
            max_nodes: Maximum total nodes to visit
            direction: "down" (callees) or "up" (callers)
            include_tests: Whether to include test callers in up direction
        """
        self._seen_nodes.clear()
        self._call_stack.clear()

        # Find entry chunk
        entry_chunk = self._find_entry(entry)
        if entry_chunk is None:
            return TraceResult(
                root=None,
                nodes=[],
                edges=[],
                readiness_summary=[f"Entry point not found: {entry}"],
                unresolved=[entry],
            )

        root = self._build_trace(
            entry_chunk,
            depth=depth,
            max_nodes=max_nodes,
            direction=direction,
            include_tests=include_tests,
        )

        # Collect all nodes and edges
        nodes = self._collect_nodes(root)
        edges = self._collect_edges(root)

        # Generate readiness summary
        readiness = self._generate_readiness(root)

        return TraceResult(
            root=root,
            nodes=nodes,
            edges=edges,
            readiness_summary=readiness,
            unresolved=[],  # TODO: track unresolved
        )

    def _find_entry(self, entry: str) -> ChunkRecord | None:
        """Find the entry chunk from a symbol, file:line, or script path."""
        # Try file:line
        if ":" in entry and entry.rsplit(":", 1)[-1].isdigit():
            filepath, line_str = entry.rsplit(":", 1)
            line = int(line_str)
            chunks: list[ChunkRecord] = self.db.get_chunks_for_file(resolve_client_path(filepath))
            # The innermost chunk containing the line: a class header spans its methods,
            # so the first match would trace the whole class for a line inside a method.
            containing = [c for c in chunks if c.start_line <= line <= c.end_line]
            if containing:
                return min(containing, key=lambda c: (c.end_line - c.start_line, c.chunk_type == "module"))
            # A blank line, comment or decorator just above a definition means that definition.
            following = [c for c in chunks if line < c.start_line <= line + TRACE_SNAP_LINES]
            return min(following, key=lambda c: c.start_line) if following else None

        # Try as filepath (script)
        if os.path.isfile(entry):
            chunks = self.db.get_chunks_for_file(entry)  # type: ignore[assignment]
            # Prefer module-level or __main__ block
            for chunk in chunks:
                if chunk.chunk_type == "module" or "__main__" in chunk.qualified_name:
                    return chunk  # type: ignore[no-any-return]
            return chunks[0] if chunks else None  # type: ignore[no-any-return]

        # A symbol: `filter_output`, or qualified `ServiceDispatcher.execute`. Chunks are
        # indexed by their last name component, so look that up and match the rest
        # against the qualified name (a dotted entry used to match nothing).
        wanted = entry.removeprefix("module.")
        last = wanted.rsplit(".", 1)[-1]
        found: dict[str, list[ChunkRecord]] = self.db.find_chunks_by_symbol(
            [last], self.allowed_projects or ["global"])
        candidates = [c for chunk_list in found.values() for c in chunk_list]
        if "." in wanted:
            candidates = [c for c in candidates
                          if c.qualified_name == wanted or c.qualified_name.endswith("." + wanted)]
        exact = [c for c in candidates if c.qualified_name == wanted]
        return (exact or candidates)[0] if (exact or candidates) else None

    def _build_trace(
        self,
        chunk: ChunkRecord,
        depth: int,
        max_nodes: int,
        direction: str,
        include_tests: bool,
    ) -> TraceNode | None:
        """Recursively build the trace tree."""
        if depth < 0 or len(self._seen_nodes) >= max_nodes:
            return None

        chunk_id = chunk.id
        if chunk_id is None:
            return None
        is_revisit = chunk_id in self._seen_nodes
        is_recursive = chunk_id in self._call_stack

        self._seen_nodes.add(chunk_id)
        self._call_stack.append(chunk_id)

        node = TraceNode(
            qualified_name=chunk.qualified_name,
            filepath=chunk.filepath,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            depth=len(self._call_stack) - 1,
            is_recursive=is_recursive,
            is_revisit=is_revisit,
        )

        if not is_recursive and not is_revisit:
            if direction == "down":
                self._add_callees(node, chunk, depth - 1, max_nodes)
            elif direction == "up":
                self._add_callers(node, chunk, depth - 1, max_nodes, include_tests)

        self._call_stack.pop()
        return node

    def _add_callees(
        self,
        node: TraceNode,
        chunk: ChunkRecord,
        depth: int,
        max_nodes: int,
    ) -> None:
        """Add callee children to the trace node."""
        from ai_db.analysis.investigate import _BUILTINS

        refs = self.db.get_refs_from(chunk.filepath, f"module.{chunk.qualified_name}")
        # Sort by seq for call order
        # Source order (get_refs_from sorts by call site).
        calls = [r for r in refs if r.ref_type == "call" and r.callee_name not in _BUILTINS]

        seen_callees: set[str] = set()
        for ref in calls:
            if len(self._seen_nodes) >= max_nodes:
                break
            if ref.callee_name in seen_callees:
                continue
            seen_callees.add(ref.callee_name)

            result = self.resolver.resolve_callees({ref.callee_name}, chunk, self.allowed_projects)
            for callee in _plausible_targets(ref, result.resolved.get(ref.callee_name, [])):
                child = self._build_trace(callee, depth, max_nodes, "down", False)
                if child:
                    child.call_line = ref.call_line or ref.caller_line
                    child.await_kind = ref.await_kind
                    child.guard = ref.guard
                    child.receiver = ref.receiver
                    node.children.append(child)

    def _add_callers(
        self,
        node: TraceNode,
        chunk: ChunkRecord,
        depth: int,
        max_nodes: int,
        include_tests: bool,
    ) -> None:
        """Add caller parents to the trace node."""
        name = chunk.qualified_name.split(".")[-1]
        callers = self.resolver.resolve_callers(name, chunk, self.allowed_projects, limit=100)

        if include_tests:
            test_callers = self.resolver.find_test_callers(name, chunk, self.allowed_projects, limit=100)
            callers.extend(test_callers)

        for caller in callers:
            if len(self._seen_nodes) >= max_nodes:
                break
            child = self._build_trace(caller, depth, max_nodes, "up", include_tests)
            if child:
                # Find the ref to get call site info
                refs = self.db.get_refs_from(caller.filepath, f"module.{caller.qualified_name}")
                matching = [r for r in refs if r.callee_name == name and r.ref_type == "call"]
                if matching:
                    ref = matching[0]
                    child.call_line = ref.caller_line
                    child.await_kind = ref.await_kind
                    child.guard = ref.guard
                    child.receiver = ref.receiver
                node.children.append(child)

    def _collect_nodes(self, root: TraceNode | None) -> list[TraceNode]:
        """Collect all nodes in the tree."""
        if root is None:
            return []
        nodes = [root]
        for child in root.children:
            nodes.extend(self._collect_nodes(child))
        return nodes

    def _collect_edges(self, root: TraceNode | None) -> list[tuple[str, str]]:
        """Collect all edges in the tree."""
        if root is None:
            return []
        edges = []
        for child in root.children:
            edges.append((root.qualified_name, child.qualified_name))
            edges.extend(self._collect_edges(child))
        return edges

    def _generate_readiness(self, root: TraceNode | None) -> list[str]:
        """Generate readiness summary for async operations."""
        if root is None:
            return []

        summary = []

        def walk(node: TraceNode, path: list[TraceNode]):
            path.append(node)

            if node.await_kind in ("spawn", "callback", "deferred"):
                # Find nearest wait on this path
                wait_found = False
                for ancestor in reversed(path[:-1]):  # exclude current
                    if self._matches_wait_pattern(ancestor.qualified_name):
                        summary.append(
                            f"`{node.qualified_name}` runs after "
                            f"`{ancestor.qualified_name}` ({ancestor.await_kind or 'sync'}, "
                            f"{os.path.basename(ancestor.filepath)}:{ancestor.call_line or ancestor.start_line})"
                        )
                        wait_found = True
                        break
                    # Also check if ancestor awaits something
                    if ancestor.await_kind == "await":
                        summary.append(
                            f"`{node.qualified_name}` spawned by "
                            f"`{ancestor.qualified_name}` (not awaited)"
                        )
                        wait_found = True
                        break
                if not wait_found:
                    summary.append(
                        f"`{node.qualified_name}` spawned by "
                        f"`{path[-2].qualified_name if len(path) >= 2 else 'unknown'}` (not awaited)"
                    )

            for child in node.children:
                walk(child, path)
            path.pop()

        walk(root, [])
        return summary

    def _matches_wait_pattern(self, name: str) -> bool:
        """Check if a name matches any wait pattern."""
        for pattern in self._wait_patterns:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name, f"*{pattern}*"):
                return True
        return False