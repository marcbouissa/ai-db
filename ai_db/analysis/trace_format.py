"""Output formatters for trace results."""
from __future__ import annotations

import json
from typing import Any

from ai_db.analysis.trace import TraceNode, TraceResult

AWAIT_MARKERS = {
    "sync": "⏵",
    "await": "⏸",
    "spawn": "⇉",
    "callback": "↺",
    "deferred": "⌁",
}


def format_tree(
    result: TraceResult,
    with_code: bool = False,
    context_lines: int = 3,
    max_nodes: int | None = None,
) -> str:
    """Format trace as an indented tree."""
    if result.root is None:
        return "No trace found."

    lines = []
    count = 0

    def write_node(node: TraceNode, prefix: str = "", is_last: bool = True):
        nonlocal count
        if max_nodes is not None and count >= max_nodes:
            return

        marker = AWAIT_MARKERS.get(node.await_kind or "sync", "⏵")
        guard_str = f" [{node.guard}]" if node.guard else ""
        recv_str = f" via {node.receiver}" if node.receiver else ""
        line_str = f" L{node.call_line}" if node.call_line else ""
        recursive_str = " (recursive)" if node.is_recursive else ""
        revisit_str = " (revisit)" if node.is_revisit else ""

        lines.append(
            f"{prefix}{marker} {node.qualified_name}{line_str}{recv_str}{guard_str}"
            f"{recursive_str}{revisit_str}"
        )
        count += 1

        if with_code and node.call_line:
            code_lines = _get_code_context(node.filepath, node.call_line, context_lines)
            for cl in code_lines:
                lines.append(f"{prefix}  │ {cl}")

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            write_node(child, child_prefix, i == len(node.children) - 1)

    write_node(result.root)

    if result.readiness_summary:
        lines.append("")
        lines.append("Readiness Summary:")
        for item in result.readiness_summary:
            lines.append(f"  • {item}")

    return "\n".join(lines)


def format_json(result: TraceResult) -> str:
    """Format trace as JSON."""
    def node_to_dict(node: TraceNode) -> dict[str, Any]:
        return {
            "qualified_name": node.qualified_name,
            "filepath": node.filepath,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "call_line": node.call_line,
            "await_kind": node.await_kind,
            "guard": node.guard,
            "receiver": node.receiver,
            "depth": node.depth,
            "is_recursive": node.is_recursive,
            "is_revisit": node.is_revisit,
            "children": [node_to_dict(c) for c in node.children],
        }

    return json.dumps({
        "root": node_to_dict(result.root) if result.root else None,
        "nodes": [node_to_dict(n) for n in result.nodes],
        "edges": [{"from": f, "to": t} for f, t in result.edges],
        "readiness_summary": result.readiness_summary,
        "unresolved": result.unresolved,
    }, indent=2)


def format_mermaid(result: TraceResult) -> str:
    """Format trace as Mermaid sequence diagram."""
    if result.root is None:
        return "sequenceDiagram\n    Note over System: No trace found"

    lines = ["sequenceDiagram", "    autonumber"]
    participants: set[str] = set()

    def collect_participants(node: TraceNode):
        participants.add(node.qualified_name)
        for child in node.children:
            collect_participants(child)

    collect_participants(result.root)

    for p in sorted(participants):
        lines.append(f"    participant {_sanitize(p)}")

    def add_edges(node: TraceNode):
        for child in node.children:
            marker = AWAIT_MARKERS.get(child.await_kind or "sync", "⏵")
            guard = f" [{child.guard}]" if child.guard else ""
            lines.append(
                f"    {_sanitize(node.qualified_name)}->>{_sanitize(child.qualified_name)}: "
                f"{marker} {child.qualified_name.split('.')[-1]}{guard}"
            )
            add_edges(child)

    add_edges(result.root)
    return "\n".join(lines)


def _sanitize(name: str) -> str:
    """Sanitize name for Mermaid."""
    return name.replace(".", "_").replace("-", "_")


def _get_code_context(filepath: str, line: int, context: int) -> list[str]:
    """Get source code lines around a line number."""
    try:
        with open(filepath, "r") as f:
            all_lines = f.readlines()
        start = max(0, line - 1 - context)
        end = min(len(all_lines), line + context)
        return [f"{i+1:4d} | {all_lines[i].rstrip()}" for i in range(start, end)]
    except OSError:
        return [f"  Could not read {filepath}:{line}"]