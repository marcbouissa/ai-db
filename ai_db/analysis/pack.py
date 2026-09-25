"""InvestigationPack: the JSON contract returned by ``investigate``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EntryPoint:
    qualified_name: str
    filepath: str
    lines: str
    why: str


@dataclass
class Evidence:
    ref: str
    filepath: str
    lines: str
    qualified_name: str
    role: str  # seed | parent | callee | caller | test
    score: float
    why: str
    body: str | None = None
    stub: str | None = None


@dataclass
class InvestigationPack:
    query: str
    mode: str
    entry_points: list[EntryPoint] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    call_graph: dict[str, list[Any]] = field(default_factory=lambda: {"nodes": [], "edges": []})
    tests: list[dict[str, Any]] = field(default_factory=list)
    recent_changes: list[dict[str, Any]] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    omitted: list[dict[str, Any]] = field(default_factory=list)
    omitted_count: int = 0
    retrieval: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    budget_tokens: int = 0
    trace_result: Any = None
    trace_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for ev in out["evidence"]:
            if ev["body"] is None:
                del ev["body"]
            if ev["stub"] is None:
                del ev["stub"]
        return out
