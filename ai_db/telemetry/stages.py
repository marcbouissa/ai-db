"""Per-stage latency statistics from the query log."""

from __future__ import annotations

from typing import Any


def _pct(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(pct / 100.0 * (len(ordered) - 1)))
    return round(ordered[idx], 2)


def stage_stats(entries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """``{stage: {count, p50_ms, p95_ms}}`` including ``total`` (computed misses only
    for stages; totals include cache hits)."""
    buckets: dict[str, list[float]] = {"total": []}
    for e in entries:
        buckets["total"].append(float(e["total_ms"]))
        for stage, ms in e.get("stages", {}).items():
            buckets.setdefault(stage, []).append(float(ms))
    return {
        stage: {"count": len(vals), "p50_ms": _pct(vals, 50), "p95_ms": _pct(vals, 95)}
        for stage, vals in buckets.items() if vals
    }
