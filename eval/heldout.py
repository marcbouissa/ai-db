"""A deterministic tune/holdout split of the golden retrieval queries.

Why this exists
---------------
Tuning a retrieval config against the same 40 queries that gate CI is fitting the
config to the test set. Whatever recall you then report is a number you
manufactured: the config was chosen to maximise it. This split exists so config
selection and config *reporting* happen on disjoint queries.

Rules, chosen so the split cannot flatter anything:

- **Deterministic.** No seed, no RNG, no shuffling. The same commit always yields
  the same split, so a reported number is reproducible months later.
- **Stratified by target module.** The 40 queries do not cluster evenly across
  the codebase -- `ai_db/storage/sqlite_backend.py` alone takes several. A naive
  every-other split would put most storage queries on one side. Each group is
  halved independently, so both halves span the same modules in the same
  proportions.
- **No new queries.** The split carves the existing 40 rather than authoring a
  fresh set. New queries would mean new ground truth, and hand-written ground
  truth has already been wrong in this project (six wrong expected files, an
  invented symbol, and a truncated call path in a first draft of
  eval/scenarios.json). Expectations here are copied from the golden file, which
  the harness tests cross-check.

Usage
-----
    split = HeldOutSplit.from_golden("eval/golden/ai_db.jsonl")
    split.tune      # config selection sees only these
    split.holdout   # reporting sees only these
    split.all       # the full 40, for continuity with the CI gate

`all` is deliberately still available. The gate scores all 40 and that number
should stay comparable; the holdout number is what may be quoted as an
uncontaminated estimate.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HeldOutSplit:
    tune: list[dict[str, Any]] = field(default_factory=list)
    holdout: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all(self) -> list[dict[str, Any]]:
        return self.tune + self.holdout

    @classmethod
    def from_golden(cls, path: str) -> HeldOutSplit:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        return cls.from_rows(rows)

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> HeldOutSplit:
        # Group by the directory holding the expected file, so the two halves
        # cover the same modules in the same proportions.
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            expected = row.get("expected") or [{}]
            filepath = expected[0].get("filepath", "")
            module = os.path.dirname(filepath) or "."
            groups.setdefault(module, []).append(row)

        tune: list[dict[str, Any]] = []
        holdout: list[dict[str, Any]] = []
        # Sorted keys: dict order is insertion order, which would make the split
        # depend on the order queries happen to appear in the file.
        for module in sorted(groups):
            members = sorted(groups[module], key=lambda r: r["query"])
            # Alternate within the group. Odd groups give the extra query to
            # holdout, so holdout is never the smaller half.
            for i, row in enumerate(members):
                (holdout if i % 2 else tune).append(row)
        return cls(tune=sorted(tune, key=lambda r: r["query"]),
                   holdout=sorted(holdout, key=lambda r: r["query"]))

    def describe(self) -> dict[str, Any]:
        """Module coverage per half, so an unbalanced split is visible."""
        def modules(rows: list[dict[str, Any]]) -> dict[str, int]:
            out: dict[str, int] = {}
            for r in rows:
                expected = r.get("expected") or [{}]
                mod = os.path.dirname(expected[0].get("filepath", "")) or "."
                out[mod] = out.get(mod, 0) + 1
            return dict(sorted(out.items()))

        return {
            "n_tune": len(self.tune),
            "n_holdout": len(self.holdout),
            "n_total": len(self.all),
            "modules_tune": modules(self.tune),
            "modules_holdout": modules(self.holdout),
            "disjoint": not ({r["query"] for r in self.tune}
                             & {r["query"] for r in self.holdout}),
            "covers_all": len(self.all) == len({r["query"] for r in self.all}),
        }


def score_recall(rows: list[dict[str, Any]], hits_for: Any) -> dict[str, Any]:
    """recall@k over `rows`, where `hits_for(row)` returns the hit file paths."""
    scored: list[float] = []
    for row in rows:
        expected = row.get("expected") or []
        want = {e["filepath"] for e in expected}
        if not want:
            continue
        got = set(hits_for(row))
        scored.append(len(want & got) / len(want))
    if not scored:
        return {"recall": None, "n": 0}
    return {
        "recall": round(sum(scored) / len(scored), 4),
        "n": len(scored),
        "perfect": sum(1 for s in scored if s == 1.0),
    }
