"""Teacher loop: turn failing benchmark runs into exemplars the next run can copy.

A failing run (an invalid spec, a layout conflict, blocking validator issues) still carries the
model's intent. Correct its spec deterministically, with the same tools build_scene uses
(validate_spec's own suggestions, then repair.relax until solve_layout succeeds), and store the
result through record_run as a successful run, so get_scene_spec_schema / search_exemplars
offer it for similar briefs. No model is called.

Sources: scene_bench.py JSONL rows (--jsonl) and failed/partial runs in the scene memory
(--memory; AI_DB_SCENE_MEMORY picks the store). Dry run by default: prints what it would store.

    python packages/ai-db-scene/eval/teacher.py --jsonl /tmp/bench.jsonl [--memory] [--write]
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from typing import Any

from ai_db_scene.layout import LayoutError, solve_layout
from ai_db_scene.repair import relax
from ai_db_scene.spec import validate_spec

_MAX_ROUNDS = 8  # validate/fix and solve/relax rounds per spec
_PATH = re.compile(r"^(setting|objects\[(\d+)\]|relations\[(\d+)\])(?:\.(\w+))?")
_MEAN = re.compile(r"Did you mean '([^']+)'")


def _fix_errors(data: dict[str, Any], errors: list[str]) -> list[str]:
    """Apply the obvious fix for each validate_spec error; returns what changed (empty: stuck)."""
    notes: list[str] = []
    drop_rel: set[int] = set()
    drop_obj: set[int] = set()
    for err in errors:
        m = _PATH.match(err)
        if m is None:
            continue
        head, oi, ri, field = m.group(1), m.group(2), m.group(3), m.group(4)
        if ri is not None:
            drop_rel.add(int(ri))  # a bad relation: the solver does fine with one fewer
            continue
        target = data.get("setting") if head == "setting" else (
            data["objects"][int(oi)] if oi is not None and int(oi) < len(data.get("objects", [])) else None)
        if not isinstance(target, dict):
            continue
        suggestion = _MEAN.search(err)
        if field and suggestion and field in target:
            notes.append(f"{head}.{field}: {target[field]!r} -> {suggestion.group(1)!r}")
            target[field] = suggestion.group(1)
        elif field and "unknown field" in err and field in target:
            notes.append(f"{head}: dropped unknown field {field}")
            del target[field]
        elif field and field in target and field not in ("id", "category", "kind", "bounds_m"):
            notes.append(f"{head}: dropped invalid {field}")
            del target[field]
        elif oi is not None:
            drop_obj.add(int(oi))
    objs = data.get("objects", [])
    gone = {objs[i].get("id") for i in drop_obj if i < len(objs) and isinstance(objs[i], dict)}
    for i in sorted(drop_obj, reverse=True):
        if i < len(objs):
            notes.append(f"dropped object {objs[i].get('id') if isinstance(objs[i], dict) else i}")
            del objs[i]
    rels = data.get("relations", [])
    for i, r in enumerate(rels):  # relations of a dropped object go with it
        if isinstance(r, dict) and (r.get("a") in gone or r.get("b") in gone):
            drop_rel.add(i)
    for i in sorted(drop_rel, reverse=True):
        if i < len(rels):
            notes.append(f"dropped relation {rels[i]}")
            del rels[i]
    return notes


def correct_spec(raw: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """(a spec that validates and solves, what was changed) or (None, why it could not be)."""
    if not isinstance(raw, dict):
        return None, ["no spec"]
    data = copy.deepcopy(raw)
    notes: list[str] = []
    for _ in range(_MAX_ROUNDS):
        spec, errors = validate_spec(data)
        if spec is not None:
            break
        changed = _fix_errors(data, errors)
        if not changed:
            return None, notes + errors[:5]
        notes += changed
    else:
        return None, notes + ["still invalid"]
    data = spec.to_dict()
    original = list(spec.setting.bounds_m)
    for _ in range(_MAX_ROUNDS):
        try:
            solve_layout(spec)
            return data, notes
        except LayoutError as e:
            fix = relax(data, e, original)
            if fix is None:
                return None, notes + [f"layout: {e}"]
            data, note = fix
            notes.append(note)
            spec, errors = validate_spec(data)
            if spec is None:
                return None, notes + errors[:5]
    return None, notes + ["layout still conflicts"]


def failures_from_jsonl(path: str) -> list[dict[str, Any]]:
    """scene_bench.py rows that did not end clean and still have a spec."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            failed = row.get("stage") != "done" or row.get("blocking_issues") or row.get("spec_errors")
            if failed and isinstance(row.get("spec"), dict):
                out.append({"brief": row.get("brief", ""), "spec": row["spec"], "source": path,
                            "errors": (row.get("spec_errors") or []) + (row.get("layout_conflicts") or [])})
    return out


def failures_from_memory(memory: Any, project: str | None = None) -> list[dict[str, Any]]:
    """Failed or partial runs in the scene memory (build_scene records every build)."""
    return [{"brief": r.brief, "spec": r.spec, "source": f"run {r.run_id}",
             "errors": [i.get("message") for rep in r.validation_reports if isinstance(rep, dict)
                        for i in rep.get("issues_left", []) if isinstance(i, dict)]}
            for r in memory.backend.list_runs(project=project, limit=1000)
            if r.outcome != "success" and r.spec and r.brief]


def teach(failures: list[dict[str, Any]], write: bool, project: str = "default",
          score: float = 4.0) -> list[dict[str, Any]]:
    """Correct each failure; with ``write``, store it through the record_run tool."""
    from ai_db_scene.tools.scene_pipeline import _handle_record_run

    out = []
    for f in failures:
        spec, notes = correct_spec(f["spec"])
        item = {"brief": f["brief"], "source": f["source"], "corrected": spec is not None, "changes": notes}
        if spec is not None and write:
            item["run_id"] = _handle_record_run({
                "brief": f["brief"], "spec": spec, "outcome": "success", "project": project,
                "validation_reports": [{"teacher": notes, "original_errors": f["errors"][:20]}],
                "judge_scores": [{"overall": score, "source": "teacher"}]})["run_id"]
        out.append(item)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jsonl", action="append", default=[], help="scene_bench.py output (repeatable)")
    ap.add_argument("--memory", action="store_true", help="also failed/partial runs of the scene memory")
    ap.add_argument("--project", default="default")
    ap.add_argument("--score", type=float, default=4.0, help="exemplar score of a corrected spec")
    ap.add_argument("--write", action="store_true", help="store the corrected specs (default: dry run)")
    args = ap.parse_args(argv)
    failures = [f for p in args.jsonl for f in failures_from_jsonl(p)]
    if args.memory:
        from ai_db_scene.tools.scene_tools import get_scene_memory

        failures += failures_from_memory(get_scene_memory(), args.project)
    results = teach(failures, args.write, args.project, args.score)
    for r in results:
        print(json.dumps(r))
    n = sum(r["corrected"] for r in results)
    print(f"{len(results)} failing runs, {n} corrected, {'stored' if args.write else 'dry run: nothing stored'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
