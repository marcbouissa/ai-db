"""Scene pipeline benchmark: a small local model writes SceneSpecs; the pipeline does the rest.

Per brief: spec (model, with up to REPAIR_ROUNDS error-feedback rounds) -> validate ->
solve_layout -> apply in Blender -> validate_scene -> snapshot -> clear -> restore ->
round-trip diff -> prompt search -> record_run -> exemplar search. One JSON line per brief.

Run (Blender with blender_ox listening on :9877, scratch DBs only):
    python packages/ai-db-scene/eval/scene_bench.py --out /tmp/bench.jsonl \
        --base-url http://10.160.30.12:20000/v1 [--briefs eval/briefs.txt] [--only N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request
from typing import Any

REPAIR_ROUNDS = 2
POS_TOL = 1e-3

BRIEFS = [
    "a small home office with a desk, an office chair, a bookshelf and a floor lamp",
    "a cozy living room: sofa facing a tv stand, coffee table in front of the sofa, rug, two armchairs",
    "a bedroom with a double bed against the wall, two nightstands beside it and a wardrobe",
    "a dining room with a table, six chairs around it and a sideboard",
    "a kitchen with counters along one wall, a fridge, an island and three bar stools",
    "a backyard patio with a grill, an outdoor table with four chairs and two potted plants",
    "a classroom with a teacher desk, a whiteboard and twelve student desks in rows",
    "a reading nook: armchair next to a window, side table, floor lamp and a bookshelf",
    "a small cafe with four round tables, eight chairs, a counter and a plant",
    "a garage workshop with a workbench, a tool cabinet, shelves and a car-sized empty space",
]


HARD_BRIEFS = [
    "a round lobby with a reception desk in the middle, four armchairs around a low table and two plants",
    "an L-shaped open-plan studio: a kitchen corner with counters, a bed area and a sofa facing a tv",
    "a park corner with three benches facing a fountain, two trees and a trash bin",
    "a tabletop still life: a vase, three apples and a book on a small table",
    "a single hero armchair in a photo studio with two softbox lights",
    "a library reading room with eight bookshelves along the walls and four tables with four chairs each",
    "a conference room with a long table, ten chairs around it, a screen on the wall and a plant",
    "a hotel bathroom with a bathtub, a sink with a mirror above it, a toilet and a towel rack",
    "a home gym with a treadmill, a bench, a rack of dumbbells and a mirror on one wall",
    "a children's bedroom with a bunk bed, a desk with a chair, a toy chest on a rug and a bookshelf",
]


def chat(base_url: str, model: str, messages: list[dict[str, str]], timeout: float) -> tuple[str, dict[str, Any]]:
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.2,
                       "max_tokens": 6000}).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"] or "", data.get("usage", {})


def extract_json(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


def call(d: Any, tool: str, /, **args: Any) -> Any:
    return d.execute(tool, args)


def transforms(info: Any) -> dict[str, tuple[float, ...]]:
    out: dict[str, tuple[float, ...]] = {}
    if isinstance(info, dict) and isinstance(info.get("result"), dict):
        info = info["result"]  # blender_ox replies {"status", "result": {...}}
    for o in (info or {}).get("objects", []):
        loc = o.get("location") or [0, 0, 0]
        rot = o.get("rotation") or o.get("rotation_euler") or [0, 0, 0]
        scl = o.get("scale") or [1, 1, 1]
        out[o["name"]] = tuple(float(v) for v in (*loc, *rot, *scl))
    return out


def roundtrip_diff(before: dict[str, tuple[float, ...]], after: dict[str, tuple[float, ...]]) -> dict[str, Any]:
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    moved = [n for n in set(before) & set(after)
             if max(abs(a - b) for a, b in zip(before[n], after[n])) > POS_TOL]
    return {"objects": len(before), "missing": missing[:10], "n_missing": len(missing),
            "extra": extra[:10], "n_extra": len(extra), "moved": sorted(moved)[:10], "n_moved": len(moved)}


def versioning(d: Any, brief: str, v1: str, placements: list[dict[str, Any]]) -> dict[str, Any]:
    """Edit -> snapshot as v2 of v1 -> version chain -> restore v1 undoes the edit -> fork."""
    out: dict[str, Any] = {}
    target = placements[0]["id"]
    before = transforms(call(d, "scene_snapshot"))
    loc = list(before[target][:3])
    moved = call(d, "modify_object", name=target, location=[loc[0] + 0.5, loc[1], loc[2]])
    out["edit_ok"] = moved.get("status") == "success"
    snap = call(d, "snapshot_scene", prompt=f"{brief} (moved {target})", project="bench",
                parent_scene_id=v1)
    v2 = snap.get("scene_id")
    out["v2_error"] = snap.get("error") or (None if v2 else str(snap)[:200])
    if v2:
        chain = [v.get("scene_id") for v in call(d, "get_scene_versions", scene_id=v2).get("versions", [])]
        out["chain_ok"] = chain[:2] == [v2, v1]
        res = call(d, "restore_scene", scene_id=v1)
        after = transforms(call(d, "scene_snapshot"))
        out["undo_ok"] = res.get("status") == "restored" and \
            max(abs(a - b) for a, b in zip(after[target][:3], loc)) < POS_TOL
        fork = call(d, "fork_scene", from_scene_id=v2, new_prompt=f"{brief} (variant)", project="bench")
        fres = call(d, "restore_scene", scene_id=fork.get("scene_id", "?"))
        after = transforms(call(d, "scene_snapshot"))
        out["fork_ok"] = fres.get("status") == "restored" and \
            abs(after[target][0] - (loc[0] + 0.5)) < POS_TOL
        out["fork_report"] = {k: fres.get(k) for k in ("status", "mismatches", "error", "message")}
    return out


def run_brief(d: Any, brief: str, args: argparse.Namespace, rec: dict[str, Any]) -> dict[str, Any]:
    rec.update(brief=brief, stage="spec")
    t0 = time.time()
    schema = call(d, "get_scene_spec_schema", brief=brief)
    system = ("You write SceneSpec JSON for a 3D layout engine. Reply with ONE JSON object and "
              "nothing else. Use only categories from the vocabulary. Units are metres.")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Brief: {brief}\n\nSchema, vocabulary and examples:\n"
                                            f"{json.dumps(schema)}"}]
    spec, errors, tokens = None, ["not attempted"], 0
    if args.replay_specs is not None:  # specs a model already wrote: no model calls at all
        spec = args.replay_specs.get(brief)
        errors = [] if spec else ["no replayed spec for this brief"]
    for attempt in range(0 if args.replay_specs is not None else REPAIR_ROUNDS + 1):
        reply, usage = chat(args.base_url, args.model, messages, args.timeout)
        tokens += int(usage.get("total_tokens", 0))
        try:
            candidate = extract_json(reply)
        except (ValueError, json.JSONDecodeError) as exc:
            errors = [f"json: {exc}"]
        else:
            res = call(d, "submit_scene_spec", spec=candidate)
            errors = list(res.get("errors") or []) if isinstance(res, dict) else ["bad response"]
            if not errors and isinstance(res, dict) and res.get("ok", True):
                spec = candidate
        rec["spec_attempts"] = attempt + 1
        if spec is not None:
            break
        messages += [{"role": "assistant", "content": reply[-4000:]},
                     {"role": "user", "content": "The spec is invalid:\n- " + "\n- ".join(errors[:15]) +
                                                 "\nReturn the corrected full JSON object only."}]
    rec.update(spec_errors=errors[:15], model_tokens=tokens)
    if spec is None:
        rec["wall_s"] = round(time.time() - t0, 1)
        return rec
    rec["n_objects_spec"] = len(spec.get("objects", []))
    rec["n_relations"] = len(spec.get("relations", []))

    rec["stage"] = "layout"
    rec["layout_conflicts"] = []
    placements = None
    for attempt in range(REPAIR_ROUNDS + 1):
        lay = call(d, "solve_layout", spec=spec, seed=0)
        placements = lay.get("placements")
        rec["layout_attempts"] = attempt + 1
        if placements:
            break
        msg = lay.get("message") or "; ".join(lay.get("errors") or [])
        rec["layout_conflicts"].append(msg[:300])
        if attempt == REPAIR_ROUNDS or args.replay_specs is not None:
            break
        # The agent loop: the solver's conflict goes back to the model for a revised spec.
        messages += [{"role": "assistant", "content": json.dumps(spec)},
                     {"role": "user", "content": f"The layout solver could not place this spec: {msg}\n"
                                                 f"Conflicting relations: {json.dumps(lay.get('conflicts'))[:1500]}\n"
                                                 "Return a corrected full JSON object only (bigger room, fewer "
                                                 "or smaller objects, or looser relations)."}]
        reply, usage = chat(args.base_url, args.model, messages, args.timeout)
        tokens += int(usage.get("total_tokens", 0))
        try:
            candidate = extract_json(reply)
        except (ValueError, json.JSONDecodeError):
            continue
        res = call(d, "submit_scene_spec", spec=candidate)
        if res.get("status") == "ok":
            spec = candidate
    rec["model_tokens"] = tokens
    rec["spec"] = spec
    if not placements:
        rec["wall_s"] = round(time.time() - t0, 1)
        return rec
    rec["n_placements"] = len(placements)

    rec["stage"] = "build"
    call(d, "clear_scene", keep_camera=False, purge_orphans=True)
    applied = call(d, "apply_layout", spec=spec, placements=placements)
    rec["apply_error"] = applied.get("error") if isinstance(applied, dict) else None
    val = call(d, "validate_scene", spec=spec)
    if val.get("status") != "success":
        rec["validate_error"] = str(val.get("message") or val.get("errors"))[:300]
    issues = (val.get("result") or {}).get("issues", [])
    rec["validate_issues"] = len(issues)
    rec["blocking_issues"] = sum(1 for i in issues if i.get("blocking") or i.get("severity") == "blocking")
    kinds: dict[str, int] = {}
    for i in issues:
        k = str(i.get("check") or i.get("kind") or i.get("type"))
        kinds[k] = kinds.get(k, 0) + 1
    rec["issue_kinds"] = kinds

    rec["stage"] = "persist"
    before = transforms(call(d, "scene_snapshot"))
    snap = call(d, "snapshot_scene", prompt=brief, project="bench", tags=["bench"])
    scene_id = snap.get("scene_id") if isinstance(snap, dict) else None
    rec["snapshot"] = {k: snap.get(k) for k in ("scene_id", "object_count", "error")} if isinstance(snap, dict) else snap
    if scene_id:
        call(d, "clear_scene", keep_camera=False, purge_orphans=True)
        restored = call(d, "restore_scene", scene_id=scene_id)
        rec["restore"] = {k: restored.get(k) for k in ("status", "restored", "error", "mismatches")} \
            if isinstance(restored, dict) else restored
        rec["roundtrip"] = roundtrip_diff(before, transforms(call(d, "scene_snapshot")))
        hits = call(d, "search_scenes_by_prompt", query=brief.split(":")[0][:60], project="bench", top_k=5)
        ids = [h.get("scene_id") for h in hits.get("scenes", [])]
        rec["search_rank"] = ids.index(scene_id) + 1 if scene_id in ids else None

    if scene_id and rec.get("roundtrip", {}).get("objects"):
        rec["stage"] = "versions"
        rec["versions"] = versioning(d, brief, scene_id, placements)

    rec["stage"] = "record"
    score = 1.0 if rec["blocking_issues"] == 0 else 0.0
    saved = call(d, "record_run", brief=brief, spec=spec, placements=placements,
                 judge_scores=[{"overall": 5.0 if score else 1.0, "source": "validator"}],
                 token_cost=tokens, wall_time_s=time.time() - t0, outcome="success" if score else "failed")
    rec["record_error"] = saved.get("error") if isinstance(saved, dict) else saved
    ex = call(d, "search_exemplars", brief=brief, min_score=0.0, top_k=3)
    rec["exemplars"] = ex.get("count")
    rec["stage"] = "done"
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://10.160.30.12:20000/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next-iq3_s")
    ap.add_argument("--scratch", default=os.path.dirname(os.path.abspath(__file__)) + "/.bench")
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--set", choices=("base", "hard", "all"), default="base")
    ap.add_argument("--replay", default=None,
                    help="JSONL of an earlier run: reuse its specs (no model calls; layout repair is skipped)")
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()

    args.replay_specs = None
    if args.replay:
        with open(args.replay, encoding="utf-8") as f:
            args.replay_specs = {r["brief"]: r["spec"] for r in map(json.loads, f) if r.get("spec")}
    os.makedirs(args.scratch, exist_ok=True)
    os.environ["AI_DB_SCENE_MEMORY"] = os.path.join(args.scratch, "scene_memory.db")
    from ai_db.config import parse_config
    from ai_db.config_template import build_template
    from ai_db.dispatcher import ServiceDispatcher

    d = ServiceDispatcher(db_path=os.path.join(args.scratch, "code.db"),
                          config=parse_config(build_template()))
    pool = {"base": BRIEFS, "hard": HARD_BRIEFS, "all": BRIEFS + HARD_BRIEFS}[args.set]
    briefs = pool[: args.only] if args.only else pool
    with open(args.out, "a", encoding="utf-8") as out:
        for brief in briefs:
            rec: dict[str, Any] = {}
            try:
                run_brief(d, brief, args, rec)
            except Exception as exc:  # noqa: BLE001 - a benchmark records failures, it does not stop
                import traceback
                frames = traceback.extract_tb(exc.__traceback__)[-4:]
                rec.update(crash=f"{type(exc).__name__}: {exc}"[:300],
                           where=[f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in frames])
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(json.dumps({k: rec.get(k) for k in ("brief", "stage", "spec_attempts", "validate_issues",
                                                      "blocking_issues", "roundtrip", "search_rank", "versions", "crash", "where")})[:700],
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
