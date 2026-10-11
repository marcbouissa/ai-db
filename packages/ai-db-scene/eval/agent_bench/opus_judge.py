"""Blind Opus judging, one judge per brief per mode, run as sub-agents of a Claude session.

    python opus_judge.py prepare     bench/_judging/<brief>/{manifest_plain,manifest_aidb}.json + key.json
    python opus_judge.py apply <agent transcripts dir>   results -> each scene's judge.json

Paired variant (cheaper: one batch per brief over only the given models' arms, judged into
judge_pairs.json so the full-field judge.json is kept):

    python opus_judge.py prepare --pairs nemotron      bench/_judging_pairs/<brief>/...
    python opus_judge.py apply --pairs '{...}'

Scenes are anonymised (S1, S2, ... shuffled per brief); key.json maps codes back to arms and
is never shown to a judge. The plain judge gets renders + brief; the aidb judge also gets the
ai-db rubric and ai-db's measurements (checklist, validator issues, inventory).
"""

from __future__ import annotations

import glob
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
import briefs  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "bench"))
JDIR = os.path.join(OUT, "_judging")


def _scenes(brief: str, models: str | None = None) -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(OUT, "*", brief))
                  if os.path.exists(os.path.join(d, "metrics.json")) and "/_" not in d
                  and (models is None or os.path.basename(os.path.dirname(d)).startswith(models)))


def prepare(models: str | None = None) -> None:
    from ai_db_scene.judge import RUBRIC

    for brief in briefs.BRIEFS:
        dirs = _scenes(brief, models)
        if not dirs:
            continue
        rng = random.Random(f"opus-judge-{brief}")
        rng.shuffle(dirs)
        key, plain, aidb = {}, [], []
        for i, d in enumerate(dirs, 1):
            code = f"S{i}"
            with open(os.path.join(d, "metrics.json"), encoding="utf-8") as f:
                m = json.load(f)
            key[code] = os.path.relpath(d, OUT)
            # copies under neutral names: the original paths name the model and the arm
            img_dir = os.path.join(_jdir(models), brief, "img")
            os.makedirs(img_dir, exist_ok=True)
            images = []
            for view, src in sorted(m["renders"].items()):
                dst = os.path.join(img_dir, f"{code}_{view}.png")
                shutil.copyfile(src, dst)
                images.append(dst)
            plain.append({"code": code, "images": images})
            q = m["quality"]
            issues = [f"{i.get('check')}: {i.get('message')}" for i in (m.get("validation") or {}).get("issues", [])][:20]
            aidb.append({"code": code, "images": images, "ai_db_measurements": {
                "checklist_found_vs_needed": q["checklist"], "validator_blocking": q["blocking"],
                "validator_warnings": q["warnings"], "validator_issues": issues, "meshes": q["meshes"],
                "triangles": q["triangles"], "camera": m["inventory"].get("camera"),
                "lights": m["inventory"].get("lights")}})
        bd = os.path.join(_jdir(models), brief)
        os.makedirs(bd, exist_ok=True)
        common = {"brief": briefs.BRIEFS[brief]["brief"],
                  "views": "agent_camera = the builder's own camera (absent if it set none); probe_interior = "
                           "eye height from a corner; probe_plan = top view with everything above 2.5 m cut away"}
        with open(os.path.join(bd, "manifest_plain.json"), "w", encoding="utf-8") as f:
            json.dump({**common, "scenes": plain}, f, indent=1)
        with open(os.path.join(bd, "manifest_aidb.json"), "w", encoding="utf-8") as f:
            json.dump({**common, "rubric_1_to_5": RUBRIC, "scenes": aidb}, f, indent=1)
        with open(os.path.join(bd, "key.json"), "w", encoding="utf-8") as f:
            json.dump(key, f, indent=1)
        print(brief, len(dirs), "scenes")


def _usage(transcript: str) -> dict[str, int]:
    """Tokens of a sub-agent transcript. A streamed message is logged several times with growing
    counts: keep each message's largest values, then sum over messages."""
    per: dict[str, dict[str, int]] = {}
    with open(transcript, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = r.get("message") or {}
            u, mid = msg.get("usage"), msg.get("id")
            if not u or not mid:
                continue
            cur = {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0),
                   "cache_write": u.get("cache_creation_input_tokens", 0),
                   "cache_read": u.get("cache_read_input_tokens", 0)}
            old = per.get(mid, {})
            per[mid] = {k: max(v, old.get(k, 0)) for k, v in cur.items()}
    return {k: sum(m[k] for m in per.values()) for k in ("input", "output", "cache_write", "cache_read")}


def _jdir(pairs: str | None) -> str:
    return os.path.join(OUT, "_judging_pairs") if pairs else JDIR


def apply(transcripts: dict[str, str], pairs: bool = False) -> None:
    """transcripts: "<brief>/<mode>" -> sub-agent transcript path (for the token cost)."""
    for brief in briefs.BRIEFS:
        bd = os.path.join(_jdir("pairs" if pairs else None), brief)
        if not os.path.exists(os.path.join(bd, "key.json")):
            continue
        with open(os.path.join(bd, "key.json"), encoding="utf-8") as f:
            key = json.load(f)
        per_scene: dict[str, dict] = {}
        for mode in ("plain", "aidb"):
            rp = os.path.join(bd, f"result_{mode}.json")
            if not os.path.exists(rp):
                continue
            with open(rp, encoding="utf-8") as f:
                res = json.load(f)
            tp = transcripts.get(f"{brief}/{mode}")
            tok = _usage(tp) if tp and os.path.exists(tp) else {}
            n = max(len(res.get("scenes", [])), 1)
            for s in res.get("scenes", []):
                arm_dir = key.get(s.get("code"))
                if not arm_dir:
                    continue
                per_scene.setdefault(arm_dir, {})[mode] = {
                    "model": "claude-opus-5-5", "parsed": s, "wall_s": 0,
                    "tokens": {k: v // n for k, v in tok.items()},  # the brief's judging cost, split evenly
                    "batch_tokens": tok, "batch_scenes": n}
        for arm_dir, judged in per_scene.items():
            path = os.path.join(OUT, arm_dir, "judge_pairs.json" if pairs else "judge.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(judged, f, indent=1)
        print(brief, len(per_scene), "scenes judged")


if __name__ == "__main__":
    pairs = "--pairs" in sys.argv
    args = [a for a in sys.argv[2:] if a != "--pairs"]
    if sys.argv[1] == "prepare":
        prepare(args[0] if pairs and args else None)
    elif sys.argv[1] == "apply":
        apply(json.loads(args[0]) if args else {}, pairs)
