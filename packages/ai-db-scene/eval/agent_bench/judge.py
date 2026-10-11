"""Judge one benchmark scene twice with a free vision model (OpenCode), and record what each
judging cost. Opus reviews these verdicts afterwards (finalists and spot checks).

- ``aidb``: ai-db's judge rubric (ai_db_scene.judge.RUBRIC, 1-5 per criterion) plus the
  measurements ai-db produced for the scene: validator issues, brief checklist, inventory.
- ``plain``: the renders and the brief only; one overall 1-10 score.

Both get the same renders attached; the judge session has no tools.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import briefs

JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL", "mimo-v2.6-flash-free")  # free, accepts images


def _ask(prompt: str, images: list[str], rd: str, label: str) -> dict[str, Any]:
    import run_opencode

    jd = os.path.join(rd, f"judge_{label}")
    os.makedirs(jd, exist_ok=True)
    t = time.monotonic()
    delay = 60.0
    while True:  # the free tier rate-limits: back off instead of failing the judgement
        try:
            res = run_opencode.run(JUDGE_MODEL, prompt, 0, jd, False, "", 900, files=images, servers={})
            break
        except run_opencode.ModelUnavailable as e:
            if "rate limit" not in str(e).lower() or delay > 900:
                raise
            time.sleep(delay)
            delay *= 2
    text = res.get("final_text") or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        parsed = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        parsed = {}
    return {"model": JUDGE_MODEL, "parsed": parsed, "raw": text[-2000:], "wall_s": round(time.monotonic() - t, 1),
            "cost_usd": 0.0, "tokens": res["tokens"], "calls": res["turns"]}


def judge_both(rd: str, brief: str) -> dict[str, Any]:
    from ai_db_scene.judge import RUBRIC

    with open(os.path.join(rd, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    images = list(m["renders"].values())
    text = briefs.BRIEFS[brief]["brief"]
    listing = "\n".join(f"- {p}" for p in images)
    out: dict[str, Any] = {}
    if not images:
        empty = {"parsed": {"overall": 1, "reason": "empty scene: nothing to render"}, "tokens": {}, "wall_s": 0}
        out = {"aidb": empty, "plain": empty}
    else:
        q = m["quality"]
        issues = [f"{i.get('check')}: {i.get('message')}" for i in (m["validation"] or {}).get("issues", [])][:25]
        rubric = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
        out["aidb"] = _ask(
            f"You judge a 3D scene built in Blender for this brief:\n{text}\n\nThe attached images are renders of it:\n{listing}\n\n"
            f"ai-db measured the scene:\n- checklist (found/needed): {json.dumps(q['checklist'])}\n"
            f"- validator: {q['blocking']} blocking, {q['warnings']} warnings: {json.dumps(issues)}\n"
            f"- meshes {q['meshes']}, triangles {q['triangles']}, camera {m['inventory'].get('camera')}, "
            f"lights {m['inventory'].get('lights')}\n\n"
            f"Score each criterion 1-5:\n{rubric}\nThen an overall score 1-10 for how well the scene "
            "delivers the brief. Reply with JSON only: {\"criteria\": {name: score}, \"overall\": n, "
            "\"issues\": [short strings]}", images, rd, "aidb")
        out["plain"] = _ask(
            f"The attached images are renders of a 3D scene built in Blender:\n{listing}\n\nThe brief was:\n{text}\n\n"
            "Score the scene 1-10 overall. Reply with JSON only: {\"overall\": n, \"reason\": \"one sentence\"}",
            images, rd, "plain")
    with open(os.path.join(rd, "judge.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return {k: {"overall": v["parsed"].get("overall"), "tokens": v["tokens"]} for k, v in out.items()}
