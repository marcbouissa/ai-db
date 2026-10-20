"""The photo experiment's report (photo_exp.py): bench/_oval_photo/REPORT.md tables and the
anonymised contact sheet the lead judges blind against the photo.

    python photo_report.py tables        markdown tables of every finished build (stdout)
    python photo_report.py sheet         blind_sheet.png + blind_key.json (random S01..S12 order)
"""

from __future__ import annotations

import json
import os
import random
import sys
from statistics import mean
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import photo_exp

OUT = os.path.join(photo_exp.REPO, "bench")


def runs(repeats: int = 2) -> list[dict[str, Any]]:
    """metrics.json of every finished photo build, with its arm parsed."""
    out = []
    for arm in photo_exp.builds(repeats):
        mp = os.path.join(OUT, arm, photo_exp.BRIEF_ID, "metrics.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            m["dir"] = os.path.dirname(mp)
            m["cellmode"] = f"{photo_exp.parse_arm(arm)['cell']}-{photo_exp.parse_arm(arm)['mode']}"
            out.append(m)
    return out


def transcript_stats(path: str) -> dict[str, int]:
    """Tool calls, errors, retries (a call to the tool that just failed), unknown tools."""
    calls = errors = retries = unknown = 0
    last_err: str | None = None
    if not os.path.exists(path):
        return {"calls": 0, "errors": 0, "retries": 0, "unknown": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "tool" not in e:
                continue
            calls += 1
            if last_err is not None and e["tool"] == last_err:
                retries += 1
            if e.get("error"):
                errors += 1
                unknown += str(e.get("result", "")).startswith(("unknown tool", "not run:"))
                last_err = e["tool"]
            else:
                last_err = None
    return {"calls": calls, "errors": errors, "retries": retries, "unknown": unknown}


#: shell, lights, cameras, terrain: not counted as the scene's objects
_SHELL = ("room_", "wall", "floor", "ceiling", "ai_db_", "ground", "moulding", "floor_line", "trim", "baseboard")


def object_count(m: dict[str, Any]) -> int:
    roots = [o for o in m["inventory"]["objects"] if o["parent"] is None and o["type"] not in ("LIGHT", "CAMERA")
             and o["name"] != "bench_probe"]
    return sum(1 for o in roots if not o["name"].lower().startswith(_SHELL)
               and str(o.get("category") or "") not in ("wall", "floor", "ceiling", "ground", "room"))


def row(m: dict[str, Any], deepen: dict[int, dict[str, Any]]) -> dict[str, Any]:
    a, q = m["agent"], m["quality"]
    p = m.get("photo") or photo_exp.parse_arm(m["arm"]) or {}
    d = deepen.get(p.get("repeat", 0)) if p.get("cell") == "B" else None
    ts = transcript_stats(os.path.join(m["dir"], "transcript.jsonl"))
    lay = q.get("layout") or {}
    return {"arm": m["arm"], "cell": m["cellmode"], "repeat": p.get("repeat"), "done": bool(a.get("ok")),
            "timed_out": bool(a.get("timed_out")), "deepen_s": d["seconds"] if d else 0.0,
            "build_s": a["wall_s"], "total_s": round(a["wall_s"] + (d["seconds"] if d else 0.0), 1),
            "model_s": a["model_s"], "tool_s": a["tool_s"], "turns": a["turns"],
            "tok_in": a["tokens"]["input"] + (d["tokens"]["input"] if d else 0),
            "tok_out": a["tokens"]["output"] + (d["tokens"]["output"] if d else 0),
            **ts, "malformed": a.get("malformed_calls", 0), "salvaged": a.get("salvaged_calls", 0),
            "rejected": bool(a.get("rejected")), "score": q["score"], "checklist_pct": q["checklist_pct"],
            "layout_pct": q.get("layout_pct"), "layout": {k: (v or {}).get("pass") for k, v in lay.items()},
            "room_shape": (m["inventory"].get("room_shape") or {}).get("shape", "?"),
            "room_fill": (m["inventory"].get("room_shape") or {}).get("fill"),
            "objects": object_count(m), "items": {i["item"]: (i["found"], i["need"]) for i in q["checklist"]},
            "blocking": q.get("blocking"), "warnings": q.get("warnings")}


def _deepens(repeats: int = 2) -> dict[int, dict[str, Any]]:
    out = {}
    for r in range(1, repeats + 1):
        p = photo_exp.deepen_path(r)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[r] = json.load(f)
    return out


def tables(repeats: int = 2) -> str:
    deep = _deepens(repeats)
    rows = [row(m, deep) for m in runs(repeats)]
    lines = ["### Per build", "",
             ("| build | done | total s (deepen+build) | model s | tool s | turns | tokens in/out | tool calls | errors "
              "| retries | malformed | score v3 | checklist % | layout % | room | objects |"),
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['cell']} r{r['repeat']} | {'yes' if r['done'] else ('timeout' if r['timed_out'] else 'no')} "
                     f"| {r['total_s']} ({r['deepen_s']}+{r['build_s']}) | {r['model_s']} | {r['tool_s']} | {r['turns']} "
                     f"| {r['tok_in']}/{r['tok_out']} | {r['calls']} | {r['errors']} | {r['retries']} | {r['malformed']} "
                     f"| {r['score']} | {r['checklist_pct']} | {r['layout_pct']} | {r['room_shape']} | {r['objects']} |")
    cells = sorted({r["cell"] for r in rows}, key=lambda c: (c[0], c[2:] != "raw"))
    lines += ["", "### Per cell (mean of repeats)", "",
              ("| cell | n | done | total s | model s | tool s | tokens in | tokens out | tool calls | errors | retries "
               "| score v3 | checklist % | layout % | oval rooms | objects |"),
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        rs = [r for r in rows if r["cell"] == c]

        def mn(k: str, rs: list[dict[str, Any]] = rs) -> str:
            vals = [r[k] for r in rs if r[k] is not None]
            return f"{mean(vals):.1f}" if vals else "–"

        lines.append(f"| {c} | {len(rs)} | {sum(r['done'] for r in rs)}/{len(rs)} | {mn('total_s')} | {mn('model_s')} "
                     f"| {mn('tool_s')} | {mn('tok_in')} | {mn('tok_out')} | {mn('calls')} | {mn('errors')} "
                     f"| {mn('retries')} | {mn('score')} | {mn('checklist_pct')} | {mn('layout_pct')} "
                     f"| {sum(r['room_shape'] == 'ellipse' for r in rs)}/{len(rs)} | {mn('objects')} |")
    labels = list(rows[0]["items"]) if rows else []
    lines += ["", "### Checklist hit rate per item (share of builds with found >= need; mean found/need)", "",
              "| item | need | " + " | ".join(cells) + " |", "|---|---|" + "---|" * len(cells)]
    for lab in labels:
        need = rows[0]["items"][lab][1]
        cols = []
        for c in cells:
            rs = [r for r in rows if r["cell"] == c]
            hit = sum(r["items"][lab][0] >= need for r in rs)
            cols.append(f"{hit}/{len(rs)} ({mean(r['items'][lab][0] for r in rs):.1f})")
        lines.append(f"| {lab} | {need} | " + " | ".join(cols) + " |")
    checks = sorted({k for r in rows for k in r["layout"]})
    lines += ["", "### Layout-sanity checks (pass / applicable)", "",
              "| check | " + " | ".join(cells) + " |", "|---|" + "---|" * len(cells)]
    for k in checks:
        cols = []
        for c in cells:
            vals = [r["layout"].get(k) for r in rows if r["cell"] == c]
            app = [v for v in vals if v is not None]
            cols.append(f"{sum(bool(v) for v in app)}/{len(app)}")
        lines.append(f"| {k} | " + " | ".join(cols) + " |")
    if deep:
        lines += ["", "### Deepen calls (cell B)", "", "| repeat | seconds | tokens in/out | words | finish |",
                  "|---|---|---|---|---|"]
        for r, d in sorted(deep.items()):
            lines.append(f"| r{r} | {d['seconds']} | {d['tokens']['input']}/{d['tokens']['output']} | {d['words']} "
                         f"| {d.get('finish_reason')} |")
    return "\n".join(lines) + "\n"


def sheet(repeats: int = 2) -> str:
    """blind_sheet.png: REF (the photo) then every build as S01..S12 in a random order, each its
    agent camera render beside its plan probe; the arm of each label only in blind_key.json."""
    from PIL import Image, ImageDraw, ImageFont

    builds = runs(repeats)
    random.SystemRandom().shuffle(builds)
    tw, th, lab = 640, 360, 44
    cols = 2
    rows_n = (len(builds) + cols - 1) // cols
    W = cols * 2 * tw + (cols + 1) * 20
    ref = Image.open(photo_exp.IMAGE).convert("RGB")
    ref_h = 2 * th
    ref = ref.resize((round(ref.width * ref_h / ref.height), ref_h))
    H = 20 + lab + ref_h + 20 + rows_n * (lab + th + 20)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 20), "REF", fill=(0, 0, 0), font=font)
    img.paste(ref, ((W - ref.width) // 2, 20 + lab))
    key = {}
    y0 = 20 + lab + ref_h + 20
    for k, m in enumerate(builds):
        label = f"S{k + 1:02d}"
        key[label] = m["arm"]
        x = 20 + (k % cols) * (2 * tw + 20)
        y = y0 + (k // cols) * (lab + th + 20)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
        for j, name in enumerate(("agent_camera.png", "probe_plan.png")):
            p = os.path.join(m["dir"], name)
            if os.path.exists(p):
                t = Image.open(p).convert("RGB")
                t.thumbnail((tw, th))
            else:
                t = Image.new("RGB", (tw, th), (200, 200, 200))
                ImageDraw.Draw(t).text((20, th // 2), "no camera" if j == 0 else "no render", fill=(0, 0, 0), font=font)
            img.paste(t, (x + j * tw, y + lab))
    path = os.path.join(photo_exp.PHOTO_DIR, "blind_sheet.png")
    img.save(path)
    with open(os.path.join(photo_exp.PHOTO_DIR, "blind_key.json"), "w", encoding="utf-8") as f:
        json.dump({"tiles": "left: the build's own camera (agent_camera.png), right: top-down plan probe",
                   "key": key}, f, indent=1)
    return path


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "tables":
        print(tables())
    elif cmd == "sheet":
        print(sheet())
    else:
        raise SystemExit(__doc__)
