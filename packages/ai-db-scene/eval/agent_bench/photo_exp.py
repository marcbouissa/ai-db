"""Photo -> scene with a local vision model (Marc's Oval Office experiment, 2026-10-21).

The builder is always Qwen2.5-VL-7B (local_model.sh, LOCAL_MODEL=qwen2.5-vl-7b) and always gets
the reference photo (bench/_oval_photo/reference.jpg) in its first message. The cells differ in
the text that comes with it (bench/_oval_photo/prompts.json, written by the lead):

* A: the ``basic`` request.
* B: a brief the same model wrote first from the photo + ``basic`` + ``deepen_instruction`` (one
  call per repeat, saved in deepen_r<N>.json with its time and tokens; B-raw and B-aidb of a
  repeat build from the same brief).
* C: the lead's ``lead_deep`` brief.

Each cell runs raw (Blender tools only) and aidb (Blender + ai-db, the aidb7 prompt hint). Arms
are ``oval-photo-<cell>-<raw|aidb>-qwen2.5vl-7b-r<N>``, the brief is ``oval_photo`` (its checklist
from prompts.json; the "oval room" item is the room's shape, see room_shape), so a scene is
``bench.py run <arm> oval_photo`` like any other and ``bench.py rescore`` scores it.

    python photo_exp.py deepen <repeat>              the B brief of a repeat (server running)
    python photo_exp.py run [--repeats N] [--workers N] [--last-start HH:MM]
                                                     deepen calls first, then every build not yet done
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import briefs

REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PHOTO_DIR = os.path.join(REPO, "bench", "_oval_photo")
PROMPTS = os.path.join(PHOTO_DIR, "prompts.json")
IMAGE = os.path.join(PHOTO_DIR, "reference.jpg")
BRIEF_ID = "oval_photo"
MODEL_KEY = "qwen2.5vl-7b"  # the arm's model part (bench.LOCAL_MODELS)
SERVED = "qwen2.5-vl-7b"  # local_model.sh's served model name
BASE_URL = os.environ.get("LOCAL_MODEL_BASE_URL", "http://127.0.0.1:8011/v1")
CELLS = ("A", "B", "C")
MODES = ("raw", "aidb")
TIMEOUT_S = 20 * 60  # per build (the standard bench gives 30)
DEEPEN_MAX_TOKENS = 1024
_ARM = re.compile(r"^oval-photo-([ABC])-(raw|aidb)-(" + re.escape(MODEL_KEY) + r")-r(\d+)$")

PHOTO_NOTE = "The photo attached to this message is the reference: recreate what it shows."


def load_prompts(path: str = PROMPTS) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def register(path: str = PROMPTS) -> None:
    """briefs.BRIEFS[oval_photo]: the basic request as the brief text, the lead's checklist."""
    if BRIEF_ID in briefs.BRIEFS or not os.path.exists(path):
        return
    p = load_prompts(path)
    briefs.BRIEFS[BRIEF_ID] = {"brief": p["basic"],
                               "checklist": [(label, int(n), tuple(toks)) for label, n, toks in p["checklist"]]}


def arm_name(cell: str, mode: str, repeat: int) -> str:
    return f"oval-photo-{cell}-{mode}-{MODEL_KEY}-r{repeat}"


def parse_arm(arm: str) -> dict[str, Any] | None:
    """The photo arm's cell, mode, model and repeat; None for any other arm."""
    m = _ARM.match(arm)
    if not m:
        return None
    return {"cell": m.group(1), "mode": m.group(2), "with_aidb": m.group(2) == "aidb", "model": m.group(3),
            "repeat": int(m.group(4))}


def deepen_path(repeat: int) -> str:
    return os.path.join(PHOTO_DIR, f"deepen_r{repeat}.json")


def build_text(cell: str, repeat: int, prompts: dict[str, Any] | None = None) -> str:
    """The text that comes with the photo in a cell."""
    p = prompts or load_prompts()
    if cell == "A":
        return p["basic"]
    if cell == "C":
        return p["lead_deep"]
    path = deepen_path(repeat)
    if not os.path.exists(path):
        raise SystemExit(f"cell B needs {path}: run `photo_exp.py deepen {repeat}` first")
    with open(path, encoding="utf-8") as f:
        text = json.load(f)["text"].strip()
    if not text:
        raise SystemExit(f"{path} has an empty brief")
    return text


def prompt(cell: str, with_aidb: bool, repeat: int, prompts: dict[str, Any] | None = None) -> str:
    """The bench's aidb6/aidb7 prompt (strata runner names) with the photo line before the task."""
    text = PHOTO_NOTE + "\n\n" + briefs.TASK.format(brief=build_text(cell, repeat, prompts)) + "\n\n"
    b = [briefs.tool_name("blender", t, "strata") for t in ("discover_servers", "discover_tools", "get_tool_schema",
                                                             "invoke_tool")]
    text += f"Blender tools are reached through {' / '.join(b)}."
    return text + (" " + briefs.hint_aidb6("strata") if with_aidb else "")


def _post(base_url: str, body: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def deepen(repeat: int, base_url: str = BASE_URL, model: str = SERVED, image: str = IMAGE,
           prompts: dict[str, Any] | None = None, post: Any = None) -> dict[str, Any]:
    """Cell B's first step: the model looks at the photo with the basic request and the deepen
    instruction and writes the brief its builds get. Saved (text, seconds, tokens) per repeat."""
    import strata_agent

    p = prompts or load_prompts()
    out = deepen_path(repeat)
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            return json.load(f)
    user = p["basic"] + "\n\n" + p["deepen_instruction"]
    body = {"model": model, "temperature": 0.2, "max_tokens": DEEPEN_MAX_TOKENS,
            "messages": [{"role": "user", "content": strata_agent.user_content(user, image)}]}
    t = time.monotonic()
    data = (post or _post)(base_url, body)
    seconds = time.monotonic() - t
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    rec = {"repeat": repeat, "model": model, "prompt": user, "text": (choice["message"].get("content") or "").strip(),
           "finish_reason": choice.get("finish_reason"), "seconds": round(seconds, 2),
           "tokens": {"input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0)},
           "words": len((choice["message"].get("content") or "").split()),
           "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return rec


# --- room shape: the "oval room" checklist item ---------------------------------------------------

#: name / category words of the room's shell; walls decide when there are any, else floor / room
WALL_WORDS = ("wall", "walls", "wainscot", "wainscoting", "baseboard")
FLOOR_WORDS = ("floor", "room", "parquet")
NOT_SHELL = ("clock", "shelf", "shelves", "sconce", "lamp", "light", "painting", "picture", "frame", "mirror",
             "art", "niche", "flag", "cabinet", "plate", "socket", "switch", "vent", "outlet", "rug", "carpet")

#: Blender code: world XY of the shell's (evaluated) vertices, rounded to 5 cm and de-duplicated
ROOM_POINTS_CODE = """
import bpy, re
scene = bpy.context.scene
dg = bpy.context.evaluated_depsgraph_get()
def words(o):
    s = (o.name + " " + str(o.get("category") or "")).lower()
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    return set(w for w in re.split(r"[^a-z]+", s) if w)
groups = {"wall": [], "floor": []}
for o in scene.objects:
    if o.type not in ("MESH", "CURVE") or o.name == "bench_probe":
        continue
    w = words(o)
    if w & NOT_SHELL:
        continue
    key = "wall" if w & WALL_WORDS else "floor" if w & FLOOR_WORDS else None
    if key is None:
        continue
    ev = o.evaluated_get(dg)
    try:
        me = ev.to_mesh()
    except Exception:
        continue
    if me is None:
        continue
    mw = o.matrix_world
    pts = set()
    for v in me.vertices:
        p = mw @ v.co
        pts.add((round(p.x / 0.05) * 0.05, round(p.y / 0.05) * 0.05))
    ev.to_mesh_clear()
    groups[key].append({"name": o.name, "points": sorted(pts)})
result = groups
"""


def room_points_code() -> str:
    return (f"WALL_WORDS = set({list(WALL_WORDS)!r})\nFLOOR_WORDS = set({list(FLOOR_WORDS)!r})\n"
            f"NOT_SHELL = set({list(NOT_SHELL)!r})\n" + ROOM_POINTS_CODE)


def convex_hull(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Counter-clockwise hull (monotone chain), collinear points dropped."""
    p = sorted(set(pts))
    if len(p) < 3:
        return p

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list[tuple[float, float]] = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def _area(poly: list[tuple[float, float]]) -> float:
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))) / 2


def _min_rect(hull: list[tuple[float, float]]) -> tuple[float, float, float]:
    """The smallest enclosing rectangle (area, long side, short side): one side lies on a hull edge."""
    best = (math.inf, 0.0, 0.0)
    for a, b in zip(hull, hull[1:] + hull[:1]):
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        c, s = math.cos(ang), math.sin(ang)
        xs = [c * x + s * y for x, y in hull]
        ys = [-s * x + c * y for x, y in hull]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w * h < best[0]:
            best = (w * h, max(w, h), min(w, h))
    return best


def classify_outline(pts: list[tuple[float, float]]) -> dict[str, Any]:
    """rect / ellipse / polygon / none from the shell's plan points. An ellipse fills pi/4 (0.785)
    of its bounding rectangle and its hull turns in many small steps; a rectangle fills it all
    with four 90 deg corners; anything else (an octagon, an L, a wedge) is a polygon."""
    hull = convex_hull([(float(x), float(y)) for x, y in pts])
    if len(hull) < 3:
        return {"shape": "none", "hull_vertices": len(hull)}
    area = _area(hull)
    rect, long_side, short_side = _min_rect(hull)
    if rect <= 1e-6 or short_side < 1.0:
        return {"shape": "none", "hull_vertices": len(hull), "size_m": [round(long_side, 2), round(short_side, 2)]}
    turns = []
    n = len(hull)
    for i in range(n):
        a, b, c = hull[i - 1], hull[i], hull[(i + 1) % n]
        t = math.degrees(abs(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(b[1] - a[1], b[0] - a[0])))
        turns.append(min(t, 360 - t))
    corners = sum(1 for t in turns if t > 20)
    fill = area / rect
    if fill >= 0.93:
        shape = "rect"
    elif 0.70 <= fill < 0.90 and corners == 0 and n >= 10:
        shape = "ellipse"
    else:
        shape = "polygon"
    return {"shape": shape, "fill": round(fill, 3), "hull_vertices": n, "corners": corners,
            "max_turn_deg": round(max(turns), 1), "size_m": [round(long_side, 2), round(short_side, 2)]}


def room_shape(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """The room's shape from ROOM_POINTS_CODE's result: from the walls when there are any (a
    rectangular floor plate under oval walls is still an oval room), else from the floor."""
    for key in ("wall", "floor"):
        objs = groups.get(key) or []
        pts = [tuple(p) for o in objs for p in o["points"]]
        if len(pts) >= 3:
            out = classify_outline(pts)  # type: ignore[arg-type]
            out.update({"from": key, "objects": [o["name"] for o in objs][:20]})
            return out
    return {"shape": "none", "from": None, "objects": []}


def probe_room_shape(client: Any) -> dict[str, Any]:
    """room_shape of the scene in a running Blender (bench.cmd_run, before the Blender stops)."""
    try:
        return room_shape(client.run_code(room_points_code()))
    except Exception as e:  # noqa: BLE001 - a failed probe must not lose the run
        return {"shape": "none", "error": str(e)[:300]}


# --- the experiment's runner -----------------------------------------------------------------------


def builds(repeats: int) -> list[str]:
    """Every build, repeat by repeat, cells interleaved and raw/aidb alternated, so parallel
    slots mix arms (no cell gets all the contention)."""
    out = []
    for r in range(1, repeats + 1):
        order = [(c, m) for c in CELLS for m in MODES]
        if r % 2 == 0:
            order = order[::-1]
        out += [arm_name(c, m, r) for c, m in order]
    return out


def _done(arm: str) -> bool:
    return os.path.exists(os.path.join(REPO, "bench", arm, BRIEF_ID, "metrics.json"))


def _log(line: str) -> None:
    with open(os.path.join(PHOTO_DIR, "runs.log"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {line}\n")
    print(line, flush=True)


def _past(hhmm: str | None) -> bool:
    if not hhmm:
        return False
    h, m = map(int, hhmm.split(":"))
    now = time.localtime()
    return (now.tm_hour, now.tm_min) >= (h, m) and now.tm_hour < 12  # an early-morning deadline


def run_all(repeats: int = 2, workers: int = 3, last_start: str | None = None) -> None:
    for r in range(1, repeats + 1):
        if not os.path.exists(deepen_path(r)):
            rec = deepen(r)
            _log(f"deepen r{r}: {rec['seconds']} s, {rec['tokens']}, {rec['words']} words")
    todo = [a for a in builds(repeats) if not _done(a)]
    py = os.path.join(REPO, ".venv", "bin", "python")
    running: dict[str, subprocess.Popen] = {}
    while todo or running:
        for arm, proc in list(running.items()):
            if proc.poll() is not None:
                _log(f"done  {arm} rc={proc.returncode}")
                del running[arm]
        while todo and len(running) < workers:
            if _past(last_start):
                _log(f"not started (past {last_start}): {todo}")
                todo = []
                break
            arm = todo.pop(0)
            logf = open(os.path.join(PHOTO_DIR, f"run_{arm}.log"), "w")  # noqa: SIM115 - the child's
            running[arm] = subprocess.Popen([py, os.path.join(HERE, "bench.py"), "run", arm, BRIEF_ID],
                                            cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
            _log(f"start {arm}")
        time.sleep(5)
    _log("all done")


def main(argv: list[str]) -> int:
    if argv and argv[0] == "deepen":
        print(json.dumps(deepen(int(argv[1])), indent=1))
        return 0
    if argv and argv[0] == "run":
        args = argv[1:]

        def opt(name: str, default: str | None) -> str | None:
            return args[args.index(name) + 1] if name in args else default

        run_all(int(opt("--repeats", "2") or 2), int(opt("--workers", "3") or 3), opt("--last-start", None))
        return 0
    raise SystemExit(__doc__)


register()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
