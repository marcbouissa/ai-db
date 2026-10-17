"""Agent benchmark: {opus, sonnet, strata} x {with ai-db, without ai-db} x briefs.

    python bench.py run <arm> <brief>      one scene: fresh Blender, agent, save + renders, quality
    python bench.py judge <arm> <brief>    both judge modes (Opus, with and without ai-db context)
    python bench.py rescore                quality of every finished scene with the current score()
    python bench.py report                 bench/REPORT.md

Arms: <model>-aidb / <model>-raw, model = strata, one of OPENCODE_FREE, one of LOCAL_MODELS (start
its server with local_model.sh first), or opus/sonnet (Claude CLI).
Outputs (kept, never committed): <repo>/bench/<arm>/<brief>/{scene.blend, *.png, metrics.json,
transcript.jsonl, judge_*.json}.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
import blender
import briefs
import run_claude

REPO = run_claude.REPO
OUT = os.path.join(REPO, "bench")
OPENCODE_FREE = ("big-pickle", "longcat-2.5-preview-free", "mimo-v2.6-flash-free", "muse-spark-1.3-contributor-free",
                 "nemotron-3-ultra-free", "nemotron-3.5-lightning-free", "space-bunny-free")
#: offered by `opencode models` on 2026-10-11 but refused by the router ("Model unavailable")
OPENCODE_UNAVAILABLE = ("exo-free", "ling-3.0-flash-fin-free", "ling-3.1-flash-free", "step-5-preview-free")
#: models served on this GPU by local_model.sh (vLLM, OpenAI-compatible), run by the Strata agent:
#: OpenCode 2 reaches MCP tools only through Code Mode (JS in one `execute` tool, ~10k tokens of
#: system prompt), which a 14B model with 32k context could not drive in the smoke test.
LOCAL_MODELS = {"qwen3-14b-local": (os.environ.get("LOCAL_MODEL_BASE_URL", "http://127.0.0.1:8011/v1"), "qwen3-14b")}
TIMEOUT_S = 30 * 60
USER_SKILLS = os.path.expanduser("~/.local/share/blender-ai/skills")


def run_dir(arm: str, brief: str) -> str:
    return os.path.join(OUT, arm, brief)


#: port -> the open lock file holding it (flock: released by the kernel if this process dies)
_PORT_LOCKS: dict[int, Any] = {}


def reserve_port() -> int:
    """A Blender port no other (parallel) run holds: an exclusive flock per port, kept until
    release_port. The old pid-file check let two runs started together claim the same port
    (both read a freed "0"); the second Blender then failed to bind and its agent drove the
    first run's Blender (ultra-aidb4 living_room got classroom furniture)."""
    import fcntl

    lock_root = os.path.join(OUT, ".portlocks")
    os.makedirs(lock_root, exist_ok=True)
    for port in range(11000, 11400):
        f = open(os.path.join(lock_root, str(port)), "w")  # noqa: SIM115 - held until release_port
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            continue
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:  # something else listens there
                f.close()
                continue
        f.write(str(os.getpid()))
        f.flush()
        _PORT_LOCKS[port] = f
        return port
    raise RuntimeError("no free Blender port in 11000-11399")


def release_port(port: int) -> None:
    f = _PORT_LOCKS.pop(port, None)
    if f is not None:
        f.close()  # drops the flock


def _prepare_aidb(rd: str) -> dict[str, Any]:
    """Private ai-db for the run (lexical: no model load), with the Blender skills indexed."""
    from ai_db.config_template import build_template

    os.makedirs(os.path.join(rd, "aidb"), exist_ok=True)
    with open(os.path.join(rd, "aidb", "config.json"), "w", encoding="utf-8") as f:
        json.dump(build_template(), f)
    env = {**os.environ, **run_claude.aidb_env(0, rd, "")}
    env.pop("AI_DB_DAEMON_URL", None)
    from ai_db_scene import skills_dir

    t = time.monotonic()
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "ai-db"), "sync-skills", "--dir", skills_dir(),
                          "--dir", USER_SKILLS], env=env, capture_output=True, text=True, timeout=600, check=False)
    return {"seconds": round(time.monotonic() - t, 1), "rc": out.returncode, "tail": out.stdout[-300:]}


def aidb_version() -> str:
    """A short hash of the ai-db-scene code (and skills) a scene ran with: ai-db changes during
    the benchmark, and before/after results must not be mixed."""
    import hashlib

    root = os.path.join(REPO, "packages", "ai-db-scene", "ai_db_scene")
    h = hashlib.sha1()
    for dirpath, _dirs, files in sorted(os.walk(root)):
        for name in sorted(files):
            if name.endswith((".py", ".md", ".json")) and "__pycache__" not in dirpath:
                with open(os.path.join(dirpath, name), "rb") as f:
                    h.update(name.encode() + f.read())
    return h.hexdigest()[:10]


def _running_elsewhere(arm: str, brief: str) -> bool:
    """Another live process is already running this scene (a parallel queue, or an older one)."""
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        if any(a.endswith(b"bench.py") for a in argv) and argv[-3:-1] == [arm.encode(), brief.encode()] \
                and b"run" in argv:
            return True
    return False


def cmd_run(arm: str, brief: str) -> dict[str, Any]:
    model, mode = arm.rsplit("-", 1)
    if mode not in ("aidb", "aidb2", "aidb3", "aidb4", "aidb5", "aidb6", "aidb7", "raw"):
        raise SystemExit(f"arm must end in -raw or -aidb, -aidb2 .. -aidb7: {arm}")
    with_aidb = mode != "raw"
    # aidb3: the aidb2 prompt and tools on the later pipeline (styles, judge, doorways, recipes,
    # skill drafting); the arm's ai-db hash in metrics.json pins the code it ran on.
    # aidb4: the realism pass (2026-10-12). aidb5 (2026-10-16): the weak-model fixes, the prompt
    # hint of the current build_scene, OpenCode with direct MCP tools and every built-in denied.
    # aidb6 (2026-10-17): compact results (verdict/next first), brief passed to build_scene, ceiling
    # grid, seating group, tool names as the runner offers them. aidb7 (2026-10-17): aidb6's prompt on
    # the outdoor terrain, round rooms, open-side camera, classroom rows
    version = {"aidb": 1, "raw": 1, "aidb5": 5, "aidb6": 6, "aidb7": 6}.get(mode, 2)
    tools = briefs.AIDB_TOOLS if version == 1 else briefs.AIDB2_TOOLS
    rd = run_dir(arm, brief)
    if os.path.exists(os.path.join(rd, "metrics.json")):
        raise SystemExit(f"{arm}/{brief} is already done (delete its metrics.json to rerun)")
    if _running_elsewhere(arm, brief):
        raise SystemExit(f"{arm}/{brief} is being run by another process; skipping")
    if os.path.isdir(rd):  # an interrupted attempt: start the scene over, keep nothing half-done
        import shutil
        shutil.rmtree(rd)
    os.makedirs(rd, exist_ok=True)
    port = reserve_port()
    setup = _prepare_aidb(rd) if with_aidb else None
    # a free cloud model (OpenCode) gets a sandboxed Blender: run_code must not reach the repos,
    # the home folder or local services
    cloud_free = not (model == "strata" or model in LOCAL_MODELS or model in run_claude.MODELS)
    proc = blender.start(port, os.path.join(rd, "blender.log"), sandbox_dir=rd if cloud_free else None)
    try:
        runner = ("strata" if model == "strata" or model in LOCAL_MODELS else
                  "claude" if model in run_claude.MODELS else "opencode")
        prompt = briefs.prompt(brief, with_aidb, version, runner)
        if model == "strata" or model in LOCAL_MODELS:
            import strata_agent
            servers = run_claude.mcp_servers(port, rd, with_aidb, tools)
            base_url, served = LOCAL_MODELS.get(model, (None, None))
            agent = strata_agent.run(prompt, servers, os.path.join(rd, "transcript.jsonl"), TIMEOUT_S,
                                     base_url=base_url, model=served)
        elif model in run_claude.MODELS:
            agent = run_claude.run(model, prompt, port, rd, with_aidb, tools, TIMEOUT_S)
        else:  # a free OpenCode model, e.g. big-pickle, mimo-v2.6-flash-free
            import run_opencode
            # truncated runs (the provider cut a started session) per scene; refusals do not count
            attempts_file = os.path.join(OUT, arm, f".{brief}.attempts")
            attempt = 0
            if os.path.exists(attempts_file):
                with open(attempts_file, encoding="utf-8") as f:
                    attempt = int(f.read() or 0)
            try:
                agent = run_opencode.run(model, prompt, port, rd, with_aidb, tools, TIMEOUT_S)
            except run_opencode.ModelUnavailable as e:
                if e.started:
                    attempt += 1
                    with open(attempts_file, "w") as f:
                        f.write(str(attempt))
                if not e.started or attempt < 4:
                    raise  # no result: the supervisor runs the scene again later
                agent = run_opencode.summarize(  # keep what there is, marked
                    os.path.join(rd, "oc_home", ".local", "share", "opencode", "opencode.db"), 0.0, 1, False,
                    os.path.join(rd, "transcript.jsonl"))
                agent["provider_limited"] = True
            agent["attempts"] = attempt
        if proc.poll() is not None:  # never score whatever else may listen on the port now
            raise RuntimeError(f"{arm}/{brief}: this run's Blender exited during the session (see blender.log)")
        t = time.monotonic()
        fin = blender.finish(port, rd)
        validation = blender.client(port).request("validate_scene", {}, timeout=300)
        harness_s = time.monotonic() - t
    finally:
        blender.stop(proc)
        release_port(port)
    quality = score(brief, fin["inventory"], validation)
    metrics = {"arm": arm, "brief": brief, "model": model, "with_aidb": with_aidb, "port": port,
               "harness": "strata-agent" if model == "strata" or model in LOCAL_MODELS else ("claude-cli" if model in run_claude.MODELS else "opencode"),
               "setup": setup, "blender_sandboxed": cloud_free, "aidb_version": aidb_version() if with_aidb else None, "agent": agent, "harness_s": round(harness_s, 1),
               "renders": fin["renders"], "inventory": fin["inventory"], "validation": validation,
               "quality": quality, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(rd, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1)
    return metrics


def _roots(inv: dict[str, Any]) -> list[dict[str, Any]]:
    return [o for o in inv["objects"] if o["parent"] is None and o["name"] != "bench_probe"]


#: score weights (sum 1.0); see score() and the report header
W_CHECKLIST, W_SIZE, W_COVERAGE, W_VALID, W_BASICS = 0.40, 0.15, 0.10, 0.25, 0.10
SIZE_SLACK = 0.10  # a size within 10% outside the prior's min..max still counts as plausible
BOX_FACES = 12  # at most this many faces: a box (6) or a box-like slab, not a modelled object
PARTS_PER_ITEM = 2  # coverage: a real object has at least two mesh parts


def _mesh_parts(inv: dict[str, Any], root: dict[str, Any]) -> list[dict[str, Any]]:
    """The root's own mesh plus every mesh below it (parent links; works for old inventories)."""
    kids: dict[str, list[dict[str, Any]]] = {}
    for o in inv["objects"]:
        if o["parent"] is not None:
            kids.setdefault(o["parent"], []).append(o)
    out, todo = [], [root]
    while todo:
        o = todo.pop()
        if o["type"] == "MESH":
            out.append(o)
        todo.extend(kids.get(o["name"], []))
    return out


def geometry_weight(inv: dict[str, Any], root: dict[str, Any]) -> float:
    """1 for a modelled object (an asset root with >= 2 mesh parts, or one mesh with more than
    BOX_FACES faces), 0.5 for a lone box, 0 for an empty or a non-mesh: a name on a placeholder
    must not score like the real thing."""
    parts = _mesh_parts(inv, root)
    if len(parts) >= 2:
        return 1.0
    if not parts:
        return 0.0
    p = parts[0]
    faces = p.get("faces", p.get("tris", 0) // 2)  # old inventories: a box is 12 triangles
    return 1.0 if faces > BOX_FACES else 0.5


def size_plausible(size: list[float], prior: Any) -> bool:
    """Height within the prior's range, footprint within it in either orientation (an object
    may be turned 90 degrees), with SIZE_SLACK."""
    lo, hi = prior.min_m, prior.max_m

    def within(v: float, a: float, b: float) -> bool:
        return a * (1 - SIZE_SLACK) <= v <= b * (1 + SIZE_SLACK)

    if not within(size[2], lo[2], hi[2]):
        return False
    w, d = size[0], size[1]
    return (within(w, lo[0], hi[0]) and within(d, lo[1], hi[1])) or \
        (within(d, lo[0], hi[0]) and within(w, lo[1], hi[1]))


def checklist(brief: str, inv: dict[str, Any]) -> list[dict[str, Any]]:
    """Per item: ``found`` = objects matching by name/category (the old count), ``real`` = the same
    weighted by geometry_weight, and the matching root names (for the size check)."""
    roots = _roots(inv)
    out = []
    for label, need, tokens in briefs.BRIEFS[brief]["checklist"]:
        hits = [o for o in roots if any(re.search(re.escape(t), f"{o['name']} {o['category']}".lower())
                                        for t in tokens)]
        out.append({"item": label, "need": need, "found": len(hits),
                    "real": round(sum(geometry_weight(inv, o) for o in hits), 2),
                    "objects": [o["name"] for o in hits]})
    return out


def score(brief: str, inv: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    """0..100 = 40% checklist (geometry-weighted) + 15% size plausibility + 10% mesh coverage
    + 25% validator + 10% basics.

    - checklist: per item min(real / need, 1), real counting an object by geometry_weight
    - size: share of checklist objects with a known ``category`` whose size is within the
      category prior; no such object -> the weight goes to the checklist term
    - coverage: min(meshes / (PARTS_PER_ITEM * total need), 1)
    - validator: 100 - 8 per blocking issue - 2 per warning
    - basics: camera, lights, share of meshes with a material
    """
    from ai_db_scene.priors import load_priors

    priors = load_priors()
    items = checklist(brief, inv)
    by_name = {o["name"]: o for o in inv["objects"]}
    cover_name = 100 * sum(min(i["found"] / i["need"], 1.0) for i in items) / len(items)
    cover = 100 * sum(min(i["real"] / i["need"], 1.0) for i in items) / len(items)
    sized = plausible = 0
    for name in sorted({n for i in items for n in i["objects"]}):  # once, even if two items match it
        o = by_name[name]
        prior = priors.get(o.get("category") or "")
        if prior is None or not o.get("size_m") or geometry_weight(inv, o) == 0:
            continue
        sized += 1
        plausible += size_plausible(o["size_m"], prior)
    size_pct = 100 * plausible / sized if sized else None
    issues = validation.get("issues", []) if isinstance(validation, dict) else []
    blocking = sum(1 for i in issues if i.get("severity") == "blocking")
    warnings = len(issues) - blocking
    valid = max(0.0, 100 - 8 * blocking - 2 * warnings)
    meshes = [o for o in inv["objects"] if o["type"] == "MESH"]
    need = sum(i["need"] for i in items)
    coverage = 100 * min(len(meshes) / (PARTS_PER_ITEM * need), 1.0) if need else 100.0
    mat_ratio = sum(1 for o in meshes if o["mats"]) / len(meshes) if meshes else 0.0
    basics = 100 * ((inv.get("camera") is not None) + (inv.get("lights", 0) > 0) + mat_ratio) / 3
    kinds: dict[str, int] = {}
    for i in issues:
        kinds[i.get("check", "?")] = kinds.get(i.get("check", "?"), 0) + 1
    w_check, w_size = (W_CHECKLIST, W_SIZE) if size_pct is not None else (W_CHECKLIST + W_SIZE, 0.0)
    total = (w_check * cover + w_size * (size_pct or 0.0) + W_COVERAGE * coverage + W_VALID * valid
             + W_BASICS * basics)
    return {"score": round(total, 1), "formula": "v2", "checklist_pct": round(cover, 1),
            "checklist_name_pct": round(cover_name, 1),
            "size_pct": None if size_pct is None else round(size_pct, 1), "sized_objects": sized,
            "coverage_pct": round(coverage, 1), "validator_pct": round(valid, 1),
            "basics_pct": round(basics, 1), "blocking": blocking, "warnings": warnings,
            "issue_kinds": kinds, "checklist": items, "meshes": len(meshes),
            "triangles": sum(o["tris"] for o in inv["objects"])}


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "run":
        m = cmd_run(sys.argv[2], sys.argv[3])
        a = m["agent"]
        print(json.dumps({"arm": m["arm"], "brief": m["brief"], "ok": a["ok"], "wall_s": a["wall_s"],
                          "tokens": a["tokens"], "cost_usd": a.get("cost_usd"), "quality": m["quality"]["score"]}))
        return 0
    if cmd == "rerender":  # probe views again from the saved scene (the .blend is not changed)
        rd = run_dir(sys.argv[2], sys.argv[3])
        port = reserve_port()
        proc = blender.start(port, os.path.join(rd, "blender_rerender.log"), os.path.join(rd, "scene.blend"))
        try:
            fin = blender.finish(port, rd, save=False)
        finally:
            blender.stop(proc)
            release_port(port)
        mp = os.path.join(rd, "metrics.json")
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        m["renders"] = fin["renders"]
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1)
        print(json.dumps(fin["renders"]))
        return 0
    if cmd == "judge":
        import judge
        print(json.dumps(judge.judge_both(run_dir(sys.argv[2], sys.argv[3]), sys.argv[3])))
        return 0
    if cmd == "rescore":  # recompute quality of every finished scene with the current formula
        import glob
        for mp in sorted(glob.glob(os.path.join(OUT, "*", "*", "metrics.json"))):
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            m["quality"] = score(m["brief"], m["inventory"], m["validation"])
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=1)
        return 0
    if cmd == "report":
        import report
        print(report.write(OUT))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
