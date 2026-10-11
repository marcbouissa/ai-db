"""Agent benchmark: {opus, sonnet, strata} x {with ai-db, without ai-db} x briefs.

    python bench.py run <arm> <brief>      one scene: fresh Blender, agent, save + renders, quality
    python bench.py judge <arm> <brief>    both judge modes (Opus, with and without ai-db context)
    python bench.py report                 bench/REPORT.md

Arms: <model>-aidb / <model>-raw, model = strata, one of OPENCODE_FREE, or opus/sonnet (Claude CLI).
Outputs (kept, never committed): <repo>/bench/<arm>/<brief>/{scene.blend, *.png, metrics.json,
transcript.jsonl, judge_*.json}.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import socket
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
import blender  # noqa: E402
import briefs  # noqa: E402
import run_claude  # noqa: E402

REPO = run_claude.REPO
OUT = os.path.join(REPO, "bench")
OPENCODE_FREE = ("big-pickle", "longcat-2.5-preview-free", "mimo-v2.6-flash-free", "muse-spark-1.3-contributor-free",
                 "nemotron-3-ultra-free", "nemotron-3.5-lightning-free", "space-bunny-free")
#: offered by `opencode models` on 2026-10-11 but refused by the router ("Model unavailable")
OPENCODE_UNAVAILABLE = ("exo-free", "ling-3.0-flash-fin-free", "ling-3.1-flash-free", "step-5-preview-free")
TIMEOUT_S = 30 * 60
USER_SKILLS = os.path.expanduser("~/.local/share/blender-ai/skills")


def run_dir(arm: str, brief: str) -> str:
    return os.path.join(OUT, arm, brief)


def reserve_port() -> int:
    """A Blender port no other (parallel) run holds: an atomic mkdir per port."""
    lock_root = os.path.join(OUT, ".ports")
    os.makedirs(lock_root, exist_ok=True)
    for port in range(11000, 11400):
        lock = os.path.join(lock_root, str(port))
        try:
            os.mkdir(lock)
        except FileExistsError:
            try:  # a lock whose owner died is free again
                with open(os.path.join(lock, "pid"), encoding="utf-8") as f:
                    owner = int(f.read().strip() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner and os.path.exists(f"/proc/{owner}"):
                continue
        with open(os.path.join(lock, "pid"), "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        release_port(port)
    raise RuntimeError("no free Blender port in 11000-11399")


def release_port(port: int) -> None:
    try:
        with open(os.path.join(OUT, ".ports", str(port), "pid"), "w", encoding="utf-8") as f:
            f.write("0")  # free; the directory stays
    except OSError:
        pass


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
                          "--dir", USER_SKILLS], env=env, capture_output=True, text=True, timeout=600)
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
    if mode not in ("aidb", "aidb2", "raw"):
        raise SystemExit(f"arm must end in -raw, -aidb or -aidb2: {arm}")
    with_aidb = mode != "raw"
    version = 2 if mode == "aidb2" else 1
    tools = briefs.AIDB2_TOOLS if version == 2 else briefs.AIDB_TOOLS
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
    proc = blender.start(port, os.path.join(rd, "blender.log"))
    try:
        prompt = briefs.prompt(brief, with_aidb, version)
        if model == "strata":
            import strata_agent
            servers = run_claude.mcp_servers(port, rd, with_aidb, tools)
            agent = strata_agent.run(prompt, servers, os.path.join(rd, "transcript.jsonl"), TIMEOUT_S)
        elif model in run_claude.MODELS:
            agent = run_claude.run(model, prompt, port, rd, with_aidb, tools, TIMEOUT_S)
        else:  # a free OpenCode model, e.g. big-pickle, mimo-v2.6-flash-free
            import run_opencode
            # truncated runs (the provider cut a started session) per scene; refusals do not count
            attempts_file = os.path.join(OUT, arm, f".{brief}.attempts")
            attempt = int(open(attempts_file).read() or 0) if os.path.exists(attempts_file) else 0
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
        t = time.monotonic()
        fin = blender.finish(port, rd)
        validation = blender.client(port).request("validate_scene", {}, timeout=300)
        harness_s = time.monotonic() - t
    finally:
        blender.stop(proc)
        release_port(port)
    quality = score(brief, fin["inventory"], validation)
    metrics = {"arm": arm, "brief": brief, "model": model, "with_aidb": with_aidb, "port": port,
               "harness": "strata-agent" if model == "strata" else ("claude-cli" if model in run_claude.MODELS else "opencode"),
               "setup": setup, "aidb_version": aidb_version() if with_aidb else None, "agent": agent, "harness_s": round(harness_s, 1),
               "renders": fin["renders"], "inventory": fin["inventory"], "validation": validation,
               "quality": quality, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(rd, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1)
    return metrics


def _roots(inv: dict[str, Any]) -> list[dict[str, Any]]:
    return [o for o in inv["objects"] if o["parent"] is None and o["name"] != "bench_probe"]


def checklist(brief: str, inv: dict[str, Any]) -> list[dict[str, Any]]:
    roots = _roots(inv)
    out = []
    for label, need, tokens in briefs.BRIEFS[brief]["checklist"]:
        hay = [f"{o['name']} {o['category']}".lower() for o in roots]
        found = sum(1 for h in hay if any(re.search(re.escape(t), h) for t in tokens))
        out.append({"item": label, "need": need, "found": found})
    return out


def score(brief: str, inv: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    """0..100: 50% checklist coverage, 30% validator, 20% basics (camera, light, materials)."""
    items = checklist(brief, inv)
    cover = 100 * sum(min(i["found"] / i["need"], 1.0) for i in items) / len(items)
    issues = validation.get("issues", []) if isinstance(validation, dict) else []
    blocking = sum(1 for i in issues if i.get("severity") == "blocking")
    warnings = len(issues) - blocking
    valid = max(0.0, 100 - 8 * blocking - 2 * warnings)
    meshes = [o for o in inv["objects"] if o["type"] == "MESH"]
    mat_ratio = sum(1 for o in meshes if o["mats"]) / len(meshes) if meshes else 0.0
    basics = 100 * ((inv.get("camera") is not None) + (inv.get("lights", 0) > 0) + mat_ratio) / 3
    kinds: dict[str, int] = {}
    for i in issues:
        kinds[i.get("check", "?")] = kinds.get(i.get("check", "?"), 0) + 1
    return {"score": round(0.5 * cover + 0.3 * valid + 0.2 * basics, 1), "checklist_pct": round(cover, 1),
            "validator_pct": round(valid, 1), "basics_pct": round(basics, 1), "blocking": blocking,
            "warnings": warnings, "issue_kinds": kinds, "checklist": items, "meshes": len(meshes),
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
    if cmd == "report":
        import report
        print(report.write(OUT))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
