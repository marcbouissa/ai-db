"""bench/REPORT.md from every run's metrics.json and judge.json.

Quality score (bench.score), formula v3 (2026-10-17): v2 x (1 - 0.25 x the share of the
applicable layout-sanity checks failed): a penalty of up to a quarter, never a bonus (a near
empty scene passes most of them trivially).

* v2 = 40 % brief checklist (geometry-weighted) + 15 % size plausibility + 10 % mesh coverage +
  25 % validator (100 - 8 / blocking - 2 / warning) + 10 % basics (camera, lights, materials).
* layout sanity (layout_sanity.py) = the share of its applicable checks a scene passes, each
  pass / fail / n.a. with its reason in metrics.json quality.layout:
  facing_focus (seats face their table / TV / fountain), seat_groups (sofas and armchairs within
  2 m of their table, benches within 3 m of their fountain), chairs_at_tables, fill_ratio
  (rooms: furniture footprint / floor 0.10-0.60), in_bounds (rooms), grounded (no floating
  objects), no_occluder (nothing in front of the subject fills the camera's frame),
  seating_fronts (the camera sees the fronts of the main seating), water_view (a scene with
  water whose brief names it: the water fills >= 15 % of the camera's frame). It reads the inventory's
  layout data (root boxes and yaws, the scene camera and a grid of rays through its frame);
  `bench.py rescore --inventory` adds that data to older runs from their scene.blend.

v2 alone saturated (aidb7 ~99) while the visual review found real layout problems; a scene
without layout data (no scene.blend) keeps formula v2.
"""

from __future__ import annotations

import json
import os
from statistics import mean
from typing import Any

import briefs

#: arm suffixes in table order: ai-db v1, ai-db v2 (build_scene/edit_scene), then without ai-db
MODES = ("aidb", "aidb2", "aidb3", "aidb4", "aidb5", "aidb6", "aidb7", "raw")


def _split(arm: str) -> tuple[str, str]:
    model, mode = arm.rsplit("-", 1)
    return model, mode


def _arms(out: str) -> list[str]:
    """Every arm folder with at least one finished scene; a model's arms side by side in MODES order."""
    found = [d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)) and not d.startswith(".")
             and "-" in d and _split(d)[1] in MODES
             and any(os.path.exists(os.path.join(out, d, b, "metrics.json")) for b in briefs.BRIEFS)]
    return sorted(found, key=lambda a: (_split(a)[0], MODES.index(_split(a)[1])))


def _version(m: dict[str, Any]) -> str:
    """Short ai-db version of a scene (a hash of the ai-db-scene code it ran with); none for raw arms."""
    v = m.get("aidb_version")
    return str(v)[:7] if v else "–"


def _thumb(out: str, m: dict[str, Any]) -> str:
    """A small linked image of the scene's interior probe render, relative to the report."""
    path = os.path.join(m["dir"], "probe_interior.png")
    if not os.path.exists(path):
        return "–"
    rel = os.path.relpath(path, out).replace(os.sep, "/")
    return f'[<img src="{rel}" width="120" alt="{m["arm"]}/{m["brief"]}">]({rel})'


def _load(out: str) -> list[dict[str, Any]]:
    rows = []
    for arm in _arms(out):
        for brief in briefs.BRIEFS:
            rd = os.path.join(out, arm, brief)
            mp = os.path.join(rd, "metrics.json")
            if not os.path.exists(mp):
                continue
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            for key, name in (("judge", "judge.json"), ("judge_pairs", "judge_pairs.json")):
                jp = os.path.join(rd, name)
                if os.path.exists(jp):
                    with open(jp, encoding="utf-8") as f:
                        m[key] = json.load(f)
            m["dir"] = rd
            rows.append(m)
    return rows


def _j(m: dict[str, Any], mode: str, key: str = "judge") -> float | None:
    v = ((m.get(key) or {}).get(mode) or {}).get("parsed", {}).get("overall")
    return float(v) if isinstance(v, (int, float)) else None


def _tok(t: dict[str, Any]) -> int:
    return sum(int(t.get(k, 0) or 0) for k in ("input", "output", "cache_write", "cache_read"))


def _fmt(x: float | None, nd: int = 1) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def write(out: str) -> str:
    rows = _load(out)
    ARMS = _arms(out)
    judges = ", ".join(sorted({(r.get("judge") or {}).get(m, {}).get("model", "?") for r in rows
                               for m in ("aidb", "plain") if (r.get("judge") or {}).get(m)})) or "none yet"
    L: list[str] = ["# Agent benchmark: with vs without ai-db", ""]
    L += [("Arms: every free OpenCode model that was available (OpenCode headless, isolated home; up to aidb4 "
          "OpenCode's built-in tools (read, shell, webfetch, ...) were still offered and the prompt forbade them, "
          "see Shell use; from aidb5 every built-in is denied, the MCP tools are offered directly instead of "
          "through Code Mode's `execute` (whose `fetch` reads local files), the working folder is an empty "
          "temporary one) and the "
          "Strata model (qwen3.8-flash, new generic agent), a local Qwen3-14B-AWQ (`qwen3-14b-local`: vLLM on "
          "this GPU via local_model.sh, the Strata agent, thinking on, 32k context), each with the Blender addon's tools only (raw) or those + "
          "ai-db scene tools (aidb; aidb2 = ai-db with build_scene/edit_scene; aidb3 = the same prompt on the later pipeline; aidb4 = the realism pass; "
          "aidb5 = the weak-model fixes with a prompt hint for the current build_scene; aidb6 = compact results with "
          "verdict/next first, the brief passed to build_scene, ceiling grid, seating group, tool names as the runner "
          "offers them; aidb7 = aidb6's prompt on outdoor terrain (sand, water, palms), round rooms, the "
          "open-side camera and classroom rows). `ai-db` = short hash of the "
          "ai-db-scene code a scene ran with (results of different versions must not be mixed). Same briefs, same Blender build, a fresh Blender per scene, 30 min cap."),
          ("Quality (formula v3) = v2 x (1 - 0.25 x share of applicable layout-sanity checks failed) "
          "(seats face their table/TV/fountain, seat groups together, chairs at tables, room fill ratio 0.10-0.60, "
          "objects inside the room, nothing floating, no foreground occluder in the camera's frame, the camera sees "
          "the fronts of the main seating, by the water the water fills >= 15% of the frame; per-check pass/fail and reason in metrics.json quality.layout). "
          "v2 = 40% brief checklist + 15% size plausibility + 10% mesh coverage + 25% "
          "validator (100 - 8/blocking - 2/warning) + 10% basics (camera, lights, materials). The checklist "
          "counts an object matched by name or category only for real geometry: an asset root with >= 2 mesh "
          "parts or one mesh with > 12 faces counts 1, a lone box 0.5, an empty 0. Size = share of matched "
          "objects with a known `category` whose extent (yaw frame) lies within the category prior "
          "(min..max, 10% slack, footprint either way round); no categorised object -> its weight goes to "
          "the checklist. Coverage = meshes / (2 x objects the brief needs), capped at 1. Scenes scored "
          "before v2 (no `formula` field) used 50% name-only checklist + 30% validator + 20% basics: "
          "run `bench.py rescore --inventory` to put every scene on v3 (v2 where no scene.blend is left). "
          f"Judges: {judges}; blind (scenes anonymised, one judge per brief and "
          "mode, comparing all scenes of the brief). `aidb` = ai-db rubric + ai-db measurements, `plain` = "
          "renders + brief only."), ""]
    L += ["## Summary per arm", "",
          "| arm | ai-db | scenes | quality | judge aidb | judge plain | wall min | model min | tool min | tokens (k) | cost $ |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        ja = [x for x in (_j(r, "aidb") for r in rs) if x is not None]
        jp = [x for x in (_j(r, "plain") for r in rs) if x is not None]
        versions = ", ".join(sorted({_version(r) for r in rs}))  # more than one: results span ai-db changes
        L.append(f"| {arm} | {versions} | {len(rs)} | {mean(r['quality']['score'] for r in rs):.1f} | "
                 f"{_fmt(mean(ja) if ja else None)} | {_fmt(mean(jp) if jp else None)} | "
                 f"{mean(r['agent']['wall_s'] for r in rs) / 60:.1f} | {mean(r['agent']['model_s'] for r in rs) / 60:.1f} | "
                 f"{mean(r['agent']['tool_s'] for r in rs) / 60:.1f} | {mean(_tok(r['agent']['tokens']) for r in rs) / 1000:.0f} | "
                 f"{sum((r['agent'].get('cost_usd') or 0) for r in rs):.2f} |")
    L += ["", "## With vs without ai-db, paired (same model, same brief: first arm minus second arm)", "",
          ("Arm averages above mix different brief sets (rate-limited models finished only the first briefs); "
          "this table compares like with like. `aidb2` is ai-db after the benchmark-driven fixes "
          "(build_scene, edit_scene); `aidb2 vs aidb` isolates what those fixes changed."), "",
          "| model | comparison | pairs | judge aidb Δ | judge plain Δ | quality Δ | wall min Δ | tokens k Δ |",
          "|---|---|---|---|---|---|---|---|"]
    by = {(r["arm"], r["brief"]): r for r in rows}
    comparisons = (("aidb", "raw"), ("aidb2", "raw"), ("aidb3", "raw"), ("aidb4", "raw"), ("aidb5", "raw"),
                   ("aidb6", "raw"), ("aidb7", "raw"), ("aidb2", "aidb"), ("aidb3", "aidb2"), ("aidb4", "aidb3"),
                   ("aidb5", "aidb4"), ("aidb6", "aidb5"), ("aidb7", "aidb6"))
    for model in sorted({_split(a)[0] for a in ARMS}):
        for first, second in comparisons:
            prs = [(by[(f"{model}-{first}", b)], by[(f"{model}-{second}", b)]) for b in briefs.BRIEFS
                   if (f"{model}-{first}", b) in by and (f"{model}-{second}", b) in by]
            if not prs:
                continue

            def d(f: Any, prs: list[Any] = prs) -> float | None:
                vals = [f(a) - f(r) for a, r in prs if f(a) is not None and f(r) is not None]
                return mean(vals) if vals else None

            ja, jp = d(lambda x: _j(x, "aidb")), d(lambda x: _j(x, "plain"))
            dq, dw = d(lambda x: x["quality"]["score"]), d(lambda x: x["agent"]["wall_s"] / 60)
            dt = d(lambda x: _tok(x["agent"]["tokens"]) / 1000)
            L.append(f"| {model} | {first} vs {second} | {len(prs)} | {_fmt(ja, 2)} | {_fmt(jp, 2)} | {_fmt(dq)} | "
                     f"{_fmt(dw)} | {_fmt(dt, 0)} |")
    paired = [r for r in rows if r.get("judge_pairs")]
    if paired:
        L += ["", "## Paired judging (one blind batch per brief over the same models' arms)", "",
              ("Judged by `opus_judge.py --pairs`: each brief's scenes from only these models, in one batch, "
              "so every arm of a model is scored side by side by the same judge."), "",
              "| model | comparison | pairs | judge Δ (plain) | judge Δ (aidb) |", "|---|---|---|---|---|"]
        pby = {(r["arm"], r["brief"]): r for r in paired}
        for model in sorted({_split(r["arm"])[0] for r in paired}):
            for first, second in comparisons:
                prs = [(pby[(f"{model}-{first}", b)], pby[(f"{model}-{second}", b)]) for b in briefs.BRIEFS
                       if (f"{model}-{first}", b) in pby and (f"{model}-{second}", b) in pby]
                if not prs:
                    continue
                cells = []
                for mode in ("plain", "aidb"):
                    vals = [x - y for x, y in ((_j(a, mode, "judge_pairs"), _j(b, mode, "judge_pairs")) for a, b in prs)
                            if x is not None and y is not None]
                    cells.append(_fmt(mean(vals), 2) if vals else "–")
                L.append(f"| {model} | {first} vs {second} | {len(prs)} | {cells[0]} | {cells[1]} |")
    L += ["", "## Best scene per brief (judge aidb, then plain, then quality)", "",
          "| brief | best arm | judge aidb | judge plain | quality | blend |", "|---|---|---|---|---|---|"]
    for brief in briefs.BRIEFS:
        rs = [r for r in rows if r["brief"] == brief]
        if not rs:
            continue
        best = max(rs, key=lambda r: (_j(r, "aidb") or 0, _j(r, "plain") or 0, r["quality"]["score"]))
        L.append(f"| {brief} | **{best['arm']}** | {_fmt(_j(best, 'aidb'))} | {_fmt(_j(best, 'plain'))} | "
                 f"{best['quality']['score']} | `{os.path.relpath(best['dir'], out)}/scene.blend` |")
    L += ["", "## Every scene", "",
          "| arm | brief | ai-db | view | done | quality | checklist | blocking | tool errors | judge aidb | judge plain | wall s | model s | tool s | turns | in | out | cache w | cache r | cost $ |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a, q, t = r["agent"], r["quality"], r["agent"]["tokens"]
        L.append(f"| {r['arm']} | {r['brief']} | {_version(r)} | {_thumb(out, r)} | "
                 f"{'yes' if a['ok'] else ('timeout' if a.get('timed_out') else 'no')} | "
                 f"{q['score']} | {q['checklist_pct']}% | {q['blocking']} | {sum(v['errors'] for v in a['tools'].values())} | {_fmt(_j(r, 'aidb'))} | {_fmt(_j(r, 'plain'))} | "
                 f"{a['wall_s']:.0f} | {a['model_s']:.0f} | {a['tool_s']:.0f} | {a.get('turns')} | {t['input']} | {t['output']} | "
                 f"{t['cache_write']} | {t['cache_read']} | {_fmt(a.get('cost_usd'), 3)} |")
    L += ["", "## Where the time went (per arm, all scenes)", ""]
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        agg: dict[str, list[float]] = {}
        for r in rs:
            for name, v in r["agent"]["tools"].items():
                s = agg.setdefault(name, [0, 0.0, 0])
                s[0] += v["calls"]
                s[1] += v["seconds"]
                s[2] += v["errors"]
        setup = sum(((r.get("setup") or {}).get("seconds") or 0) for r in rs)
        L += [f"### {arm}", "",
              f"model {sum(r['agent']['model_s'] for r in rs) / 60:.1f} min, tools {sum(r['agent']['tool_s'] for r in rs) / 60:.1f} min, "
              f"harness (save + renders + validation, not counted) {sum(r['harness_s'] for r in rs) / 60:.1f} min"
              + (f", ai-db setup (skill index, not counted) {setup:.0f} s" if setup else ""), "",
              "| tool | calls | seconds | errors |", "|---|---|---|---|"]
        for name, (c, sec, e) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:15]:
            L.append(f"| {name} | {c} | {sec:.0f} | {e} |")
        L.append("")
    flagged = [(r["arm"], r["brief"], c["command"]) for r in rows
               for c in r["agent"].get("shell_commands", []) if c.get("flagged")]
    shells = sum(len(r["agent"].get("shell_commands", [])) for r in rows)
    L += ["## Shell use (forbidden by the prompt; offered to OpenCode arms up to aidb4, denied from aidb5)", "",
          f"{shells} shell commands in all runs; {len(flagged)} touched Blender, ai-db or Python and are listed:", ""]
    L += [f"- {a}/{b}: `{c[:160]}`" for a, b, c in flagged] or ["- none"]
    L.append("")
    L += ["## Judging cost (per mode; one judge per brief, cost split evenly over its scenes)", "",
          "| mode | scenes judged | input | output | cache w | cache r | total |", "|---|---|---|---|---|---|---|"]
    for mode in ("aidb", "plain"):
        js = [r["judge"][mode] for r in rows if r.get("judge") and r["judge"].get(mode)]
        if not js:
            continue
        sums = [sum(int(j['tokens'].get(k, 0) or 0) for j in js) for k in ("input", "output", "cache_write", "cache_read")]
        L.append(f"| {mode} | {len(js)} | " + " | ".join(str(x) for x in sums) + f" | {sum(sums)} |")
    L += ["", "## Judge vs judge", ""]
    pairs = [(_j(r, "aidb"), _j(r, "plain")) for r in rows if _j(r, "aidb") is not None and _j(r, "plain") is not None]
    if pairs:
        diff = mean(abs(a - b) for a, b in pairs)
        agree = sum(1 for a, b in pairs if abs(a - b) <= 1)
        L.append(f"{len(pairs)} scenes judged both ways: mean |aidb - plain| = {diff:.2f} points; within 1 point on "
                 f"{agree}/{len(pairs)}. Mean aidb {mean(a for a, _ in pairs):.2f}, mean plain {mean(b for _, b in pairs):.2f}.")
    L += ["", "## Files", "", ("Each scene folder holds `scene.blend` (saved before any probe camera was added), "
          "`agent_camera.png` (the agent's own camera, if any), `probe_interior.png` (eye height from a corner), `probe_plan.png` (top view cut at 2.5 m), "
          "`metrics.json`, `transcript.jsonl` and `judge.json`."), ""]
    path = os.path.join(out, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path
