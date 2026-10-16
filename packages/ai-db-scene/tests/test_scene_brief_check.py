"""build_scene without a brief cannot check counts and says so (qwen3-14b-local-aidb5, 2026-10-17).

No qwen aidb5 build passed ``brief``: every verdict said done without comparing the scene with
the request. The verdict now says counts were not checked and next says to pass the brief; the
spec's own ``count`` is compared with what was placed; and the brief count reads the bench
briefs right ("a teacher desk ... twelve student desks" is 13 desks, "a workbench, safety
barriers" no count for the barriers).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ai_db_scene.layout import Placement
from ai_db_scene.priors import load_priors
from ai_db_scene.realism import brief_counts
from ai_db_scene.spec import validate_spec
from ai_db_scene.tools import scene_pipeline as sp

CLASSROOM = json.loads((Path(__file__).parent / "fixtures" / "bench_qwen_classroom_aidb5.json").read_text())["args"]
CLASSROOM_BRIEF = ("A classroom with a teacher desk and chair, a whiteboard on the front wall, and twelve "
                   "student desks with chairs in rows facing the whiteboard.")


@pytest.fixture
def fake_blender(monkeypatch):
    class Fake:
        def batch(self, cmds, timeout=None):
            return [{"ok": 1, "status": "success"} for _ in cmds]

        def request(self, handler, params, timeout=None):
            return {}

        def run_code(self, code, timeout=None):
            return {}

    class Memory:
        def record_run(self, run):
            return "run-1"

        def snapshot_scene(self, **kw):
            return "scene-1"

    monkeypatch.setattr(sp, "BlenderClient", Fake)
    monkeypatch.setattr(sp, "_memory", lambda: Memory())
    monkeypatch.setattr(sp, "_handle_validate_scene", lambda args: {"status": "success", "result": {"issues": []}})


def test_no_brief_says_counts_were_not_checked(fake_blender):
    out = sp._handle_build_scene({**CLASSROOM, "exposure_check": False})
    assert out["verdict"].startswith("done: 27 objects built") and "no brief given" in out["verdict"]
    assert "pass brief" in out["next"] and "Stop here" not in out["next"]


def test_classroom_with_its_brief_is_done(fake_blender):
    out = sp._handle_build_scene({**CLASSROOM, "exposure_check": False, "brief": CLASSROOM_BRIEF})
    assert out["verdict"].startswith("done: 27 objects built") and "brief" not in out["verdict"], out["verdict"]
    assert "Stop here and reply DONE" in out["next"]


def test_brief_counts_of_the_bench_briefs():
    cats = set(load_priors())
    assert brief_counts(CLASSROOM_BRIEF, cats) == {"whiteboard": 1, "chair": None, "desk": 13}
    wh = brief_counts("An industrial warehouse: four pallet racks in rows, pallets with crates, a forklift, a "
                      "workbench, safety barriers and ceiling lights.", cats)
    assert wh["safety_barrier"] is None and wh["pallet_rack"] == 4 and wh["workbench"] == 1
    cafe = brief_counts("A small cafe with four round tables with two chairs each, a counter with two bar stools", cats)
    assert cafe["chair"] is None and cafe["bar_stool"] == 2


def test_spec_count_placed_fewer_is_reported():
    spec, _ = validate_spec({"setting": {"kind": "indoor", "bounds_m": [4, 4, 3]},
                             "objects": [{"id": "desk", "category": "desk", "count": 3},
                                         {"id": "sofa", "category": "sofa"}]})
    assert spec is not None
    placed = [Placement("desk_1", (0, 0, 0), 0.0, (1, 1, 1), "desk"), Placement("desk_2", (2, 0, 0), 0.0, (1, 1, 1), "desk"),
              Placement("desk_1_seat_0", (0, 1, 0), 0.0, (1, 1, 1), "chair"), Placement("sofa", (3, 3, 0), 0.0, (1, 1, 1), "sofa")]
    issues = sp.count_issues(spec, placed)
    assert [i["message"][:44] for i in issues] == ["the spec asks for 3 'desk' (desk), 2 were pl"]
    verdict, _ = sp.build_verdict({"objects": 3, "issues_left": issues, "scene_id": "s"})
    assert verdict.startswith("one thing to fix: the spec asks for 3 'desk'")


def test_edit_scene_keeps_the_brief():
    assert sp.build_options({"brief": "two armchairs", "seed": 1}) == {"seed": 1, "brief": "two armchairs"}
