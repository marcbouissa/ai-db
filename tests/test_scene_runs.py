"""RunRecord persistence on both scene backends + exemplar filtering."""

from __future__ import annotations

import pytest

from ai_db.scene.factory import SceneMemoryFactory
from ai_db.scene.models import RunRecord


@pytest.fixture(params=["json_file", "sqlite_vec"])
def backend(request, tmp_path):
    path = str(tmp_path / ("j" if request.param == "json_file" else "s.db"))
    b = SceneMemoryFactory.create(request.param, {"path": path})
    yield b
    b.close()


def _run(brief, score=4.5, outcome="success", **kw):
    return RunRecord(brief=brief, outcome=outcome, spec={"x": 1},
                     judge_scores=[] if score is None else [{"overall": 2.0}, {"overall": score}], **kw)


def test_roundtrip(backend):
    r = RunRecord(brief="cozy room", spec={"a": 1}, placements=[{"id": "s"}], actions=[{"k": 1}],
                  validation_reports=[[{"issue": "x"}]], judge_scores=[{"overall": 4.0}],
                  token_cost=1.5, wall_time_s=9.0, outcome="success", project="p")
    backend.save_run(r)
    got = backend.load_run(r.run_id)
    assert got is not None and got.to_dict() == r.to_dict()
    assert got.score == 4.0
    assert backend.load_run("nope") is None


def test_list_runs_filters_and_orders(backend):
    a, b, c = _run("a", project="p1"), _run("b", project="p2"), _run("c", project="p1")
    a.created_at, b.created_at, c.created_at = 1.0, 2.0, 3.0
    for r in (a, b, c):
        backend.save_run(r)
    assert [r.brief for r in backend.list_runs()] == ["c", "b", "a"]
    assert [r.brief for r in backend.list_runs(project="p1")] == ["c", "a"]
    assert len(backend.list_runs(limit=1)) == 1


def test_save_run_upserts(backend):
    r = _run("a")
    backend.save_run(r)
    r.token_cost = 7.0
    backend.save_run(r)
    assert len(backend.list_runs()) == 1 and backend.load_run(r.run_id).token_cost == 7.0


def test_search_exemplars_never_returns_bad_runs(backend):
    good = _run("scandinavian living room with sofa")
    low = _run("scandinavian living room with sofa", score=2.5)
    failed = _run("scandinavian living room with sofa", outcome="failed")
    unjudged = _run("scandinavian living room with sofa", score=None)
    other = _run("medieval castle", score=5.0)
    for r in (good, low, failed, unjudged, other):
        backend.save_run(r)
    res = backend.search_exemplars("living room sofa scandinavian", min_score=4.0, top_k=5)
    ids = [r.run_id for r in res]
    assert good.run_id in ids and other.run_id in ids
    assert not {low.run_id, failed.run_id, unjudged.run_id} & set(ids)
    assert ids[0] == good.run_id  # most similar first
    assert len(backend.search_exemplars("living room", top_k=1)) == 1
    assert backend.search_exemplars("living room", min_score=5.1) == []
