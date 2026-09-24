"""Query log, per-stage latency and the telemetry/log surfaces."""

import json

from ai_db import VectorDB
from ai_db.cli import main
from ai_db.dispatcher import ServiceDispatcher
from ai_db.telemetry.stages import stage_stats


def _db(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "m.py").write_text("def gamma_ray():\n    return 1\n")
    db = VectorDB(str(tmp_path / "o.db"))
    db.sync(str(src), project="p", verbose=False)
    return db


def test_query_and_investigate_are_logged_with_stages(tmp_path):
    db = _db(tmp_path)
    db.query("gamma ray", project="p")
    db.query("gamma ray", project="p")
    db.investigate("gamma ray", project="p", mode="locate")
    log = db.backend.get_query_log(limit=10)
    assert [e["tool"] for e in log] == ["investigate", "query", "query"]
    miss, hit = log[2], log[1]
    assert miss["cache_hit"] is False and "bm25_ms" in miss["stages"]
    assert hit["cache_hit"] is True and hit["stages"] == {}
    assert miss["providers"] == {"retriever": "lexical", "embedding": None, "rerank": None}
    assert miss["top"] and isinstance(miss["top"][0][0], int)
    db.close()


def test_slow_filter_and_stats(tmp_path):
    db = _db(tmp_path)
    db.query("gamma", project="p")
    assert db.backend.get_query_log(min_total_ms=10**9) == []
    stats = stage_stats(db.backend.get_query_log())
    assert stats["total"]["count"] == 1 and "bm25_ms" in stats
    db.close()


def test_telemetry_includes_stages(tmp_path):
    db = _db(tmp_path)
    db.query("gamma", project="p")
    summary = ServiceDispatcher(db=db).execute("telemetry", {})
    assert "bm25_ms" in summary["stages"]
    db.close()


def test_cli_log(tmp_path, capsys):
    db = _db(tmp_path)
    db.query("gamma", project="p")
    path = db.db_path
    db.close()
    assert main(["log", "--db", path, "--format", "json"]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert entries[0]["query"] == "gamma"
