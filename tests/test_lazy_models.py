"""B1/B2: exact lookups never load models; a needed-but-missing model still raises."""

import json
import os

import pytest

from ai_db import VectorDB
from ai_db.errors import AiDbConfigError
from ai_db.quiet import is_verbose, quiet_model_load, set_verbose


def _hybrid_cfg(tmp_path, monkeypatch):
    p = tmp_path / "hyb.json"
    p.write_text(json.dumps({
        "version": 2,
        "storage": {"provider": "sqlite", "options": {"path": str(tmp_path / "h.db")}},
        "retrieval": {"mode": "hybrid"},
        "embedding": {"provider": "sentence_transformers",
                      "model": "nonexistent/model", "batch_size": 4, "device": "cpu"},
        "rerank": {"provider": "sentence_transformers", "model": "nonexistent/rr", "device": "cpu"},
    }))
    monkeypatch.setenv("AI_DB_CONFIG", str(p))
    return p


@pytest.fixture()
def boom(monkeypatch):
    calls = []

    def _raise(*a, **k):
        calls.append(1)
        raise AiDbConfigError("model must not be loaded")

    monkeypatch.setattr("ai_db.embed.registry.build_embedder", _raise)
    monkeypatch.setattr("ai_db.rerank.registry.build_reranker", _raise)
    return calls


def test_exact_lookups_never_load_models(tmp_path, monkeypatch, boom):
    src = tmp_path / "src"
    src.mkdir()
    (src / "m.py").write_text(
        "# TODO: fix me\ndef target():\n    return 1\n\ndef caller():\n    return target()\n")
    # Index with the lexical config (hybrid sync would legitimately embed).
    lex = VectorDB(str(tmp_path / "h.db"))
    lex.sync(str(src), project="p", verbose=False)
    lex.close()
    _hybrid_cfg(tmp_path, monkeypatch)
    db = VectorDB(str(tmp_path / "h.db"))
    try:
        assert boom == []
        assert db.status()
        db.query_symbol("target")
        db.query_callers("target")
        db.query_annotations()
        assert boom == []
    finally:
        db.close()


def test_missing_model_raises_when_needed(tmp_path, monkeypatch, boom):
    _hybrid_cfg(tmp_path, monkeypatch)
    db = VectorDB(str(tmp_path / "h.db"))
    try:
        with pytest.raises(AiDbConfigError):
            db.embedder.embed_query("x")
        with pytest.raises(AiDbConfigError):
            db.reranker.score("q", ["t"])
        with pytest.raises(AiDbConfigError):
            db.embed_missing()
        assert len(boom) == 3
    finally:
        db.close()


def test_quiet_load_respects_verbose(monkeypatch):
    monkeypatch.delenv("AI_DB_VERBOSE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    with quiet_model_load("some/uncached-model"):
        assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ
    set_verbose(True)
    assert is_verbose()
    with quiet_model_load("some/uncached-model"):
        assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ
