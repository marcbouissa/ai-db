"""Daemon projects share one loaded model per provider config; release_models frees it."""

from __future__ import annotations

import pytest

from ai_db.config import parse_config
from ai_db.config_template import build_template
from ai_db.embed import registry
from ai_db.facade import VectorDB
from ai_db.lazy import release_models
from tests.fake_providers import FakeEmbeddingProvider


@pytest.fixture
def builds(monkeypatch):
    made: list[FakeEmbeddingProvider] = []

    def build(opts):
        made.append(FakeEmbeddingProvider(opts))
        return made[-1]
    monkeypatch.setitem(registry.BUILTIN, "fake", build)
    release_models()
    yield made
    release_models()


def _cfg():
    data = build_template()
    data["retrieval"]["mode"] = "hybrid"
    data["embedding"] = {"provider": "fake", "model": "shared", "dim": 16, "batch_size": 8}
    return parse_config(data)


def test_projects_share_one_model_and_each_gets_its_vector_index(tmp_path, builds):
    a = VectorDB(str(tmp_path / "a.db"), config=_cfg())
    b = VectorDB(str(tmp_path / "b.db"), config=_cfg())
    try:
        assert a.embedder is not None and b.embedder is not None
        assert a.embedder.load() is b.embedder.load()
        assert len(builds) == 1
        for db in (a, b):  # on_load runs per VectorDB: each backend gets its index
            assert db.backend.get_embed_meta()["dim"] == 16
    finally:
        a.close()
        b.close()


def test_unload_keeps_the_shared_model_for_other_projects(tmp_path, builds):
    a = VectorDB(str(tmp_path / "a.db"), config=_cfg())
    b = VectorDB(str(tmp_path / "b.db"), config=_cfg())
    try:
        inner = a.embedder.load()
        a.unload_models()
        assert not a.embedder.loaded
        assert b.embedder.load() is inner
        assert b.embedder.embed_query("still works")
        assert len(builds) == 1
    finally:
        a.close()
        b.close()


def test_release_models_frees_and_next_load_rebuilds(tmp_path, builds):
    db = VectorDB(str(tmp_path / "a.db"), config=_cfg())
    try:
        db.embedder.load()
        db.unload_models()
        assert release_models() == 1
        db.embedder.load()
        assert len(builds) == 2  # rebuilt, not a released instance handed back
    finally:
        db.close()
