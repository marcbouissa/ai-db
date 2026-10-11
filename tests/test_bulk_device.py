"""GPU for large embedding runs, CPU for small ones (embedding.bulk_device)."""

from __future__ import annotations

import pytest

from ai_db import config_template
from ai_db.config import parse_config
from ai_db.config_template import build_template
from ai_db.embed.base import EmbeddingProvider
from ai_db.embed.indexing import embed_missing
from ai_db.embed.st_provider import SentenceTransformersEmbedder
from ai_db.errors import AiDbConfigError


class _Chunk:
    def __init__(self, cid: int):
        self.id, self.language, self.qualified_name, self.content = cid, "python", f"f{cid}", "x"


class _Backend:
    def __init__(self, n: int):
        self.pending = [_Chunk(i) for i in range(n)]
        self.stored: list[int] = []

    def chunks_missing_embeddings(self, limit: int):
        return self.pending[:limit]

    def upsert_embeddings(self, items):
        done = {cid for cid, _ in items}
        self.stored += sorted(done)
        self.pending = [c for c in self.pending if c.id not in done]


class _Recorder(EmbeddingProvider):
    model_id, dim = "rec", 2

    def __init__(self, threshold: int | None):
        self.bulk_threshold = threshold
        self.mode = "cpu"
        self.calls: list[tuple[str, int]] = []
        self.bulk_pending: list[int] = []

    def embed_documents(self, texts):
        self.calls.append((self.mode, len(texts)))
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]

    def bulk(self, pending):
        from contextlib import contextmanager

        @contextmanager
        def scope():
            self.bulk_pending.append(pending)
            switch = self.bulk_threshold is not None and pending >= self.bulk_threshold
            if switch:
                self.mode = "gpu"
            try:
                yield
            finally:
                self.mode = "cpu"
        return scope()


@pytest.mark.parametrize("n,expected", [(10, "cpu"), (100, "gpu")])
def test_embed_missing_uses_bulk_only_for_large_backlogs(n, expected):
    backend, emb = _Backend(n), _Recorder(threshold=50)
    assert embed_missing(backend, emb, batch_size=8) == n
    assert backend.stored == list(range(n))  # every chunk once, in order
    assert {mode for mode, _ in emb.calls} == {expected}
    assert all(size <= 8 for _, size in emb.calls)
    assert emb.mode == "cpu"  # moved back afterwards


def test_embed_missing_without_threshold_and_empty_backlog():
    emb = _Recorder(threshold=None)
    assert embed_missing(_Backend(0), emb, batch_size=8) == 0
    assert emb.bulk_pending == []  # nothing to do: no device switch at all
    assert embed_missing(_Backend(20), emb, batch_size=8) == 20
    assert {mode for mode, _ in emb.calls} == {"cpu"}


class _FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.moves: list[str] = []

    def to(self, device):
        self.device = device
        self.moves.append(device)
        return self


def _st(bulk_device="cuda", threshold=100) -> SentenceTransformersEmbedder:
    import threading
    emb = SentenceTransformersEmbedder.__new__(SentenceTransformersEmbedder)
    emb.device, emb.bulk_device, emb.bulk_threshold = "cpu", bulk_device, threshold
    emb.model = _FakeModel()
    emb._lock = threading.RLock()
    return emb


def test_st_bulk_moves_model_and_back(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    emb = _st()
    with emb.bulk(50):
        assert emb.current_device == "cpu"
    with emb.bulk(500):
        assert emb.current_device == "cuda"
    assert emb.current_device == "cpu"
    assert emb.model.moves == ["cuda", "cpu"]


def test_st_bulk_moves_back_after_error(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    emb = _st()
    with pytest.raises(RuntimeError), emb.bulk(500):
        raise RuntimeError("boom")
    assert emb.current_device == "cpu"


def test_st_without_bulk_device_never_moves():
    emb = _st(bulk_device=None, threshold=None)
    with emb.bulk(10_000):
        pass
    assert emb.model.moves == []


def _emb_cfg(**opts):
    data = build_template()
    data["retrieval"]["mode"] = "hybrid"
    data["embedding"] = {"provider": "sentence_transformers", "model": "m", "device": "cpu",
                         "batch_size": 8, **opts}
    return data


def test_config_accepts_and_validates_bulk_options():
    cfg = parse_config(_emb_cfg(bulk_device="cuda", bulk_threshold=128), check_env=False)
    assert cfg.embedding.options["bulk_device"] == "cuda"
    with pytest.raises(AiDbConfigError, match="bulk_threshold"):
        parse_config(_emb_cfg(bulk_threshold=0), check_env=False)
    with pytest.raises(AiDbConfigError, match="bulk_device"):
        parse_config(_emb_cfg(bulk_device=""), check_env=False)


def test_ai_db_device_env_pins_bulk_too(monkeypatch):
    monkeypatch.setenv("AI_DB_DEVICE", "cpu")
    cfg = parse_config(_emb_cfg(bulk_device="cuda"), check_env=False)
    assert cfg.embedding.options["device"] == "cpu"
    assert "bulk_device" not in cfg.embedding.options


@pytest.mark.parametrize("gpu,expected", [("cuda", {"device": "cpu", "bulk_device": "cuda"}),
                                          ("cpu", {"device": "cpu"})])
def test_template_uses_gpu_for_bulk_only(monkeypatch, gpu, expected):
    monkeypatch.setattr(config_template, "preferred_device", lambda: gpu)
    emb = build_template(embedding="sentence_transformers")["embedding"]
    assert {k: emb[k] for k in ("device", "bulk_device") if k in emb} == expected


def test_cpu_threads_caps_and_respects_override(monkeypatch):
    from ai_db import device
    monkeypatch.delenv("AI_DB_CPU_THREADS", raising=False)
    monkeypatch.setattr(__import__("os"), "cpu_count", lambda: 24)
    assert device.cpu_threads() == 8
    monkeypatch.setattr(__import__("os"), "cpu_count", lambda: 4)
    assert device.cpu_threads() == 2
    monkeypatch.setenv("AI_DB_CPU_THREADS", "12")
    assert device.cpu_threads() == 12
    monkeypatch.setenv("AI_DB_CPU_THREADS", "0")
    with pytest.raises(AiDbConfigError):
        device.cpu_threads()


def test_configure_cpu_threads_leaves_omp_choice_alone(monkeypatch):
    import torch

    from ai_db import device
    calls: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    monkeypatch.setattr(device, "_threads_configured", False)
    monkeypatch.delenv("AI_DB_CPU_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    device.configure_cpu_threads()
    assert calls == []
    monkeypatch.setattr(device, "_threads_configured", False)
    monkeypatch.setenv("AI_DB_CPU_THREADS", "5")
    device.configure_cpu_threads()
    device.configure_cpu_threads()  # once per process
    assert calls == [5]



def test_config_validates_max_seq_length():
    cfg = parse_config(_emb_cfg(max_seq_length=1024), check_env=False)
    assert cfg.embedding.options["max_seq_length"] == 1024
    with pytest.raises(AiDbConfigError, match="max_seq_length"):
        parse_config(_emb_cfg(max_seq_length=0), check_env=False)
