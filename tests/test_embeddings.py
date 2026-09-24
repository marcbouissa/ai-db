"""Swappable embeddings: config, hybrid retrieval, index guard, hosted HTTP providers."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ai_db import VectorDB
from ai_db.cli import main
from ai_db.config import ProviderConfig, parse_config
from ai_db.config_template import build_template
from ai_db.embed import registry
from ai_db.embed.hosted import OpenAICompatibleEmbedder, VoyageEmbedder
from ai_db.errors import AiDbConfigError
from ai_db.http_client import AiDbProviderError
from ai_db.search.retriever import HybridRetriever, LexicalRetriever
from tests.fake_providers import FakeEmbeddingProvider


def hybrid_cfg(model="bow", dim=64):
    data = build_template()
    data["retrieval"]["mode"] = "hybrid"
    data["embedding"] = {"provider": "fake", "model": model, "dim": dim, "batch_size": 8}
    return parse_config(data)


@pytest.fixture(autouse=True)
def fake_registered(monkeypatch):
    monkeypatch.setitem(registry.BUILTIN, "fake", lambda opts: FakeEmbeddingProvider(opts))


@pytest.fixture
def repo(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "cars.py").write_text("def start_car():\n    \"\"\"Start the car engine.\"\"\"\n    return 1\n")
    (src / "shop.py").write_text("def buy_item(item):\n    return pay(item)\n")
    return src


def test_lexical_mode_imports_no_embedder(tmp_path):
    db = VectorDB(str(tmp_path / "l.db"))
    assert db.embedder is None
    assert isinstance(db.query_engine.retriever, LexicalRetriever)
    db.close()


def test_hybrid_sync_embeds_and_finds_semantic_match(tmp_path, repo):
    db = VectorDB(str(tmp_path / "h.db"), config=hybrid_cfg())
    assert isinstance(db.query_engine.retriever, HybridRetriever)
    db.sync(str(repo), project="p", verbose=False)
    assert db.backend.chunks_missing_embeddings(limit=10) == []
    # 'automobile' never appears in the code; only the vector path can find it
    hits = db.query("automobile", project="p")
    assert hits and hits[0]["file"].endswith("cars.py")
    db.close()


def test_resync_keeps_vectors_for_unchanged_chunks(tmp_path, repo):
    db = VectorDB(str(tmp_path / "h.db"), config=hybrid_cfg())
    db.sync(str(repo), project="p", verbose=False)
    calls = db.embedder.calls
    (repo / "shop.py").write_text("def buy_item(item):\n    return pay(item)\n\ndef refund():\n    pass\n")
    db.sync(str(repo), project="p", verbose=False)
    assert db.embedder.calls == calls + 1  # only the new chunk batch
    db.close()


def test_model_change_requires_reindex(tmp_path, repo, monkeypatch, capsys):
    path = str(tmp_path / "h.db")
    db = VectorDB(path, config=hybrid_cfg("bow"))
    db.sync(str(repo), project="p", verbose=False)
    db.close()
    with pytest.raises(AiDbConfigError, match="reindex --embeddings"):
        VectorDB(path, config=hybrid_cfg("other"))

    cfg_file = tmp_path / "cfg.json"
    data = hybrid_cfg("other").to_dict()
    cfg_file.write_text(json.dumps(data))
    assert main(["--config", str(cfg_file), "reindex", "--embeddings", "--db", path]) == 0
    assert "fake:other" in capsys.readouterr().out
    db = VectorDB(path, config=hybrid_cfg("other"))
    assert db.backend.get_embed_meta()["model_id"] == "fake:other"
    db.close()


def test_hybrid_rejected_without_embedding():
    data = build_template()
    data["retrieval"]["mode"] = "hybrid"
    with pytest.raises(AiDbConfigError):
        parse_config(data)


def test_unknown_embedding_provider():
    with pytest.raises(AiDbConfigError, match="unknown embedding provider"):
        registry.build_embedder(ProviderConfig("nope", {}))


def test_st_provider_missing_package_is_config_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(AiDbConfigError, match="local-embed"):
        registry.build_embedder(ProviderConfig("sentence_transformers",
                                               {"model": "m", "device": "cpu", "batch_size": 1}))


# --- hosted providers against a local stub ------------------------------------

class _Stub(BaseHTTPRequestHandler):
    responses: list = []  # noqa: RUF012 - reset per test by the fixture
    requests: list = []  # noqa: RUF012

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "body": body,
                                    "auth": self.headers.get("Authorization")})
        status, payload = type(self).responses.pop(0)
        if callable(payload):
            payload = payload(body)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def stub():
    _Stub.responses, _Stub.requests = [], []
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_port}", _Stub
    server.shutdown()


def _emb(dim):
    return lambda body: {"data": [{"index": i, "embedding": [1.0] + [0.0] * (dim - 1)}
                                  for i, _ in enumerate(body["input"])]}


def test_openai_compatible_batches_and_normalizes(stub, monkeypatch):
    url, handler = stub
    monkeypatch.setenv("TEST_KEY", "k")
    handler.responses = [(200, _emb(4)), (200, _emb(4))]
    emb = OpenAICompatibleEmbedder({"model": "m", "base_url": url, "api_key_env": "TEST_KEY",
                                    "dimensions": 4, "batch_size": 2})
    vecs = emb.embed_documents(["a", "b", "c"])
    assert len(vecs) == 3 and vecs[0] == [1.0, 0.0, 0.0, 0.0]
    assert handler.requests[0]["path"] == "/embeddings"
    assert handler.requests[0]["body"]["dimensions"] == 4
    assert handler.requests[0]["auth"] == "Bearer k"


def test_retry_on_429_then_success(stub, monkeypatch):
    url, handler = stub
    monkeypatch.setenv("TEST_KEY", "k")
    monkeypatch.setattr("ai_db.http_client.time.sleep", lambda s: None)
    handler.responses = [(429, {"error": "slow down"}), (200, _emb(4))]
    emb = OpenAICompatibleEmbedder({"model": "m", "base_url": url, "api_key_env": "TEST_KEY",
                                    "dimensions": 4, "batch_size": 8})
    assert len(emb.embed_query("q")) == 4
    assert len(handler.requests) == 2


def test_client_error_is_not_retried(stub, monkeypatch):
    url, handler = stub
    monkeypatch.setenv("TEST_KEY", "k")
    handler.responses = [(400, {"error": "bad"})]
    emb = OpenAICompatibleEmbedder({"model": "m", "base_url": url, "api_key_env": "TEST_KEY",
                                    "dimensions": 4, "batch_size": 8})
    with pytest.raises(AiDbProviderError, match="HTTP 400"):
        emb.embed_query("q")


def test_dimension_mismatch_raises(stub, monkeypatch):
    url, handler = stub
    monkeypatch.setenv("TEST_KEY", "k")
    handler.responses = [(200, _emb(3))]
    emb = OpenAICompatibleEmbedder({"model": "m", "base_url": url, "api_key_env": "TEST_KEY",
                                    "dimensions": 4, "batch_size": 8})
    with pytest.raises(AiDbProviderError, match="dimension 3"):
        emb.embed_query("q")


def test_voyage_sends_input_type(stub, monkeypatch):
    url, handler = stub
    monkeypatch.setenv("VK", "v")
    handler.responses = [(200, _emb(4)), (200, _emb(4))]
    emb = VoyageEmbedder({"model": "voyage-code-3", "api_key_env": "VK", "dimensions": 4,
                          "batch_size": 8, "base_url": url + "/v1/embeddings"})
    emb.embed_query("q")
    emb.embed_documents(["d"])
    assert handler.requests[0]["body"]["input_type"] == "query"
    assert handler.requests[1]["body"]["input_type"] == "document"
    assert handler.requests[0]["body"]["output_dimension"] == 4


@pytest.mark.model
def test_sentence_transformers_real_model():
    emb = registry.build_embedder(ProviderConfig("sentence_transformers", {
        "model": "Qwen/Qwen3-Embedding-0.6B", "device": "cpu", "batch_size": 4}))
    v = emb.embed_query("hello")
    assert len(v) == emb.dim
    assert abs(sum(x * x for x in v) - 1.0) < 1e-3
