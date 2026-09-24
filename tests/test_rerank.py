"""Rerank providers, final ranking formula, graph centrality and diversity."""

import pytest

from ai_db import VectorDB
from ai_db.config import ProviderConfig, parse_config
from ai_db.config_template import build_template
from ai_db.errors import AiDbConfigError
from ai_db.rerank import registry
from ai_db.rerank.providers import CohereReranker, VoyageReranker
from ai_db.search.ranking import Ranker, diversify, last_component
from ai_db.storage.models import SearchResult
from tests.fake_providers import FakeReranker
from tests.test_embeddings import stub  # noqa: F401 - fixture


def _sr(cid, path, name, score):
    return SearchResult(chunk_id=cid, filepath=path, name=name, chunk_type="code", project="p",
                        start_line=1, end_line=2, score=score, snippet="", qualified_name=name)


def test_last_component():
    assert last_component("QueryEngine.search") == "search"
    assert last_component("imports/globals") == "imports/globals"


def test_diversify_caps_per_file():
    rs = [_sr(i, "/a.py" if i < 5 else f"/{i}.py", f"n{i}", 1.0) for i in range(8)]
    out = diversify(rs, window=5, max_per_file=3)
    assert [r.filepath for r in out[:5]].count("/a.py") == 3
    assert len(out) == 8


def test_exact_symbol_boost_without_reranker():
    class DB:
        def capabilities(self):
            return frozenset({"fts"})
    ranker = Ranker(DB())
    # exact-symbol match breaks a near tie in fused score
    cands = [_sr(1, "/a.py", "helper", 0.0200), _sr(2, "/b.py", "Engine.parse_config", 0.0199),
             _sr(3, "/c.py", "other", 0.0100)]
    out = ranker.rank("parse_config", cands, ["p"])
    assert out[0].chunk_id == 2
    assert out[0].signals["exact_symbol"] == 1.0


@pytest.fixture
def repo(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "core.py").write_text(
        "def central():\n    return 1\n\n"
        "def a():\n    return central()\n\n"
        "def b():\n    return central()\n\n"
        "def lonely():\n    return 0\n"
    )
    (src / "pay.py").write_text("def charge_card(amount):\n    '''charge the credit card'''\n    return amount\n")
    return src


def test_centrality_rebuilt_after_sync(tmp_path, repo):
    db = VectorDB(str(tmp_path / "g.db"))
    db.sync(str(repo), project="p", verbose=False)
    scores = db.backend.get_symbol_centrality(["central", "lonely"], ["p"])
    assert scores["central"] == 1.0
    assert "lonely" not in scores
    db.close()


def test_reranker_reorders_and_only_scores_top(tmp_path, repo, monkeypatch):
    fake = FakeReranker()
    monkeypatch.setitem(registry.BUILTIN, "fake", lambda opts: fake)
    data = build_template()
    data["rerank"] = {"provider": "fake"}
    db = VectorDB(str(tmp_path / "r.db"), config=parse_config(data))
    db.sync(str(repo), project="p", verbose=False)
    hits = db.query("charge credit card", project="p")
    assert hits[0]["name"] == "def charge_card"
    assert fake.calls and max(fake.calls) <= 30
    db.close()


def test_unknown_rerank_provider():
    with pytest.raises(AiDbConfigError, match="unknown rerank provider"):
        registry.build_reranker(ProviderConfig("nope", {}))


def test_voyage_rerank(stub, monkeypatch):  # noqa: F811
    url, handler = stub
    monkeypatch.setenv("VK", "v")
    handler.responses = [(200, {"data": [{"index": 1, "relevance_score": 0.9},
                                         {"index": 0, "relevance_score": 0.1}]})]
    rr = VoyageReranker({"model": "rerank-2.5", "api_key_env": "VK", "base_url": url + "/rerank"})
    assert rr.score("q", ["a", "b"]) == [0.1, 0.9]
    assert handler.requests[0]["body"]["documents"] == ["a", "b"]


def test_cohere_rerank(stub, monkeypatch):  # noqa: F811
    url, handler = stub
    monkeypatch.setenv("CK", "c")
    handler.responses = [(200, {"results": [{"index": 0, "relevance_score": 0.7}]})]
    rr = CohereReranker({"model": "rerank-v3.5", "api_key_env": "CK", "base_url": url + "/v2/rerank"})
    assert rr.score("q", ["a"]) == [0.7]


@pytest.mark.model
def test_cross_encoder_real_model():
    rr = registry.build_reranker(ProviderConfig("sentence_transformers",
                                                {"model": "BAAI/bge-reranker-v2-m3", "device": "cpu"}))
    s = rr.score("parse json config", ["def parse_config(path): return json.load(open(path))",
                                        "def draw_circle(r): pass"])
    assert s[0] > s[1]
