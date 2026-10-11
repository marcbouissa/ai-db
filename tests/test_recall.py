"""Recall extensions: the built-in ``vectors`` provider for skill and context recall."""

import importlib.metadata
import textwrap

import pytest

from ai_db.config import parse_config
from ai_db.config_template import build_template
from ai_db.constants import RECALL_VECTORS_STATE_KEY
from ai_db.errors import AiDbConfigError, AiDbStorageError
from ai_db.ext import recall as recall_mod
from ai_db.ext.recall import RecallProvider, build_recall_providers
from ai_db.ext.vector_recall import VectorRecall
from ai_db.memory.context import ContextMemory
from ai_db.search.skills import SkillRouter
from ai_db.storage.sqlite_backend import SQLiteBackend, cosine_distance

# A toy embedder with real "semantics": words map to concept axes, so "ship it to
# production" is close to a skill about deployment without sharing any keyword.
_CONCEPTS = {
    "deploy": 0, "ship": 0, "production": 0, "release": 0, "rollout": 0,
    "database": 1, "vacuum": 1, "sql": 1, "tables": 1,
    "login": 2, "oauth": 2, "token": 2, "auth": 2, "password": 2,
}


class ConceptEmbedder:
    model_id = "test:concepts"

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        v = [0.0, 0.0, 0.0, 0.01]
        for word in text.lower().replace(",", " ").split():
            if word in _CONCEPTS:
                v[_CONCEPTS[word]] += 1.0
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]

    def embed_documents(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def backend(tmp_path):
    b = SQLiteBackend(str(tmp_path / "r.db"))
    b.initialize()
    yield b
    b.close()


def _write_skill(root, name, description):
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: {description}
        ---
        # {name}
        """))


# --- extension point --------------------------------------------------------------


def test_vectors_is_registered_as_a_builtin_entry_point():
    eps = {ep.name: ep.value for ep in importlib.metadata.entry_points(group="ai_db.recall")}
    assert eps["vectors"] == "ai_db.ext.vector_recall:VectorRecall.from_options"


def test_config_default_follows_the_embedder():
    assert parse_config(build_template(), check_env=False).recall_providers == []
    hybrid = build_template()
    hybrid["embedding"] = {"provider": "sentence_transformers", "model": "x",
                           "batch_size": 8, "device": "cpu"}
    hybrid["retrieval"]["mode"] = "hybrid"
    assert parse_config(hybrid, check_env=False).recall_providers == ["vectors"]


def test_vectors_without_an_embedder_is_a_config_error():
    data = build_template()
    data["recall"] = {"providers": ["vectors"]}
    with pytest.raises(AiDbConfigError, match="needs an embedding provider"):
        parse_config(data, check_env=False)


def test_unknown_provider_is_reported(backend):
    with pytest.raises(AiDbConfigError, match="not installed"):
        build_recall_providers(["nope"], backend, None)


def test_third_party_providers_plug_in_the_same_way(backend, monkeypatch):
    class Fixed(RecallProvider):
        name = "fixed"

        def index_skill(self, skill): pass
        def index_context(self, context_id, title, summary): pass
        def sync(self): return 0
        def rank_skills(self, prompt, allowed, limit):
            return [recall_mod.SkillHit("chosen", "p", 0.99)]
        def rank_contexts(self, query, allowed, limit): return []

    class EP:
        name, value = "fixed", "pkg:Fixed"

        def load(self):
            return lambda backend, embedder, options: Fixed()

    monkeypatch.setattr(recall_mod, "entry_points", lambda group: [EP()])
    assert [p.name for p in build_recall_providers(["fixed"], backend, None)] == ["fixed"]


# --- contexts -----------------------------------------------------------------------


def test_context_recall_is_semantic_and_fused_with_bm25(backend):
    mem = ContextMemory(db=backend, recall=[VectorRecall(backend, ConceptEmbedder())])
    mem.save_context("s1", "refresh the oauth token on expiry", title="Auth work")
    mem.save_context("s2", "vacuum the sql tables nightly", title="DB upkeep")
    assert backend.conn.execute("SELECT COUNT(*) FROM context_vectors").fetchone()[0] == 2

    hits = mem.query_contexts("login password", project="global")  # no shared keyword
    assert hits[0]["session_id"] == "s1"
    assert hits[0]["match"] == ["vectors"]

    both = mem.query_contexts("oauth token", project="global")
    assert both[0]["session_id"] == "s1" and set(both[0]["match"]) == {"bm25", "vectors"}


def test_context_vector_is_deleted_with_its_context(backend):
    mem = ContextMemory(db=backend, recall=[VectorRecall(backend, ConceptEmbedder())])
    mem.save_context("s1", "deploy to production", title="T")
    mem.delete_context("s1")
    assert backend.conn.execute("SELECT COUNT(*) FROM context_vectors").fetchone()[0] == 0


def test_no_providers_means_bm25_only(backend):
    mem = ContextMemory(db=backend)
    mem.save_context("s1", "deploy to production", title="T")
    assert backend.conn.execute("SELECT COUNT(*) FROM context_vectors").fetchone()[0] == 0
    assert [h["session_id"] for h in mem.query_contexts("deploy", project="global")] == ["s1"]


def test_embedding_failures_propagate(backend):
    class Broken(ConceptEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("model unavailable")

    mem = ContextMemory(db=backend, recall=[VectorRecall(backend, Broken())])
    with pytest.raises(RuntimeError, match="model unavailable"):
        mem.save_context("s1", "x", title="T")


# --- skills -------------------------------------------------------------------------


def test_skill_routing_uses_semantic_similarity(backend, tmp_path):
    skills = tmp_path / "skills"
    _write_skill(skills, "cd-pipeline", "deploy release rollout")
    _write_skill(skills, "dba", "database vacuum")
    router = SkillRouter(db=backend, recall=[VectorRecall(backend, ConceptEmbedder())])
    router.sync_skills([str(skills)], project="p", verbose=False)

    keyword_only = SkillRouter(db=backend).route_skills("ship it to production", project="p")
    assert keyword_only == []  # no keyword overlap at all

    routed = router.route_skills("ship it to production", project="p")
    assert routed and routed[0]["name"] == "cd-pipeline"
    assert any("vectors similarity" in r for r in routed[0]["reasons"])


def test_skills_indexed_before_the_provider_are_backfilled(backend, tmp_path):
    skills = tmp_path / "skills"
    _write_skill(skills, "shipper", "deploy release")
    SkillRouter(db=backend).sync_skills([str(skills)], project="p", verbose=False)
    assert backend.conn.execute("SELECT COUNT(*) FROM skill_vectors").fetchone()[0] == 0

    router = SkillRouter(db=backend, recall=[VectorRecall(backend, ConceptEmbedder())])
    router.sync_skills([str(skills)], project="p", verbose=False)  # file unchanged
    assert backend.conn.execute("SELECT COUNT(*) FROM skill_vectors").fetchone()[0] == 1


def test_model_change_drops_and_rebuilds_the_vectors(backend):
    first = ConceptEmbedder()
    ContextMemory(db=backend, recall=[VectorRecall(backend, first)]).save_context("s1", "deploy", title="T")
    assert backend.get_state(RECALL_VECTORS_STATE_KEY) == {"model_id": "test:concepts"}

    class Other(ConceptEmbedder):
        model_id = "test:other"

        def _vec(self, text):
            return [1.0, 0.0]  # different dimension, as a real model swap would be

    provider = VectorRecall(backend, Other())
    hits = provider.rank_contexts("anything", ["global"], 5)  # would raise on mixed dims
    assert [h.session_id for h in hits] == ["s1"]
    assert backend.get_state(RECALL_VECTORS_STATE_KEY) == {"model_id": "test:other"}


def test_cosine_distance_rejects_mixed_dimensions():
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert cosine_distance([0.0, 0.0], [1.0, 1.0]) == 1.0
    with pytest.raises(AiDbStorageError, match="dimension"):
        cosine_distance([1.0], [1.0, 2.0])
