"""Phase 11 completion tests: config v1->v2 migration, skills eval, skill/context vectors."""
from __future__ import annotations

import os
import time

import pytest

from ai_db.config import AiDbConfigError, parse_config
from ai_db.config_template import build_template, migrate, migrate_legacy, migrate_v1
from ai_db.eval.harness import load_skills_golden, run_skills
from ai_db.storage.models import ContextRecord
from ai_db.storage.sqlite_backend import (
    SQLiteBackend,
    cosine_distance,
    pack_vector,
    unpack_vector,
)

FIXTURE_SKILLS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "fixtures", "skills")
GOLDEN_SKILLS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "eval", "golden", "skills.jsonl")


# ------------------------------------------------------------------ 11.1 migrate
def _v1_config() -> dict:
    return {
        "version": 1,
        "storage": {"provider": "sqlite", "options": {"path": None}},
        "retrieval": {"mode": "lexical"},
        "embedding": {"provider": "none"},
        "rerank": {"provider": "none"},
        "access": {"cross_project": {"proj": ["a", "b"]}},
        "auto_sync_paths": ["/repo"],
    }


def test_migrate_v1_adds_v2_keys():
    out = migrate_v1(_v1_config())
    assert out["version"] == 2
    assert out["retrieval"]["doc_weight"] == 0.5
    assert out["rerank"]["top_n"] == 10
    assert out["index"]["ignore"] == [".agents/**"]
    assert out["index"]["doc_weight"] == 0.5
    assert "wait_patterns" in out["trace"]


def test_migrate_v1_preserves_user_values():
    cfg = _v1_config()
    cfg["retrieval"]["doc_weight"] = 0.9
    cfg["rerank"]["top_n"] = 25
    cfg["index"] = {"ignore": ["build/**"]}
    out = migrate_v1(cfg)
    assert out["retrieval"]["doc_weight"] == 0.9
    assert out["rerank"]["top_n"] == 25
    assert out["index"]["ignore"] == ["build/**"]
    assert out["access"]["cross_project"] == {"proj": ["a", "b"]}
    assert out["auto_sync_paths"] == ["/repo"]


def test_migrate_v1_does_not_mutate_input():
    cfg = _v1_config()
    migrate_v1(cfg)
    assert cfg["version"] == 1
    assert "index" not in cfg


def test_migrate_v1_rejects_wrong_version():
    with pytest.raises(ValueError, match="expects version 1"):
        migrate_v1({"version": 2})


def test_migrate_dispatches_on_version():
    assert migrate({"cross_project_access": {}})["version"] == 2
    assert migrate(_v1_config())["version"] == 2
    cur = build_template()
    assert migrate(cur)["version"] == 2
    with pytest.raises(ValueError, match="cannot migrate from version"):
        migrate({"version": 99})


def test_migrate_legacy_still_rejects_versioned():
    with pytest.raises(ValueError, match="already has a 'version'"):
        migrate_legacy({"version": 1})


def test_v1_config_rejected_with_migrate_hint():
    with pytest.raises(AiDbConfigError, match=r"ai-db init --migrate"):
        parse_config(_v1_config())


def test_unknown_version_rejected_with_force_hint():
    with pytest.raises(AiDbConfigError, match=r"ai-db init --force"):
        parse_config({**build_template(), "version": 99})


def test_template_agrees_with_parse_defaults():
    t = build_template()
    assert t["retrieval"].get("doc_weight", 0.5) == 0.5
    assert t["rerank"].get("top_n", 10) == 10


# ------------------------------------------------------------------ 11.6 FTS expr
@pytest.mark.skipif(not os.path.exists(GOLDEN_SKILLS), reason="skills golden missing")
def test_search_contexts_uses_build_fts_identifier_expansion(tmp_path):
    b = SQLiteBackend(str(tmp_path / "c.db"))
    b.initialize()
    b.save_context(ContextRecord(session_id="s1", project="global",
                                title="Cache", summary="lookup by file hash",
                                timestamp=time.time()))
    # camelCase query: the naive whitespace OR-join would look for the literal
    # token "getUserName" and find nothing.
    assert b.search_contexts("getUserName") == []
    assert b.search_contexts("file hash")
    b.close()


# --------------------------------------------------------------- 11.7 vectors
def test_pack_unpack_roundtrip():
    v = [0.5, -0.25, 1.0, 0.0]
    assert unpack_vector(pack_vector(v)) == pytest.approx(v)


def test_cosine_distance_basics():
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    # mismatch / zero vector must not raise and must be maximally distant
    assert cosine_distance([1.0], [1.0, 2.0]) == 1.0
    assert cosine_distance([0.0, 0.0], [1.0, 1.0]) == 1.0


def test_context_vector_written_and_searched(tmp_path):
    class Emb:
        def embed_documents(self, texts):
            out = []
            for t in texts:
                v = [0.0] * 4
                for i, ch in enumerate(t.lower()):
                    v[i % 4] += ord(ch) / 1000.0
                n = sum(x * x for x in v) ** 0.5 or 1.0
                out.append([x / n for x in v])
            return out

    from ai_db.memory.context import ContextMemory
    b = SQLiteBackend(str(tmp_path / "v.db"))
    b.initialize()
    mem = ContextMemory(db=b, embedder=Emb())
    mem.save_context("s1", "OAuth Authentication token refresh", title="OAuth")
    mem.save_context("s2", "database vacuum and prune", title="DB Maint")

    rows = b.conn.execute("SELECT COUNT(*) FROM context_vectors").fetchone()[0]
    assert rows == 2, "one vector per saved context"

    hits = b.search_contexts_vector(Emb().embed_documents(["oauth token"])[0], top_k=5)
    assert next(h["title"] for h in hits) == "OAuth"
    assert hits[0]["distance"] < hits[1]["distance"]

    hyb = b.search_contexts_hybrid("oauth token",
                                   Emb().embed_documents(["oauth token"])[0], top_k=5)
    assert hyb and hyb[0]["title"] == "OAuth"
    # RRF scores must be real, positive and strictly ordered (regression: these
    # were previously abs(bm25()) and came back as 0.0)
    assert all(h["score"] > 0 for h in hyb)
    assert hyb[0]["score"] >= hyb[-1]["score"]
    b.close()


def test_context_vector_cascades_on_context_delete(tmp_path):
    class Emb:
        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    from ai_db.memory.context import ContextMemory
    b = SQLiteBackend(str(tmp_path / "d.db"))
    b.initialize()
    mem = ContextMemory(db=b, embedder=Emb())
    mem.save_context("s1", "something", title="T")
    assert b.conn.execute("SELECT COUNT(*) FROM context_vectors").fetchone()[0] == 1
    b.conn.execute("DELETE FROM contexts")
    b.conn.commit()
    assert b.conn.execute("SELECT COUNT(*) FROM context_vectors").fetchone()[0] == 0
    b.close()


def test_no_embedder_means_no_vectors(tmp_path):
    from ai_db.memory.context import ContextMemory
    b = SQLiteBackend(str(tmp_path / "n.db"))
    b.initialize()
    ContextMemory(db=b, embedder=None).save_context("s1", "x", title="T")
    assert b.conn.execute("SELECT COUNT(*) FROM context_vectors").fetchone()[0] == 0
    b.close()


# ------------------------------------------------------------------ 11.9 skills
def test_load_skills_golden_shape():
    items = load_skills_golden(GOLDEN_SKILLS)
    assert len(items) == 20
    for it in items:
        assert isinstance(it["prompt"], str) and it["prompt"]
        assert isinstance(it["expected_skill"], str) and it["expected_skill"]


def test_load_skills_golden_rejects_bad_shape(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"prompt": "", "expected_skill": "x"}\n')
    with pytest.raises(ValueError, match="'prompt' must be a non-empty string"):
        load_skills_golden(str(p))
    p.write_text('{"prompt": "hi", "expected_skill": ""}\n')
    with pytest.raises(ValueError, match="'expected_skill'"):
        load_skills_golden(str(p))


def test_run_skills_top1_accuracy(tmp_path):
    from ai_db import VectorDB
    db = VectorDB(str(tmp_path / "s.db"))
    # AI_DB_SKILL_DIRS is read at import time, so pass the fixture explicitly.
    db.sync_skills(skill_dirs=[FIXTURE_SKILLS], verbose=False)
    res = run_skills(GOLDEN_SKILLS, db)
    assert res["queries"] == 20
    assert res["skills_indexed"] == 5
    assert res["top1_accuracy"] == 1.0
    assert len(res["per_query"]) == 20
    db.close()


def test_route_skills_declines_off_topic(tmp_path):
    """A prompt matching nothing must return [], not a low-confidence guess."""
    from ai_db import VectorDB
    db = VectorDB(str(tmp_path / "s2.db"))
    db.sync_skills(skill_dirs=[FIXTURE_SKILLS], verbose=False)
    assert db.route_skills("what is the weather tomorrow", top_k=3) == []
    assert db.route_skills("build a pdf report", top_k=1)[0]["name"] == "pdf-report"
    db.close()
