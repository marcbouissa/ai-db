"""Lexical retrieval: query building, identifier expansion, weighted columns, filters."""

import pytest

from ai_db.search.query_builder import base_terms, build_fts, expand_terms, split_identifier
from ai_db.storage.models import ChunkRecord, FileRecord
from ai_db.storage.sqlite_backend import SQLiteBackend


def test_split_identifier():
    assert split_identifier("getUserName") == ["get", "user", "name"]
    assert split_identifier("get_user_name") == ["get", "user", "name"]
    assert split_identifier("HTTPServer") == ["http", "server"]


def test_expand_terms_adds_joined_forms_and_drops_stopwords():
    terms = expand_terms("where is getUserName")
    assert terms == ["get", "user", "name", "getusername", "get_user_name"]


def test_base_terms_only_words():
    assert base_terms("the getUserName of self") == ["get", "user", "name"]


def test_build_fts_shape():
    assert build_fts(["alpha"]) == '"alpha"*'
    q = build_fts(["a1", "b2", "a1b2"], core=["a1", "b2"])
    assert q == '("a1" AND "b2") OR NEAR("a1" "b2", 10) OR ("a1"* OR "b2"* OR "a1b2"*)'
    assert build_fts(['x"y']) == '"x""y"*'
    with pytest.raises(ValueError):
        build_fts([])


@pytest.fixture
def backend(tmp_path):
    b = SQLiteBackend(str(tmp_path / "lex.db"))
    b.initialize()
    yield b
    b.close()


def _add(backend, path, name, content, language="python", chunk_type="code", mtime=1.0):
    backend.upsert_file(FileRecord(path, "h" + path, mtime, 1, "p"))
    backend.insert_chunks([ChunkRecord(path, chunk_type, name, 1, 2, content, "p",
                                       qualified_name=name, language=language)])


def test_name_match_outranks_body_match(backend):
    _add(backend, "/a.py", "def helper", "calls parse_config somewhere in the body text")
    _add(backend, "/b.py", "def parse_config", "return {}")
    hits = backend.search_chunks(expand_terms("parse config"), allowed_projects=["p"], top_k=5,
                                 core_terms=base_terms("parse config"))
    assert hits[0].filepath == "/b.py"


def test_camelcase_document_matches_natural_language(backend):
    _add(backend, "/c.ts", "def loadUserProfile", "const x = fetchRemoteProfile(id)", "typescript")
    hits = backend.search_chunks(expand_terms("remote profile"), allowed_projects=["p"], top_k=5)
    assert hits and hits[0].filepath == "/c.ts"


def test_snippet_highlights_match(backend):
    _add(backend, "/d.py", "def z", "first line\n" * 50 + "the needle is here\n")
    hit = backend.search_chunks(["needle"], allowed_projects=["p"], top_k=1)[0]
    assert "«needle»" in hit.snippet


def test_language_type_and_mtime_filters(backend):
    _add(backend, "/e.py", "def token_py", "token", "python", mtime=100.0)
    _add(backend, "/e.md", "Token docs", "token", "markdown", "md", mtime=200.0)
    only_md = backend.search_chunks(["token"], allowed_projects=["p"], top_k=5, languages=["markdown"])
    assert [h.filepath for h in only_md] == ["/e.md"]
    only_code = backend.search_chunks(["token"], allowed_projects=["p"], top_k=5, chunk_types=["code"])
    assert [h.filepath for h in only_code] == ["/e.py"]
    recent = backend.search_chunks(["token"], allowed_projects=["p"], top_k=5, modified_since=150.0)
    assert [h.filepath for h in recent] == ["/e.md"]
