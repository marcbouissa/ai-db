"""SQLite passes the shipped conformance kit; third-party backends load via entry points."""

import os
import sys

import pytest

from ai_db.config import parse_config
from ai_db.config_template import build_template
from ai_db.errors import AiDbConfigError
from ai_db.storage.conformance import BackendConformance
from ai_db.storage.factory import StorageBackendFactory, available_providers
from ai_db.storage.sqlite_backend import SQLiteBackend

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "dummy_backend")


class TestSQLiteConformance(BackendConformance):
    @pytest.fixture
    def backend(self, tmp_path):
        b = SQLiteBackend(str(tmp_path / "conf.db"))
        b.initialize()
        yield b
        b.close()


@pytest.mark.parametrize("vector_index", ["exact", "vec0"])
class TestSQLiteConformanceVec0(BackendConformance):
    """The same conformance contract must hold for the sqlite-vec ANN backend.

    TODO 13.2: the only difference is the vector search implementation, so every
    storage guarantee has to be re-verified against it.
    """

    @pytest.fixture
    def backend(self, tmp_path, vector_index):
        b = SQLiteBackend(str(tmp_path / f"conf_{vector_index}.db"), vector_index=vector_index)
        b.initialize()
        yield b
        b.close()


@pytest.fixture
def dummy_on_path(monkeypatch):
    monkeypatch.syspath_prepend(FIXTURE_DIR)
    yield
    sys.modules.pop("dummy_ai_db_backend", None)


def test_sqlite_is_registered_entry_point():
    assert "sqlite" in available_providers()


def test_third_party_backend_discovered(dummy_on_path):
    assert "dummy" in available_providers()
    cfg = parse_config(build_template(storage="dummy"))
    backend = StorageBackendFactory.from_config(cfg)
    backend.initialize()
    assert backend.backend_name == "dummy"
    backend.close()


def test_unknown_provider_lists_installed():
    cfg = parse_config(build_template(storage="nope"))
    with pytest.raises(AiDbConfigError, match="installed providers"):
        StorageBackendFactory.from_config(cfg)


def test_vectordb_uses_configured_provider(dummy_on_path):
    from ai_db import VectorDB
    cfg = parse_config(build_template(storage="dummy"))
    db = VectorDB(config=cfg)
    assert db.backend.backend_name == "dummy"
    db.close()


def test_thread_local_connections(tmp_path):
    import threading
    b = SQLiteBackend(str(tmp_path / "t.db"))
    b.initialize()
    seen = []
    t = threading.Thread(target=lambda: seen.append(b.conn))
    t.start()
    t.join()
    assert seen[0] is not b.conn
    b.close()
