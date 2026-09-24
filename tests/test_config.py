"""Config loading, `ai-db init`, validation and migration."""

import json
import os

import pytest

from ai_db.cli import main
from ai_db.config import load_config, parse_config
from ai_db.config_template import build_template
from ai_db.errors import AiDbConfigError


@pytest.fixture
def cfg_file(tmp_path, monkeypatch):
    path = tmp_path / "cfgdir" / "config.json"
    monkeypatch.setenv("AI_DB_CONFIG", str(path))
    return path


def test_commands_fail_without_config(cfg_file, capsys):
    assert not cfg_file.exists()
    assert main(["status"]) == 2
    assert "run: ai-db init" in capsys.readouterr().err


def test_init_writes_valid_default(cfg_file):
    assert main(["init"]) == 0
    data = json.loads(cfg_file.read_text())
    assert data["retrieval"]["mode"] == "lexical"
    assert data["embedding"]["provider"] == "none"
    cfg = load_config()
    assert cfg.storage.provider == "sqlite"


def test_init_refuses_overwrite_without_force(cfg_file):
    assert main(["init"]) == 0
    assert main(["init"]) == 2
    assert main(["init", "--force"]) == 0


def test_init_rejects_hybrid_without_embedding(cfg_file):
    assert main(["init", "--mode", "hybrid"]) == 2
    assert not cfg_file.exists()


def test_init_hybrid_with_local_embedding(cfg_file):
    assert main(["init", "--embedding", "sentence_transformers"]) == 0
    data = json.loads(cfg_file.read_text())
    assert data["retrieval"]["mode"] == "hybrid"
    assert data["embedding"]["model"]
    assert "_note" in data["embedding"]


def test_unknown_key_rejected():
    data = build_template()
    data["surprise"] = 1
    with pytest.raises(AiDbConfigError, match="unknown key"):
        parse_config(data)


def test_unknown_version_rejected():
    data = build_template()
    data["version"] = 99
    with pytest.raises(AiDbConfigError, match="unsupported config version"):
        parse_config(data)


def test_missing_provider_option_rejected():
    data = build_template(embedding="voyage")
    del data["embedding"]["model"]
    with pytest.raises(AiDbConfigError, match="requires"):
        parse_config(data, check_env=False)


def test_hosted_provider_requires_api_key_env(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    data = build_template(embedding="voyage")
    with pytest.raises(AiDbConfigError, match="VOYAGE_API_KEY"):
        parse_config(data)
    monkeypatch.setenv("VOYAGE_API_KEY", "x")
    assert parse_config(data).embedding.provider == "voyage"


def test_lexical_with_embedding_rejected():
    data = build_template(embedding="sentence_transformers")
    data["retrieval"]["mode"] = "lexical"
    with pytest.raises(AiDbConfigError):
        parse_config(data)


def test_migrate_legacy(cfg_file):
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(json.dumps({"cross_project_access": {"a": "b"}, "auto_sync_paths": ["~/x"]}))
    with pytest.raises(AiDbConfigError):
        load_config()
    assert main(["init", "--migrate"]) == 0
    cfg = load_config()
    assert cfg.cross_project == {"a": ["b"]}
    assert cfg.auto_sync_paths == ["~/x"]


def test_config_show_masks(cfg_file, capsys, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "secret-value")
    assert main(["init", "--embedding", "voyage"]) == 0
    capsys.readouterr()
    assert main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "secret-value" not in out
    assert '"api_key": "set"' in out


def test_config_check_default(cfg_file, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_DB_PATH", str(tmp_path / "h.db"))
    assert main(["init"]) == 0
    assert main(["config", "check"]) == 0
    assert "OK   storage 'sqlite'" in capsys.readouterr().out


def test_cross_project_access_from_config(tmp_path):
    from ai_db import VectorDB
    data = build_template()
    data["access"]["cross_project"] = {"proj": ["other"]}
    cfg = parse_config(data)
    db = VectorDB(os.path.join(tmp_path, "x.db"), config=cfg)
    from ai_db.utils import get_allowed_projects
    assert "other" in get_allowed_projects("proj", None, db.query_engine.cross_project)
    db.close()
