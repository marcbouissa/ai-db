"""Command-output filtering: `ai-db run`, `ai-db filter`, MCP filter_output."""

import os
import stat
import sys

import pytest

from ai_db import output_store
from ai_db.dispatcher import ServiceDispatcher
from ai_db.output_filter import clean, filter_output, line_range
from ai_db.tokens import count_tokens


def _pytest_log(passed=300):
    lines = ["============================= test session starts ==============================",
             "platform linux -- Python 3.12.3, pytest-9.1.1", "collected 302 items", ""]
    for i in range(passed):
        pct = int(100 * i / passed)
        lines.append(f"test_many.py::test_ok[{i}] PASSED{' ' * 30}[{pct:>3}%]")
    lines += [
        "test_many.py::test_parse_price FAILED                     [ 99%]",
        "test_many.py::test_login_rejects_expired_token FAILED     [100%]",
        "",
        "=================================== FAILURES ===================================",
        "_______________________ test_parse_price _______________________",
        "",
        "    def parse_price(s):",
        ">       return float(s.replace('$', ''))",
        "E       ValueError: could not convert string to float: '1,200.50'",
        "",
        "test_many.py:10: ValueError",
        "_______________________ test_login_rejects_expired_token _______________________",
        "",
        "    def test_login_rejects_expired_token():",
        ">       assert token['exp'] > 1, 'expired token was accepted by login()'",
        "E       AssertionError: expired token was accepted by login()",
        "",
        "test_many.py:17: AssertionError",
        "=========================== short test summary info ============================",
        "FAILED test_many.py::test_parse_price - ValueError: could not convert",
        "FAILED test_many.py::test_login_rejects_expired_token - AssertionError",
        "======================== 2 failed, 300 passed in 0.09s =========================",
    ]
    return "\n".join(lines)


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setattr(output_store, "OUTPUT_STORE_DIR", str(tmp_path / "outputs"))
    # the defaults were bound at import; point them at the temp dir too
    monkeypatch.setattr(output_store.save, "__defaults__", (str(tmp_path / "outputs"),))
    monkeypatch.setattr(output_store.load, "__defaults__", (str(tmp_path / "outputs"),))


def test_short_output_is_returned_whole():
    res = filter_output("all good\n3 passed in 0.1s\n")
    assert res.text == "all good\n3 passed in 0.1s" and res.kept_lines == res.total_lines


def test_failures_and_summary_survive_and_noise_collapses():
    log = _pytest_log()
    res = filter_output(log, budget_tokens=600)
    assert res.tokens_out <= 600 < res.tokens_in
    for must in ("ValueError: could not convert", "expired token was accepted",
                 "2 failed, 300 passed", "test_login_rejects_expired_token FAILED"):
        assert must in res.text, must
    assert "x298 similar lines" in res.text  # 300 PASSED lines -> one marker
    assert res.text.count("PASSED") == 2


def test_need_selects_the_relevant_failure():
    res = filter_output(_pytest_log(), need="login token", budget_tokens=150)
    assert "expired token was accepted" in res.text
    assert "could not convert string to float: '1,200.50'" not in res.text
    assert "2 failed, 300 passed" in res.text  # the summary is always a candidate


def test_omitted_ranges_are_marked_with_line_numbers():
    res = filter_output(_pytest_log(), need="login", budget_tokens=150)
    assert "... [lines 1-" in res.text
    first = int(res.text.split("... [lines 1-")[1].split()[0])
    assert clean(_pytest_log())[first].startswith(("test_", "_", "=", "", " "))


def test_ansi_and_progress_bars_are_cleaned():
    raw = "\x1b[31mERROR\x1b[0m: boom\nDownloading 10%\rDownloading 55%\rDownloading 100%\n"
    assert clean(raw) == ["ERROR: boom", "Downloading 100%"]


def test_python_traceback_is_kept_as_one_block():
    noise = "\n".join(f"INFO step {i} done" for i in range(400))
    tb = ("Traceback (most recent call last):\n  File \"app.py\", line 3, in <module>\n"
          "    main()\n  File \"app.py\", line 2, in main\n    1 / 0\n"
          "ZeroDivisionError: division by zero")
    res = filter_output(f"{noise}\n{tb}\n{noise}", budget_tokens=200)
    assert tb in res.text


def test_line_range_is_exact():
    out = line_range("a\nb\nc\nd", 2, 3)
    assert out.splitlines() == ["     2  b", "     3  c"]


# --- tool surface ------------------------------------------------------------------


def test_filter_output_tool_stores_and_refilters(tmp_path):
    disp = ServiceDispatcher(db_path=str(tmp_path / "x.db"))
    first = disp.execute("filter_output", {"text": _pytest_log(), "command": "pytest",
                                           "exit_code": 1, "budget": 600})
    header = first.splitlines()[0]
    assert "`pytest`, exit 1" in header and "kept" in header
    output_id = header.split()[2].rstrip(":")
    again = disp.execute("filter_output", {"output_id": output_id, "need": "login", "budget": 150})
    assert "expired token was accepted" in again
    exact = disp.execute("filter_output", {"output_id": output_id, "lines": "1-2"})
    assert "test session starts" in exact
    log = os.path.join(str(tmp_path / "outputs"), f"{output_id}.log")
    assert stat.S_IMODE(os.stat(log).st_mode) == 0o600


def test_store_prunes_old_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(output_store, "OUTPUT_STORE_KEEP", 3)
    d = str(tmp_path / "outputs")
    for i in range(5):
        output_store.save(f"out {i}", {}, store_dir=d)  # each save prunes
    assert len([n for n in os.listdir(d) if n.endswith(".log")]) == 3
    assert len([n for n in os.listdir(d) if n.endswith(".json")]) == 3


def test_run_keeps_exit_code_and_filters(tmp_path, capsys, monkeypatch):
    from ai_db import cli

    script = tmp_path / "noisy.py"
    script.write_text("import sys\n"
                      "for i in range(500): print(f'compiling module {i}')\n"
                      "print('error: missing symbol frobnicate in core.c')\n"
                      "sys.exit(3)\n")
    code = cli.main(["run", "--need", "missing symbol", "--", sys.executable, str(script)])
    out = capsys.readouterr().out
    assert code == 3
    assert "exit 3" in out.splitlines()[0]
    assert "missing symbol frobnicate" in out
    assert count_tokens(out) < 300


def test_run_reports_missing_command(capsys):
    from ai_db import cli

    assert cli.main(["run", "--", "definitely-not-a-command-xyz"]) == 127


# --- semantic rerank ----------------------------------------------------------------

def _error_log():
    noise = "\n".join(f"[worker] processed batch {i}: number of records ok" for i in range(150))
    return (f"{noise}\n\nerror: number of open files exceeds limit\n  hint: nothing else\n\n"
            "ConnectionError: reset while dividing work across nodes\n\n"
            "Traceback (most recent call last):\n  File \"stats.py\", line 12, in mean\n"
            "    return total / count\nZeroDivisionError: float division by zero\n\n"
            f"{noise}\n\n3 errors, 297 batches ok")


def _fake_model(need, texts):
    """Stands in for a cross-encoder: knows that division by zero is 'dividing by nothing'."""
    return [9.0 if "ZeroDivisionError" in t else -3.0 for t in texts]


def test_semantic_rerank_finds_a_block_keywords_miss():
    need = "dividing a number by nothing"
    lexical = filter_output(_error_log(), need=need, budget_tokens=55)
    semantic = filter_output(_error_log(), need=need, budget_tokens=55, semantic=_fake_model)
    assert "ZeroDivisionError" not in lexical.text  # keyword overlap favours the distractors
    assert "ZeroDivisionError: float division by zero" in semantic.text and semantic.semantic


def test_semantic_rerank_requires_a_need():
    with pytest.raises(ValueError, match="needs a 'need'"):
        filter_output(_error_log(), semantic=_fake_model)


def test_semantic_scorer_must_score_every_block():
    with pytest.raises(ValueError, match="scores for"):
        filter_output(_error_log(), need="x", semantic=lambda need, texts: [1.0])


def _dispatcher(tmp_path, rerank="none", embedding="none"):
    from ai_db.config import parse_config
    from ai_db.config_template import build_template

    data = build_template()
    data["rerank"] = {"provider": rerank} if rerank == "none" else {
        "provider": rerank, "model": "fake/reranker", "device": "cpu"}
    if embedding != "none":
        data["embedding"] = {"provider": embedding, "model": "fake/embed",
                             "device": "cpu", "batch_size": 4}
        data["retrieval"]["mode"] = "hybrid"
    return ServiceDispatcher(db_path=str(tmp_path / "x.db"), config=parse_config(data, check_env=False))


def test_rerank_tool_uses_the_configured_reranker(tmp_path, monkeypatch):
    from ai_db import lazy

    calls = []

    def fake_score(self, query, texts):
        calls.append(query)
        return _fake_model(query, texts)

    monkeypatch.setattr(lazy.LazyReranker, "score", fake_score)
    disp = _dispatcher(tmp_path, rerank="sentence_transformers")
    out = disp.execute("filter_output", {"text": _error_log(), "rerank": True,
                                         "need": "dividing a number by nothing", "budget": 55})
    assert calls == ["dividing a number by nothing"]
    assert "(semantic rerank)" in out.splitlines()[0] and "ZeroDivisionError" in out


def test_rerank_tool_falls_back_to_embedding_cosine(tmp_path, monkeypatch):
    from ai_db import lazy

    def unit(v):
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]

    monkeypatch.setattr(lazy.LazyEmbedder, "embed_query", lambda self, t: unit([1.0, 0.0]))
    monkeypatch.setattr(lazy.LazyEmbedder, "embed_documents", lambda self, texts: [
        unit([1.0, 0.0]) if "ZeroDivisionError" in t else unit([0.0, 1.0]) for t in texts])
    disp = _dispatcher(tmp_path, embedding="sentence_transformers")
    out = disp.execute("filter_output", {"text": _error_log(), "rerank": True,
                                         "need": "dividing a number by nothing", "budget": 55})
    assert "ZeroDivisionError: float division by zero" in out


def test_rerank_without_any_model_is_a_config_error(tmp_path):
    from ai_db.errors import AiDbConfigError

    with pytest.raises(AiDbConfigError, match="needs a model"):
        _dispatcher(tmp_path).execute("filter_output", {"text": _error_log(), "need": "x",
                                                        "rerank": True})


def test_a_single_line_larger_than_the_budget_is_truncated_not_dropped():
    long_line = "files differing: [" + ", ".join(f"'ai_db/module_{i}.py'" for i in range(400)) + "]"
    res = filter_output(long_line, need="differing files", budget_tokens=120)
    assert res.text.startswith("files differing: ['ai_db/module_0.py'")
    assert "characters of this line omitted" in res.text
    assert res.text.rstrip().endswith("'ai_db/module_399.py']")
    assert res.tokens_out <= 160 and res.kept_lines == 1
