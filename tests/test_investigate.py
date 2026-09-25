"""investigate: deterministic evidence packs."""

import json

import pytest

from ai_db import VectorDB
from ai_db.analysis.investigate import is_test_path
from ai_db.analysis.resolve import qualified_of, scope_of
from ai_db.dispatcher import ServiceDispatcher
from ai_db.parser.chunker import count_tokens


@pytest.fixture
def repo(tmp_path):
    src = tmp_path / "repo"
    (src / "pkg").mkdir(parents=True)
    (src / "tests").mkdir()
    (src / "pkg" / "billing.py").write_text(
        "from pkg.tax import compute_tax\n\n"
        "class Invoice:\n"
        "    \"\"\"An invoice with line items.\"\"\"\n"
        "    currency = 'EUR'\n\n"
        "    def total_amount(self, items):\n"
        "        \"\"\"Sum line items and add tax.\"\"\"\n"
        "        subtotal = sum(items)\n"
        "        return subtotal + compute_tax(subtotal)\n\n"
        "def issue_invoice(items):\n"
        "    return Invoice().total_amount(items)\n"
    )
    (src / "pkg" / "tax.py").write_text(
        "RATE = 0.2\n\n"
        "def compute_tax(amount):\n"
        "    \"\"\"Tax for an amount.\"\"\"\n"
        "    return amount * RATE\n"
    )
    (src / "pkg" / "api.py").write_text(
        "from pkg.billing import issue_invoice\n\n"
        "def handle_checkout(req):\n"
        "    return issue_invoice(req['items'])\n"
    )
    (src / "tests" / "test_billing.py").write_text(
        "from pkg.billing import Invoice\n\n"
        "def test_total_amount():\n"
        "    assert Invoice().total_amount([10]) == 12\n"
    )
    return src


@pytest.fixture
def db(tmp_path, repo):
    d = VectorDB(str(tmp_path / "inv.db"))
    d.sync(str(repo), project="p", verbose=False)
    yield d
    d.close()


def _names(pack, role=None):
    return [e["qualified_name"] for e in pack["evidence"] if role is None or e["role"] == role]


def test_helpers():
    assert scope_of("A.b") == "module.A.b"
    assert qualified_of("module.A.b") == "A.b"
    assert qualified_of("module") is None
    assert is_test_path("/r/tests/x.py") and is_test_path("/r/test_x.py")
    assert not is_test_path("/r/pkg/testing_utils.py")


def test_explain_includes_seed_class_callee_caller_and_test(db):
    pack = db.investigate("total amount of an invoice with tax", mode="explain", project="p")
    assert pack["entry_points"][0]["qualified_name"] == "Invoice.total_amount"
    assert "Invoice" in _names(pack)  # class header (as seed or parent)
    assert "compute_tax" in _names(pack)  # callee (or lexical seed)
    assert "issue_invoice" in _names(pack)
    assert ["issue_invoice", "Invoice.total_amount"] in pack["call_graph"]["edges"]
    assert [t["test"] for t in pack["tests"]] == ["test_total_amount"]
    assert ["Invoice.total_amount", "compute_tax"] in pack["call_graph"]["edges"]
    seed = next(e for e in pack["evidence"] if e["qualified_name"] == "Invoice.total_amount")
    assert "body" in seed and "subtotal + compute_tax" in seed["body"]
    assert all(e["ref"].startswith("ref:") for e in pack["evidence"])
    assert all(("body" in e) != ("stub" in e) for e in pack["evidence"])
    assert "rank 1" in pack["entry_points"][0]["why"]


def test_impact_walks_callers_transitively(db):
    pack = db.investigate("compute_tax", mode="impact", project="p")
    assert {"issue_invoice", "handle_checkout"} <= set(_names(pack, "caller"))
    assert "Invoice.total_amount" in _names(pack)  # direct caller (also a lexical seed)
    assert "test_total_amount" in _names(pack, "test")


def test_locate_has_no_expansion(db):
    pack = db.investigate("compute tax", mode="locate", project="p")
    assert {e["role"] for e in pack["evidence"]} == {"seed"}


def test_budget_is_respected_and_refs_expand(db):
    pack = db.investigate("invoice", mode="explain", project="p", budget_tokens=500)
    assert pack["token_count"] <= 500
    assert abs(pack["token_count"] - count_tokens(json.dumps(pack, ensure_ascii=False))) <= 5
    ev = pack["evidence"][0]
    expanded = db.expand_ref(ev["ref"])
    assert expanded is not None


def test_deterministic(db):
    a = db.investigate("invoice tax", project="p")
    b = db.investigate("invoice tax", project="p")
    for pk in (a, b):
        for e in pk["evidence"]:
            e.pop("ref")
        for o in pk["omitted"]:
            o.pop("ref")
    assert a == b


def test_retrieval_block_and_validation(db):
    pack = db.investigate("invoice", project="p")
    assert pack["retrieval"] == {"mode": "lexical", "embedding_model": None, "rerank_model": None}
    with pytest.raises(ValueError):
        db.investigate("invoice", mode="bogus", project="p")
    with pytest.raises(ValueError):
        db.investigate("invoice", budget_tokens=10, project="p")


def test_dispatcher_tool(db):
    disp = ServiceDispatcher(db=db)
    tool = disp.get_tool("investigate")
    assert "INSTEAD" in tool.description
    pack = disp.execute("investigate", {"query": "compute tax", "project": "p", "mode": "locate"})
    assert pack["mode"] == "locate" and pack["evidence"]
