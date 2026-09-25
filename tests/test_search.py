"""
tests/test_search.py
Comprehensive 4-tier test suite for the Search and Retrieval subsystem.

Tiers:
  - Tier 1: Feature Coverage (BM25 ranking, exact/substring symbols, syntax check, callers, annotations, skill routing)
  - Tier 2: Boundary & Corner Cases (empty queries, special characters, project isolation, corrupt zcontent, top_k bounds)
  - Tier 3: Pairwise Combinations (sync-query update cycle, prune-query consistency, multi-project cross refs)
  - Tier 4: Real-World Workflows (developer code investigation, CI preflight gate, autonomous agent skill routing)
"""
import pytest

from ai_db import VectorDB

# ==============================================================================
# Helpers & Fixtures
# ==============================================================================

@pytest.fixture
def search_env(tmp_path):
    """Provides an isolated VectorDB instance and a temporary workspace folder."""
    db_file = str(tmp_path / "test_search.db")
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    vdb = VectorDB(db_file)
    yield {
        "vdb": vdb,
        "db_file": db_file,
        "src_dir": src_dir,
        "skills_dir": skills_dir,
        "tmp_path": tmp_path,
    }
    vdb.close()


# ==============================================================================
# Tier 1: Feature Coverage (Happy Path Isolation)
# ==============================================================================

class TestSearchTier1FeatureCoverage:
    """Happy-path verification of query, symbol lookup, callers, annotations, and skills."""

    def test_query_bm25_happy_path(self, search_env):
        """Index a Python file; query by distinct term; verify ranking, lines, and snippet."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]

        file_path = src / "metrics.py"
        file_path.write_text(
            "class MetricAggregator:\n"
            "    def calculate_percentiles(self, latencies: list[float]):\n"
            "        '''Calculates p50, p95, and p99 percentiles from measurements.'''\n"
            "        sorted_vals = sorted(latencies)\n"
            "        return sorted_vals\n"
        )
        vdb.sync(str(src), project="analytics", verbose=False)

        hits = vdb.query("calculate_percentiles", project="analytics")
        assert len(hits) > 0
        top = hits[0]
        assert "chunk_id" in top
        assert top["score"] >= 0.0
        assert top["lines"].startswith("L")
        assert "calculate_percentiles" in top["snippet"]
        assert top["project"] == "analytics"

    def test_query_symbol_exact(self, search_env):
        """Lookup an exact class name; verify symbol record fields and types."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]

        (src / "models.py").write_text(
            "class CustomerProfile:\n"
            "    def __init__(self, cid: int):\n"
            "        self.cid = cid\n"
        )
        vdb.sync(str(src), project="crm", verbose=False)

        results = vdb.query_symbol("CustomerProfile", project="crm")
        assert len(results) >= 1
        sym = results[0]
        assert sym["name"] == "CustomerProfile"
        assert sym["symbol_type"] == "class"
        assert sym["line"] == 1
        assert "class CustomerProfile" in sym["signature"]
        assert sym["project"] == "crm"

    def test_query_symbol_substring(self, search_env):
        """Substring lookup returns matches, prioritizing exact matches first."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]

        (src / "orders.py").write_text(
            "class Order:\n"
            "    pass\n\n"
            "class OrderProcessor:\n"
            "    def process_order(self):\n"
            "        pass\n"
        )
        vdb.sync(str(src), project="orders", verbose=False)

        results = vdb.query_symbol("Order", project="orders")
        assert len(results) >= 2
        # Exact match 'Order' should be ranked first
        assert results[0]["name"] == "Order"
        names = [r["name"] for r in results]
        assert "OrderProcessor" in names

    def test_check_syntax_clean(self, search_env):
        """Clean repository reports zero syntax errors."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]

        (src / "clean.py").write_text("def valid(): return 10\n")
        vdb.sync(str(src), project="clean_proj", verbose=False)

        errors = vdb.check_syntax(project="clean_proj")
        assert errors == []

    def test_check_syntax_error_detection(self, search_env):
        """Broken code is indexed in syntax_errors table with line, col, and message."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]

        (src / "bad_syntax.py").write_text("def broken(\n    pass\n")
        vdb.sync(str(src), project="bad_proj", verbose=False)

        errors = vdb.check_syntax(project="bad_proj")
        assert len(errors) == 1
        err = errors[0]
        assert "bad_syntax.py" in err["file"]
        assert err["line"] == 1
        assert isinstance(err["col"], int)
        assert len(err["message"]) > 0

    def test_query_callers_retrieval(self, search_env):
        """Finds callers and call sites referencing a target function."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]

        (src / "logger.py").write_text("def emit_log(msg: str):\n    print(msg)\n")
        (src / "app.py").write_text(
            "from logger import emit_log\n\n"
            "def run_app():\n"
            "    emit_log('App starting')\n"
        )
        vdb.sync(str(src), project="app_proj", verbose=False)

        callers = vdb.query_callers("emit_log", project="app_proj")
        assert len(callers) >= 1
        c = callers[0]
        assert "app.py" in c["file"]
        assert "run_app" in c["caller"]
        assert c["ref_type"] == "call"

    def test_query_annotations_todo(self, search_env):
        """Extracts and filters TODO/FIXME annotations from indexed files."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]

        (src / "tasks.py").write_text(
            "# TODO: implement exponential backoff\n"
            "def perform_task():\n"
            "    # FIXME: avoid hardcoded timeout\n"
            "    pass\n"
        )
        vdb.sync(str(src), project="task_proj", verbose=False)

        todos = vdb.query_annotations(kind="todo", project="task_proj")
        assert len(todos) >= 1
        assert "exponential backoff" in todos[0]["content"]

        fixmes = vdb.query_annotations(kind="fixme", project="task_proj")
        assert len(fixmes) >= 1
        assert "hardcoded timeout" in fixmes[0]["content"]

    def test_route_skills_matching(self, search_env):
        """Skill manifests are ranked and routed based on prompt intent and triggers."""
        vdb = search_env["vdb"]
        skills_dir = search_env["skills_dir"]

        deploy_dir = skills_dir / "db-deploy"
        deploy_dir.mkdir()
        (deploy_dir / "SKILL.md").write_text(
            "---\n"
            "name: db-deploy\n"
            "description: Automated database schema migrations\n"
            "actions:\n"
            "  - migrate database\n"
            "  - apply schema\n"
            "---\n"
            "# DB Deploy\nRun migrations.\n"
        )

        vdb.skill_router.sync_skills(skill_dirs=[str(skills_dir)], project="devops", verbose=False)
        routes = vdb.skill_router.route_skills("migrate database schema", project="devops", min_confidence=0.1)

        assert len(routes) >= 1
        top_skill = routes[0]
        assert top_skill["name"] == "db-deploy"
        assert top_skill["confidence"] > 0.0
        assert "db-deploy" in top_skill["filepath"]

    def test_query_bm25_differential_ranking(self, search_env):
        """Verify that a document with high keyword density is ranked before a low-density document."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]

        high_file = src / "high_density.py"
        high_file.write_text(
            "def telemetry_collector():\n"
            "    '''Telemetry pipeline processes telemetry data and logs telemetry events for telemetry monitoring.'''\n"
            "    return 'telemetry_ok'\n"
        )

        low_file = src / "low_density.py"
        low_file.write_text(
            "def background_worker():\n"
            "    '''Standard background worker with general task queue and optional telemetry reporting.'''\n"
            "    return 'worker_ok'\n"
        )

        vdb.sync(str(src), project="rank_test", verbose=False)
        hits = vdb.query("telemetry", project="rank_test")

        assert len(hits) >= 2, f"Expected at least 2 hits, got {len(hits)}"
        assert "high_density.py" in hits[0]["file"], (
            f"Differential ranking failed: high density file was not ranked first. Hits: {hits}"
        )
        assert hits[0]["score"] > hits[1]["score"], (
            f"High density score ({hits[0]['score']}) not greater than low density score ({hits[1]['score']})"
        )


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================

class TestSearchTier2BoundaryAndCorner:
    """Boundary conditions, project isolation limits, malformed queries, and corrupt data."""

    def test_query_empty_string(self, search_env):
        """Empty or whitespace queries return empty list immediately without error."""
        vdb = search_env["vdb"]
        assert vdb.query("") == []
        assert vdb.query("   \t  ") == []

    def test_query_fts_syntax_chars(self, search_env):
        """Queries containing FTS5 operator syntax are sanitized without crashing."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]
        (src / "sample.py").write_text("def sample(): pass\n")
        vdb.sync(str(src), project="fts_test", verbose=False)

        # Raw FTS operators that could crash unescaped MATCH
        res = vdb.query('def * ( [ ) ] : & | " NOT AND', project="fts_test")
        assert isinstance(res, list)

    def test_query_top_k_bounds(self, search_env):
        """Query respects top_k limits including edge values 0 and large limits."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]
        for i in range(10):
            (src / f"func_{i}.py").write_text(f"def common_worker_{i}(): pass\n")
        vdb.sync(str(src), project="limits", verbose=False)

        assert len(vdb.query("common_worker", top_k=0, project="limits")) == 0
        assert len(vdb.query("common_worker", top_k=2, project="limits")) <= 2
        assert len(vdb.query("common_worker", top_k=50, project="limits")) <= 10

    def test_query_project_isolation_boundary(self, search_env):
        """Queries restricted to project A strictly exclude chunks from project B."""
        vdb = search_env["vdb"]
        tmp = search_env["tmp_path"]

        dir_a = tmp / "alpha"
        dir_a.mkdir()
        (dir_a / "secret_a.py").write_text("def proprietary_algorithm_alpha(): pass\n")

        dir_b = tmp / "beta"
        dir_b.mkdir()
        (dir_b / "secret_b.py").write_text("def proprietary_algorithm_beta(): pass\n")

        vdb.sync(str(dir_a), project="alpha", verbose=False)
        vdb.sync(str(dir_b), project="beta", verbose=False)

        # Querying from alpha must NOT return beta's algorithm
        alpha_hits = vdb.query("proprietary_algorithm", project="alpha")
        alpha_files = [h["file"] for h in alpha_hits]
        assert any("secret_a.py" in f for f in alpha_files)
        assert not any("secret_b.py" in f for f in alpha_files)

    def test_query_allow_project_boundary(self, search_env):
        """Using allowed_projects enables explicit cross-project search access."""
        vdb = search_env["vdb"]
        tmp = search_env["tmp_path"]

        dir_a = tmp / "proj_a"
        dir_a.mkdir()
        (dir_a / "shared_a.py").write_text("def compute_shared_hash(): pass\n")

        dir_b = tmp / "proj_b"
        dir_b.mkdir()
        (dir_b / "shared_b.py").write_text("def compute_shared_hash(): pass\n")

        vdb.sync(str(dir_a), project="proj_a", verbose=False)
        vdb.sync(str(dir_b), project="proj_b", verbose=False)

        # Allowed projects include both
        hits = vdb.query("compute_shared_hash", project="proj_a", allowed_projects=["proj_b"])
        projects = {h["project"] for h in hits}
        assert "proj_a" in projects
        assert "proj_b" in projects

    def test_query_nonexistent_symbol(self, search_env):
        """Looking up non-existent symbol returns empty list without error."""
        vdb = search_env["vdb"]
        assert vdb.query_symbol("__non_existent_symbol_xyz__") == []

    def test_check_syntax_nonexistent_path(self, search_env):
        """Syntax check on a non-existent target path returns empty list safely."""
        vdb = search_env["vdb"]
        assert vdb.check_syntax(target_path="/non/existent/path/file.py") == []

    def test_corrupt_zcontent_handling(self, search_env):
        """Chunks with corrupted binary zcontent raise AiDbStorageError instead of hiding it."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]
        (src / "valid.py").write_text("def healthy_function(): pass\n")
        vdb.sync(str(src), project="corrupt_test", verbose=False)

        # Manually corrupt a chunk in SQLite
        cur = vdb.conn.cursor()
        cur.execute("UPDATE chunks SET zcontent = ? WHERE project = 'corrupt_test'", [b"NOT_ZLIB_DATA"])
        vdb.conn.commit()

        from ai_db.errors import AiDbStorageError
        assert vdb.query("healthy_function", project="corrupt_test")  # FTS snippet only
        with pytest.raises(AiDbStorageError):
            vdb.backend.get_chunks_for_file(str(src / "valid.py"))

    def test_query_unicode_multilingual_terms(self, search_env):
        """Verify queries containing non-ASCII Unicode terms return indexed documents."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]

        intl_file = src / "international.py"
        intl_file.write_text(
            "# Unicode support test\n"
            "def berechne_übertrag():\n"
            "    '''Berechnet den Übertrag für Währungen.'''\n"
            "    return 'übertrag_erledigt'\n\n"
            "def функция_расчета():\n"
            "    '''Функция для расчета баланса.'''\n"
            "    return 'баланс_ок'\n",
            encoding="utf-8"
        )

        vdb.sync(str(src), project="intl_proj", verbose=False)

        # Test German umlaut search
        hits_de = vdb.query("übertrag", project="intl_proj")
        assert len(hits_de) > 0, "Query for 'übertrag' returned 0 hits"
        assert "berechne_übertrag" in hits_de[0]["snippet"] or "übertrag" in hits_de[0]["snippet"]

        # Test Cyrillic search
        hits_ru = vdb.query("функция", project="intl_proj")
        assert len(hits_ru) > 0, "Query for Cyrillic 'функция' returned 0 hits"



# ==============================================================================
# Tier 3: Pairwise Combinations
# ==============================================================================

class TestSearchTier3Combinations:
    """Pairwise interactions between synchronization, search querying, and deletion."""

    def test_search_sync_and_query_cycle(self, search_env):
        """Verify incremental sync: modifying code updates chunk search hits and line numbers."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]
        target = src / "cycle.py"

        # Version 1
        target.write_text("def worker_task():\n    return 100\n")
        vdb.sync(str(src), project="cycle_proj", verbose=False)
        hits_v1 = vdb.query("worker_task", project="cycle_proj")
        assert len(hits_v1) == 1
        assert "return 100" in hits_v1[0]["snippet"]

        # Version 2: append comment and change return
        target.write_text("# Updated header\n# Line 2\ndef worker_task():\n    return 200\n")
        vdb.sync(str(src), project="cycle_proj", verbose=False)
        hits_v2 = vdb.query("worker_task", project="cycle_proj")
        assert len(hits_v2) == 1
        assert "return 200" in hits_v2[0]["snippet"]
        # Line numbers updated (leading comments are attached to the definition)
        assert hits_v2[0]["lines"] == "L1-4"

    def test_search_prune_and_symbol_consistency(self, search_env):
        """Deleting a file and re-syncing removes its symbols and chunks from search."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]
        ephemeral = src / "ephemeral.py"

        ephemeral.write_text("class EphemeralRecord:\n    pass\n")
        vdb.sync(str(src), project="prune_test", verbose=False)
        assert len(vdb.query_symbol("EphemeralRecord", project="prune_test")) == 1

        # Delete file and re-sync
        ephemeral.unlink()
        vdb.sync(str(src), project="prune_test", verbose=False)

        assert len(vdb.query_symbol("EphemeralRecord", project="prune_test")) == 0
        assert len(vdb.query("EphemeralRecord", project="prune_test")) == 0

    def test_search_multi_project_cross_referencing(self, search_env):
        """Cross-project symbol references where caller is in Project A and callee is in Project B."""
        vdb = search_env["vdb"]
        tmp = search_env["tmp_path"]

        lib_dir = tmp / "lib"
        lib_dir.mkdir()
        (lib_dir / "utils.py").write_text("def compute_checksum(data):\n    pass\n")

        app_dir = tmp / "app"
        app_dir.mkdir()
        (app_dir / "main.py").write_text(
            "from utils import compute_checksum\n\n"
            "def run():\n"
            "    compute_checksum('payload')\n"
        )

        vdb.sync(str(lib_dir), project="lib_proj", verbose=False)
        vdb.sync(str(app_dir), project="app_proj", verbose=False)

        # Looking up callers for compute_checksum across projects
        callers = vdb.query_callers("compute_checksum", project="app_proj", allowed_projects=["lib_proj"])
        assert len(callers) >= 1
        assert any("main.py" in c["file"] for c in callers)

    def test_search_annotations_with_syntax_errors(self, search_env):
        """File with syntax error is captured in syntax_errors while annotations are preserved."""
        vdb = search_env["vdb"]
        src = search_env["src_dir"]

        (src / "broken_annotated.py").write_text(
            "# TODO: restore syntax\n"
            "def broken_syntax(\n"
            "    pass\n"
        )
        vdb.sync(str(src), project="hybrid_test", verbose=False)

        errs = vdb.check_syntax(project="hybrid_test")
        assert len(errs) == 1

        todos = vdb.query_annotations(kind="todo", project="hybrid_test")
        assert len(todos) == 1
        assert "restore syntax" in todos[0]["content"]


# ==============================================================================
# Tier 4: Real-World Application Workflows
# ==============================================================================

class TestSearchTier4Workflows:
    """Realistic end-to-end user and agent journeys through the search engine."""

    def test_workflow_developer_code_investigation(self, search_env):
        """
        Developer investigation workflow:
        1. Full-text search for domain concept 'payment'
        2. Identify symbol 'PaymentService'
        3. Lookup exact definition and signature
        4. Inspect call sites and usages
        5. Check pending tech debt annotations
        """
        vdb = search_env["vdb"]
        src = search_env["src_dir"]

        (src / "payment.py").write_text(
            "# TODO: add idempotent payment keys\n"
            "class PaymentService:\n"
            "    def execute_payment(self, amount: float):\n"
            "        return True\n"
        )
        (src / "checkout.py").write_text(
            "from payment import PaymentService\n\n"
            "def checkout_cart():\n"
            "    svc = PaymentService()\n"
            "    svc.execute_payment(99.0)\n"
        )
        vdb.sync(str(src), project="ecommerce", verbose=False)

        # 1. Search concept
        hits = vdb.query("execute_payment", project="ecommerce")
        assert len(hits) >= 1

        # 2 & 3. Locate symbol definition
        syms = vdb.query_symbol("PaymentService", project="ecommerce")
        assert len(syms) >= 1
        assert syms[0]["symbol_type"] == "class"

        # 4. Check callers
        callers = vdb.query_callers("execute_payment", project="ecommerce")
        assert len(callers) >= 1
        assert "checkout.py" in callers[0]["file"]

        # 5. Check debt
        todos = vdb.query_annotations(kind="todo", project="ecommerce")
        assert any("idempotent payment keys" in t["content"] for t in todos)

    def test_workflow_ci_preflight_check(self, search_env):
        """
        CI preflight gate workflow:
        Checks repository for syntax regressions before approving merge.
        """
        vdb = search_env["vdb"]
        src = search_env["src_dir"]

        # Stage 1: Clean build
        (src / "feature.py").write_text("def new_feature():\n    return 'success'\n")
        vdb.sync(str(src), project="ci_check", verbose=False)
        assert len(vdb.check_syntax(project="ci_check")) == 0

        # Stage 2: Developer introduces syntax error
        (src / "regression.py").write_text("class Regression( :\n    pass\n")
        vdb.sync(str(src), project="ci_check", verbose=False)

        gate_errors = vdb.check_syntax(project="ci_check")
        assert len(gate_errors) == 1
        assert "regression.py" in gate_errors[0]["file"]

    def test_workflow_skill_routing_for_autonomous_agent(self, search_env):
        """
        Autonomous agent routes varied prompts to the correct skill manifests.
        """
        vdb = search_env["vdb"]
        skills_dir = search_env["skills_dir"]

        # Register Docker skill
        docker_dir = skills_dir / "docker-helper"
        docker_dir.mkdir()
        (docker_dir / "SKILL.md").write_text(
            "---\n"
            "name: docker-helper\n"
            "description: Build and containerize applications\n"
            "actions:\n"
            "  - docker build\n"
            "  - containerize app\n"
            "---\n"
            "# Docker Helper\n"
        )

        # Register Typing skill
        typing_dir = skills_dir / "mypy-lint"
        typing_dir.mkdir()
        (typing_dir / "SKILL.md").write_text(
            "---\n"
            "name: mypy-lint\n"
            "description: Typecheck Python code using mypy\n"
            "actions:\n"
            "  - typecheck python\n"
            "  - run mypy\n"
            "---\n"
            "# Mypy Lint\n"
        )

        vdb.skill_router.sync_skills(skill_dirs=[str(skills_dir)], project="agent_skills", verbose=False)

        # Routing prompt 1: typing
        match_typing = vdb.skill_router.route_skills("typecheck python with mypy", project="agent_skills")
        assert len(match_typing) >= 1
        assert match_typing[0]["name"] == "mypy-lint"

        # Routing prompt 2: containerization
        match_docker = vdb.skill_router.route_skills("containerize app using docker build", project="agent_skills")
        assert len(match_docker) >= 1
        assert match_docker[0]["name"] == "docker-helper"
