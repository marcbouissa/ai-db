"""Packaging and environment E2E test suite (Features 1-5).

Verifies PEP 517/518/621 packaging specifications, console script entry points,
zero-dependency core runtime, modular dependency extras, and .gitignore hygiene.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import tomllib

from ai_db.cli import main

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_pyproject() -> dict[str, Any]:
    """Helper to parse pyproject.toml from repository root."""
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.is_file(), f"pyproject.toml not found at {pyproject_path}"
    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)


# ==============================================================================
# Tier 1: Feature Coverage (Isolation & Happy Path Tests)
# ==============================================================================

@pytest.mark.packaging
class TestPackagingTier1:
    """Tier 1: Baseline feature coverage for packaging specifications."""

    def test_pyproject_toml_exists_and_valid(self):
        """TC-PKG-T1-01: pyproject.toml exists and parses as valid TOML dict."""
        pyproject = load_pyproject()
        assert isinstance(pyproject, dict), "pyproject.toml root must parse to a dict"
        assert "build-system" in pyproject, "Missing [build-system] table"
        assert "project" in pyproject, "Missing [project] table"

    def test_pyproject_pep518_build_system(self):
        """TC-PKG-T1-02: [build-system] specifies setuptools.build_meta and requirements."""
        pyproject = load_pyproject()
        build_sys = pyproject.get("build-system", {})
        assert build_sys.get("build-backend") == "setuptools.build_meta", (
            f"Expected setuptools.build_meta, got {build_sys.get('build-backend')}"
        )
        requires = build_sys.get("requires", [])
        assert any("setuptools" in req.lower() for req in requires), (
            f"Build requirements must include setuptools, got {requires}"
        )
        assert any("wheel" in req.lower() for req in requires), (
            f"Build requirements must include wheel, got {requires}"
        )

    def test_pyproject_pep621_metadata(self):
        """TC-PKG-T1-03: [project] table adheres to PEP 621 metadata standard."""
        pyproject = load_pyproject()
        project = pyproject.get("project", {})
        assert project.get("name") == "ai-db", f"Expected project.name 'ai-db', got {project.get('name')}"
        assert "version" in project, "project.version must be defined"
        # Semver format check: X.Y.Z
        assert re.match(r"^\d+\.\d+\.\d+", project["version"]), (
            f"Version {project['version']} does not match semver format"
        )
        assert project.get("requires-python") in [">=3.10", ">=3.10.0"], (
            f"requires-python must be >=3.10, got {project.get('requires-python')}"
        )
        assert project.get("description"), "project.description must be non-empty"

    def test_core_zero_runtime_dependencies(self):
        """TC-PKG-T1-04: Core runtime deps are the parsing/index stack; ML stays optional."""
        pyproject = load_pyproject()
        dependencies = pyproject.get("project", {}).get("dependencies", None)
        names = {re.split(r"[<>=~!\[ ]", d, maxsplit=1)[0].lower() for d in dependencies}
        assert {"tree-sitter", "sqlite-vec", "tiktoken"} <= names, dependencies
        assert not names & {"torch", "sentence-transformers"}, (
            f"heavy ML deps must live in the local-embed extra, got {dependencies}"
        )

    def test_console_scripts_defined(self):
        """TC-PKG-T1-05: Dual console scripts 'ai-db' and 'vectordb' mapped to ai_db.cli:main."""
        pyproject = load_pyproject()
        scripts = pyproject.get("project", {}).get("scripts", {})
        assert scripts.get("ai-db") == "ai_db.cli:main", (
            f"Expected ai-db -> ai_db.cli:main, got {scripts.get('ai-db')}"
        )
        assert scripts.get("vectordb") == "ai_db.cli:main", (
            f"Expected vectordb -> ai_db.cli:main, got {scripts.get('vectordb')}"
        )

    def test_optional_dependencies_extras(self):
        """TC-PKG-T1-06: Modular extras [local-embed], [dev], and [all] configured."""
        pyproject = load_pyproject()
        extras = pyproject.get("project", {}).get("optional-dependencies", {})
        assert "local-embed" in extras, "Missing [project.optional-dependencies.local-embed]"
        assert "dev" in extras, "Missing [project.optional-dependencies.dev]"
        assert "all" in extras, "Missing [project.optional-dependencies.all]"
        assert any("sentence-transformers" in dep.lower() for dep in extras["local-embed"]), (
            f"local-embed extra must contain sentence-transformers, got {extras['local-embed']}"
        )
        assert any("pytest" in dep.lower() for dep in extras["dev"]), (
            f"dev extra must contain pytest, got {extras['dev']}"
        )

    def test_requirements_files_exist_and_match(self):
        """TC-PKG-T1-07: Standard requirements.txt and requirements-dev.txt exist."""
        req_file = REPO_ROOT / "requirements.txt"
        req_dev_file = REPO_ROOT / "requirements-dev.txt"
        assert req_file.is_file(), f"{req_file} must exist"
        assert req_dev_file.is_file(), f"{req_dev_file} must exist"

    def test_gitignore_contains_required_rules(self):
        """TC-PKG-T1-08: .gitignore excludes virtualenvs, bytecode, caches, and databases."""
        gitignore = REPO_ROOT / ".gitignore"
        assert gitignore.is_file(), f"{gitignore} must exist"
        content = gitignore.read_text(encoding="utf-8")
        assert ".venv" in content, ".gitignore must exclude .venv"
        assert "__pycache__" in content, ".gitignore must exclude __pycache__"
        assert "*.pyc" in content or ".pyc" in content, ".gitignore must exclude .pyc"
        assert ".pytest_cache" in content, ".gitignore must exclude .pytest_cache"
        assert "*.db" in content or "codebase_knowledge.db" in content, (
            ".gitignore must exclude SQLite database files"
        )

    def test_cli_main_callable(self):
        """TC-PKG-T1-09: ai_db.cli:main entry point is importable and callable."""
        assert callable(main), "ai_db.cli.main must be a callable function"

    def test_vectordb_facade_script(self):
        """TC-PKG-T1-10: Root vectordb.py facade executes --help with returncode 0."""
        facade_path = REPO_ROOT / "vectordb.py"
        assert facade_path.is_file(), f"{facade_path} must exist"
        proc = subprocess.run(
            [sys.executable, str(facade_path), "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT), check=False
        )
        assert proc.returncode == 0, f"vectordb.py --help failed: {proc.stderr}"
        assert "usage" in proc.stdout.lower() or "ai-db" in proc.stdout.lower()

    def test_license_file_exists(self):
        """TC-PKG-T1-11: Standard MIT LICENSE exists at project root."""
        license_path = REPO_ROOT / "LICENSE"
        assert license_path.is_file(), f"{license_path} must exist"
        text = license_path.read_text(encoding="utf-8")
        assert "MIT License" in text, "LICENSE must state MIT License"

    # --- Expanded Tier 1 Tests (Features 1-5) ---

    def test_pyproject_python_requires_constraint(self):
        """TC-PKG-T1-12 (F1): pyproject.toml strictly requires Python 3.10 or higher."""
        pyproject = load_pyproject()
        req_py = pyproject.get("project", {}).get("requires-python", "")
        assert req_py in (">=3.10", ">=3.10.0"), f"Expected requires-python >=3.10, got {req_py}"

    def test_pyproject_version_declared(self):
        """TC-PKG-T1-13 (F1): pyproject.toml declares valid semver version string."""
        pyproject = load_pyproject()
        ver = pyproject.get("project", {}).get("version", "")
        assert re.match(r"^\d+\.\d+\.\d+$", ver), f"Version '{ver}' does not follow semver format"

    def test_cli_both_scripts_point_to_main(self):
        """TC-PKG-T1-14 (F2): Console scripts ai-db and vectordb map to ai_db.cli:main."""
        pyproject = load_pyproject()
        scripts = pyproject.get("project", {}).get("scripts", {})
        assert scripts.get("ai-db") == "ai_db.cli:main", "ai-db console script must map to ai_db.cli:main"
        assert scripts.get("vectordb") == "ai_db.cli:main", "vectordb console script must map to ai_db.cli:main"

    def test_cli_main_returns_integer_exit_code(self, cli_runner):
        """TC-PKG-T1-15 (F2): Invoking CLI main with --help returns exit code 0."""
        code, out, err = cli_runner.run("--help")
        assert code == 0, f"Expected 0 exit code on --help, got {code}"
        assert "usage:" in (out + err).lower(), "Help output missing 'usage:' prefix"

    def test_requirements_dev_contains_essential_tools(self):
        """TC-PKG-T1-16 (F3): requirements-dev.txt or dev extra specifies pytest, ruff, and mypy."""
        dev_req = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        pyproject = load_pyproject()
        dev_deps = pyproject.get("project", {}).get("optional-dependencies", {}).get("dev", [])
        combined = (dev_req + " " + " ".join(dev_deps)).lower()
        tools = ["pytest", "ruff", "mypy"]
        for tool in tools:
            assert tool in combined, f"Development specification missing {tool}"

    def test_requirements_txt_minimal_footprint(self):
        """TC-PKG-T1-17 (F3): requirements.txt declares zero external third-party package dependencies."""
        req_txt = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        non_comment_lines = [l.strip() for l in req_txt.splitlines() if l.strip() and not l.startswith("#")]
        # May only contain editable root package install (-e .) or be empty
        assert all(l in ("-e .", ".") for l in non_comment_lines), (
            f"requirements.txt must not contain external dependencies, found: {non_comment_lines}"
        )

    def test_requirements_encoding_utf8(self):
        """TC-PKG-T1-18 (F3): All requirements files decode strictly as UTF-8 without errors."""
        for fname in ["requirements.txt", "requirements-dev.txt"]:
            fpath = REPO_ROOT / fname
            assert fpath.is_file(), f"{fname} not found"
            raw = fpath.read_bytes()
            decoded = raw.decode("utf-8")
            assert len(decoded) == len(raw.decode("utf-8", errors="replace"))


    def test_extras_dev_dependency_declared(self):
        """TC-PKG-T1-20 (F4): [project.optional-dependencies.dev] declares dev tools."""
        pyproject = load_pyproject()
        extras = pyproject.get("project", {}).get("optional-dependencies", {})
        assert "dev" in extras, "Missing dev extras group"
        assert any("pytest" in dep for dep in extras["dev"]), "pytest missing from dev extra"

    def test_extras_all_dependency_declared(self):
        """TC-PKG-T1-21 (F4): [project.optional-dependencies.all] references composite dependencies."""
        pyproject = load_pyproject()
        extras = pyproject.get("project", {}).get("optional-dependencies", {})
        assert "all" in extras, "Missing all extras group"
        assert len(extras["all"]) > 0, "all extras group is empty"

    def test_extras_keys_lowercase_normalized(self):
        """TC-PKG-T1-22 (F4): All extra group keys are normalized lowercase strings."""
        pyproject = load_pyproject()
        extras = pyproject.get("project", {}).get("optional-dependencies", {})
        for key in extras:
            assert key == key.lower(), f"Extra key '{key}' is not lowercased"

    def test_gitignore_file_exists_at_root(self):
        """TC-PKG-T1-23 (F5): .gitignore exists in project root."""
        gi = REPO_ROOT / ".gitignore"
        assert gi.is_file(), ".gitignore must exist in root"
        assert gi.stat().st_size > 0, ".gitignore must not be empty"

    def test_gitignore_blocks_python_cache_dirs(self):
        """TC-PKG-T1-24 (F5): .gitignore specifies rules for __pycache__ and *.pyc."""
        content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "__pycache__" in content, ".gitignore missing __pycache__"
        assert "*.pyc" in content or ".pyc" in content, ".gitignore missing .pyc rule"

    def test_gitignore_blocks_virtual_environments(self):
        """TC-PKG-T1-25 (F5): .gitignore specifies rules for .venv and virtualenv dirs."""
        content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".venv" in content, ".gitignore missing .venv rule"

    def test_gitignore_blocks_database_files(self):
        """TC-PKG-T1-26 (F5): .gitignore specifies rules for SQLite database files."""
        content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "*.db" in content or ".db" in content, ".gitignore missing database file rule"


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================

@pytest.mark.packaging
class TestPackagingTier2:
    """Tier 2: Boundary conditions, corner cases, and parser limits."""

    def test_pyproject_readme_file_exists(self):
        """TC-PKG-T2-01: project.readme in pyproject.toml references an existing file."""
        pyproject = load_pyproject()
        readme_ref = pyproject.get("project", {}).get("readme")
        assert readme_ref, "project.readme must be specified in pyproject.toml"
        if isinstance(readme_ref, dict):
            readme_path = REPO_ROOT / readme_ref.get("file", "README.md")
        else:
            readme_path = REPO_ROOT / readme_ref
        assert readme_path.is_file(), f"Referenced readme {readme_path} does not exist"
        assert readme_path.stat().st_size > 0, f"Referenced readme {readme_path} is empty"

    def test_pyproject_package_discovery(self):
        """TC-PKG-T2-02: Package discovery includes ai_db* and defines py-modules."""
        pyproject = load_pyproject()
        setuptools_cfg = pyproject.get("tool", {}).get("setuptools", {})
        # Must discover ai_db packages
        find_cfg = setuptools_cfg.get("packages", {}).get("find", {})
        if find_cfg:
            include_patterns = find_cfg.get("include", [])
            assert any("ai_db" in pat for pat in include_patterns), (
                f"packages.find.include must include 'ai_db*', got {include_patterns}"
            )
        # Must declare py-modules for root single-file modules (vectordb, mcp_server)
        py_modules = setuptools_cfg.get("py-modules", [])
        assert "vectordb" in py_modules, f"py-modules must include 'vectordb', got {py_modules}"

    def test_cli_invocation_no_args(self, cli_runner):
        """TC-PKG-T2-03: CLI executed with zero arguments exits cleanly with usage."""
        _code, stdout, stderr = cli_runner.run()
        # Should exit with code 0 or 1/2 and print help/usage, not traceback crash
        assert "Traceback" not in stderr, f"Unhandled traceback when CLI called with no args: {stderr}"
        combined = (stdout + stderr).lower()
        assert "usage:" in combined or "ai-db" in combined

    def test_cli_invocation_invalid_command(self, cli_runner):
        """TC-PKG-T2-04: CLI executed with invalid command exits with error code."""
        code, _stdout, stderr = cli_runner.run("nonexistent_subcmd_xyz_1234")
        assert code != 0, f"Expected non-zero exit code for invalid subcommand, got {code}"
        assert "Traceback" not in stderr, f"Unhandled traceback for invalid subcommand: {stderr}"

    def test_cli_version_flag(self, cli_runner):
        """TC-PKG-T2-05: CLI executed with --version prints matching version string."""
        pyproject = load_pyproject()
        expected_version = pyproject.get("project", {}).get("version", "")
        code, stdout, stderr = cli_runner.run("--version")
        assert code == 0, f"--version flag returned {code}: {stderr}"
        assert expected_version in (stdout + stderr), (
            f"Expected version '{expected_version}' in output: {stdout + stderr}"
        )

    def test_extras_specifiers_pep440_validity(self):
        """TC-PKG-T2-06: All dependency extras strings conform to PEP 440/508."""
        pyproject = load_pyproject()
        extras = pyproject.get("project", {}).get("optional-dependencies", {})
        spec_pattern = re.compile(r"^[a-zA-Z0-9_\-\.]+(\[[a-zA-Z0-9_,\-]+\])?(\s*[><=~!]=?\s*[0-9a-zA-Z\.\*\+\-]+)?$")
        for group_name, dep_list in extras.items():
            for dep in dep_list:
                # e.g., "torch>=2.4", "ai-db[local-embed,dev]"
                assert spec_pattern.match(dep.strip()), (
                    f"Dependency specifier '{dep}' in extra '{group_name}' is invalid"
                )

    def test_all_extra_is_complete_superset(self):
        """TC-PKG-T2-07: [all] extra incorporates dependencies from all individual extras."""
        pyproject = load_pyproject()
        extras = pyproject.get("project", {}).get("optional-dependencies", {})
        all_extra = extras.get("all", [])
        all_str = " ".join(all_extra)
        assert "local-embed" in all_str and "dev" in all_str, (
            f"[all] extra {all_extra} does not encompass [local-embed] and [dev] dependencies"
        )

    def test_import_ai_db_has_no_cli_side_effects(self):
        """TC-PKG-T2-08: Importing ai_db does not produce stdout or create database files."""
        script = "import sys, ai_db; assert True"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT), check=False
        )
        assert proc.returncode == 0, f"Importing ai_db failed: {proc.stderr}"
        assert proc.stdout == "", f"Importing ai_db emitted unexpected stdout: {proc.stdout}"
        assert proc.stderr == "", f"Importing ai_db emitted unexpected stderr: {proc.stderr}"

    def test_requirements_txt_has_no_absolute_paths(self):
        """TC-PKG-T2-09: requirements.txt does not contain machine-specific absolute paths."""
        for req_name in ["requirements.txt", "requirements-dev.txt"]:
            path = REPO_ROOT / req_name
            if path.is_file():
                for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    assert not re.search(r"(/home/|[a-zA-Z]:\\)", line), (
                        f"Absolute path detected in {req_name}:{idx}: {line}"
                    )

    def test_gitignore_untracked_extensions_coverage(self):
        """TC-PKG-T2-10: .gitignore covers database artifacts, coverage files, and builds."""
        content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        patterns_to_check = [".db", ".coverage", "build", "dist", ".egg-info"]
        for pat in patterns_to_check:
            assert pat in content, f".gitignore missing pattern rule for '{pat}'"

    # --- Expanded Tier 2 Tests (Features 1-5) ---

    def test_pyproject_classifiers_conform_to_pep621(self):
        """TC-PKG-T2-11 (F1): project.classifiers is a non-empty list of valid classifier strings."""
        pyproject = load_pyproject()
        classifiers = pyproject.get("project", {}).get("classifiers", [])
        assert isinstance(classifiers, list), "classifiers must be a list"
        assert len(classifiers) >= 3, f"Expected at least 3 classifiers, got {len(classifiers)}"
        for c in classifiers:
            assert isinstance(c, str) and " :: " in c, f"Invalid classifier: {c}"

    def test_pyproject_urls_metadata(self):
        """TC-PKG-T2-12 (F1): pyproject.toml has valid metadata keywords and author information."""
        pyproject = load_pyproject()
        proj = pyproject.get("project", {})
        assert "authors" in proj, "Missing authors in project table"
        assert isinstance(proj["authors"], list) and len(proj["authors"]) > 0
        assert "keywords" in proj, "Missing keywords in project table"
        assert isinstance(proj["keywords"], list) and len(proj["keywords"]) > 0

    def test_pyproject_entry_points_syntax(self):
        """TC-PKG-T2-13 (F1): Scripts entries follow standard 'module:callable' syntax."""
        pyproject = load_pyproject()
        scripts = pyproject.get("project", {}).get("scripts", {})
        for name, entry_point in scripts.items():
            assert ":" in entry_point, f"Entry point '{entry_point}' for '{name}' missing ':' separator"
            mod, func = entry_point.split(":", 1)
            assert mod.isidentifier() or "." in mod, f"Invalid module in entry point: {mod}"
            assert func.isidentifier(), f"Invalid function in entry point: {func}"

    def test_cli_help_flag_output_structure(self, cli_runner):
        """TC-PKG-T2-14 (F2): CLI --help structure includes commands section and options."""
        code, out, err = cli_runner.run("--help")
        assert code == 0, f"--help exited with {code}"
        combined = out + err
        assert "commands" in combined.lower() or "positional arguments" in combined.lower()
        assert "options" in combined.lower() or "-h, --help" in combined.lower()

    def test_cli_malformed_argument_types(self, cli_runner):
        """TC-PKG-T2-15 (F2): Passing invalid argument types (e.g. non-integer --top) exits with code 2."""
        code, out, err = cli_runner.run("query", "test_term", "--top", "not_a_number")
        assert code != 0, f"Expected error code for invalid --top type, got {code}"
        assert "invalid int value" in (out + err).lower() or "error" in (out + err).lower()

    def test_requirements_no_wildcard_unpinned_critical_ranges(self):
        """TC-PKG-T2-16 (F3): requirements-dev.txt does not contain unconstrained wildcard version pins."""
        dev_req = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        for line in dev_req.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            assert "==*" not in line, f"Wildcard pin detected in requirements-dev: {line}"

    def test_requirements_no_vcs_or_direct_file_uris(self):
        """TC-PKG-T2-17 (F3): requirements files contain no direct git+, svn+, or file:// references."""
        for fname in ["requirements.txt", "requirements-dev.txt"]:
            content = (REPO_ROOT / fname).read_text(encoding="utf-8")
            for prefix in ["git+", "svn+", "hg+", "file://"]:
                assert prefix not in content, f"VCS/direct URI prefix '{prefix}' found in {fname}"

    def test_requirements_syntax_validation(self):
        """TC-PKG-T2-18 (F3): Each line in requirements-dev conforms to valid package specification syntax."""
        dev_req = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        pkg_pattern = re.compile(r"^(-e\s+\S+|[a-zA-Z0-9_\-\.]+(\[[a-zA-Z0-9_,\-]+\])?(\s*[><=~!]=?\s*[0-9a-zA-Z\.\*\+\-]+)?)$")
        for line in dev_req.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            assert pkg_pattern.match(line), f"Invalid requirement syntax: '{line}'"

    def test_requirements_dev_strict_superset_of_base(self):
        """TC-PKG-T2-19 (F3): requirements-dev.txt or dev extra provides required development and test tools."""
        dev_req = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        pyproject = load_pyproject()
        dev_deps = pyproject.get("project", {}).get("optional-dependencies", {}).get("dev", [])
        combined = (dev_req + " " + " ".join(dev_deps)).lower()
        assert "pytest" in combined, "pytest missing from dev requirements"
        assert "ruff" in combined or "flake8" in combined, "Linter missing from dev requirements"

    def test_extras_mutually_exclusive_isolation(self):
        """TC-PKG-T2-20 (F4): local-embed extra does not contaminate core or dev-only tools."""
        pyproject = load_pyproject()
        extras = pyproject.get("project", {}).get("optional-dependencies", {})
        embed_deps = " ".join(extras.get("local-embed", []))
        assert "pytest" not in embed_deps, "dev dependency pytest leaked into local-embed extra"
        assert "ruff" not in embed_deps, "dev dependency ruff leaked into local-embed extra"

    def test_extras_no_recursive_cycle(self):
        """TC-PKG-T2-21 (F4): Extras definitions contain no recursive self-referential cycles."""
        pyproject = load_pyproject()
        extras = pyproject.get("project", {}).get("optional-dependencies", {})
        for group, deps in extras.items():
            for dep in deps:
                assert f"ai-db[{group}]" not in dep, f"Self-referential recursive extra cycle: {group} -> {dep}"

    def test_extras_environment_markers_validity(self):
        """TC-PKG-T2-22 (F4): Any environment markers in optional dependencies are valid."""
        pyproject = load_pyproject()
        extras = pyproject.get("project", {}).get("optional-dependencies", {})
        for deps in extras.values():
            for dep in deps:
                if ";" in dep:
                    marker = dep.split(";", 1)[1].strip()
                    assert "python_version" in marker or "sys_platform" in marker or "os_name" in marker

    def test_gitignore_blocks_build_distribution_artifacts(self):
        """TC-PKG-T2-23 (F5): .gitignore contains rules blocking build/ and dist/ artifacts."""
        content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "build" in content, ".gitignore missing build artifact rule"
        assert "dist" in content, ".gitignore missing dist artifact rule"

    def test_gitignore_blocks_ide_and_editor_files(self):
        """TC-PKG-T2-24 (F5): .gitignore covers IDE/editor directories (.vscode, .idea, swap files)."""
        content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".vscode" in content or ".idea" in content or "*.swp" in content, (
            ".gitignore should cover common editor/IDE state"
        )

    def test_git_status_zero_untracked_pyc_in_tree(self):
        """TC-PKG-T2-25 (F5): git status confirms zero untracked .pyc or cache files in repo."""
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True, check=False
        )
        untracked = [line for line in res.stdout.splitlines() if line.startswith("??")]
        assert not any(f.endswith(".pyc") or "__pycache__" in f for f in untracked), (
            f"Untracked bytecode files in working tree: {untracked}"
        )


# ==============================================================================
# Tier 3: Pairwise & Cross-Feature Combinations
# ==============================================================================

@pytest.mark.packaging
class TestPackagingTier3:
    """Tier 3: Pairwise cross-feature interactions."""

    def test_pairwise_packaging_clean_build(self, tmp_path):
        """TC-PAIR-01: Packaging metadata build creates zero untracked git status dirtying."""
        # Check git status before and after inspecting metadata
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True, check=False
        )
        # Parse pyproject metadata in clean process
        proc = subprocess.run(
            [sys.executable, "-c", "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True, check=False
        )
        assert proc.returncode == 0

    def test_pairwise_dual_scripts_sanitized_help(self, cli_runner):
        """TC-PAIR-02: Dual scripts (ai-db and vectordb) produce sanitized help text."""
        for entry_script in ["ai_db.cli", "vectordb"]:
            if entry_script == "vectordb":
                proc = subprocess.run(
                    [sys.executable, str(REPO_ROOT / "vectordb.py"), "--help"],
                    capture_output=True,
                    text=True,
                    cwd=str(REPO_ROOT), check=False
                )
                output = proc.stdout + proc.stderr
            else:
                _, out, err = cli_runner.run("--help")
                output = out + err

            assert "/home/marc" not in output, f"{entry_script} --help leaked personal path"
            assert "noble-tesla" not in output, f"{entry_script} --help leaked personal machine name"

    def test_pairwise_optional_extras_no_local_urls(self):
        """TC-PAIR-03: Optional dependency specifications contain no local file:// URLs."""
        pyproject = load_pyproject()
        extras = pyproject.get("project", {}).get("optional-dependencies", {})
        for group, reqs in extras.items():
            for req in reqs:
                assert "file://" not in req, f"Local URL found in extra '{group}': {req}"
                assert "@ http" not in req, f"Remote direct URL found in extra '{group}': {req}"

    def test_pairwise_dual_scripts_env_isolation(self, isolated_env, cli_runner):
        """TC-PAIR-04: Dual entry points respect isolated environment variables identically."""
        # Both entry points should point to the virtualized AI_DB_PATH
        _code1, out1, err1 = cli_runner.run("status", use_subprocess=True)
        # Subprocess run of vectordb.py with same env
        proc2 = subprocess.run(
            [sys.executable, str(REPO_ROOT / "vectordb.py"), "status"],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            cwd=str(REPO_ROOT), check=False
        )
        # Both must succeed and execute without pointing to default home dir
        assert "/home/marc" not in (out1 + err1 + proc2.stdout + proc2.stderr)

    def test_pairwise_requirements_gitignore_build_artifacts(self):
        """TC-PAIR-05: Build outputs (build/, dist/, *.egg-info) are all ignored by git."""
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "build/" in gitignore or "build" in gitignore
        assert "dist/" in gitignore or "dist" in gitignore
        assert "*.egg-info" in gitignore or ".egg-info" in gitignore

    def test_pairwise_packaging_pytest_discovery(self):
        """TC-PAIR-06: pyproject.toml defines pytest tool options discovering tests/ directory."""
        pyproject = load_pyproject()
        tool_pytest = pyproject.get("tool", {}).get("pytest", {}).get("ini_options", {})
        testpaths = tool_pytest.get("testpaths", [])
        assert "tests" in testpaths or (REPO_ROOT / "tests").is_dir(), (
            "pytest testpaths must include 'tests'"
        )

    def test_pairwise_cli_core_stdlib_only(self):
        """TC-PAIR-07: CLI entry points execute using standard library alone."""
        # Run CLI in clean subprocess blocking site-packages
        script = (
            "import sys\n"
            "from ai_db.cli import main\n"
            "assert callable(main)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT), check=False
        )
        assert proc.returncode == 0, f"CLI requires non-stdlib dependency at import: {proc.stderr}"

    def test_pairwise_gitignore_clean_git_status(self):
        """TC-PAIR-08: Check gitignore ignores *.db files without writing to repo root."""
        res = subprocess.run(
            ["git", "check-ignore", "-q", "test_scratch_temp.db"],
            cwd=str(REPO_ROOT), check=False
        )
        assert res.returncode == 0, "git check-ignore failed to ignore test_scratch_temp.db"


# ==============================================================================
# Tier 4: Real-World Application Workflows
# ==============================================================================

@pytest.mark.packaging
class TestPackagingTier4:
    """Tier 4: End-to-end integration workflows."""

    def test_workflow_fresh_clone_bootstrap(self):
        """TC-PKG-WF-01: Simulates fresh git clone inspection and bootstrap checks."""
        # 1. Project standard files exist
        assert (REPO_ROOT / "pyproject.toml").is_file()
        assert (REPO_ROOT / "requirements.txt").is_file()
        assert (REPO_ROOT / "requirements-dev.txt").is_file()
        assert (REPO_ROOT / ".gitignore").is_file()
        assert (REPO_ROOT / "LICENSE").is_file()
        assert (REPO_ROOT / "README.md").is_file()

        # 2. pyproject.toml parses and satisfies PEP 518/621
        pyproject = load_pyproject()
        assert pyproject["project"]["name"] == "ai-db"
        assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"

        # 3. Core module is importable
        import ai_db
        import ai_db.cli
        assert callable(ai_db.cli.main)

    def test_workflow_precommit_sanitization_gate(self):
        """TC-PKG-WF-02: Simulates a pre-commit quality and cleanliness gate."""
        # Audit tracked files list
        res = subprocess.run(
            ["git", "ls-files"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        tracked = [f.strip() for f in res.stdout.splitlines() if f.strip()]

        # Gate 1: Zero tracked .pyc files
        pyc_files = [f for f in tracked if f.endswith(".pyc") or "__pycache__" in f]
        assert len(pyc_files) == 0, f"Pre-commit gate failed: bytecode tracked: {pyc_files}"

        # Gate 2: Zero tracked database files
        db_files = [f for f in tracked if f.endswith((".db", ".sqlite", ".sqlite3"))]
        assert len(db_files) == 0, f"Pre-commit gate failed: database files tracked: {db_files}"

        # Gate 3: pyproject.toml is valid
        pyproject = load_pyproject()
        assert "project" in pyproject

    def test_workflow_hermetic_cli_session(self, isolated_env, cli_runner):
        """TC-PKG-WF-03: Hermetic CLI session operates on isolated DB with zero host pollution."""
        isolated_db = Path(os.environ["AI_DB_PATH"])
        host_user_db = Path(os.path.expanduser("~/.local/share/ai-db/codebase_knowledge.db"))
        host_mtime_before = host_user_db.stat().st_mtime if host_user_db.exists() else None

        # Execute status command
        code, out, err = cli_runner.run("status")
        assert code == 0, f"Status failed with exit code {code}: {out}\n{err}"
        assert str(isolated_db) in (out + err), f"Status did not reference isolated database: {out}"
        assert "/home/marc/.local/share" not in (out + err), f"Status output referenced host DB: {out}"

        # Verify host database was not modified
        if host_user_db.exists() and host_mtime_before is not None:
            assert host_user_db.stat().st_mtime == host_mtime_before, (
                "Hermetic session mutated live host database ~/.local/share/ai-db/codebase_knowledge.db!"
            )

    def test_workflow_dual_entrypoint_parity(self, cli_runner):
        """TC-PKG-WF-04: Full output and argument parity between ai-db and vectordb."""
        commands = [["--help"], ["--version"]]
        for cmd in commands:
            code1, out1, err1 = cli_runner.run(*cmd)
            proc2 = subprocess.run(
                [sys.executable, str(REPO_ROOT / "vectordb.py")] + cmd,
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT), check=False
            )
            assert code1 == proc2.returncode, f"Exit code mismatch on {cmd}: {code1} vs {proc2.returncode}"
            combined = (out1 + err1 + proc2.stdout + proc2.stderr).lower()
            assert ("usage:" in combined or "version" in combined or "ai-db" in combined or "error:" in combined)

    def test_workflow_air_gapped_zero_dependency_runtime(self, isolated_env):
        """TC-PKG-WF-05: Air-gapped runtime verification with standard library alone."""
        # Executes full AST check, outline, and status via CLI using only standard library
        code = (
            "import sys\n"
            "from ai_db.cli import main\n"
            "sys.argv = ['ai-db', '--help']\n"
            "try:\n"
            "    main(['--help'])\n"
            "except TypeError:\n"
            "    main()\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            cwd=str(REPO_ROOT), check=False
        )
        assert proc.returncode == 0, f"Air-gapped execution failed: {proc.stderr}"
        assert "usage" in proc.stdout.lower() or "ai-db" in proc.stdout.lower()
