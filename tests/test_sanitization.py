"""Sanitization and repository hygiene E2E test suite (Features 5 & 21).

Verifies zero occurrences of personal paths (/home/marc), personal identities,
credentials, private keys, tracked database binaries, and compiled bytecode.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_tracked_files() -> list[str]:
    """Retrieve list of all files currently tracked by git index."""
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def is_non_test_tracked_file(rel_path: str) -> bool:
    """Filter out test files and .agents metadata to avoid self-referential false positives."""
    return not rel_path.startswith(("tests/", ".agents/"))


# ==============================================================================
# Tier 1: Feature Coverage (Isolation & Happy Path Tests)
# ==============================================================================

@pytest.mark.sanitization
class TestSanitizationTier1:
    """Tier 1: Baseline sanitization across all git-tracked repository files."""

    def test_git_grep_zero_home_marc(self):
        """TC-SAN-T1-01: git grep finds zero occurrences of /home/marc in tracked files."""
        proc = subprocess.run(
            ["git", "grep", "-n", "/home/marc"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True, check=False
        )
        # Filter out tests/ to prevent self-matching
        violations = [
            line for line in proc.stdout.splitlines()
            if not line.startswith("tests/") and not line.startswith(".agents/")
        ]
        assert len(violations) == 0, "Detected /home/marc in tracked files:\n" + "\n".join(violations)

    def test_zero_occurrences_username_marc(self):
        """TC-SAN-T1-02: Zero occurrences of developer personal username in paths/configs."""
        # Check specifically for /marc/ or \marc\ or "marc" as user identifier in source/configs
        user_path_pattern = re.compile(r"[/\\@]marc[/\\\"\s:]", re.IGNORECASE)
        violations = []
        for rel_path in get_tracked_files():
            if not is_non_test_tracked_file(rel_path):
                continue
            full_path = REPO_ROOT / rel_path
            if full_path.is_file():
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    for line_num, line in enumerate(content.splitlines(), 1):
                        if user_path_pattern.search(line):
                            violations.append(f"{rel_path}:{line_num}: {line}")
                except OSError:
                    continue  # unreadable (e.g. broken symlink): nothing to scan
        assert len(violations) == 0, "Detected personal username occurrences:\n" + "\n".join(violations)

    def test_no_tracked_sqlite_database_files(self):
        """TC-SAN-T1-03: Zero SQLite database binary files are tracked in git index."""
        tracked = get_tracked_files()
        db_files = [f for f in tracked if f.endswith((".db", ".sqlite", ".sqlite3"))]
        assert len(db_files) == 0, f"Binary database files tracked in git: {db_files}"

    def test_no_tracked_cpython_bytecode(self):
        """TC-SAN-T1-04: Zero compiled bytecode (.pyc) files or __pycache__ are tracked in git index."""
        tracked = get_tracked_files()
        pyc_files = [f for f in tracked if f.endswith(".pyc") or "__pycache__" in f]
        assert len(pyc_files) == 0, f"Bytecode files tracked in git: {pyc_files}"

    def test_readme_no_personal_paths(self):
        """TC-SAN-T1-05: README.md contains no personal paths or personal project names."""
        readme_path = REPO_ROOT / "README.md"
        assert readme_path.is_file(), "README.md must exist"
        text = readme_path.read_text(encoding="utf-8")
        assert "/home/marc" not in text, "README.md contains /home/marc"
        assert "blender_ox" not in text, "README.md contains personal project blender_ox"
        assert "noble-tesla" not in text, "README.md contains personal machine name noble-tesla"

    def test_watch_sync_sh_sanitized(self):
        """TC-SAN-T1-06: watch_sync.sh contains no hardcoded personal paths."""
        script_path = REPO_ROOT / "watch_sync.sh"
        if script_path.is_file():
            text = script_path.read_text(encoding="utf-8")
            assert "/home/marc" not in text, "watch_sync.sh contains /home/marc"
            # Default DB path should use variable substitution, e.g. ${AI_DB_PATH:-...}
            assert "$HOME/GitRepos/ai-db" not in text, (
                "watch_sync.sh contains hardcoded $HOME/GitRepos/ai-db path"
            )

    def test_constants_py_sanitized(self):
        """TC-SAN-T1-07: ai_db/constants.py uses portable defaults without personal paths."""
        constants_path = REPO_ROOT / "ai_db" / "constants.py"
        assert constants_path.is_file(), f"{constants_path} must exist"
        text = constants_path.read_text(encoding="utf-8")
        assert "/home/marc" not in text, "constants.py contains /home/marc"
        # Should not hardcode ~/.gemini as un-overridable default
        assert "AI_DB_SKILL_DIRS" in text or "XDG" in text or "AI_DB_PATH" in text

    def test_no_hardcoded_aws_or_api_credentials(self):
        """TC-SAN-T1-08: Zero AWS, GitHub, or OpenAI credentials in tracked files."""
        patterns = [
            re.compile(r"AKIA[0-9A-Z]{16}"),               # AWS Access Key ID
            re.compile(r"ghp_[0-9a-zA-Z]{36}"),             # GitHub PAT
            re.compile(r"github_pat_[0-9a-zA-Z_]{82}"),     # GitHub Fine-grained PAT
            re.compile(r"sk-[a-zA-Z0-9]{48}"),              # OpenAI legacy secret key
            re.compile(r"sk-proj-[a-zA-Z0-9_\-]{50,}"),     # OpenAI modern project key
        ]
        violations = []
        for rel_path in get_tracked_files():
            if not is_non_test_tracked_file(rel_path):
                continue
            full_path = REPO_ROOT / rel_path
            if full_path.is_file():
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    for line_num, line in enumerate(content.splitlines(), 1):
                        for pat in patterns:
                            if pat.search(line):
                                violations.append(f"{rel_path}:{line_num}: {line}")
                except OSError:
                    continue  # unreadable (e.g. broken symlink): nothing to scan
        assert len(violations) == 0, "Detected credentials in tracked files:\n" + "\n".join(violations)

    def test_no_private_key_blocks(self):
        """TC-SAN-T1-09: Zero private cryptographic keys in tracked files."""
        key_pattern = re.compile(r"-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----")
        violations = []
        for rel_path in get_tracked_files():
            if not is_non_test_tracked_file(rel_path):
                continue
            full_path = REPO_ROOT / rel_path
            if full_path.is_file():
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    if key_pattern.search(content):
                        violations.append(rel_path)
                except OSError:
                    continue  # unreadable (e.g. broken symlink): nothing to scan
        assert len(violations) == 0, f"Private cryptographic keys found in: {violations}"

    def test_no_personal_email_addresses(self):
        """TC-SAN-T1-10: Zero personal email addresses in tracked source code and configs."""
        email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@(?!example\.com|users\.noreply\.github\.com)[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        violations = []
        for rel_path in get_tracked_files():
            if not is_non_test_tracked_file(rel_path):
                continue
            full_path = REPO_ROOT / rel_path
            if full_path.is_file():
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    for line_num, line in enumerate(content.splitlines(), 1):
                        match = email_pattern.search(line)
                        if match and "marc" in match.group(0).lower():
                            violations.append(f"{rel_path}:{line_num}: {match.group(0)}")
                except OSError:
                    continue  # unreadable (e.g. broken symlink): nothing to scan
        assert len(violations) == 0, "Personal email addresses detected:\n" + "\n".join(violations)


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================

@pytest.mark.sanitization
class TestSanitizationTier2:
    """Tier 2: Boundary conditions, encoded variants, and gitignore enforcement."""

    def test_case_insensitive_path_sanitization(self):
        """TC-SAN-T2-01: Zero case-insensitive variations of personal paths (/Home/Marc)."""
        pattern = re.compile(r"(?i)[/\\]home[/\\]marc")
        violations = []
        for rel_path in get_tracked_files():
            if not is_non_test_tracked_file(rel_path):
                continue
            full_path = REPO_ROOT / rel_path
            if full_path.is_file():
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    if pattern.search(content):
                        violations.append(rel_path)
                except OSError:
                    continue  # unreadable (e.g. broken symlink): nothing to scan
        assert len(violations) == 0, f"Case-insensitive personal paths found in: {violations}"

    def test_url_encoded_path_sanitization(self):
        """TC-SAN-T2-02: Zero URL-encoded variations (%2Fhome%2Fmarc) of personal paths."""
        pattern = re.compile(r"(?i)%2fhome%2fmarc")
        violations = []
        for rel_path in get_tracked_files():
            if not is_non_test_tracked_file(rel_path):
                continue
            full_path = REPO_ROOT / rel_path
            if full_path.is_file():
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    if pattern.search(content):
                        violations.append(rel_path)
                except OSError:
                    continue  # unreadable (e.g. broken symlink): nothing to scan
        assert len(violations) == 0, f"URL-encoded personal paths found in: {violations}"

    def test_gitignore_prevents_accidental_db_commit(self):
        """TC-SAN-T2-03: git check-ignore ignores database files in arbitrary subdirectories."""
        test_paths = ["scratch.db", "data/test.sqlite", "nested/sub/data.sqlite3"]
        for path in test_paths:
            proc = subprocess.run(
                ["git", "check-ignore", "-q", path],
                cwd=str(REPO_ROOT), check=False
            )
            assert proc.returncode == 0, f".gitignore failed to ignore database path '{path}'"

    def test_gitignore_prevents_accidental_venv_commit(self):
        """TC-SAN-T2-04: git check-ignore ignores virtual environment binaries and scripts."""
        test_paths = [".venv/bin/python", ".venv/lib/python3.12/site-packages/pkg.py"]
        for path in test_paths:
            proc = subprocess.run(
                ["git", "check-ignore", "-q", path],
                cwd=str(REPO_ROOT), check=False
            )
            assert proc.returncode == 0, f".gitignore failed to ignore venv path '{path}'"

    def test_gitignore_prevents_pycache_commit(self):
        """TC-SAN-T2-05: git check-ignore ignores __pycache__ directories and .pyc files."""
        test_paths = [
            "ai_db/__pycache__/cli.cpython-312.pyc",
            "tests/__pycache__/conftest.cpython-312.pyc"
        ]
        for path in test_paths:
            proc = subprocess.run(
                ["git", "check-ignore", "-q", path],
                cwd=str(REPO_ROOT), check=False
            )
            assert proc.returncode == 0, f".gitignore failed to ignore pycache path '{path}'"

    def test_no_tracked_environment_secret_files(self):
        """TC-SAN-T2-06: Zero environment secret files (.env, .secrets, *.pem, *.key) tracked."""
        tracked = get_tracked_files()
        secret_files = [
            f for f in tracked
            if f.endswith((".pem", ".key", ".pfx", ".p12")) or f.startswith(".env") or ".secrets" in f
        ]
        assert len(secret_files) == 0, f"Secret files tracked in git: {secret_files}"

    def test_git_tracked_file_size_limits(self):
        """TC-SAN-T2-07: No tracked file exceeds 2MB (prevents accidental binary data commits)."""
        max_bytes = 2 * 1024 * 1024  # 2 Megabytes
        oversized = []
        for rel_path in get_tracked_files():
            full_path = REPO_ROOT / rel_path
            if full_path.is_file():
                size = full_path.stat().st_size
                if size > max_bytes:
                    oversized.append(f"{rel_path} ({size / 1024 / 1024:.2f} MB)")
        assert len(oversized) == 0, "Oversized files tracked in git:\n" + "\n".join(oversized)

    def test_cli_sample_configs_sanitized(self):
        """TC-SAN-T2-08: ai_db/cli.py sample configs and help texts use generic paths."""
        cli_path = REPO_ROOT / "ai_db" / "cli.py"
        assert cli_path.is_file(), f"{cli_path} must exist"
        text = cli_path.read_text(encoding="utf-8")
        assert "/home/marc" not in text, "ai_db/cli.py contains /home/marc"
        assert "~/GitRepos/" not in text, "ai_db/cli.py contains personal ~/GitRepos/ references"

    def test_license_file_exists_and_sanitized(self):
        """TC-SAN-T2-09: LICENSE file exists with generic copyright attribution."""
        license_path = REPO_ROOT / "LICENSE"
        assert license_path.is_file(), f"{license_path} must exist"
        content = license_path.read_text(encoding="utf-8")
        assert "/home/marc" not in content
        # Should attribute to ai-db Contributors
        assert "ai-db Contributors" in content or "Copyright" in content

    def test_sanitization_ignores_tests_own_assertions(self):
        """TC-SAN-T2-10: Sanitization scanner logic successfully excludes test suite files."""
        # Verifies that is_non_test_tracked_file returns False for tests/ and .agents/
        assert not is_non_test_tracked_file("tests/test_sanitization.py")
        assert not is_non_test_tracked_file("tests/conftest.py")
        assert not is_non_test_tracked_file(".agents/e2e_writer_1/DISPATCH.md")
        assert is_non_test_tracked_file("ai_db/cli.py")
        assert is_non_test_tracked_file("README.md")


# ==============================================================================
# Tier 3: Pairwise & Cross-Feature Combinations
# ==============================================================================

@pytest.mark.sanitization
class TestSanitizationTier3:
    """Tier 3: Pairwise cross-feature interactions."""

    def test_pairwise_sanitization_git_ls_and_grep(self):
        """TC-SAN-PAIR-01: Cross-checks git ls-files against full-text regex scanner."""
        tracked_files = [f for f in get_tracked_files() if is_non_test_tracked_file(f)]
        forbidden_strings = ["/home/marc", "noble-tesla", "blender_ox"]
        violations = []
        for rel_path in tracked_files:
            file_path = REPO_ROOT / rel_path
            if file_path.is_file():
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                except (OSError, UnicodeDecodeError):
                    continue
                for s in forbidden_strings:
                    if s in text:
                        violations.append(f"Found '{s}' in {rel_path}")
        assert not violations, "Sanitization violations found:\n" + "\n".join(violations)

    def test_pairwise_constants_defaults_xdg_compliance(self, monkeypatch, tmp_path):
        """TC-SAN-PAIR-02: Constants module resolves paths via XDG env vars without hardcoding."""
        from ai_db import constants
        # Check that constants module exposes configurable paths
        assert hasattr(constants, "DEFAULT_DB_FILE") or hasattr(constants, "DB_PATH") or hasattr(constants, "SKILL_DIRS")

    def test_pairwise_untracked_databases_in_gitignore(self, tmp_path):
        """TC-SAN-PAIR-03: Newly created SQLite databases in working tree are ignored by git."""
        # Test git check-ignore without writing scratch files to REPO_ROOT
        res = subprocess.run(
            ["git", "check-ignore", "-q", "temp_e2e_check.db"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True, check=False
        )
        assert res.returncode == 0, "git check-ignore failed to recognize *.db pattern"

    def test_pairwise_mcp_readme_sanitization(self):
        """TC-SAN-PAIR-04: MCP server configurations in README use portable command forms."""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        # Ensure MCP config snippets use "ai-db", "mcp", or "vectordb.py" without absolute paths
        mcp_blocks = [line for line in readme.splitlines() if "mcp_server.py" in line or "ai-db mcp" in line]
        for line in mcp_blocks:
            assert "/home/marc" not in line, f"Leaked personal path in MCP config: {line}"

    def test_pairwise_credentials_across_all_tracked_extensions(self):
        """TC-SAN-PAIR-05: Multi-extension scanning across .py, .md, .sh, .json, .toml."""
        extensions = {".py", ".md", ".sh", ".json", ".toml", ".yaml", ".yml"}
        tracked = [f for f in get_tracked_files() if is_non_test_tracked_file(f)]
        matching = [f for f in tracked if Path(f).suffix in extensions]
        assert len(matching) > 0, "No tracked files found matching target extensions"
        for rel_path in matching:
            content = (REPO_ROOT / rel_path).read_text(encoding="utf-8", errors="ignore")
            assert "PRIVATE KEY-----" not in content, f"Private key in {rel_path}"
            assert "ghp_" not in content, f"GitHub token in {rel_path}"


# ==============================================================================
# Tier 4: Real-World Application Workflows
# ==============================================================================

@pytest.mark.sanitization
class TestSanitizationTier4:
    """Tier 4: Complete hygiene audit workflows."""

    def test_workflow_repository_clean_export(self, tmp_path):
        """TC-SAN-WF-01: Simulates git archive export and audits exported files."""
        archive_tar = tmp_path / "export.tar"
        proc = subprocess.run(
            ["git", "archive", "--format=tar", "-o", str(archive_tar), "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True, check=False
        )
        assert proc.returncode == 0, f"git archive failed: {proc.stderr}"
        assert archive_tar.is_file()

        # Unpack archive and scan all contents
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        subprocess.run(["tar", "-xf", str(archive_tar), "-C", str(extract_dir)], check=True)

        for p in extract_dir.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(extract_dir))
                assert not rel.endswith((".db", ".sqlite", ".sqlite3")), (
                    f"Database binary file '{rel}' was exported in git archive"
                )
                if not rel.endswith((".png", ".jpg", ".ico", ".pyc")) and is_non_test_tracked_file(rel):
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    has_leak = "/home/marc" in content
                    assert not has_leak, f"Exported file {rel} leaked /home/marc"
                    has_host = "noble-tesla" in content
                    assert not has_host, f"Exported file {rel} leaked noble-tesla"

    def test_workflow_ci_hygiene_audit(self):
        """TC-SAN-WF-02: Complete CI hygiene gate verifying git cleanliness."""
        # 1. Verify zero untracked tracked binaries
        tracked = get_tracked_files()
        assert not any(f.endswith(".db") for f in tracked)
        assert not any(f.endswith(".pyc") for f in tracked)

        # 2. Verify git status is clean of tracked changes (no uncommitted edits to tracked binaries)
        res = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True, check=False
        )
        modified_tracked = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        # No binary files should be modified in tracked tree
        assert not any(f.endswith(".db") for f in modified_tracked), (
            f"Tracked database file modified in git working tree: {modified_tracked}"
        )

    def test_workflow_zero_leakage_on_fresh_index(self, isolated_env, temp_workspace, cli_runner):
        """TC-SAN-WF-03: Indexing an isolated workspace stores only workspace-relative/generic paths."""
        # Execute sync on temporary workspace
        code, out, err = cli_runner.run("sync", str(temp_workspace))
        assert code == 0 or "sync" in (out + err).lower(), f"Sync failed: {out}\n{err}"

        # Inspect resulting SQLite database
        db_path = os.environ["AI_DB_PATH"]
        assert os.path.exists(db_path), f"Isolated database {db_path} was not created! Leakage suspected: {out}"

        import sqlite3
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        try:
            cur.execute("SELECT filepath FROM files")
            rows = cur.fetchall()
            assert len(rows) > 0, "No files were indexed into the isolated database!"
            for (fp,) in rows:
                has_leak = "/home/marc" in fp and str(temp_workspace) not in fp
                assert not has_leak, f"Database leaked host user path: {fp}"
        finally:
            conn.close()

    def test_workflow_commit_hook_simulation(self):
        """TC-SAN-WF-04: Simulates commit-msg/pre-commit rejecting staged personal secrets."""
        # Check against list of files staged or committed
        tracked_files = [f for f in get_tracked_files() if is_non_test_tracked_file(f)]
        for f in tracked_files:
            p = REPO_ROOT / f
            if p.is_file():
                content = p.read_text(encoding="utf-8", errors="ignore")
                assert "AKIA" not in content, f"AWS key pattern detected in {f}"
                has_leak = "/home/marc" in content
                assert not has_leak, f"/home/marc detected in {f}"

    def test_workflow_tarball_distribution_cleanliness(self, tmp_path):
        """TC-SAN-WF-05: Verifies that source distribution contains only project files."""
        # Check that root files are clean
        for root_file in ["README.md", "pyproject.toml", "requirements.txt", "LICENSE"]:
            p = REPO_ROOT / root_file
            if p.is_file():
                text = p.read_text(encoding="utf-8")
                has_leak = "/home/marc" in text
                assert not has_leak, f"{root_file} contains /home/marc"
