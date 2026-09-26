"""
ai_db.parser.linters
Runs external linter tools on file content and returns (line, col, message) errors.
Validators are opt-in: each only fires if the CLI tool is found on PATH.
Custom validators can be added via config.json: { "validators": { ".ts": ["npx","tsc","--noEmit","{file}"] } }
"""
import ast
import os
import re
import shutil
import subprocess
import tempfile

# Built-in validator registry: ext -> command template (use {file} for the temp filepath)
BUILTIN_VALIDATORS: dict[str, list[str]] = {
    ".js":   ["node", "--check", "{file}"],
    ".ts":   ["npx", "--yes", "tsc", "--noEmit", "--allowJs", "--checkJs",
               "--target", "ESNext", "--moduleResolution", "node", "{file}"],
    ".sh":   ["bash", "-n", "{file}"],
    ".bash": ["bash", "-n", "{file}"],
    ".yml":  ["python3", "-c",
               "import yaml,sys; yaml.safe_load(open(sys.argv[1]))", "{file}"],
    ".yaml": ["python3", "-c",
               "import yaml,sys; yaml.safe_load(open(sys.argv[1]))", "{file}"],
}


def _tool_available(cmd: str) -> bool:
    """Returns True if a shell command is available on PATH."""
    return shutil.which(cmd) is not None


def _parse_error_location(output: str) -> tuple[int, int]:
    """Extracts (line, col) from common linter error output formats."""
    # Matches: :12:5:  or  line 12, col 5  or  (12,5)
    patterns = [
        r":(\d+):(\d+):",
        r"line (\d+).*?col(?:umn)? (\d+)",
        r"\((\d+),(\d+)\)",
    ]
    for pat in patterns:
        m = re.search(pat, output, re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
    # Try just line number
    m = re.search(r":(\d+):", output)
    if m:
        return int(m.group(1)), 1
    return 1, 1


class ExternalLinter:
    """Validates non-Python source files using external CLI tools."""

    def __init__(self, config_validators: dict[str, list[str]] | None = None):
        self.validators: dict[str, list[str]] = dict(BUILTIN_VALIDATORS)
        if config_validators:
            self.validators.update(config_validators)

    def validate(self, filepath: str, content: str) -> tuple[int, int, str] | None:
        """
        Returns (line, col, message) on error, None if file is clean or no validator is available.
        Always a no-op for Python files (handled by validate_python_syntax separately).
        """
        ext = os.path.splitext(filepath)[1].lower()
        cmd_template = self.validators.get(ext)
        if not cmd_template:
            return None

        # Skip if the base tool is not installed
        base_tool = cmd_template[0]
        if not _tool_available(base_tool):
            return None

        # Write content to a temp file with the correct extension
        try:
            with tempfile.NamedTemporaryFile(
                suffix=ext, mode="w", encoding="utf-8", delete=False
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
        except OSError as exc:
            raise OSError(f"cannot write temp file for linting {filepath}: {exc}") from exc

        try:
            cmd = [c.replace("{file}", tmp_path) for c in cmd_template]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15, check=False,
            )
            if result.returncode != 0:
                output = (result.stderr or result.stdout or "").strip()
                line, col = _parse_error_location(output)
                msg = output.splitlines()[0] if output.splitlines() else "Lint error"
                # Truncate very long error messages
                msg = msg[:200]
                return (line, col, msg)
        except subprocess.TimeoutExpired:
            # A linter that never answers is not a clean file: report it as an
            # error so the gap is visible instead of silently passing.
            return (1, 1, "linter timeout")
        finally:
            os.unlink(tmp_path)

        return None


def validate_python_syntax(content: str, filepath: str) -> tuple[int, int, str] | None:
    """
    Validates Python syntax using the built-in ast module.
    Returns (line, col, message) on error, None if valid.
    """
    try:
        ast.parse(content, filename=filepath)
    except SyntaxError as e:
        return (e.lineno or 1, e.offset or 1, e.msg)
    except ValueError as e:
        return (1, 1, str(e))
    return None


# Module-level singleton — avoids re-reading config on every file
_linter: ExternalLinter | None = None


def get_linter(config_validators: dict[str, list[str]] | None = None) -> ExternalLinter:
    global _linter
    if _linter is None:
        _linter = ExternalLinter(config_validators)
    return _linter


def parse_source(
    filepath: str, content: str
) -> tuple[list[dict], list[dict], tuple[int, int, str] | None]:
    """Parse one file and report its syntax verdict. Pure: touches no storage.

    Returns ``(symbols, refs, syntax_error)`` where ``syntax_error`` is
    ``(line, col, message)`` or ``None``.

    This is the single parse path. Both the indexer (which needs the symbols and
    refs) and ``ai-db check <path>`` (which needs only the verdict) go through
    here, so a file's syntax cannot be reported one way when checked on disk and
    another way when read back from the index.

    Languages tree-sitter handles are parsed with tree-sitter. Everything else
    goes to the configured linter for its extension, which is a no-op when no
    validator is registered -- an unvalidatable file is clean, not broken.
    """
    from ai_db.parser.ts_graph import extract_graph, language_for

    if language_for(filepath) is not None:
        symbols, refs, errors = extract_graph(filepath, content)
        return symbols, refs, (errors[0] if errors else None)
    return [], [], get_linter().validate(filepath, content)
