"""The generated mode table must match the code.

`docs/MODES.md` is produced by `eval/render_modes_table.py`, which reads the mode
lists out of `argparse` and the parser's own constants. A generator is only worth
anything if the committed output matches what the generator would produce now, so
these tests re-derive the file and diff it.

Without this, a mode added to the code would leave a stale doc that still reads
authoritatively -- the failure mode docs usually have.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(REPO, "docs/MODES.md")
GENERATOR = os.path.join(REPO, "eval/render_modes_table.py")


def test_table_exists() -> None:
    assert os.path.isfile(TABLE), "docs/MODES.md missing; run eval/render_modes_table.py"


def test_table_is_up_to_date() -> None:
    """Regenerating must be a no-op. Run with --regen to accept a real change."""
    with open(TABLE, encoding="utf-8") as fh:
        before = fh.read()
    proc = subprocess.run([sys.executable, GENERATOR], capture_output=True, text=True,
                          cwd=REPO, check=False)
    assert proc.returncode == 0, proc.stderr
    with open(TABLE, encoding="utf-8") as fh:
        after = fh.read()
    if before != after:
        raise AssertionError(
            "docs/MODES.md is stale; regenerate with "
            "`uv run python eval/render_modes_table.py` and commit the result")


def test_generator_cross_checks_hold() -> None:
    """The assertions inside the generator are the real guard; make them run."""
    proc = subprocess.run([sys.executable, GENERATOR], capture_output=True, text=True,
                          cwd=REPO, check=False)
    assert proc.returncode == 0, (
        f"generator cross-check failed: {proc.stderr.strip()}")


class TestTableContent:
    """Spot-check that the claims in the table are the ones the code supports."""

    def test_investigate_modes_all_documented(self) -> None:
        from ai_db.analysis.investigate import MODES

        with open(TABLE, encoding="utf-8") as fh:
            text = fh.read()
        for mode in MODES:
            assert f"| `{mode}` |" in text, f"investigate mode {mode} not in the table"

    def test_every_retrieval_mode_documented(self) -> None:
        from ai_db.config import RETRIEVAL_MODES

        with open(TABLE, encoding="utf-8") as fh:
            text = fh.read()
        for mode in RETRIEVAL_MODES:
            assert f"`{mode}`" in text, f"retrieval mode {mode} not in the table"

    def test_language_count_is_not_hardcoded_wrong(self) -> None:
        """The coverage row is generated; assert it agrees with the parser."""
        from ai_db.parser.ts_graph import SUPPORTED_LANGUAGES

        langs = {v for v in SUPPORTED_LANGUAGES.values()}
        with open(TABLE, encoding="utf-8") as fh:
            text = fh.read()
        row = next(line for line in text.splitlines() if "Languages parsed" in line)
        assert f"{len(langs)} grammars" in row, row
        assert f"{len(SUPPORTED_LANGUAGES)} extensions" in row, row

    def test_no_placeholder_text_left(self) -> None:
        with open(TABLE, encoding="utf-8") as fh:
            text = fh.read()
        for bad in ("TODO", "FIXME", "XXX", "None", "{}"):
            assert bad not in text, f"placeholder {bad!r} in the table"


def test_anchors_resolve() -> None:
    """Every in-page link in the table must have a heading to land on."""
    with open(TABLE, encoding="utf-8") as fh:
        text = fh.read()
    anchors = re.findall(r"\]\(#([a-z0-9\-]+)\)", text)
    headings = []
    for line in text.splitlines():
        if line.startswith("#"):
            h = re.sub(r"[^\w\s\-]", "", line.lstrip("#").strip().lower())
            headings.append(h.replace(" ", "-"))
    missing = [a for a in anchors if a not in headings]
    assert not missing, f"broken anchors: {missing}"
