import difflib
import hashlib
import os
import time
from typing import Any

from ai_db.logger import _logger
from ai_db.storage.models import AnalysisRefRecord


class ReferenceStore:
    def __init__(self, db: Any = None, conn: Any = None):
        self.db = db if db is not None else conn

    def _store_analysis_ref(
        self, filepath: str, name: str, start_line: int, end_line: int,
        kind: str, body_text: str
    ) -> str:
        """Stores a node body in analysis_refs and returns an opaque handle `ref:<sha1_hex>`."""
        hasher = hashlib.sha1()
        hasher.update(f"{filepath}:{start_line}:{end_line}:{body_text}".encode())
        ref_id = f"ref:{hasher.hexdigest()[:8]}"

        record = AnalysisRefRecord(
            ref_id=ref_id,
            filepath=filepath,
            name=name,
            start_line=start_line,
            end_line=end_line,
            kind=kind,
            body_text=body_text,
            timestamp=time.time()
        )
        self.db.store_analysis_ref(record)
        return ref_id

    def expand_ref(
        self, ref_id: str, depth: str = "full", span: tuple[int, int] | None = None
    ) -> dict[str, Any] | None:
        """F7 Progressive Disclosure: Expands an opaque ref handle pay-per-section."""
        record = self.db.get_analysis_ref(ref_id)
        if not record:
            return None

        body = record.body_text
        body_lines = body.splitlines()
        start_line = record.start_line
        end_line = record.end_line

        if span:
            req_start, req_end = span
            offset_start = max(0, req_start - start_line)
            offset_end = min(len(body_lines), req_end - start_line + 1)
            selected_lines = body_lines[offset_start:offset_end]
            body = "\n".join(selected_lines)
            start_line = start_line + offset_start
            end_line = start_line + len(selected_lines) - 1

        tokens_est = max(1, len(body) // 4)
        return {
            "ref": record.ref_id,
            "file": record.filepath,
            "name": record.name,
            "kind": record.kind,
            "span": [start_line, end_line],
            "body": body,
            "meta": {"tokens_out": tokens_est}
        }

    def _diff_spans(self, filepath: str, current_content: str, since: str | None) -> dict[str, Any]:
        """F8 Diff Mode: Identifies changed, added, or removed line spans.

        Resolution order:
        1. If `since` is a git ref (hash/branch/tag), run `git show <since>:<relpath>`.
        2. If git fails or `since` is None/'last', reconstruct old text from stored DB chunks.
        3. If nothing stored, treat entire file as added.
        """
        import subprocess as _sp

        old_text: str | None = None

        # 1. Try git-based diff when since looks like a git ref
        if since and since not in ("last", "db"):
            try:
                repo_root_res = _sp.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    capture_output=True, text=True, timeout=5,
                    cwd=os.path.dirname(os.path.abspath(filepath)), check=False
                )
                if repo_root_res.returncode == 0:
                    repo_root = repo_root_res.stdout.strip()
                    rel_path = os.path.relpath(os.path.abspath(filepath), repo_root)
                    result = _sp.run(
                        ["git", "show", f"{since}:{rel_path}"],
                        capture_output=True, text=True, timeout=10,
                        cwd=repo_root, check=False
                    )
                    if result.returncode == 0:
                        old_text = result.stdout
            except (OSError, _sp.TimeoutExpired) as e:
                # git missing/slow: `since` cannot be resolved from git; the stored
                # chunks below are the documented source for this case
                _logger.debug(f"_diff_spans git unavailable: {e}")

        # 2. Fallback: reconstruct from stored DB chunks using get_chunks_for_file
        if old_text is None:
            chunks = self.db.get_chunks_for_file(filepath)
            if not chunks:
                return {"added": [[1, len(current_content.splitlines())]], "removed": [], "changed": []}
            old_text = "\n".join(c.content for c in chunks)

        old_lines = old_text.splitlines()
        new_lines = current_content.splitlines()
        if old_lines == new_lines:
            return {"added": [], "removed": [], "changed": []}

        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        added_spans, removed_spans, changed_spans = [], [], []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "insert":
                added_spans.append([j1 + 1, j2])
            elif tag == "delete":
                removed_spans.append([i1 + 1, i2])
            elif tag == "replace":
                changed_spans.append([j1 + 1, j2])

        return {"added": added_spans, "removed": removed_spans, "changed": changed_spans}
