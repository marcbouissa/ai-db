import difflib
import hashlib
import os
import time
from typing import Any

from ai_db.constants import REF_ID_HEX_LEN
from ai_db.parser.languages import SUPPORTED_LANGUAGES
from ai_db.storage.models import AnalysisRefRecord


class ReferenceStore:
    def __init__(self, db: Any = None, conn: Any = None):
        self.db = db if db is not None else conn
        # index.languages, so a file is re-chunked the way the indexer chunked it.
        self.languages: frozenset[str] = SUPPORTED_LANGUAGES

    def store_ref(
        self, filepath: str, name: str, start_line: int, end_line: int,
        kind: str, body_text: str
    ) -> str:
        """Stores a node body in analysis_refs and returns an opaque handle `ref:<sha1_hex>`."""
        hasher = hashlib.sha1()
        hasher.update(f"{filepath}:{start_line}:{end_line}:{body_text}".encode())
        ref_id = f"ref:{hasher.hexdigest()[:REF_ID_HEX_LEN]}"

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

    def diff_spans(self, filepath: str, current_content: str, since: str | None) -> dict[str, Any]:
        """F8 Diff Mode: Identifies changed, added, or removed line spans.

        Resolution order:
        1. If `since` is a git ref (hash/branch/tag), run `git show <since>:<relpath>`.
        2. If `since` is "last" or "db", reconstruct old text from stored DB chunks.
        3. Any other `since` value requires git; if not in a repo or git fails, raise AiDbConfigError.
        """
        import subprocess as _sp

        from ai_db.errors import AiDbConfigError

        old_text: str | None = None

        # 1. Try git-based diff when since looks like a git ref (not "last" or "db")
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
                    else:
                        raise AiDbConfigError(f"--since {since} needs the file to be in a git repository: git show failed")
                else:
                    raise AiDbConfigError(f"--since {since} needs the file to be in a git repository: not in a git repo")
            except (OSError, _sp.TimeoutExpired) as e:
                raise AiDbConfigError(f"--since {since} needs the file to be in a git repository: {e}")

        # 2. "last"/"db": compare with what the index holds.
        if old_text is None:
            if since and since not in ("last", "db"):
                raise AiDbConfigError(f"--since {since} needs the file to be in a git repository")
            return self._diff_against_index(filepath, current_content)

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

    def _diff_against_index(self, filepath: str, current_content: str) -> dict[str, Any]:
        """Spans that changed since the last index, at definition (chunk) granularity.

        The index does not keep a file's raw text, and its chunks are compacted and skip
        the lines between definitions, so joining them never reproduces the file: that
        reported changes in files that had none. Instead: an unchanged hash means no
        change; otherwise the current text is chunked the way the indexer chunks it and
        chunks are matched by content hash. A new chunk whose name was indexed is
        ``changed``, an unknown name is ``added``; an indexed chunk with no match is
        ``removed`` (its old lines).
        """
        from ai_db.parser.chunker import chunk_file
        from ai_db.utils import compute_sha256

        record = self.db.get_file(filepath)
        old = self.db.get_chunks_for_file(filepath)
        lines = current_content.splitlines()
        if record is None or not old:
            return {"added": [[1, len(lines)]] if lines else [], "removed": [], "changed": []}
        if os.path.isfile(filepath) and record.sha256 == compute_sha256(filepath):
            return {"added": [], "removed": [], "changed": []}
        new = chunk_file(filepath, current_content, self.languages)
        old_hashes = {c.content_hash for c in old}
        new_hashes = {c["content_hash"] for c in new}
        old_names = {c.name for c in old}
        added: list[list[int]] = []
        changed: list[list[int]] = []
        for c in new:
            if c["content_hash"] in old_hashes:
                continue
            (changed if c["name"] in old_names else added).append([c["start_line"], c["end_line"]])
        removed = [[c.start_line, c.end_line] for c in old if c.content_hash not in new_hashes
                   and c.name not in {n["name"] for n in new}]
        return {"added": added, "removed": removed, "changed": changed}
