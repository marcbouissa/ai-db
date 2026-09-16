import os
import time
import zlib
import difflib
import hashlib
import sqlite3
from typing import Dict, Any, Tuple, Optional

class ReferenceStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def _store_analysis_ref(self, filepath: str, name: str, start_line: int, end_line: int,
                            kind: str, body_text: str) -> str:
        """Stores a node body in analysis_refs and returns an opaque handle `ref:<sha1_hex>`."""
        hasher = hashlib.sha1()
        hasher.update(f"{filepath}:{start_line}:{end_line}:{body_text}".encode("utf-8"))
        ref_id = f"ref:{hasher.hexdigest()[:8]}"
        zbody = zlib.compress(body_text.encode("utf-8"), level=9)
        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO analysis_refs (ref_id, filepath, name, start_line, end_line, kind, zbody, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ref_id, filepath, name, start_line, end_line, kind, zbody, time.time())
        )
        return ref_id

    def expand_ref(self, ref_id: str, depth: str = "full", span: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
        """F7 Progressive Disclosure: Expands an opaque ref handle pay-per-section."""
        cur = self.conn.cursor()
        cur.execute("SELECT ref_id, filepath, name, start_line, end_line, kind, zbody FROM analysis_refs WHERE ref_id = ?", (ref_id,))
        row = cur.fetchone()
        if not row:
            return None

        try:
            body = zlib.decompress(row["zbody"]).decode("utf-8", errors="replace")
        except Exception:
            body = ""

        body_lines = body.splitlines()
        start_line = row["start_line"]
        end_line = row["end_line"]

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
            "ref": row["ref_id"],
            "file": row["filepath"],
            "name": row["name"],
            "kind": row["kind"],
            "span": [start_line, end_line],
            "body": body,
            "meta": {"tokens_out": tokens_est}
        }

    def _diff_spans(self, filepath: str, current_content: str, since: Optional[str]) -> Dict[str, Any]:
        """F8 Diff Mode: Identifies changed, added, or removed line spans since previous hash/ts."""
        if not since:
            return {"added": [], "removed": [], "changed": []}

        cur = self.conn.cursor()
        cur.execute("SELECT zcontent, sha256 FROM chunks WHERE filepath = ? ORDER BY id ASC", (filepath,))
        stored_chunks = cur.fetchall()
        if not stored_chunks:
            return {"added": [[1, len(current_content.splitlines())]], "removed": [], "changed": []}

        old_text = ""
        for sc in stored_chunks:
            try:
                old_text += zlib.decompress(sc["zcontent"]).decode("utf-8", errors="replace") + "\n"
            except Exception:
                pass

        old_lines = old_text.splitlines()
        new_lines = current_content.splitlines()
        if old_lines == new_lines:
            return {"added": [], "removed": [], "changed": []}

        # Terse span calculation
        changed_spans = []
        import difflib
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        added_spans = []
        removed_spans = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "insert":
                added_spans.append([j1 + 1, j2])
            elif tag == "delete":
                removed_spans.append([i1 + 1, i2])
            elif tag == "replace":
                changed_spans.append([j1 + 1, j2])

        return {"added": added_spans, "removed": removed_spans, "changed": changed_spans}
