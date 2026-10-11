import hashlib
import time
from typing import Any

from ai_db.constants import REF_ID_HEX_LEN
from ai_db.parser.languages import SUPPORTED_LANGUAGES
from ai_db.storage.models import AnalysisRefRecord
from ai_db.tokens import count_tokens


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

        tokens_est = max(1, count_tokens(body))
        return {
            "ref": record.ref_id,
            "file": record.filepath,
            "name": record.name,
            "kind": record.kind,
            "span": [start_line, end_line],
            "body": body,
            "meta": {"tokens_out": tokens_est}
        }
