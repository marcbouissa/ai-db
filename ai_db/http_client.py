"""Minimal JSON-over-HTTPS client for hosted model providers (stdlib only).

Retries the *same* request on HTTP 429 and 5xx (3 attempts, exponential backoff),
then raises. Any other failure raises immediately.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from ai_db.errors import AiDbError

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class AiDbProviderError(AiDbError):
    """A hosted model provider returned an error or an invalid response."""


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str],
              timeout: float = 60.0, attempts: int = 3, backoff: float = 1.0) -> Any:
    body = json.dumps(payload).encode("utf-8")
    all_headers = {"Content-Type": "application/json", **headers}
    last_error = ""
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=body, headers=all_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {exc.code} from {url}: {detail}"
            if exc.code not in RETRY_STATUSES or attempt == attempts - 1:
                raise AiDbProviderError(last_error) from exc
        except urllib.error.URLError as exc:
            raise AiDbProviderError(f"cannot reach {url}: {exc.reason}") from exc
        time.sleep(backoff * (2 ** attempt))
    raise AiDbProviderError(last_error)
