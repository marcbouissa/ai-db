"""Token counting (tiktoken o200k_base), shared by the chunker and the output filter."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _encoder() -> Any:
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))
