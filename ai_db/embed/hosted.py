"""Hosted embeddings: OpenAI-compatible ``/embeddings`` endpoints and Voyage AI."""

from __future__ import annotations

import os
from typing import Any

from ai_db.embed.base import EmbeddingProvider, l2_normalize
from ai_db.errors import AiDbConfigError
from ai_db.http_client import AiDbProviderError, post_json

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"


def _api_key(env_name: str) -> str:
    key = os.environ.get(env_name)
    if not key:
        raise AiDbConfigError(f"environment variable {env_name} is not set")
    return key


def _vectors(resp: Any, expected: int, dim: int, source: str) -> list[list[float]]:
    data = resp.get("data") if isinstance(resp, dict) else None
    if not isinstance(data, list) or len(data) != expected:
        raise AiDbProviderError(f"{source}: expected {expected} embeddings, got {resp!r:.300}")
    ordered = sorted(data, key=lambda d: d.get("index", 0))
    out = []
    for item in ordered:
        vec = item.get("embedding")
        if not isinstance(vec, list) or len(vec) != dim:
            got = len(vec) if isinstance(vec, list) else type(vec).__name__
            raise AiDbProviderError(f"{source}: embedding has dimension {got}, expected {dim}")
        out.append(l2_normalize([float(v) for v in vec]))
    return out


class _Batched(EmbeddingProvider):
    batch_size: int

    def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed(texts[i:i + self.batch_size], "document"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


class OpenAICompatibleEmbedder(_Batched):
    """Any server implementing ``POST {base_url}/embeddings`` (OpenAI, vLLM, TEI, Ollama)."""

    def __init__(self, options: dict[str, Any]):
        self.model = options["model"]
        self.base_url = options["base_url"].rstrip("/")
        self.api_key = _api_key(options["api_key_env"])
        self.dim = int(options["dimensions"])
        self.batch_size = int(options["batch_size"])
        self.model_id = f"openai_compatible:{self.base_url}:{self.model}"

    def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        resp = post_json(
            f"{self.base_url}/embeddings",
            {"model": self.model, "input": texts, "dimensions": self.dim},
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return _vectors(resp, len(texts), self.dim, self.base_url)


class VoyageEmbedder(_Batched):
    def __init__(self, options: dict[str, Any]):
        self.model = options["model"]
        self.api_key = _api_key(options["api_key_env"])
        self.dim = int(options["dimensions"])
        self.batch_size = int(options["batch_size"])
        self.url = options.get("base_url", VOYAGE_URL)
        self.model_id = f"voyage:{self.model}"

    def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        resp = post_json(
            self.url,
            {"model": self.model, "input": texts, "input_type": kind, "output_dimension": self.dim},
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return _vectors(resp, len(texts), self.dim, "voyage")
