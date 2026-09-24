"""Built-in rerankers: local cross-encoders and the Voyage / Cohere rerank APIs."""

from __future__ import annotations

import os
from typing import Any

from ai_db.errors import AiDbConfigError
from ai_db.http_client import AiDbProviderError, post_json
from ai_db.rerank.base import RerankProvider

VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


def _api_key(env_name: str) -> str:
    key = os.environ.get(env_name)
    if not key:
        raise AiDbConfigError(f"environment variable {env_name} is not set")
    return key


class CrossEncoderReranker(RerankProvider):
    def __init__(self, options: dict[str, Any]):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise AiDbConfigError(
                "rerank.provider 'sentence_transformers' needs: pip install 'ai-db[local-embed]'"
            ) from exc
        self.model_name = options["model"]
        self.model = CrossEncoder(self.model_name, device=options["device"])
        self.model_id = f"sentence_transformers:{self.model_name}"

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(query, t) for t in texts], show_progress_bar=False)
        return [float(s) for s in scores]


def _indexed_scores(items: Any, n: int, key: str, source: str) -> list[float]:
    if not isinstance(items, list) or len(items) != n:
        raise AiDbProviderError(f"{source}: expected {n} rerank results, got {items!r:.300}")
    scores = [0.0] * n
    for item in items:
        scores[int(item["index"])] = float(item[key])
    return scores


class VoyageReranker(RerankProvider):
    def __init__(self, options: dict[str, Any]):
        self.model = options["model"]
        self.api_key = _api_key(options["api_key_env"])
        self.url = options.get("base_url", VOYAGE_RERANK_URL)
        self.model_id = f"voyage:{self.model}"

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        resp = post_json(self.url, {"model": self.model, "query": query, "documents": texts},
                         {"Authorization": f"Bearer {self.api_key}"})
        return _indexed_scores(resp.get("data"), len(texts), "relevance_score", "voyage rerank")


class CohereReranker(RerankProvider):
    def __init__(self, options: dict[str, Any]):
        self.model = options["model"]
        self.api_key = _api_key(options["api_key_env"])
        self.url = options.get("base_url", COHERE_RERANK_URL)
        self.model_id = f"cohere:{self.model}"

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        resp = post_json(self.url, {"model": self.model, "query": query, "documents": texts},
                         {"Authorization": f"Bearer {self.api_key}"})
        return _indexed_scores(resp.get("results"), len(texts), "relevance_score", "cohere rerank")
