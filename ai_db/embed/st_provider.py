"""Local embeddings via sentence-transformers (``pip install ai-db[local-embed]``)."""

from __future__ import annotations

from typing import Any

from ai_db.embed.base import EmbeddingProvider
from ai_db.errors import AiDbConfigError


class SentenceTransformersEmbedder(EmbeddingProvider):
    def __init__(self, options: dict[str, Any]):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise AiDbConfigError(
                "embedding.provider 'sentence_transformers' needs: pip install 'ai-db[local-embed]'"
            ) from exc
        self.model_name: str = options["model"]
        self.batch_size: int = options["batch_size"]
        self.query_prompt: str = options.get("query_prompt", "")
        self.model = SentenceTransformer(self.model_name, device=options["device"])
        # renamed in sentence-transformers 6 (get_sentence_embedding_dimension -> get_embedding_dimension)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        dim = get_dim()
        if not dim:
            raise AiDbConfigError(f"model {self.model_name} does not report an embedding dimension")
        self.dim = int(dim)
        self.model_id = f"sentence_transformers:{self.model_name}"

    def _encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        vecs = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            prompt=prompt or None, convert_to_numpy=True, show_progress_bar=False,
        )
        return [v.astype("float32").tolist() for v in vecs]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, "")

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self.query_prompt)[0]
