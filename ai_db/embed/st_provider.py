"""Local embeddings via sentence-transformers (``pip install ai-db[local-embed]``)."""

from __future__ import annotations

from typing import Any

from ai_db.device import resolve_device, resolve_dtype
from ai_db.embed.base import EmbeddingProvider
from ai_db.errors import AiDbConfigError

#: Config-facing dtype names -> torch dtype attribute names.
_TORCH_DTYPES = {"float32": "float32", "bfloat16": "bfloat16", "float16": "float16"}


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
        device = resolve_device(options.get("device"), where="embedding")
        self.device = device

        # Most modern embedding models ship in bfloat16. On a GPU that is right.
        # On a CPU without avx512_bf16 or AMX it is emulated and *slower* than
        # float32 -- measured 0.24 vs 0.50 chunks/s on an i7-11800H, so leaving
        # it alone costs a 2x indexing slowdown on exactly the hardware most
        # people run. resolve_dtype probes and decides, and returns None on an
        # accelerator so the checkpoint's native dtype is left alone.
        # `embedding.dtype` overrides both. See ai_db.device.resolve_dtype.
        self.dtype: str | None = resolve_dtype(device, options.get("dtype"))
        model_kwargs: dict[str, Any] = {}
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = _TORCH_DTYPES.get(self.dtype, self.dtype)
        self.model = SentenceTransformer(self.model_name, device=device,
                                         model_kwargs=model_kwargs or None)
        self.loaded_dtype: str = str(
            next(self.model.parameters()).dtype).removeprefix("torch.")
        # renamed in sentence-transformers 6 (get_sentence_embedding_dimension -> get_embedding_dimension)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        dim = get_dim()
        if not dim:
            raise AiDbConfigError(f"model {self.model_name} does not report an embedding dimension")
        self.dim = int(dim)
        # NOTE: the dtype is deliberately NOT part of model_id. Changing it
        # changes the vectors slightly, but folding it in would invalidate every
        # cached query on upgrade for a sub-1% numeric difference. Stored
        # vectors stay as they are until `ai-db reindex --embeddings`.
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
