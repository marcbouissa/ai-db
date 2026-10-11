"""Local embeddings via sentence-transformers (``pip install ai-db[local-embed]``)."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ai_db.constants import EMBED_BULK_THRESHOLD
from ai_db.device import CUDA, configure_cpu_threads, resolve_device, resolve_dtype
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
        configure_cpu_threads()  # queries and small syncs run on the CPU even with a bulk GPU
        # Optional second device for large runs (see bulk()). Validated now, so a
        # config asking for cuda without a GPU fails at load, not mid-index.
        bulk = options.get("bulk_device")
        self.bulk_device: str | None = (
            resolve_device(bulk, where="embedding.bulk_device") if bulk else None)
        if self.bulk_device == device:
            self.bulk_device = None
        self.bulk_threshold = (int(options.get("bulk_threshold", EMBED_BULK_THRESHOLD))
                               if self.bulk_device else None)
        # Guards the model against encoding while it moves between devices
        # (the daemon serves queries from other threads during a sync).
        self._lock = threading.RLock()

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
        from ai_db.quiet import quiet_model_load
        with quiet_model_load(self.model_name):
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

    @contextmanager
    def bulk(self, pending: int) -> Iterator[None]:
        """Run a large embed on ``bulk_device``, then move the model back.

        The dtype is kept: it was chosen for ``device``, and keeping it means a
        vector embedded on the GPU matches one later embedded on the CPU, so
        the index stays consistent. Queries arriving meanwhile run on whichever
        device the model is on. VRAM is released afterwards.
        """
        if self.bulk_device is None or self.bulk_threshold is None or pending < self.bulk_threshold:
            yield
            return
        self._move(self.bulk_device)
        try:
            yield
        finally:
            self._move(self.device)

    def _move(self, device: str) -> None:
        with self._lock:
            self.model.to(device)
            if not device.startswith(CUDA):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    @property
    def current_device(self) -> str:
        return str(self.model.device)

    def _encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        with self._lock:
            vecs = self.model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True,
                prompt=prompt or None, convert_to_numpy=True, show_progress_bar=False,
            )
        return [v.astype("float32").tolist() for v in vecs]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, "")

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self.query_prompt)[0]
