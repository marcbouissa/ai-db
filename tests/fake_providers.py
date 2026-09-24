"""Deterministic in-process providers for tests (no network, no models)."""

import hashlib
import math
import re

from ai_db.embed.base import EmbeddingProvider

SYNONYMS = {"automobile": "car", "vehicle": "car", "purchase": "buy", "payment": "pay"}


class FakeEmbeddingProvider(EmbeddingProvider):
    """Hashed bag-of-words with a synonym table, so 'automobile' ~ 'car'."""

    def __init__(self, options=None):
        self.dim = int((options or {}).get("dim", 64))
        self.model_id = f"fake:{(options or {}).get('model', 'bow')}"
        self.calls = 0

    def _vec(self, text):
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z]+", text.lower()):
            w = SYNONYMS.get(w, w)
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        if n == 1.0 and not any(v):
            v[0] = 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)
