"""Suggested models, used ONLY by ``ai-db init`` templates and docs.

Nothing in ai-db selects a model automatically from this file at runtime; the model
is always read from the user's config.

Verify these against the MTEB-Code / CoIR leaderboards before each release.
"""

MODEL_NOTE = "suggested default; verify on MTEB-Code/CoIR before use"

SUGGESTED_EMBEDDING: dict[str, dict[str, object]] = {
    "sentence_transformers": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "device": "cpu",
        "batch_size": 32,
        "query_prompt": "Instruct: Given a code search query, retrieve relevant code\nQuery: ",
    },
    "openai_compatible": {
        "model": "text-embedding-3-large",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "dimensions": 1024,
        "batch_size": 64,
    },
    "voyage": {
        "model": "voyage-code-3",
        "api_key_env": "VOYAGE_API_KEY",
        "batch_size": 64,
    },
}

SUGGESTED_RERANK: dict[str, dict[str, object]] = {
    "sentence_transformers": {"model": "BAAI/bge-reranker-v2-m3", "device": "cpu"},
    "voyage": {"model": "rerank-2.5", "api_key_env": "VOYAGE_API_KEY"},
    "cohere": {"model": "rerank-v3.5", "api_key_env": "COHERE_API_KEY"},
}
