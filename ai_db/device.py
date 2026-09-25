"""Resolve the torch device for local embedding / rerank providers.

Kept in one place so ``build_template``, the sentence-transformers embedder and the
cross-encoder reranker cannot disagree about what ``"cuda"`` means. torch is
imported lazily: ai-db's core is stdlib-only and must not pay for a ~2 GB import
just to validate a config.
"""

from __future__ import annotations

from typing import Any

from ai_db.errors import AiDbConfigError

CUDA = "cuda"
CPU = "cpu"


def cuda_available() -> bool:
    """True only if torch is importable *and* reports a usable CUDA device."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken driver must not crash config
        return False


def torch_version_cuda() -> str | None:
    """``torch.version.cuda`` (e.g. ``"12.8"``), or None if torch is absent."""
    try:
        import torch
    except ImportError:
        return None
    return getattr(torch.version, "cuda", None)


def preferred_device() -> str:
    """Device to write into a fresh config: cuda when usable, else cpu."""
    return CUDA if cuda_available() else CPU


def resolve_device(requested: str | None, *, where: str) -> str:
    """Validate a configured device string.

    A config asking for ``cuda`` on a machine without a usable GPU is a hard
    error (TODO rule 1: no silent downgrade) -- silently falling back to CPU
    would turn a large index into a multi-hour one with no explanation.
    """
    if requested is None:
        return CPU
    if not isinstance(requested, str) or not requested.strip():
        raise AiDbConfigError(f"{where}: device must be a non-empty string")
    device = requested.strip().lower()
    if device == CPU:
        return CPU
    if device.startswith(CUDA):
        if not cuda_available():
            raise AiDbConfigError(
                f"{where}: device '{requested}' requested but no CUDA device is "
                f"available. Install a CUDA build of torch, or set device to 'cpu'."
            )
        return device
    # sentence-transformers also accepts 'mps', 'xpu', ...
    return device


def describe(where: str) -> dict[str, Any]:
    """Device facts for `ai-db config check`."""
    return {
        "torch_installed": _torch_installed(),
        "cuda_available": cuda_available(),
        "torch_cuda": torch_version_cuda(),
        "preferred_device": preferred_device(),
        "scope": where,
    }


def _torch_installed() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True
