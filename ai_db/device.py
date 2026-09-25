"""Resolve the torch device for local embedding / rerank providers.

Kept in one place so ``build_template``, the sentence-transformers embedder and the
cross-encoder reranker cannot disagree about what ``"cuda"`` means. torch is
imported lazily: ai-db's core is stdlib-only and must not pay for a ~2 GB import
just to validate a config.
"""

from __future__ import annotations

from typing import Any

from ai_db.constants import (
    BF16_PROBE_MARGIN,
    DTYPE_PROBE_ITERS,
    DTYPE_PROBE_REPEATS,
    DTYPE_PROBE_SIZE,
)
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


_DTYPE_CACHE: dict[tuple[str, str | None], str] = {}


def _probe_cpu_dtypes(candidates: tuple[str, ...]) -> dict[str, float]:
    """Seconds per matmul iteration for each dtype, best of N repeats.

    Deliberately a timing probe rather than an ISA-flag lookup. ``/proc/cpuinfo``
    does not exist on macOS or Windows, flag names differ between Intel, AMD and
    ARM, and any hardcoded table goes stale the moment a new extension ships.
    Timing the operation we actually care about is portable by construction and
    cannot be wrong about a CPU it has never heard of.
    """
    import time

    import torch

    timings: dict[str, float] = {}
    n = DTYPE_PROBE_SIZE
    for name in candidates:
        dtype = getattr(torch, name)
        # (n, n) matrices, NOT torch.randn(n) -- that is a 1-D vector, and
        # vector @ vector is a dot product, which costs nothing and makes the
        # probe measure pure dispatch overhead. Two dimensions are required.
        a = torch.randn(n, n, dtype=dtype, device="cpu")
        b = torch.randn(n, n, dtype=dtype, device="cpu")
        a @ b  # warm up oneDNN kernel selection and the thread pool
        best = float("inf")
        for _ in range(DTYPE_PROBE_REPEATS):
            start = time.perf_counter()
            for _ in range(DTYPE_PROBE_ITERS):
                a @ b
            best = min(best, time.perf_counter() - start)
        # best-of, not mean: one scheduler hiccup on a loaded machine should not
        # decide the dtype for the rest of the process's life.
        timings[name] = best
    return timings


def resolve_dtype(device: str, requested: str | None = None) -> str | None:
    """Choose the torch dtype to load a transformer in, or None to keep native.

    ``requested`` is the user's explicit ``embedding.dtype`` and always wins.

    On an accelerator this returns ``None``, meaning "load whatever the model
    ships in". GPU tensor cores are built for bf16/fp16, so forcing float32
    there would waste VRAM and bandwidth for nothing. The native dtype is never
    second-guessed on hardware designed around it.

    On CPU it measures. bfloat16 is what most modern embedding models ship in,
    but it is only fast on CPUs with ``avx512_bf16`` or AMX; without them every
    matmul is emulated and runs *slower* than float32 -- measured 0.24 vs 0.50
    chunks/s on an i7-11800H. So pick the fastest dtype the hardware actually
    delivers, rather than the one in the checkpoint.

    The probe is memoised: it measures a property of the machine, not of the
    call, and re-running it per embedder would be pure waste.
    """
    if requested is not None:
        return requested
    if device != CPU:
        return None

    key = (device, requested)
    cached = _DTYPE_CACHE.get(key)
    if cached is not None:
        return cached

    from ai_db.constants import CPU_DTYPES

    try:
        timings = _probe_cpu_dtypes(CPU_DTYPES)
    except Exception:  # noqa: BLE001 - a failed probe must not block startup
        timings = {}
    if not timings:
        chosen = "float32"
    else:
        fastest = min(timings, key=lambda name: timings[name])
        # Require the winner to be clear of the runner-up, so measurement noise
        # on a loaded machine cannot flip the dtype between runs.
        chosen = fastest
        for other, seconds in timings.items():
            if other == fastest:
                continue
            if seconds < timings[fastest] * BF16_PROBE_MARGIN:
                # Too close to call: fall back to float32, the safe default that
                # is never emulated on any CPU.
                chosen = "float32"
                break

    _DTYPE_CACHE[key] = chosen
    return chosen


def dtype_report(device: str, chosen: str | None) -> dict[str, Any]:
    """Diagnostics for ``ai-db config check``: why this dtype was chosen."""
    out: dict[str, Any] = {"device": device,
                           "dtype": chosen or "model default (native)"}
    if device != CPU or chosen is None:
        return out
    from ai_db.constants import CPU_DTYPES

    try:
        timings = _probe_cpu_dtypes(CPU_DTYPES)
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return out
    out["probe_seconds"] = {k: round(v, 6) for k, v in timings.items()}
    out["reason"] = ("fastest measured CPU dtype; bfloat16 in a checkpoint is "
                     "emulated here and runs slower than float32")
    return out


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
