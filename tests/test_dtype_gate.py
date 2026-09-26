"""Compute-dtype gate: pick a torch dtype the hardware can actually run fast.

Most modern embedding checkpoints ship in bfloat16. On a GPU that is correct.
On a CPU without ``avx512_bf16`` or AMX it is emulated in software and runs
*slower* than float32 -- measured on an i7-11800H at 0.24 vs 0.50 chunks/s, so
leaving it alone costs a 2x indexing slowdown on exactly the hardware most
people run. ``ai_db.device.resolve_dtype`` probes and decides.
"""

from __future__ import annotations

import pytest

from ai_db import constants
from ai_db import device as device_mod
from ai_db.errors import AiDbConfigError


@pytest.fixture(autouse=True)
def _clear_cache():
    """The probe memoises per (device, requested); tests must not inherit it."""
    device_mod._DTYPE_CACHE.clear()
    yield
    device_mod._DTYPE_CACHE.clear()


# ==============================================================================
# the probe itself
# ==============================================================================

def test_probe_uses_real_matrices_not_vectors(monkeypatch):
    """Regression: torch.randn(n) is 1-D, and vector @ vector is a dot product.

    The probe shipped measuring a 512-element dot product -- ~0 FLOPs -- so it
    reported float32 and bfloat16 as indistinguishable and the gate never fired.
    Caught by asserting the probe's own arithmetic is physically plausible.
    """
    import time

    import torch

    n = constants.DTYPE_PROBE_SIZE
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    a @ b
    t0 = time.perf_counter()
    a @ b
    one = time.perf_counter() - t0
    # One 512^3 matmul is 2*n^3 FLOPs. Anything above ~1 TFLOP/s sustained on
    # this class of CPU is a measurement artefact, not a real matmul.
    assert one > 0, "probe would be measuring nothing"
    flops = 2 * n ** 3 / one
    assert flops < 1e12, f"probe implausible: {flops / 1e9:.0f} GFLOPS for one matmul"

    timings = device_mod._probe_cpu_dtypes(("float32",))
    assert list(timings) == ["float32"]
    assert timings["float32"] > 0


def test_probe_is_best_of_n_not_mean(monkeypatch):
    """A single slow repeat must not decide the dtype for the process lifetime."""
    import time

    import torch

    real = torch.matmul
    calls = {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(0.05)  # a scheduler hiccup on the first repeat
        return real(a, b)

    monkeypatch.setattr(torch, "matmul", flaky, raising=False)
    timings = device_mod._probe_cpu_dtypes(("float32", "bfloat16"))
    # The best-of must be far below the mean that includes the 50 ms stall.
    assert timings["float32"] < 0.05 / constants.DTYPE_PROBE_REPEATS


# ==============================================================================
# the gate
# ==============================================================================

def test_cpu_picks_the_faster_dtype(monkeypatch):
    """With float32 measured faster, the gate must choose float32."""
    monkeypatch.setattr(device_mod, "_probe_cpu_dtypes",
                        lambda c: {"float32": 0.001, "bfloat16": 0.004})
    assert device_mod.resolve_dtype("cpu", None) == "float32"


def test_cpu_keeps_bfloat16_when_the_cpu_is_fast_at_it(monkeypatch):
    """On a CPU with real bf16 support (AMX, avx512_bf16), do not downgrade."""
    monkeypatch.setattr(device_mod, "_probe_cpu_dtypes",
                        lambda c: {"float32": 0.004, "bfloat16": 0.001})
    assert device_mod.resolve_dtype("cpu", None) == "bfloat16"


def test_cpu_falls_back_to_float32_when_too_close_to_call(monkeypatch):
    """A near-tie is measurement noise, and float32 is never emulated."""
    monkeypatch.setattr(device_mod, "_probe_cpu_dtypes",
                        lambda c: {"float32": 0.001, "bfloat16": 0.0011})
    assert device_mod.resolve_dtype("cpu", None) == "float32"


def test_explicit_dtype_always_wins(monkeypatch):
    """`embedding.dtype` overrides the probe, in both directions."""
    monkeypatch.setattr(device_mod, "_probe_cpu_dtypes",
                        lambda c: {"float32": 0.001, "bfloat16": 0.004})
    assert device_mod.resolve_dtype("cpu", "bfloat16") == "bfloat16"
    assert device_mod.resolve_dtype("cpu", "float16") == "float16"


def test_accelerator_keeps_the_native_dtype(monkeypatch):
    """Never second-guess hardware built around bf16/fp16 tensor cores."""
    def explode(_candidates):
        raise AssertionError("must not probe on cuda")

    monkeypatch.setattr(device_mod, "_probe_cpu_dtypes", explode)
    assert device_mod.resolve_dtype("cuda", None) is None
    # An explicit request is still honoured on cuda.
    assert device_mod.resolve_dtype("cuda", "float32") == "float32"


def test_failed_probe_does_not_block_startup(monkeypatch):
    """A probe failure must degrade to float32, never raise at startup."""
    def boom(_candidates):
        raise RuntimeError("torch exploded")

    monkeypatch.setattr(device_mod, "_probe_cpu_dtypes", boom)
    assert device_mod.resolve_dtype("cpu", None) == "float32"


def test_probe_is_memoised(monkeypatch):
    """It measures the machine, not the call; re-probing per embedder is waste."""
    calls = {"n": 0}

    def counting(candidates):
        calls["n"] += 1
        return {"float32": 0.001, "bfloat16": 0.004}

    monkeypatch.setattr(device_mod, "_probe_cpu_dtypes", counting)
    for _ in range(5):
        device_mod.resolve_dtype("cpu", None)
    assert calls["n"] == 1


# ==============================================================================
# config validation
# ==============================================================================

def _cfg(embedding: dict) -> dict:
    return {"version": 2,
            "storage": {"provider": "sqlite", "options": {"path": None}},
            "retrieval": {"mode": "hybrid"},   # an embedding provider requires hybrid
            "embedding": embedding,
            "rerank": {"provider": "none"}}


def test_config_accepts_a_valid_dtype():
    from ai_db.config import parse_config

    cfg = parse_config(_cfg({"provider": "sentence_transformers",
                             "model": "m", "device": "cpu", "batch_size": 8,
                             "dtype": "float32"}))
    assert cfg.embedding.options["dtype"] == "float32"


@pytest.mark.parametrize("bad", ["float64", "int8", "FLOAT32", "", "bf16"])
def test_config_rejects_an_invalid_dtype(bad):
    from ai_db.config import parse_config

    with pytest.raises(AiDbConfigError, match="dtype"):
        parse_config(_cfg({"provider": "sentence_transformers",
                           "model": "m", "device": "cpu", "batch_size": 8,
                           "dtype": bad}))


def test_config_dtype_is_optional():
    from ai_db.config import parse_config

    cfg = parse_config(_cfg({"provider": "sentence_transformers",
                             "model": "m", "device": "cpu", "batch_size": 8}))
    assert "dtype" not in cfg.embedding.options


# ==============================================================================
# float16 is never auto-selected
# ==============================================================================

def test_float16_is_not_a_cpu_candidate():
    """Most CPUs have no fp16 accumulate; measured 0.11 chunks/s here.

    It stays selectable by hand for the rare CPU that does support it, but the
    gate must never pick it on its own.
    """
    assert "float16" not in constants.CPU_DTYPES


# ==============================================================================
# rerank config: every shape used to be rejected
# ==============================================================================

def _hybrid_cfg(rerank: dict) -> dict:
    return {"version": 2,
            "storage": {"provider": "sqlite", "options": {"path": None}},
            "retrieval": {"mode": "hybrid"},
            "embedding": {"provider": "sentence_transformers", "model": "m",
                          "device": "cpu", "batch_size": 8},
            "rerank": rerank}


def test_rerank_provider_options_are_configurable():
    """Regression: no shape of a non-`none` rerank section used to validate.

    The section was key-checked against {provider, options, top_n}, which
    rejected the flat sibling shape, while the option extraction looked for
    required keys inside a nested "options" dict that nothing ever wrote. Every
    shape failed, so rerank could not be switched on from a config file at all.
    Options are siblings of "provider", exactly as in the `embedding` section.
    """
    from ai_db.config import parse_config

    cfg = parse_config(_hybrid_cfg({"provider": "sentence_transformers",
                                    "model": "m", "device": "cpu"}))
    assert cfg.rerank.provider == "sentence_transformers"
    assert cfg.rerank.options == {"model": "m", "device": "cpu"}
    assert cfg.rerank.top_n == 10


def test_rerank_top_n_is_a_sibling_not_an_option():
    from ai_db.config import parse_config

    cfg = parse_config(_hybrid_cfg({"provider": "sentence_transformers",
                                    "model": "m", "device": "cpu", "top_n": 25}))
    assert cfg.rerank.top_n == 25
    assert "top_n" not in cfg.rerank.options


@pytest.mark.parametrize("bad,match", [
    ({"provider": "sentence_transformers", "model": "m"}, "device"),
    ({"provider": "sentence_transformers", "model": "m", "device": "cpu", "typo": 1}, "typo"),
    ({"provider": "sentence_transformers", "model": "m", "device": "cpu", "top_n": 0}, "top_n"),
    ({"provider": "sentence_transformers",
      "options": {"model": "m", "device": "cpu"}}, "requires"),
])
def test_rerank_config_errors_are_still_useful(bad, match):
    from ai_db.config import parse_config

    with pytest.raises(AiDbConfigError, match=match):
        parse_config(_hybrid_cfg(bad))
