"""Silence Hugging Face / tqdm noise during model loads unless ``--verbose`` (TODO B2)."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator

VERBOSE_ENV = "AI_DB_VERBOSE"
_NOISY_LOGGERS = ("transformers", "sentence_transformers", "huggingface_hub", "tqdm",
                  "httpx", "httpcore", "urllib3", "filelock", "torch")


def is_verbose() -> bool:
    return os.environ.get(VERBOSE_ENV, "") not in ("", "0")


def set_verbose(on: bool) -> None:
    if on:
        os.environ[VERBOSE_ENV] = "1"


def _is_cached(model: str) -> bool:
    if os.path.isdir(model):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(model, "config.json"), str)
    except (ImportError, OSError, ValueError):
        return False


@contextlib.contextmanager
def quiet_model_load(model: str) -> Iterator[None]:
    """Quiet loggers/progress bars; load offline when the model is already cached."""
    if is_verbose():
        yield
        return
    saved_env = {k: os.environ.get(k) for k in ("HF_HUB_DISABLE_PROGRESS_BARS", "HF_HUB_OFFLINE",
                                                "TRANSFORMERS_VERBOSITY", "TOKENIZERS_PARALLELISM")}
    saved_levels = {n: logging.getLogger(n).level for n in _NOISY_LOGGERS}
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    offline = _is_cached(model)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    for n in _NOISY_LOGGERS:
        logging.getLogger(n).setLevel(logging.ERROR)
    restore_hub = []
    try:
        import huggingface_hub.constants as hc
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
        if offline:
            restore_hub.append(("HF_HUB_OFFLINE", hc.HF_HUB_OFFLINE))
            hc.HF_HUB_OFFLINE = True
    except (ImportError, AttributeError):
        pass
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
    except (ImportError, AttributeError):
        pass
    try:
        yield
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for n, lv in saved_levels.items():
            logging.getLogger(n).setLevel(lv)
        try:
            import huggingface_hub.constants as hc
            for attr, val in restore_hub:
                setattr(hc, attr, val)
        except (ImportError, AttributeError):
            pass
