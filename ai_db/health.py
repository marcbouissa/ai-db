"""``ai-db config check``: build every enabled component and make one real call."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_db.config import AppConfig
from ai_db.errors import AiDbError


@dataclass
class HealthReport:
    ok: bool = True
    lines: list[str] = field(default_factory=list)

    def passed(self, msg: str) -> None:
        self.lines.append(f"OK   {msg}")

    def failed(self, msg: str) -> None:
        self.ok = False
        self.lines.append(f"FAIL {msg}")


def check_config(cfg: AppConfig) -> HealthReport:
    report = HealthReport()
    report.passed(f"config {cfg.source_path} (version {cfg.version}, mode={cfg.retrieval_mode})")

    from ai_db.storage.factory import StorageBackendFactory

    try:
        backend = StorageBackendFactory.from_config(cfg)
        backend.initialize()
        caps = sorted(backend.capabilities())
        backend.close()
        report.passed(f"storage '{cfg.storage.provider}' capabilities={caps}")
    except AiDbError as exc:
        report.failed(f"storage '{cfg.storage.provider}': {exc}")

    if cfg.embedding.enabled:
        from ai_db.embed.registry import build_embedder

        _report_device(report, "embedding", cfg.embedding.options.get("device"))
        try:
            emb = build_embedder(cfg.embedding)
            assert emb is not None
            vec = emb.embed_query("ai-db health check")
            if len(vec) != emb.dim:
                report.failed(f"embedding returned dim {len(vec)}, expected {emb.dim}")
            else:
                device = getattr(emb, "device", None) or cfg.embedding.options.get("device")
                report.passed(f"embedding '{emb.model_id}' dim={emb.dim} device={device}")
        except AiDbError as exc:
            report.failed(f"embedding '{cfg.embedding.provider}': {exc}")

    if cfg.rerank.enabled:
        from ai_db.rerank.registry import build_reranker

        _report_device(report, "rerank", cfg.rerank.options.get("device"))
        try:
            rr = build_reranker(cfg.rerank)
            assert rr is not None
            scores = rr.score("health check", ["def health_check(): pass"])
            report.passed(f"rerank '{rr.model_id}' score={scores[0]:.3f}")
        except AiDbError as exc:
            report.failed(f"rerank '{cfg.rerank.provider}': {exc}")

    return report


def _report_device(report: HealthReport, scope: str, configured: str | None) -> None:
    """Print the resolved device and torch's CUDA build for a local provider."""
    from ai_db.device import describe

    d = describe(scope)
    torch_cuda = d["torch_cuda"] or "n/a"
    if not d["torch_installed"]:
        report.passed(
            f"{scope} device={configured or 'cpu'} (torch not installed; "
            f"torch.version.cuda={torch_cuda})")
    elif d["cuda_available"]:
        report.passed(
            f"{scope} device={configured or d['preferred_device']} "
            f"(cuda available, torch.version.cuda={torch_cuda})")
    else:
        report.passed(
            f"{scope} device={configured or d['preferred_device']} "
            f"(no CUDA device; torch.version.cuda={torch_cuda})")
