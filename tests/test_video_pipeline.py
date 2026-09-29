"""Video processing: the processor contract, entry-point loading, and the pipeline.

Covers ``ai_db.video``:

  - ``processors.base``  registry, dependency gating, entry-point discovery
  - ``pipeline``        stage sequencing, error collection, result merging
  - ``factory``         entry-point loading

The shipped processors depend on heavy optional packages (opencv, torch, openai)
that are not installed everywhere, so the pipeline tests use stub processors
registered directly rather than the real ones.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ai_db.video.pipeline import VideoPipeline
from ai_db.video.processors.base import (
    Keyframe,
    VideoProcessor,
    VideoProcessorResult,
    get_video_processor,
    list_video_processors,
    register_video_processor,
)


# ==============================================================================
# Stubs
# ==============================================================================

class StubProcessor(VideoProcessor):
    """Minimal concrete processor. Deliberately declares no `requires`."""

    name = "stub"
    version = "1.0.0"
    provides = ["keyframes"]

    def process(self, video_path: Path, config: dict) -> VideoProcessorResult:
        n = config.get("count", 1)
        return VideoProcessorResult(
            keyframes=[Keyframe(index=i, timestamp=float(i), filepath="f.png", width=2, height=2)
                       for i in range(n)],
            metadata={"count": n},
        )


class MissingDepsProcessor(StubProcessor):
    name = "missing_deps"
    requires = ["a_package_that_does_not_exist"]

    def process(self, video_path: Path, config: dict) -> VideoProcessorResult:
        raise AssertionError("must not run when dependencies are missing")


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video")
    return path


@pytest.fixture
def stub_registry(monkeypatch):
    """A VideoPipeline over known stubs, leaving the global registry untouched."""
    pipeline = VideoPipeline()
    monkeypatch.setattr(pipeline, "processors", {
        "stub": StubProcessor(),
        "missing_deps": MissingDepsProcessor(),
    })
    return pipeline


# ==============================================================================
# Processor contract
# ==============================================================================

class TestVideoProcessorContract:
    def test_defaults_are_real_collections(self):
        """Regression: `requires`/`provides` were declared with dataclasses.field()
        in a class that is not a dataclass, so they were `Field` objects and
        iterating `requires` raised TypeError for any subclass that did not
        override them.
        """
        proc = StubProcessor()
        assert proc.requires == []
        assert proc.check_dependencies() == []

    def test_declared_dependencies_are_reported_missing(self):
        assert MissingDepsProcessor().check_dependencies() == [
            "a_package_that_does_not_exist"
        ]

    def test_satisfied_dependencies_report_nothing(self):
        class Installed(StubProcessor):
            name = "installed"
            requires = ["json", "pathlib"]  # stdlib, always importable

        assert Installed().check_dependencies() == []

    def test_can_handle_matches_known_video_extensions(self):
        proc = StubProcessor()
        assert proc.can_handle(Path("a.mp4"))
        assert proc.can_handle(Path("a.MOV"))
        assert proc.can_handle(Path("a.webm"))
        assert not proc.can_handle(Path("a.txt"))

    def test_abstract_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            VideoProcessor()  # type: ignore[abstract]


# ==============================================================================
# Registry
# ==============================================================================

class TestVideoProcessorRegistry:
    def test_register_and_get_round_trip(self):
        proc = StubProcessor()
        register_video_processor(proc)
        try:
            assert get_video_processor("stub") is proc
            assert "stub" in list_video_processors()
        finally:
            from ai_db.video.processors.base import _VIDEO_PROCESSORS
            _VIDEO_PROCESSORS.pop("stub", None)

    def test_registering_without_a_name_is_rejected(self):
        class Unnamed(StubProcessor):
            name = ""

        with pytest.raises(ValueError, match="name"):
            register_video_processor(Unnamed())

    def test_unknown_processor_is_none(self):
        assert get_video_processor("definitely-not-registered") is None

    def test_entry_points_loaded_the_shipped_processors(self):
        """The three processors declared in pyproject.toml are discovered, and
        any that are skipped are skipped because of missing optional deps.
        """
        names = set(list_video_processors())
        assert {"frame_extractor", "vlm_analyzer", "depth_estimator"} <= names


# ==============================================================================
# Pipeline
# ==============================================================================

class TestVideoPipeline:
    def test_missing_video_file_raises(self, stub_registry):
        with pytest.raises(FileNotFoundError, match="Video not found"):
            stub_registry.run("/no/such/video.mp4")

    def test_unknown_pipeline_lists_presets(self, stub_registry, fake_video):
        with pytest.raises(ValueError, match="Unknown pipeline"):
            stub_registry.run(fake_video, pipeline="nope")

    def test_runs_a_processor_and_merges_results(self, stub_registry, fake_video, tmp_path):
        res = stub_registry.run(fake_video, processors=["stub"], config={"stub": {"count": 3}})

        assert res.stages_run == ["stub"]
        assert res.errors == {}
        assert len(res.keyframes) == 3
        assert res.metadata["stub"] == {"count": 3}

    def test_output_dir_is_created_and_passed_down(self, stub_registry, fake_video, tmp_path):
        out = tmp_path / "nested" / "out"
        stub_registry.run(fake_video, processors=["stub"], output_dir=out)
        assert out.is_dir()

    def test_missing_processor_is_collected_not_raised(self, stub_registry, fake_video):
        res = stub_registry.run(fake_video, processors=["stub", "ghost"])

        assert "ghost" in res.errors
        assert "Processor not found" in res.errors["ghost"]
        # The stages that could run still ran.
        assert res.stages_run == ["stub"]

    def test_missing_dependencies_are_collected_not_raised(self, stub_registry, fake_video):
        res = stub_registry.run(fake_video, processors=["missing_deps"])

        assert "missing_deps" in res.errors
        assert "Missing dependencies" in res.errors["missing_deps"]
        assert res.stages_run == []

    def test_preset_selects_the_processor_list(self, stub_registry, fake_video):
        res = stub_registry.run(fake_video, pipeline="basic")
        # "basic" is frame_extractor, which is not in the stub registry.
        assert res.errors  # reported, not raised
        assert "frame_extractor" in res.errors

    def test_results_are_handed_to_later_stages(self, fake_video):
        seen: dict = {}

        class Second(VideoProcessor):
            name = "second"

            def process(self, video_path: Path, config: dict) -> VideoProcessorResult:
                seen.update(config)
                return VideoProcessorResult()

        pipeline = VideoPipeline()
        pipeline.processors = {"stub": StubProcessor(), "second": Second()}
        pipeline.run(fake_video, processors=["stub", "second"],
                     config={"stub": {"count": 2}})

        assert len(seen["keyframes"]) == 2


# ==============================================================================
# MCP tool surface
# ==============================================================================

VIDEO_TOOLS = {"video_to_scene", "list_video_processors", "video_extract_keyframes"}


class TestVideoMCPTools:
    def test_registration_exposes_every_video_tool(self):
        from ai_db.dispatcher import ServiceDispatcher
        from ai_db.mcp_tools.video_tools import register_video_tools

        dispatcher = ServiceDispatcher(db_path=":memory:")
        register_video_tools(dispatcher)
        assert VIDEO_TOOLS <= set(dispatcher._tools)
        assert all(dispatcher._tools[n].category == "video" for n in VIDEO_TOOLS)

    def test_list_video_processors_tool_returns_presets(self):
        from ai_db.dispatcher import ServiceDispatcher
        from ai_db.mcp_tools.video_tools import register_video_tools

        dispatcher = ServiceDispatcher(db_path=":memory:")
        register_video_tools(dispatcher)
        res = dispatcher.execute("list_video_processors", {})
        assert "presets" in res
        assert "basic" in res["presets"]


# ==============================================================================
# stdout hygiene
# ==============================================================================

class TestVideoLoadingKeepsStdoutClean:
    def test_importing_the_factory_prints_nothing_to_stdout(self):
        """The stdio MCP server is a JSON-RPC channel on stdout.

        Importing this module runs entry-point discovery, which used to print
        skip/load notices there and made the client fail its parse.
        """
        code = (
            "import ai_db.video.processors.base, ai_db.video.factory, sys;"
            "sys.stderr.close()"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert proc.stdout == ""
