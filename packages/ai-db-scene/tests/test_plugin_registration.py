"""The plugin's tools reach ai-db through the ai_db.tools entry point."""

import importlib.metadata
import subprocess
import sys

from ai_db.dispatcher import ServiceDispatcher


def test_entry_point_is_installed():
    names = {ep.value for ep in importlib.metadata.entry_points(group="ai_db.tools")}
    assert "ai_db_scene.tools:register" in names


def test_every_dispatcher_gets_scene_and_video_tools(tmp_path):
    tools = set(ServiceDispatcher(db_path=str(tmp_path / "x.db"))._tools)
    assert {"snapshot_scene", "get_scene_spec_schema", "solve_layout", "video_to_scene"} <= tools


def test_registration_does_not_import_heavy_dependencies(tmp_path):
    """Checked in a fresh interpreter: other tests in this session import processors."""
    code = (
        "import sys\n"
        "from ai_db.dispatcher import ServiceDispatcher\n"
        f"ServiceDispatcher(db_path={str(tmp_path / 'x.db')!r})\n"
        "heavy = [m for m in ('torch', 'transformers', 'cv2', "
        "'ai_db_scene.video.processors.depth_estimator') if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == ""
