"""Outline lists only module/class-scope entries, indents members, and has no duplicates."""
from ai_db.parser.ast_visitor import extract_file_outline

PLANNER = '''\
from dataclasses import dataclass, field


@dataclass
class Subtask:
    id: str
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        tmp: list[int] = list()
        return {}


@dataclass
class Plan:
    goal: str
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {}


def plan_scene(goal: str):
    known: dict[str, int] = dict()
    cam_x: float = float(1)

    def helper():
        return known

    return cam_x


def plan_asset():
    z_offset: float = float(2)
    return z_offset
'''


def test_outline_has_classes_members_functions_and_no_locals():
    outline = extract_file_outline("planner.py", PLANNER)
    assert [label for _, label in outline] == [
        "class Subtask",
        "  type dependencies",
        "  def to_dict",
        "class Plan",
        "  type missing",
        "  def to_dict",
        "def plan_scene",
        "def plan_asset",
    ]
    assert len(outline) == len(set(outline))
