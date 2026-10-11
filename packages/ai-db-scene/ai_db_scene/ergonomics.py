"""How people use the furniture, as checks (the realistic-modeling skill's proportion table).

A layout can be collision-free and still wrong: a coffee table out of reach of the sofa, a
coffee table taller than the seat, a dining table so close to the wall that nobody can pull a
chair out. These come from furniture sizing and clearance guides (ranges, not standards), and
are reported as warnings with the numbers, never blocking.
"""

from __future__ import annotations

from typing import Any

SEATING = ("sofa", "armchair")
SEAT_HEIGHT_M = 0.45            # typical upholstered seat top (guides: 43-48 cm)
COFFEE_REACH_M = (0.25, 0.75)   # gap seat front -> coffee table (guides: 40-45 cm; tolerant)
COFFEE_HEIGHT_DELTA_M = (-0.15, 0.05)  # table top vs seat top: at or a little below it
SIDE_TABLE_HEIGHT_M = (0.45, 0.75)     # about armrest height
DINING_WALL_CLEAR_M = 0.75      # table edge -> wall to pull a chair out (guides: 0.9-1.0)


def _footprint(p: Any) -> tuple[float, float, float, float]:
    """Axis-aligned (xmin, xmax, ymin, ymax) of a placement (quarter-turn rotations)."""
    w, d = p.dims[0], p.dims[1]
    if round(p.rotation_deg_z / 90) % 2:
        w, d = d, w
    x, y = p.location[0], p.location[1]
    return x - w / 2, x + w / 2, y - d / 2, y + d / 2


def _gap(a: Any, b: Any) -> float:
    ax0, ax1, ay0, ay1 = _footprint(a)
    bx0, bx1, by0, by1 = _footprint(b)
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return float((dx * dx + dy * dy) ** 0.5)


def _warn(objects: list[str], message: str) -> dict[str, Any]:
    return {"check": "ergonomics", "severity": "warning", "objects": objects, "message": message}


def check(spec: Any, placements: list[Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    seats = [p for p in placements if p.category in SEATING]
    for t in (p for p in placements if p.category == "coffee_table"):
        if seats:
            near = min(seats, key=lambda s: _gap(s, t))
            g = _gap(near, t)
            lo, hi = COFFEE_REACH_M
            if not lo <= g <= hi:
                issues.append(_warn([t.id, near.id], f"{t.id} is {g:.2f} m from {near.id}; within reach is "
                                                     f"{lo:.2f}-{hi:.2f} m (about 0.45 m)"))
        dh = t.dims[2] - SEAT_HEIGHT_M
        if not COFFEE_HEIGHT_DELTA_M[0] <= dh <= COFFEE_HEIGHT_DELTA_M[1]:
            issues.append(_warn([t.id], f"{t.id} top is {t.dims[2]:.2f} m; a coffee table sits at or up to "
                                        f"0.15 m below seat height (~{SEAT_HEIGHT_M:.2f} m)"))
    for t in (p for p in placements if p.category == "side_table"):
        lo, hi = SIDE_TABLE_HEIGHT_M
        if not lo <= t.dims[2] <= hi:
            issues.append(_warn([t.id], f"{t.id} is {t.dims[2]:.2f} m tall; side tables match the armrest "
                                        f"({lo:.2f}-{hi:.2f} m)"))
    st = spec.setting
    if st.kind == "indoor" and st.shape == "rect" and not st.rooms:
        hw, hd = st.bounds_m[0] / 2, st.bounds_m[1] / 2
        chairs = [p for p in placements if p.category in ("dining_chair", "chair")]
        for t in (p for p in placements if p.category in ("dining_table", "conference_table")):
            x0, x1, y0, y1 = _footprint(t)
            # sides that have a chair need room behind it
            for side, clear in (("x-", x0 + hw), ("x+", hw - x1), ("y-", y0 + hd), ("y+", hd - y1)):
                has_chair = any((side == "x-" and c.location[0] < x0) or (side == "x+" and c.location[0] > x1)
                                or (side == "y-" and c.location[1] < y0) or (side == "y+" and c.location[1] > y1)
                                for c in chairs if _gap(c, t) < 0.6)
                if has_chair and clear < DINING_WALL_CLEAR_M:
                    issues.append(_warn([t.id], f"{t.id}: {clear:.2f} m from the table edge to the wall behind its "
                                                f"chairs; pulling a chair out needs about 0.9 m"))
    return issues
