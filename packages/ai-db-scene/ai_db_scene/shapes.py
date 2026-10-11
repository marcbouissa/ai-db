"""Room footprints for the layout solver: rect (walls handled in layout), ellipse, polygon.

All shapes live inside the box [0, W] x [0, D]. A *wall reference* is ``(p, n)``: a point on
the wall and the inward unit normal there; a box placed flush has its back face on the
tangent line through ``p``.
"""

from __future__ import annotations

import math
import re

Pt = tuple[float, float]
_WALL_N = re.compile(r"^wall_(\d+)$")


def polygon_wall_names(n: int) -> list[str]:
    return [f"wall_{i}" for i in range(1, n + 1)]


def _area(pts: list[Pt]) -> float:
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1], strict=True)) / 2


class RoomShape:
    def __init__(self, kind: str, w: float, d: float, points: tuple[Pt, ...] = ()) -> None:
        self.kind = kind
        self.w, self.d = w, d
        self.curved = kind == "ellipse"
        self.cx, self.cy = w / 2, d / 2
        self.pts: list[Pt] = []
        if kind == "polygon":
            pts = list(points)
            if _area(pts) < 0:
                pts = pts[::-1]  # CCW so the inward normal is always the left of an edge
            self.pts = pts
            a = _area(pts)
            if abs(a) > 1e-12:
                self.cx = sum((p[0] + q[0]) * (p[0] * q[1] - q[0] * p[1]) for p, q in zip(pts, pts[1:] + pts[:1], strict=True)) / (6 * a)
                self.cy = sum((p[1] + q[1]) * (p[0] * q[1] - q[0] * p[1]) for p, q in zip(pts, pts[1:] + pts[:1], strict=True)) / (6 * a)
        self._orig = list(points)

    # ---- containment ----
    def contains(self, x: float, y: float, eps: float) -> bool:
        if self.kind == "ellipse":
            a, b = self.w / 2 + eps, self.d / 2 + eps
            return ((x - self.cx) / a) ** 2 + ((y - self.cy) / b) ** 2 <= 1.0
        ins = False
        n = len(self.pts)
        for i in range(n):
            (x1, y1), (x2, y2) = self.pts[i], self.pts[(i + 1) % n]
            if self._seg_dist(x, y, x1, y1, x2, y2)[0] <= eps:
                return True
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                ins = not ins
        return ins

    @staticmethod
    def _seg_dist(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> tuple[float, Pt]:
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
        px, py = x1 + t * dx, y1 + t * dy
        return math.hypot(x - px, y - py), (px, py)

    # ---- walls ----
    def wall_names(self) -> list[str]:
        if self.kind == "ellipse":
            return ["wall"]
        return ["wall", *polygon_wall_names(len(self.pts))]

    def _ell_point(self, th: float) -> tuple[Pt, Pt]:
        a, b = self.w / 2, self.d / 2
        p = (self.cx + a * math.cos(th), self.cy + b * math.sin(th))
        gx, gy = math.cos(th) / a, math.sin(th) / b
        g = math.hypot(gx, gy)
        return p, (-gx / g, -gy / g)

    def wall_ref(self, name: str, at_deg: float | None, x: float, y: float) -> tuple[Pt, Pt]:
        """(point on wall, inward normal) for the wall `name`; `x, y` is the object center."""
        if self.kind == "ellipse":
            if at_deg is not None:
                return self._ell_point(math.radians(at_deg))
            th0 = math.atan2((y - self.cy) / (self.d / 2), (x - self.cx) / (self.w / 2))
            best = min((th0 + math.radians(k * 2.0) for k in range(-90, 91)), key=lambda t: self._dist2(t, x, y))
            lo, hi = best - math.radians(2), best + math.radians(2)
            for _ in range(30):
                m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
                if self._dist2(m1, x, y) < self._dist2(m2, x, y):
                    hi = m2
                else:
                    lo = m1
            return self._ell_point((lo + hi) / 2)
        n = len(self.pts)
        m = _WALL_N.match(name)
        if m and 1 <= int(m.group(1)) <= n:
            return self._edge_ref(self._edge_index(int(m.group(1)) - 1), x, y)
        if at_deg is not None:
            return self._ray_ref(math.radians(at_deg))
        i = min(range(n), key=lambda k: self._seg_dist(x, y, *self.pts[k], *self.pts[(k + 1) % n])[0])
        return self._edge_ref(i, x, y)

    def _edge_index(self, orig_i: int) -> int:
        """Index in the CCW-normalised list of the original edge `orig_i`."""
        n = len(self._orig)
        if self.pts == self._orig:
            return orig_i
        return n - 2 - orig_i if orig_i <= n - 2 else n - 1  # reversed list: edge i -> n-2-i (mod n)

    def _edge_ref(self, i: int, x: float, y: float) -> tuple[Pt, Pt]:
        (x1, y1), (x2, y2) = self.pts[i], self.pts[(i + 1) % len(self.pts)]
        _, p = self._seg_dist(x, y, x1, y1, x2, y2)
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        return p, (-dy / L, dx / L)

    def _ray_ref(self, ang: float) -> tuple[Pt, Pt]:
        ux, uy = math.cos(ang), math.sin(ang)
        best: tuple[float, int] | None = None
        n = len(self.pts)
        for i in range(n):
            (x1, y1), (x2, y2) = self.pts[i], self.pts[(i + 1) % n]
            ex, ey = x2 - x1, y2 - y1
            den = ux * ey - uy * ex
            if abs(den) < 1e-12:
                continue
            t = ((x1 - self.cx) * ey - (y1 - self.cy) * ex) / den
            s = ((x1 - self.cx) * uy - (y1 - self.cy) * ux) / den
            if t > 0 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t < best[0]):
                best = (t, i)
        i = best[1] if best else 0
        t = best[0] if best else 0.0
        return self._edge_ref(i, self.cx + ux * t, self.cy + uy * t)

    def _dist2(self, th: float, x: float, y: float) -> float:
        a, b = self.w / 2, self.d / 2
        return (self.cx + a * math.cos(th) - x) ** 2 + (self.cy + b * math.sin(th) - y) ** 2
