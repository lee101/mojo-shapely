"""Honest same-input benchmarks against Shapely/GEOS."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import shapely
from shapely import geometry as sg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojoshapely as msh  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def cases():
    polygon_coords = np.array(
        [(0, 0), (100, 0), (100, 100), (70, 100), (70, 30), (30, 30), (30, 100), (0, 100)],
        dtype=np.float64,
    )
    ours = msh.Polygon(polygon_coords)
    theirs = sg.Polygon(polygon_coords)
    rng = np.random.default_rng(42)
    x = rng.uniform(-20, 120, 2_000_000)
    y = rng.uniform(-20, 120, 2_000_000)
    yield (
        "contains_xy, 2M points",
        lambda: msh.contains_xy(ours, x, y),
        lambda: shapely.contains_xy(theirs, x, y),
    )

    angles = np.linspace(0, 2 * np.pi, 1_000, endpoint=False)
    line_coords = np.column_stack((angles, np.sin(angles * 17))) * [1000, 10]
    ours_line = msh.LineString(line_coords)
    their_line = sg.LineString(line_coords)
    yield (
        "LineString.length, 1k vertices",
        lambda: ours_line.length,
        lambda: their_line.length,
    )

    def poly(cx, cy, phase):
        angles = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        radius = 10 + 1.5 * np.sin(angles * 7 + phase)
        return np.column_stack((cx + radius * np.cos(angles), cy + radius * np.sin(angles)))

    ac, bc = poly(0, 0, 0.1), poly(5, 2, 0.8)
    ma, mb = msh.Polygon(ac), msh.Polygon(bc)
    sa, sb = sg.Polygon(ac), sg.Polygon(bc)
    yield (
        "Polygon.intersection, 256 vertices",
        lambda: ma.intersection(mb),
        lambda: sa.intersection(sb),
    )
    yield (
        "Polygon.union, 256 vertices",
        lambda: ma.union(mb),
        lambda: sa.union(sb),
    )


def main():
    rows = []
    for name, mojo, geos in cases():
        mojo()
        geos()
        mt = timeit(mojo)
        st = timeit(geos)
        rows.append((name, mt, st, st / mt))
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| Case | mojo-shapely | Shapely/GEOS | Relative |")
    print("|---|---:|---:|---:|")
    for name, mt, st, ratio in rows:
        label = "faster" if ratio >= 1 else "slower"
        print(f"| {name} | {mt * 1e3:.3f} ms | {st * 1e3:.3f} ms | {ratio:.2f}x {label} |")


if __name__ == "__main__":
    main()
