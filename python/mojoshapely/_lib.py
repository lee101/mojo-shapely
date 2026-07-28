from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-shapely.so")
I = ctypes.c_int64
F = ctypes.c_double
LOCATE_PARALLEL_WORK = 1 << 20
SEGMENT_PARALLEL_WORK = 1 << 18

_SIGNATURES = {
    "msh_locate_points": ([I, I, I, I, I, I], None),
    "msh_ring_area": ([I, I], F),
    "msh_line_length": ([I, I, I], F),
    "msh_segment_intersections": ([I, I, I, I, I, I, I], None),
    "msh_orient_batch": ([I, I, I, I, I], None),
}

_LIB = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        if not os.path.exists(LIB):
            raise RuntimeError("Mojo library not built; run `pixi run build`")
        _LIB = ctypes.CDLL(LIB)
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_LIB, name)
            fn.argtypes = args
            fn.restype = result
    return _LIB


def f64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    if not value.size:
        raise ValueError("cannot take an FFI address for an empty buffer")
    return int(value.ctypes.data)


def _pairs(value, name: str) -> np.ndarray:
    array = f64(value)
    if array.size % 2:
        raise ValueError(f"{name} must contain coordinate pairs")
    array = array.reshape(-1, 2)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _segments(value, name: str) -> np.ndarray:
    array = f64(value)
    if array.size % 4:
        raise ValueError(f"{name} must contain x0, y0, x1, y1 rows")
    array = array.reshape(-1, 4)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def ring_area(coords: np.ndarray) -> float:
    coords = _pairs(coords, "ring coordinates")
    if len(coords) < 3:
        return 0.0
    return float(lib().msh_ring_area(addr(coords), len(coords)))


def line_length(coords: np.ndarray, closed: bool = False) -> float:
    coords = _pairs(coords, "line coordinates")
    if len(coords) < 2:
        return 0.0
    return float(lib().msh_line_length(addr(coords), len(coords), int(closed)))


def locate_points(rings: list[np.ndarray], points) -> np.ndarray:
    if not rings:
        return np.zeros(len(_pairs(points, "points")), dtype=np.float64)
    rings = [_pairs(ring, "ring coordinates") for ring in rings]
    coords = f64(np.concatenate(rings, axis=0))
    offsets = np.cumsum(
        np.asarray([0] + [len(ring) for ring in rings], dtype=np.int64),
        dtype=np.int64,
    )
    return locate_points_buffer(coords, offsets, points)


def locate_points_buffer(
    coords: np.ndarray, offsets: np.ndarray, points
) -> np.ndarray:
    coords = _pairs(coords, "ring coordinates")
    offsets = np.asarray(offsets)
    if offsets.dtype != np.int64:
        raise TypeError("offsets must have dtype int64")
    offsets = np.ascontiguousarray(offsets)
    if offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("offsets must contain a start and at least one ring end")
    if offsets[0] != 0 or offsets[-1] != len(coords) or np.any(np.diff(offsets) < 0):
        raise ValueError("offsets must be monotonic and span the coordinate buffer")
    pts = _pairs(points, "points")
    if not len(pts):
        return np.empty(0, dtype=np.float64)
    result = np.empty(len(pts), dtype=np.float64)
    lib().msh_locate_points(
        addr(coords), addr(offsets), len(offsets) - 1, addr(pts), addr(result), len(pts)
    )
    return result


def segment_intersections(a: np.ndarray, b: np.ndarray):
    a = _segments(a, "left segments")
    b = _segments(b, "right segments")
    b_soa = np.ascontiguousarray(b.T)
    shape = (len(a), len(b))
    ts = np.empty(shape, dtype=np.float64)
    us = np.empty(shape, dtype=np.float64)
    kinds = np.empty(shape, dtype=np.int64)
    if a.size and b.size:
        lib().msh_segment_intersections(
            addr(a), addr(b_soa), addr(ts), addr(us), addr(kinds), len(a), len(b)
        )
    else:
        ts.fill(0)
        us.fill(0)
        kinds.fill(0)
    return ts, us, kinds
