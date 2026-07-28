from __future__ import annotations

import math

import numpy as np

from . import _lib
from .geometry import (
    BaseGeometry,
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)


def _polygons(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection) and geometry.is_empty:
        return []
    raise TypeError("operation requires Polygon or MultiPolygon operands")


def _rings(polygon: Polygon, oriented: bool = False) -> list[np.ndarray]:
    rings = [polygon._shell, *polygon._holes]
    if not oriented:
        return rings
    result = []
    for index, ring in enumerate(rings):
        area = _lib.ring_area(ring)
        want_positive = index == 0
        result.append(ring if (area > 0) == want_positive else ring[::-1].copy())
    return result


def _segments_from_rings(rings: list[np.ndarray]) -> np.ndarray:
    count = sum(len(ring) for ring in rings)
    result = np.empty((count, 4), dtype=np.float64)
    offset = 0
    for ring in rings:
        stop = offset + len(ring)
        if stop > offset:
            result[offset:stop, :2] = ring
            result[offset : stop - 1, 2:] = ring[1:]
            result[stop - 1, 2:] = ring[0]
        offset = stop
    return result


def _line_segments(line: LineString) -> np.ndarray:
    if len(line._coords) < 2:
        return np.empty((0, 4), dtype=np.float64)
    return np.ascontiguousarray(
        np.column_stack((line._coords[:-1], line._coords[1:])).reshape(-1, 4)
    )


def _geometry_segments(geometry) -> np.ndarray:
    if isinstance(geometry, Polygon):
        return _segments_from_rings(_rings(geometry))
    if isinstance(geometry, LineString):
        return _line_segments(geometry)
    return np.empty((0, 4), dtype=np.float64)


def _location(polygon: Polygon, points) -> np.ndarray:
    return _lib.locate_points_buffer(
        polygon._location_coords, polygon._location_offsets, points
    )


def contains_xy(geometry, x, y=None):
    if not isinstance(geometry, Polygon):
        raise TypeError("contains_xy currently supports Polygon geometries")
    if y is None:
        points = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        shape = (len(points),)
    else:
        xb, yb = np.broadcast_arrays(x, y)
        shape = xb.shape
        points = np.column_stack((xb.ravel(), yb.ravel()))
    result = (_location(geometry, points) == 1).reshape(shape)
    return bool(result.item()) if result.ndim == 0 else result


def intersects_xy(geometry, x, y=None):
    if not isinstance(geometry, Polygon):
        raise TypeError("intersects_xy currently supports Polygon geometries")
    if y is None:
        points = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        shape = (len(points),)
    else:
        xb, yb = np.broadcast_arrays(x, y)
        shape = xb.shape
        points = np.column_stack((xb.ravel(), yb.ravel()))
    result = (_location(geometry, points) != 0).reshape(shape)
    return bool(result.item()) if result.ndim == 0 else result


def _point_on_line(point: Point, line: LineString, strict=False) -> bool:
    if point.is_empty or line.is_empty:
        return False
    p = point._coords[0]
    for segment in _line_segments(line):
        a = segment[:2]
        b = segment[2:]
        ab = b - a
        ap = p - a
        cross = ab[0] * ap[1] - ab[1] * ap[0]
        eps = 1e-12 * (1.0 + np.abs(segment).sum() + np.abs(p).sum())
        if (
            abs(cross) <= eps
            and np.all(p >= np.minimum(a, b) - eps)
            and np.all(p <= np.maximum(a, b) + eps)
        ):
            if strict and (np.array_equal(p, line._coords[0]) or np.array_equal(p, line._coords[-1])):
                return False
            return True
    return False


def _segment_matrix(a, b):
    sa = _geometry_segments(a)
    sb = _geometry_segments(b)
    return (*_lib.segment_intersections(sa, sb), sa, sb)


def intersects(a, b, **kwargs) -> bool:
    if not isinstance(a, BaseGeometry) or not isinstance(b, BaseGeometry):
        raise TypeError("intersects expects geometry objects")
    if a.is_empty or b.is_empty:
        return False
    if isinstance(a, Point) and isinstance(b, Point):
        return bool(np.array_equal(a._coords, b._coords))
    if isinstance(a, Point) and isinstance(b, Polygon):
        return bool(_location(b, a._coords)[0] != 0)
    if isinstance(b, Point) and isinstance(a, Polygon):
        return intersects(b, a)
    if isinstance(a, Point) and isinstance(b, LineString):
        return _point_on_line(a, b)
    if isinstance(b, Point) and isinstance(a, LineString):
        return _point_on_line(b, a)
    if isinstance(a, MultiPolygon):
        return any(intersects(part, b) for part in a.geoms)
    if isinstance(b, MultiPolygon):
        return any(intersects(a, part) for part in b.geoms)
    if isinstance(a, (Polygon, LineString)) and isinstance(b, (Polygon, LineString)):
        _, _, kinds, _, _ = _segment_matrix(a, b)
        if np.any(kinds):
            return True
        if isinstance(a, Polygon) and len(b._coordinate_arrays()[0]):
            if _location(a, b._coordinate_arrays()[0][:1])[0] != 0:
                return True
        if isinstance(b, Polygon) and len(a._coordinate_arrays()[0]):
            if _location(b, a._coordinate_arrays()[0][:1])[0] != 0:
                return True
        return False
    if isinstance(a, GeometryCollection):
        return any(intersects(g, b) for g in a.geoms)
    if isinstance(b, GeometryCollection):
        return any(intersects(a, g) for g in b.geoms)
    return False


def disjoint(a, b, **kwargs) -> bool:
    return not intersects(a, b)


def covers(a, b, **kwargs) -> bool:
    if a.is_empty or b.is_empty:
        return False
    if isinstance(a, Point):
        return isinstance(b, Point) and intersects(a, b)
    if isinstance(a, LineString) and isinstance(b, Point):
        return _point_on_line(b, a)
    if isinstance(a, Polygon) and isinstance(b, Point):
        return bool(_location(a, b._coords)[0] != 0)
    if isinstance(a, Polygon) and isinstance(b, (LineString, Polygon)):
        points = b._coordinate_arrays()
        if not all(np.all(_location(a, array) != 0) for array in points if len(array)):
            return False
        boundary = _geometry_segments(b)
        container = _geometry_segments(a)
        ts, _, kinds = _lib.segment_intersections(boundary, container)
        parameters = _split_parameters(boundary, container, ts, kinds, True)
        fragments = _fragments(boundary, parameters)
        if not len(fragments):
            return True
        midpoints = np.asarray([(start + stop) * 0.5 for start, stop in fragments])
        return bool(np.all(_location(a, midpoints) != 0))
    if isinstance(a, MultiPolygon):
        return any(covers(part, b) for part in a.geoms)
    return False


def contains(a, b, **kwargs) -> bool:
    if not covers(a, b):
        return False
    if isinstance(a, Polygon) and isinstance(b, Point):
        return bool(_location(a, b._coords)[0] == 1)
    if isinstance(a, LineString) and isinstance(b, Point):
        return _point_on_line(b, a, strict=True)
    if isinstance(a, Polygon) and isinstance(b, Polygon):
        if equals(a, b):
            return True
        samples = np.concatenate(b._coordinate_arrays())
        if np.any(_location(a, samples) == 1):
            return True
        center = samples.mean(axis=0, keepdims=True)
        return bool(_location(a, center)[0] == 1)
    return covers(a, b)


def within(a, b, **kwargs) -> bool:
    return contains(b, a)


def covered_by(a, b, **kwargs) -> bool:
    return covers(b, a)


def touches(a, b, **kwargs) -> bool:
    if not intersects(a, b):
        return False
    if isinstance(a, Polygon) and isinstance(b, Point):
        return bool(_location(a, b._coords)[0] == -1)
    if isinstance(b, Polygon) and isinstance(a, Point):
        return touches(b, a)
    if isinstance(a, LineString) and isinstance(b, Point):
        return _point_on_line(b, a) and not _point_on_line(b, a, strict=True)
    if isinstance(b, LineString) and isinstance(a, Point):
        return touches(b, a)
    if isinstance(a, Polygon) and isinstance(b, Polygon):
        ts, us, kinds, _, _ = _segment_matrix(a, b)
        proper = (kinds == 1) & (ts > 1e-10) & (ts < 1 - 1e-10)
        proper &= (us > 1e-10) & (us < 1 - 1e-10)
        if np.any(proper):
            return False
        if np.any(_location(a, b._shell) == 1) or np.any(_location(b, a._shell) == 1):
            return False
        return True
    return not crosses(a, b)


def overlaps(a, b, **kwargs) -> bool:
    if type(a) is not type(b) and not (
        isinstance(a, (Polygon, MultiPolygon)) and isinstance(b, (Polygon, MultiPolygon))
    ):
        return False
    return intersects(a, b) and not covers(a, b) and not covers(b, a) and not touches(a, b)


def crosses(a, b, **kwargs) -> bool:
    if isinstance(a, LineString) and isinstance(b, LineString):
        ts, us, kinds, _, _ = _segment_matrix(a, b)
        proper = (kinds == 1) & (ts > 1e-10) & (ts < 1 - 1e-10)
        proper &= (us > 1e-10) & (us < 1 - 1e-10)
        return bool(np.any(proper))
    if isinstance(a, Polygon) and isinstance(b, LineString):
        locations = _location(a, b._coords)
        return bool(np.any(locations == 1) and np.any(locations == 0))
    if isinstance(b, Polygon) and isinstance(a, LineString):
        return crosses(b, a)
    return False


def equals(a, b, **kwargs) -> bool:
    if type(a) is Point and type(b) is Point:
        return a.is_empty == b.is_empty and (
            a.is_empty or bool(np.array_equal(a._coords, b._coords))
        )
    if isinstance(a, (Polygon, MultiPolygon)) and isinstance(b, (Polygon, MultiPolygon)):
        if not math.isclose(a.area, b.area, rel_tol=1e-10, abs_tol=1e-12):
            return False
        if isinstance(a, MultiPolygon) or isinstance(b, MultiPolygon):
            left = _polygons(a)
            right = _polygons(b)
            if len(left) != len(right):
                return False
            unmatched = list(right)
            for polygon in left:
                match = next(
                    (index for index, candidate in enumerate(unmatched) if equals(polygon, candidate)),
                    None,
                )
                if match is None:
                    return False
                unmatched.pop(match)
            return True
        result = symmetric_difference(a, b)
        return result.is_empty or result.area <= 1e-10 * max(1.0, a.area, b.area)
    if isinstance(a, LineString) and isinstance(b, LineString):
        return bool(
            np.array_equal(a._coords, b._coords)
            or np.array_equal(a._coords, b._coords[::-1])
        )
    return False


def _split_parameters(segments, other, params, kinds, along_a):
    result = [{0.0, 1.0} for _ in range(len(segments))]
    rows, columns = np.nonzero(kinds == 1)
    targets, candidates = (rows, columns) if along_a else (columns, rows)
    for target, row, column in zip(targets, rows, columns):
        result[target].add(float(np.clip(params[row, column], 0.0, 1.0)))

    rows, columns = np.nonzero(kinds == 2)
    targets, candidates = (rows, columns) if along_a else (columns, rows)
    if len(targets):
        selected = segments[targets]
        direction = selected[:, 2:] - selected[:, :2]
        denom = np.einsum("ij,ij->i", direction, direction)
        endpoints = other[candidates]
        for endpoint in (endpoints[:, :2], endpoints[:, 2:]):
            values = np.einsum(
                "ij,ij->i", endpoint - selected[:, :2], direction
            ) / denom
            for target, value in zip(targets, values):
                if -1e-12 <= value <= 1 + 1e-12:
                    result[target].add(float(np.clip(value, 0.0, 1.0)))
    return result


def _fragments(segments, parameters):
    simple = np.fromiter(
        (len(values) == 2 for values in parameters), dtype=bool, count=len(parameters)
    )
    parts = [segments[simple].reshape(-1, 2, 2)]
    for segment, values in zip(segments[~simple], np.asarray(parameters, dtype=object)[~simple]):
        start = segment[:2]
        delta = segment[2:] - start
        ordered = np.asarray(sorted(values))
        valid = np.diff(ordered) > 1e-12
        if np.any(valid):
            parts.append(
                np.stack(
                    (
                        start + ordered[:-1][valid, None] * delta,
                        start + ordered[1:][valid, None] * delta,
                    ),
                    axis=1,
                )
            )
    return np.concatenate(parts) if parts else np.empty((0, 2, 2))


def _boundary_fragments(a: Polygon, b: Polygon, operation: str):
    sa = _segments_from_rings(_rings(a, oriented=True))
    sb = _segments_from_rings(_rings(b, oriented=True))
    ts, us, kinds = _lib.segment_intersections(sa, sb)
    pa = _split_parameters(sa, sb, ts, kinds, True)
    pb = _split_parameters(sb, sa, us, kinds, False)
    candidates = np.concatenate((_fragments(sa, pa), _fragments(sb, pb)))
    if not len(candidates):
        return np.empty((0, 2, 2)), kinds, sa, sb
    scale = max(1.0, *(abs(v) for v in (*a.bounds, *b.bounds)))
    delta = candidates[:, 1] - candidates[:, 0]
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    valid = lengths > 1e-13 * scale
    candidates = candidates[valid]
    delta = delta[valid]
    lengths = lengths[valid]
    midpoints = candidates.mean(axis=1)
    normals = np.column_stack((-delta[:, 1], delta[:, 0])) / lengths[:, None]
    offsets = np.maximum(1e-9 * scale, 1e-7 * lengths)
    probes = np.empty((2 * len(candidates), 2), dtype=np.float64)
    probes[0::2] = midpoints + normals * offsets[:, None]
    probes[1::2] = midpoints - normals * offsets[:, None]
    la = _location(a, probes)
    lb = _location(b, probes)
    inside_a = la == 1
    inside_b = lb == 1
    if operation == "intersection":
        truth = inside_a & inside_b
    elif operation == "union":
        truth = inside_a | inside_b
    elif operation == "difference":
        truth = inside_a & ~inside_b
    else:
        truth = inside_a != inside_b
    left, right = truth[0::2], truth[1::2]
    forward = left & ~right
    reverse = right & ~left
    kept = np.concatenate((candidates[forward], candidates[reverse, ::-1]))
    return kept, kinds, sa, sb


def _loops(fragments, scale):
    tolerance = 1e-9 * max(1.0, scale)
    fragment_keys = np.rint(fragments / tolerance).astype(np.int64)
    edges = {}
    points = {}
    for (start, stop), (ka_array, kb_array) in zip(fragments, fragment_keys):
        ka = (int(ka_array[0]), int(ka_array[1]))
        kb = (int(kb_array[0]), int(kb_array[1]))
        if ka == kb:
            continue
        edge = (ka, kb)
        reverse = (kb, ka)
        if reverse in edges:
            del edges[reverse]
            continue
        edges[edge] = True
        points.setdefault(ka, start)
        points.setdefault(kb, stop)
    outgoing = {}
    for edge in edges:
        outgoing.setdefault(edge[0], []).append(edge[1])
    unused = set(edges)
    loops = []
    while unused:
        first = next(iter(unused))
        unused.remove(first)
        keys = [first[0], first[1]]
        previous, current = first
        while current != keys[0]:
            choices = [n for n in outgoing.get(current, []) if (current, n) in unused]
            if not choices:
                break
            if len(choices) == 1:
                nxt = choices[0]
            else:
                incoming = points[current] - points[previous]
                angles = []
                for candidate in choices:
                    direction = points[candidate] - points[current]
                    cross = incoming[0] * direction[1] - incoming[1] * direction[0]
                    angle = math.atan2(cross, incoming @ direction)
                    angles.append(angle if angle >= 0 else angle + 2 * math.pi)
                nxt = choices[int(np.argmax(angles))]
            unused.remove((current, nxt))
            keys.append(nxt)
            previous, current = current, nxt
            if len(keys) > len(edges) + 2:
                break
        if current == keys[0] and len(keys) >= 4:
            ring = np.asarray([points[k] for k in keys[:-1]])
            if abs(_lib.ring_area(ring)) > tolerance * tolerance:
                loops.append(ring)
    return loops


def _assemble(loops):
    shells = [ring for ring in loops if _lib.ring_area(ring) > 0]
    holes = [ring for ring in loops if _lib.ring_area(ring) < 0]
    polygons = [Polygon(shell) for shell in shells]
    assignments = [[] for _ in shells]
    for hole in holes:
        candidates = []
        for index, shell in enumerate(shells):
            if _lib.locate_points([shell], hole[:1])[0] != 0:
                candidates.append((abs(_lib.ring_area(shell)), index))
        if candidates:
            assignments[min(candidates)[1]].append(hole)
    polygons = [Polygon(shell, assignments[i]) for i, shell in enumerate(shells)]
    if not polygons:
        return GeometryCollection()
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)


def _lower_dimensional_intersection(kinds, sa, sb):
    points = []
    lines = []
    ts, us, exact = _lib.segment_intersections(sa, sb)
    for i, j in zip(*np.where(exact == 1)):
        point = sa[i, :2] + ts[i, j] * (sa[i, 2:] - sa[i, :2])
        if not any(np.allclose(point, p, atol=1e-10, rtol=0) for p in points):
            points.append(point)
    for i, j in zip(*np.where(exact == 2)):
        a, b = sa[i], sb[j]
        direction = a[2:] - a[:2]
        denom = direction @ direction
        if denom == 0:
            continue
        values = sorted(
            np.clip(
                [
                    (b[:2] - a[:2]) @ direction / denom,
                    (b[2:] - a[:2]) @ direction / denom,
                ],
                0,
                1,
            )
        )
        if values[1] - values[0] > 1e-12:
            lines.append(LineString([a[:2] + values[0] * direction, a[:2] + values[1] * direction]))
    if lines:
        return lines[0] if len(lines) == 1 else GeometryCollection(lines)
    if len(points) == 1:
        return Point(points[0])
    if points:
        return GeometryCollection([Point(p) for p in points])
    return GeometryCollection()


def _binary_polygon(a, b, operation):
    if isinstance(a, MultiPolygon) or isinstance(b, MultiPolygon):
        if operation == "intersection":
            pieces = [
                _binary_polygon(pa, pb, operation)
                for pa in _polygons(a)
                for pb in _polygons(b)
            ]
            polygons = [p for piece in pieces for p in _polygons(piece) if not p.is_empty]
            return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
        raise NotImplementedError("union, difference, and xor currently require Polygon operands")
    if a.is_empty:
        return b if operation in ("union", "symmetric_difference") else GeometryCollection()
    if b.is_empty:
        return a if operation in ("union", "difference", "symmetric_difference") else GeometryCollection()
    fragments, kinds, sa, sb = _boundary_fragments(a, b, operation)
    scale = max(1.0, *(abs(v) for v in (*a.bounds, *b.bounds)))
    result = _assemble(_loops(fragments, scale))
    if result.is_empty and operation == "intersection":
        return _lower_dimensional_intersection(kinds, sa, sb)
    return result


def intersection(a, b, **kwargs):
    if isinstance(a, Point):
        return a if intersects(a, b) else Point()
    if isinstance(b, Point):
        return b if intersects(a, b) else Point()
    return _binary_polygon(a, b, "intersection")


def union(a, b, **kwargs):
    if isinstance(a, Point) or isinstance(b, Point):
        point, other = (a, b) if isinstance(a, Point) else (b, a)
        return other if intersects(point, other) else GeometryCollection([other, point])
    return _binary_polygon(a, b, "union")


def difference(a, b, **kwargs):
    if isinstance(a, Point):
        return Point() if intersects(a, b) else a
    if isinstance(b, Point):
        return a
    return _binary_polygon(a, b, "difference")


def symmetric_difference(a, b, **kwargs):
    if isinstance(a, Point) or isinstance(b, Point):
        return union(a, b)
    left = difference(a, b)
    right = difference(b, a)
    pieces = _polygons(left) + _polygons(right)
    if not pieces:
        return GeometryCollection()
    return pieces[0] if len(pieces) == 1 else MultiPolygon(pieces)
