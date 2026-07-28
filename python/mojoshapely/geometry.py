from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from . import _lib


def _coords(value, *, minimum: int = 0) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    array = np.ascontiguousarray(array.reshape(-1, 2))
    if len(array) < minimum:
        raise ValueError(f"requires at least {minimum} coordinate pairs")
    if not np.isfinite(array).all():
        raise ValueError("coordinates must be finite")
    return array


def _ring(value) -> np.ndarray:
    array = _coords(value)
    if len(array) and np.array_equal(array[0], array[-1]):
        array = array[:-1].copy()
    if len(array) and len(array) < 3:
        raise ValueError("a linear ring requires at least 3 distinct vertices")
    return array


def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.15g}"


def _coord_text(coords: np.ndarray) -> str:
    return ", ".join(f"{_number(x)} {_number(y)}" for x, y in coords)


class CoordinateSequence(Sequence):
    def __init__(self, coords: np.ndarray, closed: bool = False):
        self._coords = coords
        self._closed = closed

    def __len__(self):
        return len(self._coords) + int(self._closed and len(self._coords) > 0)

    def __getitem__(self, item):
        data = self._array()
        if isinstance(item, slice):
            return [tuple(row) for row in data[item]]
        return tuple(data[item])

    def __iter__(self):
        return (tuple(row) for row in self._array())

    def __array__(self, dtype=None, copy=None):
        result = self._array()
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        return result.copy() if copy else result

    def _array(self):
        if self._closed and len(self._coords):
            return np.vstack((self._coords, self._coords[0]))
        return self._coords


class BaseGeometry:
    geom_type = "Geometry"

    @property
    def is_empty(self) -> bool:
        raise NotImplementedError

    @property
    def area(self) -> float:
        return 0.0

    @property
    def length(self) -> float:
        return 0.0

    @property
    def bounds(self):
        arrays = self._coordinate_arrays()
        if not arrays or not any(len(a) for a in arrays):
            return ()
        data = np.concatenate([a for a in arrays if len(a)])
        return (
            float(data[:, 0].min()),
            float(data[:, 1].min()),
            float(data[:, 0].max()),
            float(data[:, 1].max()),
        )

    @property
    def wkt(self) -> str:
        raise NotImplementedError

    def _coordinate_arrays(self) -> list[np.ndarray]:
        return []

    def intersects(self, other) -> bool:
        from .ops import intersects

        return intersects(self, other)

    def disjoint(self, other) -> bool:
        from .ops import disjoint

        return disjoint(self, other)

    def contains(self, other) -> bool:
        from .ops import contains

        return contains(self, other)

    def within(self, other) -> bool:
        from .ops import within

        return within(self, other)

    def covers(self, other) -> bool:
        from .ops import covers

        return covers(self, other)

    def covered_by(self, other) -> bool:
        from .ops import covered_by

        return covered_by(self, other)

    def touches(self, other) -> bool:
        from .ops import touches

        return touches(self, other)

    def overlaps(self, other) -> bool:
        from .ops import overlaps

        return overlaps(self, other)

    def crosses(self, other) -> bool:
        from .ops import crosses

        return crosses(self, other)

    def equals(self, other) -> bool:
        from .ops import equals

        return equals(self, other)

    def intersection(self, other):
        from .ops import intersection

        return intersection(self, other)

    def union(self, other):
        from .ops import union

        return union(self, other)

    def difference(self, other):
        from .ops import difference

        return difference(self, other)

    def symmetric_difference(self, other):
        from .ops import symmetric_difference

        return symmetric_difference(self, other)

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.difference(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def __bool__(self):
        return not self.is_empty

    def __repr__(self):
        return f"<{self.geom_type} {self.wkt}>"


class Point(BaseGeometry):
    geom_type = "Point"

    def __init__(self, x=None, y=None):
        if x is None:
            self._coords = np.empty((0, 2), dtype=np.float64)
        elif y is None:
            self._coords = _coords(x, minimum=1)[:1]
        else:
            self._coords = _coords([[x, y]], minimum=1)

    @property
    def x(self):
        if self.is_empty:
            raise ValueError("empty point has no x coordinate")
        return float(self._coords[0, 0])

    @property
    def y(self):
        if self.is_empty:
            raise ValueError("empty point has no y coordinate")
        return float(self._coords[0, 1])

    @property
    def coords(self):
        return CoordinateSequence(self._coords)

    @property
    def is_empty(self):
        return not len(self._coords)

    @property
    def wkt(self):
        return "POINT EMPTY" if self.is_empty else f"POINT ({_coord_text(self._coords)})"

    def _coordinate_arrays(self):
        return [self._coords]


class LineString(BaseGeometry):
    geom_type = "LineString"

    def __init__(self, coordinates=None):
        self._coords = _coords([] if coordinates is None else coordinates)
        if len(self._coords) == 1:
            raise ValueError("a line string requires 0 or at least 2 coordinates")

    @property
    def coords(self):
        return CoordinateSequence(self._coords)

    @property
    def is_empty(self):
        return not len(self._coords)

    @property
    def length(self):
        return _lib.line_length(self._coords)

    @property
    def wkt(self):
        return (
            "LINESTRING EMPTY"
            if self.is_empty
            else f"LINESTRING ({_coord_text(self._coords)})"
        )

    def _coordinate_arrays(self):
        return [self._coords]


class LinearRing(LineString):
    geom_type = "LinearRing"

    def __init__(self, coordinates=None):
        self._coords = _ring([] if coordinates is None else coordinates)

    @property
    def coords(self):
        return CoordinateSequence(self._coords, closed=True)

    @property
    def length(self):
        return _lib.line_length(self._coords, closed=True)

    @property
    def wkt(self):
        if self.is_empty:
            return "LINEARRING EMPTY"
        closed = np.vstack((self._coords, self._coords[0]))
        return f"LINEARRING ({_coord_text(closed)})"


class Polygon(BaseGeometry):
    geom_type = "Polygon"

    def __init__(self, shell=None, holes=None):
        if isinstance(shell, LinearRing):
            self._shell = shell._coords.copy()
        else:
            self._shell = _ring([] if shell is None else shell)
        self._holes = []
        for hole in holes or []:
            self._holes.append(_ring(hole))
        if not len(self._shell) and self._holes:
            raise ValueError("an empty polygon cannot have holes")
        rings = [self._shell, *self._holes]
        self._location_coords = np.ascontiguousarray(
            np.concatenate(rings, axis=0), dtype=np.float64
        )
        self._location_offsets = np.asarray(
            np.cumsum([0] + [len(ring) for ring in rings]), dtype=np.int64
        )

    @property
    def exterior(self):
        return LinearRing(self._shell)

    @property
    def interiors(self):
        return tuple(LinearRing(hole) for hole in self._holes)

    @property
    def is_empty(self):
        return not len(self._shell)

    @property
    def area(self):
        if self.is_empty:
            return 0.0
        return abs(_lib.ring_area(self._shell)) - sum(
            abs(_lib.ring_area(hole)) for hole in self._holes
        )

    @property
    def length(self):
        return _lib.line_length(self._shell, True) + sum(
            _lib.line_length(hole, True) for hole in self._holes
        )

    @property
    def is_valid(self):
        if self.is_empty:
            return True
        from .ops import _lib, _segments_from_rings

        rings = self._coordinate_arrays()
        ring_segments = []
        for ring in rings:
            if abs(_lib.ring_area(ring)) <= 1e-14:
                return False
            segments = _segments_from_rings([ring])
            ring_segments.append(segments)
            _, _, kinds = _lib.segment_intersections(segments, segments)
            count = len(segments)
            for i, j in zip(*np.where(kinds != 0)):
                if i == j or (i - j) % count in (1, count - 1):
                    continue
                return False
        for index, hole in enumerate(self._holes, start=1):
            if np.any(_lib.locate_points([self._shell], hole) != 1):
                return False
            if np.any(_lib.segment_intersections(ring_segments[0], ring_segments[index])[2]):
                return False
        for left in range(1, len(rings)):
            for right in range(left + 1, len(rings)):
                if np.any(
                    _lib.segment_intersections(
                        ring_segments[left], ring_segments[right]
                    )[2]
                ):
                    return False
                if (
                    _lib.locate_points([rings[left]], rings[right][:1])[0] != 0
                    or _lib.locate_points([rings[right]], rings[left][:1])[0] != 0
                ):
                    return False
        return True

    @property
    def wkt(self):
        if self.is_empty:
            return "POLYGON EMPTY"
        rings = [self._shell, *self._holes]
        text = []
        for ring in rings:
            closed = np.vstack((ring, ring[0]))
            text.append(f"({_coord_text(closed)})")
        return f"POLYGON ({', '.join(text)})"

    def _coordinate_arrays(self):
        return [self._shell, *self._holes]


class MultiPolygon(BaseGeometry, Sequence):
    geom_type = "MultiPolygon"

    def __init__(self, polygons=None):
        self.geoms = tuple(
            p if isinstance(p, Polygon) else Polygon(*p) if isinstance(p, tuple) else Polygon(p)
            for p in (polygons or [])
        )

    def __len__(self):
        return len(self.geoms)

    def __getitem__(self, item):
        return self.geoms[item]

    @property
    def is_empty(self):
        return not self.geoms

    @property
    def area(self):
        return sum(p.area for p in self.geoms)

    @property
    def length(self):
        return sum(p.length for p in self.geoms)

    @property
    def wkt(self):
        if self.is_empty:
            return "MULTIPOLYGON EMPTY"
        parts = []
        for polygon in self.geoms:
            rings = [polygon._shell, *polygon._holes]
            text = []
            for ring in rings:
                closed = np.vstack((ring, ring[0]))
                text.append(f"({_coord_text(closed)})")
            parts.append(f"({', '.join(text)})")
        return f"MULTIPOLYGON ({', '.join(parts)})"

    def _coordinate_arrays(self):
        return [a for polygon in self.geoms for a in polygon._coordinate_arrays()]


class GeometryCollection(BaseGeometry, Sequence):
    geom_type = "GeometryCollection"

    def __init__(self, geometries=None):
        self.geoms = tuple(geometries or [])

    def __len__(self):
        return len(self.geoms)

    def __getitem__(self, item):
        return self.geoms[item]

    @property
    def is_empty(self):
        return not self.geoms or all(g.is_empty for g in self.geoms)

    @property
    def area(self):
        return sum(g.area for g in self.geoms)

    @property
    def length(self):
        return sum(g.length for g in self.geoms)

    @property
    def wkt(self):
        if self.is_empty:
            return "GEOMETRYCOLLECTION EMPTY"
        return f"GEOMETRYCOLLECTION ({', '.join(g.wkt for g in self.geoms)})"

    def _coordinate_arrays(self):
        return [a for geometry in self.geoms for a in geometry._coordinate_arrays()]


def box(minx, miny, maxx, maxy, ccw=True):
    coordinates = [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    return Polygon(coordinates if ccw else coordinates[::-1])
