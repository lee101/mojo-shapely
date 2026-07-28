"""A Mojo-accelerated subset of Shapely's planar geometry API."""

from .geometry import (
    BaseGeometry,
    GeometryCollection,
    LinearRing,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from .ops import (
    contains,
    contains_xy,
    covered_by,
    covers,
    crosses,
    difference,
    disjoint,
    equals,
    intersection,
    intersects,
    intersects_xy,
    overlaps,
    symmetric_difference,
    touches,
    union,
    within,
)

__version__ = "0.1.0"


def area(geometry, **kwargs):
    return geometry.area


def length(geometry, **kwargs):
    return geometry.length


def bounds(geometry, **kwargs):
    return geometry.bounds


def is_empty(geometry, **kwargs):
    return geometry.is_empty


def is_valid(geometry, **kwargs):
    return getattr(geometry, "is_valid", True)

__all__ = [
    "BaseGeometry",
    "GeometryCollection",
    "LinearRing",
    "LineString",
    "MultiPolygon",
    "Point",
    "Polygon",
    "area",
    "bounds",
    "box",
    "contains",
    "contains_xy",
    "covered_by",
    "covers",
    "crosses",
    "difference",
    "disjoint",
    "equals",
    "intersection",
    "intersects",
    "intersects_xy",
    "is_empty",
    "is_valid",
    "length",
    "overlaps",
    "symmetric_difference",
    "touches",
    "union",
    "within",
]
