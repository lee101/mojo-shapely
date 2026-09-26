"""Behavioural and numerical parity with Shapely 2.1 on the covered subset."""

import numpy as np
import pytest
import shapely
from shapely import geometry as sg

import mojoshapely as msh
from mojoshapely import _lib


def upstream(geometry):
    return shapely.from_wkt(geometry.wkt)


def assert_same_geometry(ours, theirs, tolerance=1e-9):
    converted = upstream(ours)
    assert converted.is_valid
    assert converted.symmetric_difference(theirs).area <= tolerance
    assert converted.length == pytest.approx(theirs.length, abs=tolerance)


@pytest.fixture
def square():
    return msh.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])


def test_point_constructor_and_wkt():
    ours = msh.Point(1.5, -2)
    theirs = sg.Point(1.5, -2)
    assert ours.x == theirs.x
    assert ours.y == theirs.y
    assert ours.bounds == theirs.bounds
    assert upstream(ours).equals(theirs)
    assert msh.Point().is_empty


def test_line_measurement_and_coordinates():
    coords = [(0, 0), (3, 4), (6, 4), (6, -2)]
    ours = msh.LineString(coords)
    theirs = sg.LineString(coords)
    assert ours.length == pytest.approx(theirs.length)
    assert ours.bounds == theirs.bounds
    assert list(ours.coords) == list(theirs.coords)


def test_polygon_measurements_with_hole():
    shell = [(0, 0), (6, 0), (6, 5), (0, 5)]
    hole = [(1, 1), (1, 3), (4, 3), (4, 1)]
    ours = msh.Polygon(shell, [hole])
    theirs = sg.Polygon(shell, [hole])
    assert ours.area == pytest.approx(theirs.area)
    assert ours.length == pytest.approx(theirs.length)
    assert ours.bounds == theirs.bounds
    assert len(ours.interiors) == len(theirs.interiors)
    assert upstream(ours).equals(theirs)


def test_closed_ring_is_not_duplicated():
    coords = [(0, 0), (2, 0), (1, 1), (0, 0)]
    ours = msh.Polygon(coords)
    assert len(ours.exterior.coords) == 4
    assert ours.area == pytest.approx(sg.Polygon(coords).area)


def test_top_level_measurements_box_and_validity():
    rectangle = msh.box(-2, 1, 3, 5)
    assert msh.area(rectangle) == 20
    assert msh.length(rectangle) == 18
    assert msh.bounds(rectangle) == (-2.0, 1.0, 3.0, 5.0)
    assert msh.is_valid(rectangle)
    assert not msh.is_empty(rectangle)
    bowtie = msh.Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])
    assert not msh.is_valid(bowtie)
    outside_hole = msh.Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        [[(3, 1), (5, 1), (5, 2), (3, 2)]],
    )
    overlapping_holes = msh.Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        [
            [(1, 1), (4, 1), (4, 4), (1, 4)],
            [(3, 3), (6, 3), (6, 6), (3, 6)],
        ],
    )
    assert not msh.is_valid(outside_hole)
    assert not msh.is_valid(overlapping_holes)


def test_documented_geometry_types_and_accessors():
    ring = msh.LinearRing([(0, 0), (2, 0), (1, 1)])
    polygon = msh.Polygon(ring)
    multi = msh.MultiPolygon([polygon, msh.box(3, 0, 4, 1)])
    collection = msh.GeometryCollection([msh.Point(9, 9), msh.LineString([(0, 0), (1, 0)])])
    for geometry, expected_type in [
        (ring, "LinearRing"),
        (polygon, "Polygon"),
        (multi, "MultiPolygon"),
        (collection, "GeometryCollection"),
    ]:
        assert geometry.geom_type == expected_type
        assert upstream(geometry).geom_type == expected_type
    assert list(polygon.exterior.coords) == list(ring.coords)
    assert len(polygon.interiors) == 0
    assert len(multi.geoms) == 2
    assert len(collection.geoms) == 2


def test_ffi_rejects_invalid_layout_metadata_and_values(square):
    with pytest.raises(TypeError, match="int64"):
        _lib.locate_points_buffer(
            square._location_coords,
            square._location_offsets.astype(np.float64),
            [[1, 1]],
        )
    with pytest.raises(ValueError, match="span"):
        _lib.locate_points_buffer(
            square._location_coords,
            np.array([0, 99], dtype=np.int64),
            [[1, 1]],
        )
    with pytest.raises(ValueError, match="finite"):
        _lib.segment_intersections([[0, 0, np.nan, 1]], [[0, 0, 1, 1]])


def test_contains_xy_scalar_and_boundary(square):
    theirs = sg.Polygon(square._shell)
    for x, y in [(2, 2), (0, 2), (5, 2)]:
        assert msh.contains_xy(square, x, y) == bool(shapely.contains_xy(theirs, x, y))
        assert msh.intersects_xy(square, x, y) == bool(shapely.intersects_xy(theirs, x, y))


def test_contains_xy_vectorized_with_hole():
    ours = msh.Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [[(3, 3), (7, 3), (7, 7), (3, 7)]],
    )
    theirs = upstream(ours)
    rng = np.random.default_rng(4)
    x = rng.uniform(-2, 12, 20_000)
    y = rng.uniform(-2, 12, 20_000)
    assert np.array_equal(msh.contains_xy(ours, x, y), shapely.contains_xy(theirs, x, y))
    assert np.array_equal(
        msh.intersects_xy(ours, x, y), shapely.intersects_xy(theirs, x, y)
    )


def test_simd_tails_for_measurement_location_and_segments():
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    coords = np.column_stack((3 * np.cos(angles), 2 * np.sin(angles)))
    ours_line = msh.LineString(coords)
    theirs_line = sg.LineString(coords)
    assert ours_line.length == pytest.approx(theirs_line.length)

    ours_polygon = msh.Polygon(coords)
    theirs_polygon = sg.Polygon(coords)
    assert ours_polygon.area == pytest.approx(theirs_polygon.area)
    points = np.column_stack(
        (np.linspace(-4, 4, 13), np.linspace(-2.5, 2.5, 13))
    )
    assert np.array_equal(
        msh.intersects_xy(ours_polygon, points),
        shapely.intersects_xy(theirs_polygon, points[:, 0], points[:, 1]),
    )

    horizontal = np.array([[0.0, y, 2.0, y] for y in range(3)])
    vertical = np.array([[x, -1.0, x, 3.0] for x in np.linspace(-1, 3, 7)])
    ts, us, kinds = _lib.segment_intersections(horizontal, vertical)
    expected_hits = (vertical[:, 0] >= 0) & (vertical[:, 0] <= 2)
    assert kinds.dtype == np.uint8
    assert np.all(kinds == expected_hits[None, :])
    assert np.all((ts[kinds == 1] >= 0) & (ts[kinds == 1] <= 1))
    assert np.all((us[kinds == 1] >= 0) & (us[kinds == 1] <= 1))


def test_large_inputs_match_expected_results():
    polygon = msh.box(0, 0, 1, 1)
    point_count = _lib.LOCATE_LARGE_INPUT // len(polygon._shell)
    x = np.linspace(-0.5, 1.5, point_count)
    y = np.full(point_count, 0.5)
    truncated = msh.contains_xy(polygon, x[:-1], y[:-1])
    full = msh.contains_xy(polygon, x, y)
    assert np.array_equal(truncated, (x[:-1] > 0) & (x[:-1] < 1))
    assert np.array_equal(full, (x > 0) & (x < 1))

    side = int(np.sqrt(_lib.SEGMENT_LARGE_INPUT))
    horizontal = np.column_stack(
        (
            np.zeros(side),
            np.arange(side),
            np.ones(side),
            np.arange(side),
        )
    )
    vertical = np.tile([0.5, -1.0, 0.5, side + 1.0], (side, 1))
    _, _, truncated_kinds = _lib.segment_intersections(horizontal[:-1], vertical)
    _, _, full_kinds = _lib.segment_intersections(horizontal, vertical)
    assert np.all(truncated_kinds == 1)
    assert np.all(full_kinds == 1)


@pytest.mark.parametrize(
    "point, expected",
    [
        ((2, 2), (True, True, False)),
        ((0, 2), (True, False, True)),
        ((5, 2), (False, False, False)),
    ],
)
def test_point_polygon_predicates(square, point, expected):
    p = msh.Point(point)
    assert (msh.intersects(square, p), msh.contains(square, p), msh.touches(square, p)) == expected
    a, b = upstream(square), upstream(p)
    assert msh.covers(square, p) == a.covers(b)
    assert msh.within(p, square) == b.within(a)
    assert msh.covered_by(p, square) == b.covered_by(a)
    assert msh.disjoint(square, p) == a.disjoint(b)


@pytest.mark.parametrize(
    "coords",
    [
        [(2, -1), (5, 2), (2, 5), (-1, 2)],
        [(4, 1), (6, 1), (6, 3), (4, 3)],
        [(1, 1), (3, 1), (3, 3), (1, 3)],
        [(5, 5), (7, 5), (7, 7), (5, 7)],
    ],
)
def test_polygon_predicates(square, coords):
    other = msh.Polygon(coords)
    a, b = upstream(square), upstream(other)
    for name in ("intersects", "disjoint", "contains", "within", "covers", "covered_by", "touches", "overlaps"):
        assert getattr(msh, name)(square, other) == getattr(shapely, name)(a, b), name


def test_equal_polygons_ignore_start_and_orientation(square):
    other = msh.Polygon([(4, 4), (4, 0), (0, 0), (0, 4)])
    assert msh.equals(square, other)
    assert msh.contains(square, other)
    assert msh.covers(square, other)


def test_line_predicates():
    a = msh.LineString([(0, 0), (4, 4)])
    b = msh.LineString([(0, 4), (4, 0)])
    endpoint = msh.Point(0, 0)
    middle = msh.Point(2, 2)
    assert msh.crosses(a, b) == sg.LineString(a._coords).crosses(sg.LineString(b._coords))
    assert msh.touches(a, endpoint)
    assert not msh.contains(a, endpoint)
    assert msh.contains(a, middle)


def test_concave_polygon_does_not_cover_crossing_edge():
    concave = msh.Polygon([(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4)])
    bridge = msh.LineString([(0.5, 3), (3.5, 3)])
    assert msh.covers(concave, bridge) == upstream(concave).covers(upstream(bridge))
    assert not msh.covers(concave, bridge)


@pytest.mark.parametrize(
    "a_coords,b_coords",
    [
        (
            [(0, 0), (5, 0), (5, 4), (0, 4)],
            [(3, -1), (7, -1), (7, 2), (3, 2)],
        ),
        (
            [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)],
            [(0.5, 0.5), (3, 0.5), (3, 3), (0.5, 3)],
        ),
        (
            [(0, 0), (6, 0), (6, 6), (0, 6)],
            [(2, 2), (4, 2), (4, 4), (2, 4)],
        ),
        (
            [(0, 0), (2, 0), (2, 2), (0, 2)],
            [(3, 0), (5, 0), (5, 2), (3, 2)],
        ),
    ],
)
def test_polygon_boolean_operations(a_coords, b_coords):
    a = msh.Polygon(a_coords)
    b = msh.Polygon(b_coords)
    ua, ub = sg.Polygon(a_coords), sg.Polygon(b_coords)
    for name in ("intersection", "union", "difference", "symmetric_difference"):
        ours = getattr(msh, name)(a, b)
        theirs = getattr(shapely, name)(ua, ub)
        assert_same_geometry(ours, theirs)


def test_touching_intersection_is_line():
    a = msh.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    b = msh.Polygon([(2, 0), (4, 0), (4, 2), (2, 2)])
    result = msh.intersection(a, b)
    expected = upstream(a).intersection(upstream(b))
    assert isinstance(result, msh.LineString)
    assert upstream(result).equals(expected)


def test_corner_intersection_is_point():
    a = msh.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    b = msh.Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    result = a & b
    assert isinstance(result, msh.Point)
    assert upstream(result).equals(upstream(a).intersection(upstream(b)))


@pytest.mark.parametrize("point", [(1, 1), (0, 1), (5, 5)])
def test_point_polygon_boolean_operations(square, point):
    ours_point = msh.Point(point)
    theirs_polygon = upstream(square)
    theirs_point = upstream(ours_point)
    for left, right, upstream_left, upstream_right in [
        (square, ours_point, theirs_polygon, theirs_point),
        (ours_point, square, theirs_point, theirs_polygon),
    ]:
        for name in ("intersection", "union", "difference", "symmetric_difference"):
            ours = getattr(msh, name)(left, right)
            theirs = getattr(shapely, name)(upstream_left, upstream_right)
            assert upstream(ours).equals(theirs), name


def test_difference_creates_hole():
    a = msh.Polygon([(0, 0), (8, 0), (8, 8), (0, 8)])
    b = msh.Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
    result = a - b
    assert isinstance(result, msh.Polygon)
    assert len(result.interiors) == 1
    assert_same_geometry(result, upstream(a).difference(upstream(b)))


def test_boolean_operation_preserves_existing_hole():
    a = msh.Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        [[(2, 2), (2, 6), (6, 6), (6, 2)]],
    )
    b = msh.Polygon([(4, -1), (9, -1), (9, 9), (4, 9)])
    assert_same_geometry(a & b, upstream(a).intersection(upstream(b)))
    assert_same_geometry(a - b, upstream(a).difference(upstream(b)))


def test_disjoint_union_is_multipolygon():
    a = msh.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    b = msh.Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    result = a | b
    assert isinstance(result, msh.MultiPolygon)
    assert len(result.geoms) == 2
    assert_same_geometry(result, upstream(a).union(upstream(b)))


def test_symmetric_difference_produces_valid_components():
    a = msh.Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    b = msh.Polygon([(2, 1), (4, 1), (4, 2), (2, 2)])
    result = a ^ b
    assert isinstance(result, msh.MultiPolygon)
    assert_same_geometry(result, upstream(a).symmetric_difference(upstream(b)))


def test_operator_aliases(square):
    other = msh.Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
    assert (square & other).equals(msh.intersection(square, other))
    assert (square | other).equals(msh.union(square, other))
    assert (square - other).equals(msh.difference(square, other))
    assert (square ^ other).equals(msh.symmetric_difference(square, other))
