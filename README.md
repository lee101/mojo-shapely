# mojo-shapely

`mojo-shapely` is a standalone Mojo implementation of a focused, useful subset
of [Shapely](https://shapely.readthedocs.io/): planar geometry predicates and
boolean operations on valid simple polygons. It does not link to GEOS at
runtime. Shapely is installed only as a development dependency so that the test
suite and benchmarks can compare both implementations on identical inputs.

This is an early subset, not a replacement for all of Shapely. The public Python
module is named `mojoshapely`, so both libraries can be installed together.

## Coverage

The following Shapely-compatible surface is implemented:

- Geometry types: `Point`, `LineString`, `LinearRing`, `Polygon`,
  `MultiPolygon`, and `GeometryCollection`
- Constructors and accessors: `box`, `coords`, `exterior`, `interiors`,
  `geoms`, `geom_type`, `wkt`, `bounds`, `area`, `length`, `is_empty`, and
  polygon `is_valid`
- Predicates: `intersects`, `disjoint`, `contains`, `within`, `covers`,
  `covered_by`, `touches`, `crosses`, `overlaps`, and `equals`
- Vectorized predicates: `contains_xy` and `intersects_xy`
- Polygon boolean operations: `intersection`, `union`, `difference`, and
  `symmetric_difference`, including `&`, `|`, `-`, and `^`
- Polygon results with holes and disconnected `MultiPolygon` results
- Point/polygon boolean operations and point or line results where two polygons
  meet without an area overlap

Boolean polygon inputs must be finite, valid, simple polygons with linear
boundaries. Polygon holes are supported. The clipping implementation uses a
floating-point tolerance and is intended for ordinary Cartesian data; it does
not provide GEOS's adaptive exact predicates for adversarial near-degenerate
coordinates.

Not covered are buffering, distance calculations, affine transforms, prepared
geometries, spatial indexes, coordinate reference systems, Z/M coordinates,
WKB parsing, arbitrary line boolean operations, and union/difference of
`MultiPolygon` operands. Shapely's NumPy-style broadcasting is currently
limited to `contains_xy` and `intersects_xy`.

## Install

Install the pinned Mojo nightly, Python, NumPy, Shapely, and test tools:

```bash
pixi install
pixi run build
```

The build creates `dist/libmojo-shapely.so`. Run the parity suite with:

```bash
pixi run test
```

## Usage

```python
import mojoshapely as shapely

park = shapely.Polygon([(0, 0), (6, 0), (6, 5), (0, 5)])
pond = shapely.box(2, 1, 4, 3)
dry_land = park - pond

assert dry_land.area == 26.0
assert len(dry_land.interiors) == 1
assert shapely.contains_xy(dry_land, 1, 1)
assert not shapely.contains_xy(dry_land, 3, 2)
print(dry_land.wkt)
```

Run the example from the repository with
`pixi run python -c '<the code above>'`, or put it in a script and use
`pixi run python script.py`.

## How it works

All compute kernels live in one Mojo compilation unit. Python validates shapes,
finiteness, dtypes, and contiguous layout before passing NumPy buffers across
`ctypes` as integer addresses. The Python call frame keeps every buffer alive
until Mojo returns. Coordinates are C-contiguous `float64` values in interleaved
row-major layout (`x0, y0, x1, y1, ...`); ring offsets and intersection kind
outputs are C-contiguous `int64` values. Mojo writes only into caller-allocated
output buffers.

Mojo performs orientation, point-on-segment, even/odd point-in-polygon,
measurement, and all-pairs segment-intersection kernels. The Python topology
layer splits polygon edges at intersections, classifies both sides of every
fragment with the Mojo point locator, and polygonizes the retained directed
boundary. Float64 measurement, point-location, and segment-pair loops use the
host SIMD width with scalar remainder loops. Point location and segment-pair
work use CPU parallelism only above measured size thresholds; smaller calls
stay serial to avoid dispatch overhead. Polygon coordinate buffers are cached
and passed across the FFI without a copy.

There is no GPU path. The available kernels are branch-heavy streaming
operations, and this project currently targets the CPU only.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. Times are the best of five same-process runs on identical data.
These are real results from this checkout:

| Case | mojo-shapely | Shapely/GEOS | Relative |
|---|---:|---:|---:|
| contains_xy, 2M points | 63.165 ms | 203.500 ms | 3.22x faster |
| LineString.length, 1k vertices | 0.016 ms | 0.009 ms | 0.55x slower |
| Polygon.intersection, 256 vertices | 6.656 ms | 0.267 ms | 0.04x slower |
| Polygon.union, 256 vertices | 7.096 ms | 0.114 ms | 0.02x slower |

The SIMD and thresholded parallel point kernel is about three times faster than
GEOS for the 2M-point case on this machine. The short length call is slower
because its fixed Python/FFI cost dominates. Boolean operations are about 25
and 62 times slower than GEOS in these two cases because they materialize an
all-pairs intersection matrix and assemble topology in Python.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

`tests/test_parity.py` checks numerical values, predicate truth tables, geometry
topology, holes, disconnected results, and lower-dimensional contacts directly
against Shapely 2.1/GEOS.
