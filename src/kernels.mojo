"""Planar geometry kernels exposed through a small C ABI."""

from std.algorithm import parallelize
from std.math import sqrt
from std.sys.info import num_physical_cores, simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IntPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime LOCATE_PARALLEL_WORK = 1048576
comptime SEGMENT_PARALLEL_WORK = 262144


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IntPtr:
    return IntPtr(unsafe_from_address=addr)


def orient(
    ax: Float64, ay: Float64, bx: Float64, by: Float64, cx: Float64, cy: Float64
) -> Float64:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def tolerance(
    ax: Float64, ay: Float64, bx: Float64, by: Float64, cx: Float64, cy: Float64
) -> Float64:
    return 1.0e-12 * (
        1.0
        + abs(ax)
        + abs(ay)
        + abs(bx)
        + abs(by)
        + abs(cx)
        + abs(cy)
    )


def point_on_segment(
    px: Float64, py: Float64, ax: Float64, ay: Float64, bx: Float64, by: Float64
) -> Bool:
    if abs(orient(ax, ay, bx, by, px, py)) > tolerance(ax, ay, bx, by, px, py):
        return False
    var eps = tolerance(ax, ay, bx, by, px, py)
    return (
        px >= min(ax, bx) - eps
        and px <= max(ax, bx) + eps
        and py >= min(ay, by) - eps
        and py <= max(ay, by) + eps
    )


def locate_point(coords: Ptr, offsets: IntPtr, nrings: Int, px: Float64, py: Float64) -> Int:
    var inside = False
    for r in range(nrings):
        var start = Int(offsets[r])
        var stop = Int(offsets[r + 1])
        if stop - start < 3:
            continue
        var j = stop - 1
        for i in range(start, stop):
            var ax = coords[2 * j]
            var ay = coords[2 * j + 1]
            var bx = coords[2 * i]
            var by = coords[2 * i + 1]
            if point_on_segment(px, py, ax, ay, bx, by):
                return -1
            if (ay > py) != (by > py):
                var xhit = ax + (py - ay) * (bx - ax) / (by - ay)
                if xhit > px:
                    inside = not inside
            j = i
    return 1 if inside else 0


def locate_point_batch(
    coords: Ptr, offsets: IntPtr, nrings: Int, points: Ptr, base: Int
) -> SIMD[DType.float64, W]:
    var packed = points.load[width=W](2 * base).join(
        points.load[width=W](2 * base + W)
    )
    var px = SIMD[DType.float64, W]()
    var py = SIMD[DType.float64, W]()
    comptime if W == 2:
        px = packed.shuffle[0, 2, 0, 0]().slice[W]()
        py = packed.shuffle[1, 3, 0, 0]().slice[W]()
    elif W == 4:
        px = packed.shuffle[0, 2, 4, 6, 0, 0, 0, 0]().slice[W]()
        py = packed.shuffle[1, 3, 5, 7, 0, 0, 0, 0]().slice[W]()
    elif W == 8:
        px = packed.shuffle[
            0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0
        ]().slice[W]()
        py = packed.shuffle[
            1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0
        ]().slice[W]()
    else:
        for lane in range(W):
            px[lane] = points[2 * (base + lane)]
            py[lane] = points[2 * (base + lane) + 1]
    var inside = px.eq(px) & px.ne(px)
    var boundary = inside
    for r in range(nrings):
        var start = Int(offsets[r])
        var stop = Int(offsets[r + 1])
        if stop - start < 3:
            continue
        var j = stop - 1
        for i in range(start, stop):
            var ax = coords[2 * j]
            var ay = coords[2 * j + 1]
            var bx = coords[2 * i]
            var by = coords[2 * i + 1]
            var cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            var eps = 1.0e-12 * (
                1.0
                + abs(ax)
                + abs(ay)
                + abs(bx)
                + abs(by)
                + abs(px)
                + abs(py)
            )
            var on_segment = (
                abs(cross).le(eps)
                & px.ge(min(ax, bx) - eps)
                & px.le(max(ax, bx) + eps)
                & py.ge(min(ay, by) - eps)
                & py.le(max(ay, by) + eps)
            )
            boundary = boundary | on_segment
            var crosses = py.lt(ay) ^ py.lt(by)
            var xhit = ax + (py - ay) * (bx - ax) / (by - ay)
            inside = inside ^ (crosses & xhit.gt(px))
            j = i
    var zeros = SIMD[DType.float64, W](0.0)
    var ones = SIMD[DType.float64, W](1.0)
    var negative_ones = SIMD[DType.float64, W](-1.0)
    return boundary.select(negative_ones, inside.select(ones, zeros))


@export("msh_locate_points")
def msh_locate_points(
    coords_addr: Int,
    offsets_addr: Int,
    nrings: Int,
    points_addr: Int,
    dst_addr: Int,
    npoints: Int,
) abi("C"):
    var coords = p(coords_addr)
    var offsets = ip(offsets_addr)
    var points = p(points_addr)
    var dst = p(dst_addr)
    var nvertices = Int(offsets[nrings])
    var batches = npoints // W

    @parameter
    @__copy_capture(coords, offsets, nrings, points, dst)
    @always_inline
    def locate_batch(i: Int):
        dst.store(
            i * W,
            locate_point_batch(coords, offsets, nrings, points, i * W),
        )

    if npoints * nvertices >= LOCATE_PARALLEL_WORK:
        parallelize[locate_batch](
            batches, min(batches, num_physical_cores())
        )
    else:
        for i in range(batches):
            locate_batch(i)
    for i in range(batches * W, npoints):
        dst[i] = Float64(
            locate_point(coords, offsets, nrings, points[2 * i], points[2 * i + 1])
        )


@export("msh_ring_area")
def msh_ring_area(coords_addr: Int, n: Int) abi("C") -> Float64:
    var coords = p(coords_addr)
    if n < 3:
        return 0.0
    var acc = 0.0
    var i = 0
    while i + W <= n - 1:
        var current = coords.load[width=W](2 * i).join(
            coords.load[width=W](2 * i + W)
        )
        var following = coords.load[width=W](2 * i + 2).join(
            coords.load[width=W](2 * i + W + 2)
        )
        var x0, y0 = current.deinterleave()
        var x1, y1 = following.deinterleave()
        acc += (x0 * y1 - x1 * y0).reduce_add()
        i += W
    while i < n - 1:
        acc += coords[2 * i] * coords[2 * i + 3]
        acc -= coords[2 * i + 2] * coords[2 * i + 1]
        i += 1
    acc += coords[2 * n - 2] * coords[1]
    acc -= coords[0] * coords[2 * n - 1]
    return 0.5 * acc


@export("msh_line_length")
def msh_line_length(coords_addr: Int, n: Int, closed: Int) abi("C") -> Float64:
    var coords = p(coords_addr)
    if n < 2:
        return 0.0
    var acc = 0.0
    var i = 0
    while i + W <= n - 1:
        var current = coords.load[width=W](2 * i).join(
            coords.load[width=W](2 * i + W)
        )
        var following = coords.load[width=W](2 * i + 2).join(
            coords.load[width=W](2 * i + W + 2)
        )
        var x0, y0 = current.deinterleave()
        var x1, y1 = following.deinterleave()
        var dx = x1 - x0
        var dy = y1 - y0
        acc += sqrt(dx * dx + dy * dy).reduce_add()
        i += W
    while i < n - 1:
        var dx = coords[2 * i + 2] - coords[2 * i]
        var dy = coords[2 * i + 3] - coords[2 * i + 1]
        acc += sqrt(dx * dx + dy * dy)
        i += 1
    if closed != 0:
        var dx = coords[0] - coords[2 * n - 2]
        var dy = coords[1] - coords[2 * n - 1]
        acc += sqrt(dx * dx + dy * dy)
    return acc


@export("msh_segment_intersections")
def msh_segment_intersections(
    a_addr: Int,
    b_addr: Int,
    t_addr: Int,
    u_addr: Int,
    kind_addr: Int,
    na: Int,
    nb: Int,
) abi("C"):
    var a = p(a_addr)
    var b = p(b_addr)
    var ts = p(t_addr)
    var us = p(u_addr)
    var kinds = ip(kind_addr)

    @parameter
    @__copy_capture(a, b, ts, us, kinds, nb)
    @always_inline
    def intersect_row(i: Int):
        var ax = a[4 * i]
        var ay = a[4 * i + 1]
        var bx = a[4 * i + 2]
        var by = a[4 * i + 3]
        var rx = bx - ax
        var ry = by - ay
        var j = 0
        while j + W <= nb:
            var k = i * nb + j
            var cx = b.load[width=W](j)
            var cy = b.load[width=W](j + nb)
            var dx = b.load[width=W](j + 2 * nb)
            var dy = b.load[width=W](j + 3 * nb)
            var sx = dx - cx
            var sy = dy - cy
            var den = rx * sy - ry * sx
            var qpx = cx - ax
            var qpy = cy - ay
            var scale = 1.0 + abs(rx) + abs(ry) + abs(sx) + abs(sy)
            var eps = 1.0e-12 * scale
            var zeros = SIMD[DType.float64, W](0.0)
            var ones = SIMD[DType.float64, W](1.0)
            var bbox_miss = (
                SIMD[DType.float64, W](max(ax, bx)).lt(min(cx, dx))
                | max(cx, dx).lt(SIMD[DType.float64, W](min(ax, bx)))
                | SIMD[DType.float64, W](max(ay, by)).lt(min(cy, dy))
                | max(cy, dy).lt(SIMD[DType.float64, W](min(ay, by)))
            )
            var nonparallel = abs(den).gt(eps)
            var collinear = abs(qpx * ry - qpy * rx).le(eps)
            var tv = (qpx * sy - qpy * sx) / den
            var uv = (qpx * ry - qpy * rx) / den
            var hit = (
                ~bbox_miss
                & nonparallel
                & tv.ge(-eps)
                & tv.le(1.0 + eps)
                & uv.ge(-eps)
                & uv.le(1.0 + eps)
            )
            ts.store(k, hit.select(min(ones, max(zeros, tv)), zeros))
            us.store(k, hit.select(min(ones, max(zeros, uv)), zeros))
            kinds.store(
                k,
                hit.select(
                    SIMD[DType.int64, W](1),
                    ((~bbox_miss) & (~nonparallel) & collinear).select(
                        SIMD[DType.int64, W](2), SIMD[DType.int64, W](0)
                    ),
                )
            )
            j += W
        while j < nb:
            var k = i * nb + j
            kinds[k] = 0
            ts[k] = 0.0
            us[k] = 0.0
            var cx = b[j]
            var cy = b[j + nb]
            var dx = b[j + 2 * nb]
            var dy = b[j + 3 * nb]
            j += 1
            if (
                max(ax, bx) < min(cx, dx)
                or max(cx, dx) < min(ax, bx)
                or max(ay, by) < min(cy, dy)
                or max(cy, dy) < min(ay, by)
            ):
                continue
            var sx = dx - cx
            var sy = dy - cy
            var den = rx * sy - ry * sx
            var qpx = cx - ax
            var qpy = cy - ay
            var scale = 1.0 + abs(rx) + abs(ry) + abs(sx) + abs(sy)
            var eps = 1.0e-12 * scale
            if abs(den) <= eps:
                if abs(qpx * ry - qpy * rx) <= eps:
                    kinds[k] = 2
                continue
            var tv = (qpx * sy - qpy * sx) / den
            var uv = (qpx * ry - qpy * rx) / den
            if tv >= -eps and tv <= 1.0 + eps and uv >= -eps and uv <= 1.0 + eps:
                ts[k] = min(1.0, max(0.0, tv))
                us[k] = min(1.0, max(0.0, uv))
                kinds[k] = 1

    if na * nb >= SEGMENT_PARALLEL_WORK:
        parallelize[intersect_row](na, min(na, num_physical_cores()))
    else:
        for i in range(na):
            intersect_row(i)


@export("msh_orient_batch")
def msh_orient_batch(
    a_addr: Int, b_addr: Int, c_addr: Int, dst_addr: Int, n: Int
) abi("C"):
    var a = p(a_addr)
    var b = p(b_addr)
    var c = p(c_addr)
    var dst = p(dst_addr)
    for i in range(n):
        dst[i] = orient(
            a[2 * i],
            a[2 * i + 1],
            b[2 * i],
            b[2 * i + 1],
            c[2 * i],
            c[2 * i + 1],
        )
