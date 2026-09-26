"""Planar geometry kernels exposed through a small C ABI."""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IntPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime BytePtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IntPtr:
    return IntPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BytePtr:
    return BytePtr(unsafe_from_address=addr)


def orient(
    ax: Float64, ay: Float64, bx: Float64, by: Float64, cx: Float64, cy: Float64
) -> Float64:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def tolerance(
    ax: Float64, ay: Float64, bx: Float64, by: Float64, cx: Float64, cy: Float64
) -> Float64:
    return 1.0e-12 * (
        1.0 + abs(ax) + abs(ay) + abs(bx) + abs(by) + abs(cx) + abs(cy)
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


def locate_point(
    coords: Ptr, offsets: IntPtr, nrings: Int, px: Float64, py: Float64
) -> Int:
    var inside = False
    for r in range(nrings):
        var start = Int(offsets[unsafe_offset=r])
        var stop = Int(offsets[unsafe_offset=r + 1])
        if stop - start < 3:
            continue
        var j = stop - 1
        for i in range(start, stop):
            var ax = coords[unsafe_offset=2 * j]
            var ay = coords[unsafe_offset=2 * j + 1]
            var bx = coords[unsafe_offset=2 * i]
            var by = coords[unsafe_offset=2 * i + 1]
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
    var packed = points.unsafe_load[width=W](2 * base).join(
        points.unsafe_load[width=W](2 * base + W)
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
            px[lane] = points[unsafe_offset=2 * (base + lane)]
            py[lane] = points[unsafe_offset=2 * (base + lane) + 1]
    var inside = px.eq(px) & px.ne(px)
    var boundary = inside
    for r in range(nrings):
        var start = Int(offsets[unsafe_offset=r])
        var stop = Int(offsets[unsafe_offset=r + 1])
        if stop - start < 3:
            continue
        var j = stop - 1
        for i in range(start, stop):
            var ax = coords[unsafe_offset=2 * j]
            var ay = coords[unsafe_offset=2 * j + 1]
            var bx = coords[unsafe_offset=2 * i]
            var by = coords[unsafe_offset=2 * i + 1]
            var cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            var eps = 1.0e-12 * (
                1.0 + abs(ax) + abs(ay) + abs(bx) + abs(by) + abs(px) + abs(py)
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
    var batches = npoints // W
    for i in range(batches):
        dst.unsafe_store(
            i * W,
            locate_point_batch(coords, offsets, nrings, points, i * W),
        )

    for i in range(batches * W, npoints):
        dst[unsafe_offset=i] = Float64(
            locate_point(
                coords,
                offsets,
                nrings,
                points[unsafe_offset=2 * i],
                points[unsafe_offset=2 * i + 1],
            )
        )


@export("msh_ring_area")
def msh_ring_area(coords_addr: Int, n: Int) abi("C") -> Float64:
    var coords = p(coords_addr)
    if n < 3:
        return 0.0
    var acc = 0.0
    var i = 0
    while i + W <= n - 1:
        var current = coords.unsafe_load[width=W](2 * i).join(
            coords.unsafe_load[width=W](2 * i + W)
        )
        var following = coords.unsafe_load[width=W](2 * i + 2).join(
            coords.unsafe_load[width=W](2 * i + W + 2)
        )
        var x0, y0 = current.deinterleave()
        var x1, y1 = following.deinterleave()
        acc += (x0 * y1 - x1 * y0).reduce_add()
        i += W
    while i < n - 1:
        acc += coords[unsafe_offset=2 * i] * coords[unsafe_offset=2 * i + 3]
        acc -= coords[unsafe_offset=2 * i + 2] * coords[unsafe_offset=2 * i + 1]
        i += 1
    acc += coords[unsafe_offset=2 * n - 2] * coords[unsafe_offset=1]
    acc -= coords[unsafe_offset=0] * coords[unsafe_offset=2 * n - 1]
    return 0.5 * acc


@export("msh_line_length")
def msh_line_length(coords_addr: Int, n: Int, closed: Int) abi("C") -> Float64:
    var coords = p(coords_addr)
    if n < 2:
        return 0.0
    var acc = 0.0
    var i = 0
    while i + W <= n - 1:
        var current = coords.unsafe_load[width=W](2 * i).join(
            coords.unsafe_load[width=W](2 * i + W)
        )
        var following = coords.unsafe_load[width=W](2 * i + 2).join(
            coords.unsafe_load[width=W](2 * i + W + 2)
        )
        var x0, y0 = current.deinterleave()
        var x1, y1 = following.deinterleave()
        var dx = x1 - x0
        var dy = y1 - y0
        acc += sqrt(dx * dx + dy * dy).reduce_add()
        i += W
    while i < n - 1:
        var dx = coords[unsafe_offset=2 * i + 2] - coords[unsafe_offset=2 * i]
        var dy = (
            coords[unsafe_offset=2 * i + 3] - coords[unsafe_offset=2 * i + 1]
        )
        acc += sqrt(dx * dx + dy * dy)
        i += 1
    if closed != 0:
        var dx = coords[unsafe_offset=0] - coords[unsafe_offset=2 * n - 2]
        var dy = coords[unsafe_offset=1] - coords[unsafe_offset=2 * n - 1]
        acc += sqrt(dx * dx + dy * dy)
    return acc




def segment_intersect_row(
    a: Ptr, b: Ptr, ts: Ptr, us: Ptr, kinds: BytePtr, i: Int, nb: Int
):
    var ax = a[unsafe_offset=4 * i]
    var ay = a[unsafe_offset=4 * i + 1]
    var bx = a[unsafe_offset=4 * i + 2]
    var by = a[unsafe_offset=4 * i + 3]
    var rx = bx - ax
    var ry = by - ay
    var j = 0
    while j + W <= nb:
        var k = i * nb + j
        var cx = b.unsafe_load[width=W](j)
        var cy = b.unsafe_load[width=W](j + nb)
        var dx = b.unsafe_load[width=W](j + 2 * nb)
        var dy = b.unsafe_load[width=W](j + 3 * nb)
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
        var any_bbox_hit = False
        for lane in range(W):
            if not Bool(bbox_miss[lane]):
                any_bbox_hit = True
        if not any_bbox_hit:
            ts.unsafe_store(k, zeros)
            us.unsafe_store(k, zeros)
            kinds.unsafe_store(k, SIMD[DType.uint8, W](0))
            j += W
            continue
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
        ts.unsafe_store(k, hit.select(min(ones, max(zeros, tv)), zeros))
        us.unsafe_store(k, hit.select(min(ones, max(zeros, uv)), zeros))
        kinds.unsafe_store(
            k,
            hit.select(
                SIMD[DType.uint8, W](1),
                ((~bbox_miss) & (~nonparallel) & collinear).select(
                    SIMD[DType.uint8, W](2), SIMD[DType.uint8, W](0)
                ),
            ),
        )
        j += W
    while j < nb:
        var k = i * nb + j
        kinds[unsafe_offset=k] = 0
        ts[unsafe_offset=k] = 0.0
        us[unsafe_offset=k] = 0.0
        var cx = b[unsafe_offset=j]
        var cy = b[unsafe_offset=j + nb]
        var dx = b[unsafe_offset=j + 2 * nb]
        var dy = b[unsafe_offset=j + 3 * nb]
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
                kinds[unsafe_offset=k] = 2
            continue
        var tv = (qpx * sy - qpy * sx) / den
        var uv = (qpx * ry - qpy * rx) / den
        if (
            tv >= -eps
            and tv <= 1.0 + eps
            and uv >= -eps
            and uv <= 1.0 + eps
        ):
            ts[unsafe_offset=k] = min(1.0, max(0.0, tv))
            us[unsafe_offset=k] = min(1.0, max(0.0, uv))
            kinds[unsafe_offset=k] = 1


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
    var kinds = bp(kind_addr)

    for i in range(na):
        segment_intersect_row(a, b, ts, us, kinds, i, nb)



@export("msh_orient_batch")
def msh_orient_batch(
    a_addr: Int, b_addr: Int, c_addr: Int, dst_addr: Int, n: Int
) abi("C"):
    var a = p(a_addr)
    var b = p(b_addr)
    var c = p(c_addr)
    var dst = p(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = orient(
            a[unsafe_offset=2 * i],
            a[unsafe_offset=2 * i + 1],
            b[unsafe_offset=2 * i],
            b[unsafe_offset=2 * i + 1],
            c[unsafe_offset=2 * i],
            c[unsafe_offset=2 * i + 1],
        )
