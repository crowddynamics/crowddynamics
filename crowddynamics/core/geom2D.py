import numba


@numba.jit(['f8(f8[:, :])'], nopython=True, nogil=True, cache=True)
def polygon_area(vertices):
    r"""Shoelace formula for computing area of polygon

    .. math::
        A = \sum_{i=1}^{n} x_i \left(y_{i+1} - y_{i-1}\right), \quad i\mod n

    References:
        - https://en.wikipedia.org/wiki/Shoelace_formula
        - https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon/717367#717367
        - https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

    Args:
        vertices (numpy.ndarray): Vertices of the polygon

    Returns:
        float: Area of the polygon
    """
    n = len(vertices)
    area = 0.0
    if n < 3:
        return area
    x, y = vertices[:, 0], vertices[:, 1]
    for i in range(1, n-1):
        area += x[i] * (y[i + 1] - y[i - 1])
    # i=n-1
    area += x[n-1] * (y[0] - y[n - 2])
    # i=n
    area += x[0] * (y[1] - y[n - 1])
    return 0.5 * abs(area)


@numba.jit(['boolean(f8[:], f8[:], f8[:], f8[:])'],
           nopython=True, nogil=True, cache=True)
def line_intersect(x0, x1, y0, y1):
    """Test if two lines intersect. Assumes that lines have finite length.

    Args:
        x0 (numpy.ndarray): Start point of first line
        x1 (numpy.ndarray): End point of first line
        y0 (numpy.ndarray): Start point of second line
        y1 (numpy.ndarray): End point of second line

    Returns:
        bool:
    """
    # FIXME: if u or v is zero vector
    u = x1 - x0
    v = y1 - y0
    b = y0 - x0
    d = u[0] * v[1] - u[1] * v[0]
    if d == 0:
        return False
    t0 = b[0] * v[1] - b[1] * v[0]
    t1 = b[0] * u[1] - b[1] * u[0]
    return 0 <= t0 / d <= 1 and 0 <= t1 / d <= 1
