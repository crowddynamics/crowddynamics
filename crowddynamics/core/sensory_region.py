import numba
from numba.types import f8, boolean
from numba.typing.typeof import typeof

from crowddynamics.core.geom2D import line_intersect
from crowddynamics.core.structures import obstacle_type_linear


@numba.jit([boolean(f8[:], f8[:], typeof(obstacle_type_linear)[:])],
           nopython=True, nogil=True, cache=True)
def is_obstacle_between_points(p0, p1, obstacles):
    """Tests if there is obstacles between the two points p0 and p1."""
    for obstacle in obstacles:
        if line_intersect(p0, p1, obstacle['p0'], obstacle['p1']):
            return True
    return False
