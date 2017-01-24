"""
Module to generate various input values for testing code using ``Hypothesis``
library.
"""
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from shapely.geometry import LineString

from crowddynamics.multiagent import Agent
from crowddynamics.core.vector2D import length, wrap_to_pi


def real(min_value=None, max_value=None, exclude_zero=None):
    """Real number strategy using 64-bit floating point numbers excluding
    ``nan`` and ``inf``.

    Args:
        min_value:
        max_value:
        exclude_zero (str, optional):
            Choices: (None, 'exact', 'near')

    """
    strategy = st.floats(min_value, max_value, False, False)
    # TODO: Maybe use assume instead?
    if exclude_zero == 'exact':
        return strategy.filter(lambda x: x != 0.0)
    if exclude_zero == 'near':
        return strategy.filter(lambda x: not np.isclose(x, 0.0))
    return strategy


def vector(dtype=np.float64, shape=2, elements=real()):
    """

    Args:
        dtype (type):
        shape (int|Tuple[Int]):
        elements (SearchStrategy):

    Returns:

    """
    return arrays(dtype, shape, elements)


@st.composite
def vectors(draw, elements=real(), maxsize=100, dim=2):
    size = draw(st.integers(1, maxsize))
    values = draw(arrays(np.float64, (size, dim), elements))
    return values


@st.composite
def unit_vector(draw, start=0, end=2 * np.pi):
    phi = draw(st.floats(start, end, False, False))
    return np.array((np.cos(phi), np.sin(phi)), dtype=np.float64)


@st.composite
def polygons(draw, min_value=-1.0, max_value=1.0, num_points=5,
             buffer=real(0.1, 0.2), convex_hull=False):
    """
    Generate a random polygon. Polygon should have area > 0.

    Args:
        draw:
        min_value (float):
        max_value (float):
        num_points (int):
        buffer (SearchStrategy):
        convex_hull (Boolean):

    Returns:
        Polygon: Random polygon

    """
    points = draw(arrays(np.float64, (num_points, 2),
                         real(min_value, max_value, exclude_zero='near')))
    l = draw(buffer)
    poly = LineString(points).buffer(l)
    if convex_hull:
        return poly.convex_hull
    else:
        return poly


@st.composite
def field(draw, domain_strategy=polygons(), min_target_length=0.1):
    """
    SearchStrategy that generates a domain, targets and obstacles. Domain
    is created as polygon, then targets are chosen from the exterior of the
    polygon and rest of the exterior is obstacles.

    Args:
        draw:
        domain_strategy:

    Returns:
        (Polygon, LineString, LineString):

    """
    domain = draw(domain_strategy)
    targets = None
    obstacles = None

    pts = np.asarray(domain.exterior)
    lens = [length(pts[j], pts[j+1]) for j in range(len(pts) - 1)]
    i = np.random.uniform(0, len(pts) - 1)

    l = 0
    target_pts = []
    obstacle_pts = []
    while l < min_target_length:
        target_pts += pts[i]
        i += 1

    return domain, targets, obstacles


@st.composite
def agent(draw, size):
    r"""
    Agent SearchStrategy

    Args:
        draw:
        size (int):

    Returns:
        SearchStrategy:

    """
    kwargs = dict(
        size=size,
        mass=draw(vector(shape=size, elements=real(1.0, 100.0))),
        radius=draw(vector(shape=size, elements=real(0.1, 1.0))),
        ratio_rt=draw(vector(shape=size, elements=real(0.1, 1.0))),
        ratio_rs=draw(vector(shape=size, elements=real(0.1, 1.0))),
        ratio_ts=draw(vector(shape=size, elements=real(0.1, 1.0))),
        inertia_rot=draw(vector(shape=size, elements=real())),
        target_velocity=draw(vector(shape=size, elements=real())),
        target_angular_velocity=draw(vector(shape=size, elements=real(-10, 10)))
    )
    agent = Agent(**kwargs)
    agent.position[:] = draw(vector(shape=(size, 2), elements=real(-100, 100)))
    agent.velocity[:] = draw(vector(shape=(size, 2), elements=real(-100, 100)))
    agent.force[:] = draw(vector(shape=(size, 2), elements=real(-100, 100)))

    agent.angle[:] = draw(vector(shape=size, elements=real(-np.pi, np.pi)))
    agent.angular_velocity[:] = draw(vector(shape=size, elements=real(-100, 100)))
    agent.torque[:] = draw(vector(shape=size, elements=real(-100, 100)))

    agent.update_shoulder_positions()
    return agent
