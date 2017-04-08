"""Obstacles

Attributes:
    obstacle_type_linear: Linear obstacle defined by two points.
"""
import numpy as np
from crowddynamics.core.geometry import geom_to_pairs


obstacle_type_linear = np.dtype([
    ('p0', np.float64, 2),
    ('p1', np.float64, 2),
])


def geom_to_linear_obstacles(geom):
    """Converts shape(s) to array of linear obstacles."""
    return np.array(geom_to_pairs(geom), dtype=obstacle_type_linear)
