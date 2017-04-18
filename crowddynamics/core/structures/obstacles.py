"""Obstacles

Attributes:
    obstacle_type_linear: Linear obstacle defined by two points.
"""
import numpy as np

obstacle_type_linear = np.dtype([
    ('p0', np.float64, 2),
    ('p1', np.float64, 2),
])
