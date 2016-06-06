from collections import namedtuple

import numpy as np
from scipy.stats import truncnorm

"""
Generate parameters for testing algorithms.
===========================================
"""


def _truncnorm(loc, scale, size):
    tn = truncnorm(-3, 3)
    vals = tn.rvs(size)
    vals *= scale / 3
    vals += loc
    return vals


"""
Field
"""
dim = namedtuple('dim', ['width', 'height'])
lim = namedtuple('lim', ['min', 'max'])
d = dim(100.0, 100.0)
x = lim(0.0, d.width)
y = lim(0.0, d.height)

"""
Agents
"""
size = 10
mass = _truncnorm(loc=70.0, scale=10.0, size=size)
radius = _truncnorm(loc=0.22, scale=0.01, size=size)
goal_velocity = 5.0

"""
Walls
"""
round_params = None
linear_params = None
