import numba
import numpy as np

from crowddynamics.core.vector.vector import vector_type


# Default values
tau_adj = 0.5
tau_rot = 0.2
k_soc = 1.5
tau_0 = 3.0
mu = 1.2e5
kappa = 4e4
damping = 500
std_rand_force = 0.1
std_rand_torque = 0.1


# Limits
sight_soc = 3.0
sight_wall = 3.0
f_soc_ij_max = 2e3
f_soc_iw_max = 2e3


# Agent types
translational = [
    ('mass', np.float64),
    ('radius', np.float64),
    ('position', vector_type),
    ('velocity', vector_type),
    ('target_velocity', np.float64),
    ('target_direction', vector_type),
    ('force', vector_type),
    ('tau_adj', np.float64),
    ('k_soc', np.float64),
    ('tau_0', np.float64),
    ('mu', np.float64),
    ('kappa', np.float64),
    ('damping', np.float64),
    ('std_rand_force', np.float64),
    ('f_soc_ij_max', np.float64),
    ('f_soc_iw_max', np.float64),
    ('sight_soc', np.float64),
    ('sight_wall', np.float64),
]

rotational = [
    ('inertia_rot', np.float64),
    ('orientation', np.float64),
    ('angular_velocity', np.float64),
    ('target_orientation', np.float64),
    ('target_angular_velocity', np.float64),
    ('torque', np.float64),
    ('tau_rot', np.float64),
    ('std_rand_torque', np.float64),
]

three_circle = [
    ('r_t', np.float64),
    ('r_s', np.float64),
    ('r_ts', np.float64),
]

agent_type_circular = np.dtype(
    translational
)

agent_type_three_circle = np.dtype(
    translational +
    rotational +
    three_circle
)


# Linear obstacle defined by two points
obstacle_type_linear = np.dtype([
    ('p0', vector_type),
    ('p1', vector_type),
])
