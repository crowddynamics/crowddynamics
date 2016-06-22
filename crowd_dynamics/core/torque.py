import numpy as np
import numba

from .vector2d import wrap_to_pi


@numba.jit(nopython=True, nogil=True)
def torque_random(agent):
    """Random torque."""
    for i in range(agent.size):
        agent.torque[i] += np.random.uniform(-1, 1)


@numba.jit(nopython=True, nogil=True)
def torque_adjust(agent):
    """Adjusting torque."""
    # TODO: Needs fixing
    agent.torque += agent.inertia_rot / agent.tau_adj_rot * (
        wrap_to_pi(agent.target_angle - agent.angle) / np.pi *
        agent.target_angular_velocity - agent.angular_velocity)


@numba.jit(nopython=True, nogil=True)
def torque(radius, force):
    """Torque for 2D vectors. Right corner from cross product."""
    return radius[0] * force[1] - radius[1] * force[0]
