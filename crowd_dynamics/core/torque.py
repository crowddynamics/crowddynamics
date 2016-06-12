import numpy as np
import numba


@numba.jit(nopython=True, nogil=True)
def torque_random(agent):
    """Random torque."""
    for i in range(agent.size):
        agent.torque += np.random.uniform(-1, 1)


@numba.jit(nopython=True, nogil=True)
def torque_adjust(agent, constant):
    """Adjusting torque."""
    agent.torque += agent.inertia_rot / constant.tau_adj_torque * (
        (agent.angle - agent.target_angle) *
        agent.target_angular_velocity - agent.angular_velocity)


@numba.jit(nopython=True, nogil=True)
def torque(radius, force):
    """Torque for 2D vectors. Right corner from cross product."""
    return radius[0] * force[1] - radius[1] * force[0]
