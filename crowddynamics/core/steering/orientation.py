r"""
Orientation
-----------
Target orientation :math:`\varphi_{0}` finding algorithms.
"""
import numba
from numba import void, typeof

from crowddynamics.simulation.agents import agent_type_three_circle
from crowddynamics.core.vector2D import angle


def orientation():
    return NotImplementedError


@numba.jit(void(typeof(agent_type_three_circle)[:]),
           nopython=True, nogil=True, cache=True)
def orient_towards_target_direction(agents):
    for agent in agents:
        agent['target_orientation'] = angle(agent['target_direction'])
