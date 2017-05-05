import numba
import numpy as np
from loggingtools import log_with
from numba import f8

from crowddynamics.core.steering.quickest_path import distance_map, \
    direction_map


@log_with(arguments=False, timed=True)
@numba.jit((f8[:, :], numba.types.Tuple((f8[:, :], f8[:, :])),
            numba.types.Tuple((f8[:, :], f8[:, :])), f8, f8),
           nopython=True, nogil=True, cache=True)
def obstacle_handling(dmap, dir_map_obs, dir_map_targets, radius, strength):
    r"""Merges direction maps

    Function that merges two direction maps together. Let distance map from
    obstacles be :math:`\Phi(\mathbf{x})` and :math:`\lambda(\Phi(\mathbf{x}))`
    be any decreasing function :math:`\frac{\partial}{\partial\Phi}\lambda(\Phi(\mathbf{x})) < 0` of
    distance from obstacles such that

    .. math::
       \lambda(\Phi) &=
       \begin{cases}
       1 & \Phi = 0 \\
       0 & \Phi > M > 0
       \end{cases}

    Then merged direction map :math:`\hat{\mathbf{e}}_{merged}` is

    .. math::
       p &= \lambda(\Phi(\mathbf{x})) \\
       \hat{\mathbf{e}}_{merged} &= p \hat{\mathbf{e}}_{obs} + (1 - p) \hat{\mathbf{e}}_{exits}

    Args:
        dmap:
        dir_map_obs:
        dir_map_targets:
        radius (float):
            Radius
        strength (float):
            Value between (0, 1). Value denoting the strength of dir_map1 at
            distance of radius.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
    """
    # FIXME: artifacts near radius distance from obstacles
    u1, v1 = dir_map_obs
    u2, v2 = dir_map_targets
    u_out, v_out = np.copy(u2), np.copy(v2)

    n, m = dmap.shape
    for i in range(n):
        for j in range(m):
            # Distance from the obstacles
            x = -dmap[i, j]
            if 0 < x < radius:
                # Decreasing function
                p = strength ** (x / radius)
                u_out[i, j] = - p * u1[i, j] + (1 - p) * u2[i, j]
                v_out[i, j] = - p * v1[i, j] + (1 - p) * v2[i, j]

    # Normalize the output
    l = np.hypot(u_out, v_out)
    return u_out / l, v_out / l


def direction_map_obstacles(mgrid, obstacles):
    """Vector field towards obstacles"""
    dmap_obs = distance_map(mgrid, obstacles, None)
    dir_map_obs = direction_map(dmap_obs)
    return dir_map_obs, dmap_obs
