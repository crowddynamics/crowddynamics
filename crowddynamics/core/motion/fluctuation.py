r"""
Fluctuation force and torque are stochastic in nature and analogous to heat in
particle systems. For modeling fluctuation we use :math:`\mathcal{U}(a, b)` for
continuous uniform distribution and :math:`\mathcal{N}(\mu, \sigma^{2})` for
truncated normal distribution.
"""
import numpy as np
from scipy.stats import truncnorm


def force_fluctuation(mass, scale):
    r"""
    Truncated normal distribution with standard deviation of 3.

    .. math::
       \boldsymbol{\xi} = \xi \cdot \mathbf{\hat{e}}(\varphi)

    where

    - :math:`\xi \sim \mathcal{N}(\mu, \sigma^{2})`
    - :math:`\varphi \sim \mathcal{U}(-\pi, \pi)`
    - :math:`\mathbf{\hat{e}}(\varphi)` is unit vector to direction of
      :math:`\varphi`.

    Args:
        mass (float):
            Mass

        scale (float):
            Standard deviation of truncated normal distribution

    Returns:
        numpy.ndarray: Fluctuation force
    """
    if isinstance(mass, np.ndarray):
        size = len(mass)
    elif isinstance(scale, np.ndarray):
        size = len(scale)
    else:
        size = 1

    magnitude = truncnorm.rvs(0.0, 3.0, loc=0.0, scale=scale, size=size)
    angle = np.random.uniform(0.0, 2.0 * np.pi, size=size)
    force = magnitude * np.array((np.cos(angle), np.sin(angle)))
    return force.T * mass


def torque_fluctuation(inertia_rot, scale):
    r"""
    Random torque.

    .. math::
       \eta \sim \mathcal{N}(\mu, \sigma^{2})

    Args:
        inertia_rot (float):
            Rotational intertial

        scale (float):
            Standard deviation of truncated normal distribution

    Returns:
        numpy.ndarray: Fluctuation torque
    """
    if isinstance(inertia_rot, np.ndarray):
        size = len(inertia_rot)
    elif isinstance(scale, np.ndarray):
        size = len(scale)
    else:
        size = 1

    torque = truncnorm.rvs(-3.0, 3.0, loc=0.0, scale=scale, size=size)
    return torque * inertia_rot
