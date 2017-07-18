r"""
Fluctuation
-----------
Fluctuation force and torque are stochastic in nature and analogous to heat in
particle systems. For modeling fluctuation we use :math:`\mathcal{U}(a, b)` for
continuous uniform distribution and :math:`\mathcal{N}(\mu, \sigma^{2})` for
truncated normal distribution.
"""
import numpy as np

from crowddynamics.core.rand import truncnorm


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
        mass (numpy.ndarray):
            Mass

        scale (numpy.ndarray):
            Standard deviation of truncated normal distribution

    Returns:
        numpy.ndarray: Fluctuation force vector
    """
    size = len(mass)
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=size)
    unit_vector = np.array((np.cos(phi), np.sin(phi)))
    magnitude = truncnorm(0.0, 3.0, loc=0.0, scale=scale, size=size)
    return (mass * magnitude * unit_vector).T


def torque_fluctuation(inertia_rot, scale):
    r"""Random torque

    .. math::
       \eta \sim \mathcal{N}(\mu, \sigma^{2})

    Args:
        inertia_rot (numpy.ndarray):
            Rotational intertial

        scale (numpy.ndarray):
            Standard deviation of truncated normal distribution

    Returns:
        numpy.ndarray: Fluctuation torque scalar
    """
    size = len(inertia_rot)
    return inertia_rot * truncnorm(-3.0, 3.0, loc=0.0, scale=scale, size=size)
