from collections import OrderedDict

import numba
from numba import float64


spec_constant = OrderedDict(
    dt=float64,
    dt_min=float64,
    dx_max=float64,
    tau_adj=float64,
    k=float64,
    tau_0=float64,
    mu=float64,
    kappa=float64,
    a=float64,
    b=float64,
    tau_adj_torque=float64,
    f_random_fluctuation_max=float64,
    f_adjust_max=float64,
    f_soc_ij_max=float64,
    f_c_ij_max=float64,
    f_soc_iw_max=float64,
    f_c_iw_max=float64,
)

constant_attr_names = [key for key in spec_constant.keys()]


@numba.jitclass(spec_constant)
class Constant(object):
    """
    Structure for constants.
    """

    def __init__(self):
        # Integrator
        # TODO: Lower than this raises warning
        # TODO: Relative to crowd density and target_velocity * dt
        self.dx_max = 0.05

        self.dt = 0.01
        self.dt_min = 0.001

        # Force related constants
        self.tau_adj = 0.5
        self.k = 1.5 * 70
        self.tau_0 = 3.0
        self.mu = 1.2e5
        self.kappa = 2.4e5
        self.a = 2e3
        self.b = 0.08

        # Rotational constants
        self.tau_adj_torque = 0.2

        # Limits
        self.f_random_fluctuation_max = 1.0
        self.f_adjust_max = 1e3
        self.f_soc_ij_max = 2e3
        self.f_c_ij_max = 2e5
        self.f_soc_iw_max = 2e3
        self.f_c_iw_max = 2e5
