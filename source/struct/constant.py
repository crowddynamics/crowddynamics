from collections import OrderedDict

from numba import float64, jitclass


spec_constant = OrderedDict(
    tau_adj=float64,
    k=float64,
    tau_0=float64,
    mu=float64,
    kappa=float64,
    a=float64,
    b=float64,
    f_max=float64,
)


@jitclass(spec_constant)
class Constant(object):
    """
    Structure for constants.
    """

    def __init__(self):
        # Force related constants
        self.tau_adj = 0.5
        self.k = 1.5 * 70
        self.tau_0 = 3.0
        self.mu = 1.2e5
        self.kappa = 2.4e5
        self.a = 2e3
        self.b = 0.08
        # Limits
        self.f_max = 1e3
