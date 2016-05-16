from numba import jitclass, float64


spec_constant = dict(
    tau_adj=float64,
    k=float64,
    tau_0=float64,
    sight=float64,
    f_max=float64,
    mu=float64,
    kappa=float64,
    a=float64,
    b=float64,
)


@jitclass(spec_constant)
class Constant(object):
    def __init__(self):
        self.sight = 7.0
        self.f_max = 1e3
        self.tau_adj = 0.5
        self.k = 1.5 * 70
        self.tau_0 = 3.0
        self.mu = 1.2e5
        self.kappa = 2.4e5
        self.a = 2e3
        self.b = 0.08


constant = {
    'tau_adj': 0.5,
    'k': 1.5 * 70,
    'tau_0': 3.0,
    'sight': 7.0,
    'f_max': 1e3,
    'mu': 1.2e5,
    'kappa': 2.4e5,
    'a': 2e3,
    'b': 0.08
}

bounds = {
    'acceleration_max': None,
}

system_params = {
    't_delta': 0.01,
}

simulation_params = {
    'seed': None,
}

field_params = {
    'amount': 100,
    'x_dims': (0, 10),
    'y_dims': (0, 10),
}

agent_params = {
    'mass': (60, 80),
    'radius': (0.2, 0.3),
    'goal_velocity': 2.5
}

wall_params = {
    'round_params':
        (),
    'linear_params': (
        ((0, 0), (0, 10)),
        ((0, 0), (10, 0)),
        ((0, 10), (10, 10))
    )
}
