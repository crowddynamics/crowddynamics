constants = {
    'tau_adj': 0.5,
    'tau_0': 3.0,
    'sight': 7.0,
    'f_max': 5.0,
    'mu': 1.2e5,
    'kappa': 2.4e5,
    'a': 2e3,
    'b': 0.08
}

system_params = {
    't_delta': 0.01,
}

simulation_params = {
    'seed': None,
}

field_params = {
    'amount': 10,
    'x_dims': (0, 10),
    'y_dims': (0, 10),
}

agent_params = {
    'mass': 1,
    'radius': (0.2, 0.3),
    # 'position': None,
    # 'velocity': None,
    'goal_velocity': 1.5
}

wall_params = {
    'round_params':
        (),
    'linear_params': (
        ((0, 0), (0, 4)),
        ((0, 0), (4, 0)),
        ((0, 4), (4, 4))
    )
}
