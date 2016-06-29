from crowd_dynamics.simulation import Simulation


def initialize(**kwargs):
    simulation = Simulation(**kwargs)
    return simulation
