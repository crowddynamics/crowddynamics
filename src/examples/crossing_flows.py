from src.simulation import MultiAgentSimulation


def initialize(**kwargs):
    simulation = MultiAgentSimulation(**kwargs)
    return simulation
