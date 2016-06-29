from crowd_dynamics.simulation import Simulation
from crowd_dynamics.parameters import Parameters
from crowd_dynamics.structure.agent import Agent


def initialize(size=100, width=25, height=25, path="", **kwargs):
    name = "outdoor"

    x = (0.0, width)
    y = (0.0, height)

    params = Parameters(width, height)

    # Agents
    agent = Agent(*params.agent(size))
    params.random_position(agent.position, agent.radius, x, y)

    agent.velocity += agent.target_velocity * params.random_unit_vector(size)
    agent.target_direction += params.random_unit_vector(size)

    return Simulation(agent, name=name, dirpath=path, **kwargs)
