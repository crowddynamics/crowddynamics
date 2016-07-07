from crowd_dynamics.parameters import Parameters, populate
from crowd_dynamics.simulation import Simulation
from crowd_dynamics.structure.agent import Agent
from crowd_dynamics.environment import Rectangle


def initialize(size, width, height, path="", name="outdoor", **kwargs):
    params = Parameters(width, height)

    bounds = Rectangle((0.0, width), (0.0, height))

    # Agents:
    agent = Agent(*params.agent(size))

    populate(agent, agent.size, bounds)
    agent.velocity[:] = agent.target_velocity * params.random_unit_vector(size)

    target_direction = params.random_unit_vector(size)

    agent.update_shoulder_positions()

    return Simulation(agent, name=name, dirpath=path, bounds=bounds,
                      direction_update=target_direction, **kwargs)
