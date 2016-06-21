import numpy as np

from crowd_dynamics.parameters import Parameters
from crowd_dynamics.structure.constant import Constant
from crowd_dynamics.structure.agent import Agent
from crowd_dynamics.structure.wall import LinearWall
from crowd_dynamics.area import GoalRectangle


# Path and name for saving simulation data
name = None
path = None

# Dimensions of the simulation.
bounds = None
width = None
height = None

# Parameters
parameters = Parameters()
constant = Constant()

# Walls
linear_wall = None
round_wall = None
walls = (linear_wall, round_wall)

# Areas
goals = None

# Agents
size = None
agent = Agent()
