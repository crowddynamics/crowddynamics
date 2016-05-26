import os
import numpy as np

from source.struct.agent import agent_struct, random_position
from source.struct.constant import Constant
from source.struct.obstacle import LinearWall
from source.struct.area import GoalRectangle

# Path to this folder
path = os.path.abspath(__file__).split()[0]

# Constants
constant = Constant()

# Field
x_dims = (0, 50)
y_dims = (0, 50)

# Walls
linear_params = np.array(
    (
        ((0, 0), (0, 50)),
        ((0, 0), (50, 0)),
        ((0, 50), (50, 50)),
        ((50, 0), (50, 24.5)),
        ((50, 26.5), (50, 50)),
    ),
    dtype=np.float64
)

linear_wall = LinearWall(linear_params)
round_wall = None

# Agents
size = 200
mass = np.random.normal(loc=70.0, scale=10.0, size=size)
radius = np.random.normal(loc=0.22, scale=0.01, size=size)
goal_velocity = 5.0

agent = agent_struct(size, mass, radius, goal_velocity)

random_position(agent, x_dims, y_dims, linear_wall)

agent.herding_flag = False
agent.herding_tendency = np.ones(size)

# Goal
goal_point = np.array((53.0, 25.0))
goal = GoalRectangle(center=np.array((52.5, 25.0)),
                     radius=np.array((2.5, 5.0)))
