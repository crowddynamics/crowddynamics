import os
import numpy as np

from source.struct.agent import agent_struct, random_position
from source.struct.constant import Constant
from source.struct.wall import LinearWall
from source.struct.area import GoalRectangle

# Path to this folder
path = os.path.abspath(__file__).split()[0]

# Constants
constant = Constant()

# Field
length = 50
width = 10
x_dims = (0, 50)
y_dims = (0, 50)

# Walls
linear_params = np.array(
    (
        ((0, 0), (length, 0)),
        ((0, width), (length, width)),
    ),
    dtype=np.float64
)

linear_wall = LinearWall(linear_params)
round_wall = None
