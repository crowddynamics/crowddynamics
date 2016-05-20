import numpy as np

from source.struct.agent import agent_struct, initial_position, initial_velocity
from source.struct.constant import Constant
from source.struct.wall import LinearWall

# np.random.seed(seed=1111)

constant = Constant()

# Field
x_dims = (-0.1, 100.1)
y_dims = (-0.1, 100.1)

# Walls
linear_params = np.array(
    (((0, 0), (0, 100)),
     ((0, 0), (100, 0)),
     ((100, 0), (100, 100)),
     ((0, 100), (100, 100))),
    dtype=np.float64
)

linear_wall = LinearWall(linear_params)
round_wall = None

# Agents
amount = 200
goal_velocity = 5.0

mass = np.random.uniform(60.0, 80.0, amount)
radius = np.random.uniform(0.2, 0.3, amount)
# mass = 70
# radius = 0.25


position = initial_position(amount, x_dims, y_dims, radius, linear_wall)
velocity = initial_velocity(amount)
goal_direction = np.copy(velocity)

herding_tendency = 0.7 * np.ones(amount)

agent = agent_struct(mass, radius, position, velocity, goal_velocity,
                     goal_direction, herding_tendency)
