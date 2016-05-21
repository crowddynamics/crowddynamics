import numpy as np

from source.struct.agent import agent_struct, initial_position
from source.struct.constant import Constant
from source.struct.wall import LinearWall

# np.random.seed(seed=1111)

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
        ((50, 0), (50, 24)),
        ((50, 26), (50, 50)),
    ),
    dtype=np.float64
)

linear_wall = LinearWall(linear_params)
round_wall = None

# Agents
amount = 200
goal_velocity = 5.0
goal_point = np.array((53, 25))

mass = np.random.normal(loc=70.0, scale=10.0, size=amount)
radius = np.random.normal(loc=0.22, scale=0.01, size=amount)
position = initial_position(amount, x_dims, y_dims, radius, linear_wall)
mass.sort()
radius.sort()

agent = agent_struct(mass, radius, position, goal_velocity)

agent.herding_flag = 0
agent.herding_tendency = 0.7 * np.ones(amount)
