import numpy as np
from src.struct.wall import LinearWall
from tests.config import Params

params = Params(100, 100)
lwp = params.linear_wall(10)
linear_wall = LinearWall(lwp)


size = 100
position = params.random_2D_coordinates(size)
velocity = params.random_unit_vector(size)

for w in range(linear_wall.size):
    x = position[0]
    v = np.zeros(2)
    print(linear_wall.relative_position(w, x, v))


for i in range(len(position)):
    for w in range(linear_wall.size):
        x = position[i]
        v = velocity[i]
        print(linear_wall.relative_position(w, x, v))
