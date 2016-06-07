from src.struct.wall import LinearWall
from tests.config import Params

params = Params(100, 100)
linear_wall = LinearWall(params.linear_wall(10))

size = 10
position = params.random_2D_coordinates(size)
velocity = params.random_unit_vector(size)


for i in range(len(position)):
    for w in range(linear_wall.size):
        x = position[i]
        v = velocity[i]
        # print(v)
        print(linear_wall.relative_position(w, x, v))
