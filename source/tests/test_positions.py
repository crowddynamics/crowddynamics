import numpy as np
import matplotlib.pyplot as plt

from source.core.positions import set_positions
from source.field import linear_wall


lws = [[[0, 0], [0, 10]],
       [[0, 0], [10, 0]]]

params = dict(
    amount=50,
    x_dims=(0, 10),
    y_dims=(0, 10),
    radius=0.2,
    linear_walls=list(map(linear_wall, lws)),
    seed=None
)

x = params['x_dims']
y = params['y_dims']
rad = params['radius']
amount = params['amount']
square = (x[1] - x[0]) * (y[1] - y[0])

area = np.pi * rad ** 2
if isinstance(area, np.ndarray):
    area = np.sum(area)
else:
    area = amount * area

print('Area: ', square)
print('Agents: ', area)
print('Fill: ', area / square)

position = set_positions(**params)

plt.figure()
plt.scatter(position[:, 0], position[:, 1], s=area, alpha=0.5)
plt.show()