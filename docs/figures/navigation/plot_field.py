import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from crowddynamics.logging import setup_logging
from crowddynamics.plot import plot_field
from shapely.geometry import LineString, Polygon

setup_logging()

step = 0.01
height = 3.0
width = 4.0
radius = 0.3
value = 0.3

domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
targets = LineString([(0, height / 2), (0, height)])
obstacles = LineString([(0, height / 2), (width * 3 / 4, height / 2)]) | \
            domain.exterior - targets

fig, ax = plt.subplots(figsize=(12, 12))
# plt.xlim((0.0, 4.0))
# plt.ylim((0.0, 4.0))

# This locator puts ticks at regular intervals
loc = plticker.MultipleLocator(base=0.5)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)

plot_field(fig, ax, domain, obstacles, targets)
plt.show()
