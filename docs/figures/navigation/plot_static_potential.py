import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from crowddynamics.logging import setup_logging
from shapely.geometry import LineString, Polygon

from crowddynamics.core.steering.navigation import meshgrid, distance_map, \
    fill_missing, direction_map, merge_dir_maps
from crowddynamics.plot import plot_navigation

setup_logging()

step = 0.02
height = 3.0
width = 4.0
radius = 0.3
value = 0.3

domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
targets = LineString([(0, height / 2), (0, height)])
obstacles = LineString([(0, height / 2), (width * 3 / 4, height / 2)]) | \
            domain.exterior - targets

# Compute meshgrid for solving distance maps.
mgrid = meshgrid(step, *domain.bounds)

# Direction map (vector field) for guiding agents towards targets without
# them walking into obstacles.
obstacles_buffered = obstacles.buffer(radius).intersection(domain)
dmap_exits = distance_map(mgrid, targets, obstacles_buffered)
dir_map_exits = direction_map(dmap_exits)
fill_missing(mgrid, dir_map_exits)

# Direction map guiding agents away from the obstacles
dmap_obs = distance_map(mgrid, obstacles, None)
dir_map_obs = direction_map(dmap_obs)

# Direction map that combines the two direction maps
dir_map_merged = merge_dir_maps(dmap_obs, dir_map_obs, dir_map_exits, radius, value)


for dmap, dir_map, name in zip([dmap_exits, dmap_obs, dmap_exits],
                               [dir_map_exits, dir_map_obs, dir_map_merged],
                               ['exits', 'obstacles', 'merged']):
    # Figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # This locator puts ticks at regular intervals
    loc = plticker.MultipleLocator(base=0.5)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    plot_navigation(fig, ax, mgrid.values, dmap, dir_map, 2)
    # for fmt in ('png', 'pdf'):
    #     plt.savefig('navigation_' + name + '.' + fmt)
plt.show()
