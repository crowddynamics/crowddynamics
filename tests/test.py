import numpy as np
from descartes import PolygonPatch
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point

from src.core.sampling import PolygonSample

height = 1
width = 1
poly = Polygon([(0, 0),
                (0, height),
                (width, height),
                (width, 0),
                (width / 2, -1)])

sample = PolygonSample(poly)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set(ylim=(-2, 2), xlim=(-2, 2))
patch = PolygonPatch(poly, alpha=0.4)
ax.add_patch(patch)
for _ in range(1000):
    p = PolygonPatch(sample.draw().buffer(0.01))
    ax.add_patch(p)

plt.show()
