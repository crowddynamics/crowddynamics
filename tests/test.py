import numpy as np
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import Polygon

from crowddynamics.core.sampling import PolygonSample


a = np.random.uniform(size=(10, 2))
poly = Polygon(a).convex_hull
sample = PolygonSample(poly)

fig, ax = plt.subplots(figsize=(8, 8))
patch = PolygonPatch(poly, alpha=0.4)
ax.add_patch(patch)

for _ in range(1000):
    p = PolygonPatch(sample.draw().buffer(0.01))
    ax.add_patch(p)

plt.show()
