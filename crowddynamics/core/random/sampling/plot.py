import os

import numpy as np
from bokeh.io import output_file, save
from bokeh.plotting import figure

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = "output"


def save_plot(name, polygon, points):
    """

    Args:
        name (str):
        polygon (Polygon):
        points (List[Point]):

    Returns:
        None:
    """
    # TODO: name unique
    ext = ".html"
    os.makedirs(os.path.join(BASE_DIR, OUTPUT_FOLDER), exist_ok=True)
    filename = os.path.join(OUTPUT_FOLDER, name) + ext
    title = name.replace("_", "").capitalize()

    output_file(filename, title)

    # Figure
    p = figure()

    # Polygon as a patch
    values = np.asarray(polygon.exterior)
    p.patch(values[:, 0], values[:, 1], alpha=0.5, line_width=0.1)

    # Points as circles
    for point in points:
        x, y = point.xy
        p.circle(x, y)

    # TODO: save html
    save(p, filename, title=title)