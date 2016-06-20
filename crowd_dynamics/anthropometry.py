from collections import namedtuple


"""
Tables of anthropometric (human measure) data. These can be used to generate
agents.
"""

# TODO: Body mass and rotational moment values table
# column = namedtuple("column", ("adult", "male", "female", "child", "eldery"))
# radius, dr, torso, shoulder, shoulder distance, walking speed, dv
index = namedtuple(
    "index", ("r", "dr", "k_t", "k_s", "k_ts", "v", "dv", "mass", "mass_scale"))
body_types = dict(
    adult=index(0.255, 0.035, 0.5882, 0.3725, 0.6275, 1.25, 0.30, 75, 7),
    male=index(0.270, 0.020, 0.5926, 0.3704, 0.6296, 1.35, 0.20, 82, 10),
    female=index(0.240, 0.020, 0.5833, 0.3750, 0.6250, 1.15, 0.20, 67, 5),
    child=index(0.210, 0.015, 0.5714, 0.3333, 0.6667, 0.90, 0.30, 57, 5),
    eldery=index(0.250, 0.020, 0.6000, 0.3600, 0.6400, 0.80, 0.30, None, None),
)

# Moment of inertial scale for human
inertia_rot_scale = 4.0 / (80.0 * 0.255 ** 2)
