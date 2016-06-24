import os
import numpy as np
import pandas

"""
Tables of anthropometric (human measure) data. These can be used to generate
agents.

            adult    male    female    child    eldery
----------  -------  ------  --------  -------  --------
radius      0.255    0.27    0.24      0.21     0.25
dr          0.035    0.02    0.02      0.015    0.02
k_t         0.5882   0.5926  0.5833    0.5714   0.6
k_s         0.3725   0.3704  0.375     0.3333   0.36
k_ts        0.6275   0.6296  0.625     0.6667   0.64
v           1.25     1.35    1.15      0.9      0.8
dv          0.3      0.2     0.2       0.3      0.3
mass        73.5     80      67        57       70
mass_scale  8        8       6.7       5.7      7

radius_torso = k_t * radius
radius_shoulder = k_s * radius
distance_torso_shoulder = k_ts * radius

"""

# Loads csv with values shown above table
path = os.path.abspath(__file__)
path, _ = os.path.split(path)
path = os.path.join(path, "body_types.csv")
body_types = pandas.read_csv(path, index_col=[0])

# Moment of inertial scale for human. TODO: from csv
inertia_rot_value = 4.0
walking_speed_max = 5.0
angular_velocity_max = 4.0 * np.pi
inertia_rot_scale = inertia_rot_value / (80.0 * 0.255 ** 2)
