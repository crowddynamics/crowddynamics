import sys

import numpy as np

try:
    # Find "Source" module to perform import
    module_path = '/home/jaan/Dropbox/Projects/Crowd-Dynamics/'
    sys.path.append(module_path)
    from source.parameters import *
    from source.field import set_field
    from source import visualization
    from source.core.system import system
except:
    pass


if __name__ == '__main__':
    """
    1. Parameters
    2. Set Field
        2.1 Set Walls
        2.2 Set Agents
    3. System
        3.1 Forces
        3.2 Game
    4. Visualization
    """
    np.random.seed(simulation_params['seed'])
    agents, walls = set_field(field_params, wall_params, agent_params)
    simulation = system(agents, walls, constants, **system_params)
    # TODO: goal_direction
    next(simulation)
    # visualization.func_plot(*simulation)
