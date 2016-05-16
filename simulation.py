import numpy as np

import seaborn

from source import visualization
from source.core.system import system
from source.field.field import set_field

import source.parameters as params


seaborn.set()
np.random.seed(params.simulation_params['seed'])


"""
1. Parameters

2. Field
    1.1 Walls
        1.1.1 Linear wall
        1.1.2 Round wall
    1.2 Agents

3. System
    3.1 Forces
    3.2 Game

4. Visualization

"""

if __name__ == '__main__':
    agents, walls = set_field(params.field_params,
                              params.wall_params,
                              params.agent_params)

    # constants = params.Constants
    simulation = system(agents,
                        walls,
                        params.constant,
                        # constants
                        **params.system_params)

    visualization.plot_animation(simulation,
                                 agents,
                                 params.wall_params,
                                 params.field_params)
