from source import visualization
from source.core.system import system
from source.parameters import constant, linear_wall, agent, x_dims, y_dims

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
    # visualization.plot_field(agent, x_dims, y_dims, linear_wall)
    simulation = system(constant, agent, linear_wall)
    visualization.plot_animation(simulation, agent, linear_wall, x_dims, y_dims)
