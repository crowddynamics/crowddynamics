from source.core.system import system
from source.parameters import constant, linear_wall, agent

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
    simulation = system(constant, agent, linear_wall)
    i = 0
    for _ in simulation:
        i += 1
        if i > 1000:
            break

    """
    visualization.plot_animation()
    """
