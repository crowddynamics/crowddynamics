
"""
Function that returns unit vector for agent.goal_direction

Vector field
- Incompressible, irrotational and inviscosid fluid flow
- Poisson equation, Heat equation, Navier-Stokes
"""


def navigation(agent, goal_point):
    # TODO: Navigation
    agent.set_goal_direction(goal_point)

