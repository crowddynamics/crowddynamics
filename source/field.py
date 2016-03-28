# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


walls = {
    'circles': None,  # p, radius
    'lines': None     # p0, p1
}

agents = {
    'amount': None,      # Number of agents in the field
    'positions': None,   # x and y coordinates of the positions
    'velocities': None,  # x and y components of the velocities
    'goal': None,        # x and y components of the goal direction
    'masses': None,      # Masses of the agents
    'radii': None,       # Radii of the agents
}

field = {
    'dimensions': (4, 4),  # x and y dimensions of the field
    'walls': walls,        # Walls
    'agents': agents,      # Agents
}


def populate_agents():
    """
    Populate the positions of the agents in to the field so that they don't
    overlap each others or the walls.
    """
    pass


def init_field():
    """
    Initialise the field.
    """
    pass
