import numpy as np


# TODO: Spring, Void


class Area(object):
    def __init__(self):
        """Abstract base class for area object."""
        pass

    def size(self):
        """:return: Total area."""
        return NotImplemented

    def random(self):
        """:return: Random point inside the area."""
        return NotImplemented


class Rectangle(Area):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def size(self):
        return np.diff(self.x) * np.diff(self.y)

    def random(self):
        pos = np.zeros(2)
        pos[0] = np.random.uniform(self.x[0], self.x[1])
        pos[1] = np.random.uniform(self.y[0], self.y[1])
        return pos


class Circle(Area):
    def __init__(self, phi, radius, center):
        super().__init__()
        self.phi = phi
        self.radius = radius
        self.center = center

    def size(self):
        phi = np.array(self.phi) % (2 * np.pi)
        return np.pi * np.diff(phi) / (2 * np.pi) * self.radius[1]**2

    def random(self):
        phi = np.random.uniform(self.phi[0], self.phi[1])
        radius = np.random.power(2) * self.radius[1]
        return radius * np.array([np.cos(phi), np.sin(phi)]) + self.center


class Bounds:
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=np.float64)
        self.radius = np.array(radius, dtype=np.float64)

    def update(self, agent):
        """Agents outside bounds are updated to inactive state."""
        vec = np.abs(self.center - agent.position) <= self.radius
        condition = vec[:, 0] & vec[:, 1]
        agent.active &= condition


class Goal(object):
    def __init__(self, center, radius):
        """Rectangle shaped."""
        self.center = np.array(center, dtype=np.float64)
        self.radius = np.array(radius, dtype=np.float64)

    def update(self, agent):
        """Updates agent that have reached goal.

        :param agent:
        :return: Number of agent that reached the goal.
        """
        vec = np.abs(self.center - agent.position) <= self.radius
        condition = vec[:, 0] & vec[:, 1]
        prev_num = np.sum(agent.goal_reached)
        agent.goal_reached |= condition
        num = np.sum(agent.goal_reached) - prev_num
        return num
