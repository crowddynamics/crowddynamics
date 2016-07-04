import numpy as np


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