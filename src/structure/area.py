import numpy as np

from src.core.vector2d import length, angle, angle_nx2, length_nx2


class Area(object):
    def __init__(self):
        """Abstract base class for area object."""
        # TODO: contains, update
        # TODO: spring, void
        pass

    def contains(self, array):
        """If area contains item"""
        pass

    def update(self):
        """Update"""
        pass

    def size(self):
        """:return: Total area."""
        pass

    def random(self):
        """:return: Random point inside the area."""
        pass


class Rectangle(Area):
    def __init__(self, x, y):
        """Rectangle.
        :param x: 2D array like: (xmin, xmax)
        :param y: 2D array like: (ymin, ymax)
        """
        super().__init__()
        self.x = x
        self.y = y

    def contains(self, array):
        if len(array.shape) == 1:
            x = array[0]
            y = array[1]
        else:
            x = array[:, 0]
            y = array[:, 1]
        return (x >= self.x[0]) & (x <= self.x[1]) & \
               (y >= self.y[0]) & (y <= self.y[1])

    def size(self):
        return np.asscalar(np.diff(self.x) * np.diff(self.y))

    def random(self):
        pos = np.zeros(2)
        pos[0] = np.random.uniform(self.x[0], self.x[1])
        pos[1] = np.random.uniform(self.y[0], self.y[1])
        return pos


class Circle(Area):
    def __init__(self, phi, radius, center):
        """

        :param phi:
        :param radius:
        :param center:
        """
        super().__init__()
        self.phi = phi  # [0, 2 * pi]
        self.radius = radius
        self.center = center

    def contains(self, array):
        c = array - self.center
        if len(array.shape) == 1:
            phi = angle(c)
            return (length(c) <= self.radius) & \
                   (phi >= self.phi[0]) & \
                   (phi <= self.phi[1])
        else:
            phi = angle_nx2(c)
            return (length_nx2(c) <= self.radius) & \
                   (phi >= self.phi[0]) & \
                   (phi <= self.phi[1])

    def size(self):
        phi = np.array(self.phi) % (2 * np.pi)
        return np.asscalar(np.pi * np.diff(phi) / (2 * np.pi) * self.radius[1]**2)

    def random(self):
        phi = np.random.uniform(self.phi[0], self.phi[1])
        radius = np.random.power(2) * self.radius[1]
        return radius * np.array([np.cos(phi), np.sin(phi)]) + self.center
