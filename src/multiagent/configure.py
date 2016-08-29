import logging
from numbers import Number

import numpy as np
from collections import Iterable
from scipy.spatial import Delaunay
from scipy.stats import truncnorm
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union

from src.config import Load
from src.core.vector2d import angle, reflect, rotate90, rotate270
from src.multiagent.agent import Agent

pi = np.pi


class PolygonSample:
    """Draw random uniform point from inside of polygon.
     - Breaks the polygon into triangular mesh (Delaunay triangulation)
     - Draw random uniform triangle weighted by its area
     - Draw random sample from inside the triangle

    .. [1]: http://mathworld.wolfram.com/TrianglePointPicking.html \n
    .. [2]: http://gis.stackexchange.com/questions/6412/generate-points-that-lie-inside-polygon
    .. [3]: http://docs.scipy.org/doc/scipy/reference/spatial.html
    .. [4]: https://en.wikipedia.org/wiki/Reflection_(mathematics)#Reflection_across_a_line_in_the_plane
    """

    def __init__(self, polygon):
        self.polygon = polygon

        # Triangle mesh
        self.tria_points = None
        self.triangles = []
        self.areas = None  # Areas
        self.area_sum = None
        self.area_cumsum = None  # Cumulative sum of areas of the triangles
        self.triangulation()

    def sample_trianle(self, p, tria):
        # Sample inside parallellogram
        x = np.random.uniform(size=2)  # Random variables
        sample = x[0] * (p[1] - p[0]) + x[1] * (p[2] - p[0]) + p[0]

        # Reflect points if they are not inside desired triangle
        if not tria.contains(Point(sample)):
            # FIXME
            l = p[2] - p[1]
            l2 = rotate90(l) + l / 2
            sample = reflect(sample, l)
            sample = reflect(sample, l2)

        return Point(sample)

    def triangulation(self):
        points = np.asarray(self.polygon.exterior)
        delaunay = Delaunay(points)  # Delaunay triangulation
        self.tria_points = points[delaunay.simplices]

        areas = []
        triangles = []
        for array in self.tria_points:
            tria = Polygon(array)
            triangles.append(tria)
            areas.append(tria.area)

        self.triangles = triangles
        self.areas = np.array(areas)
        self.area_sum = self.areas.sum()
        self.area_cumsum = self.areas.cumsum()

    def draw(self):
        # Draw random triangle weighted by the area of the triangle
        x = np.random.uniform(high=self.area_sum)
        i = np.searchsorted(self.area_cumsum, x)
        point = self.sample_trianle(self.tria_points[i], self.triangles[i])
        return point


class ConfigField:
    def __init__(self):
        self.domain = None  # Shapely.Polygon
        self.goals = []
        self.obstacles = []  # shapely.LineString
        self.exits = []  # shapely.LineString

    def set_domain(self, polygon):
        if polygon.is_valid and polygon.is_simple:
            self.domain = polygon

    def _set_goal(self, polygon):
        if polygon.is_valid and polygon.is_simple:
            self.goals.append(polygon)

    def _set_obstacle(self, linestring):
        if linestring.is_valid and linestring.is_simple:
            self.obstacles.append(linestring)

    def _set_exit(self, linestring):
        if linestring.is_valid and linestring.is_simple:
            self.exits.append(linestring)

    def set_goals(self, polygon):
        if isinstance(polygon, Iterable):
            for poly in polygon:
                self._set_goal(poly)
        else:
            self._set_goal(polygon)

    def set_obstacles(self, linestring):
        if isinstance(linestring, Iterable):
            for ls in linestring:
                self._set_obstacle(ls)
        else:
            self._set_obstacle(linestring)

    def set_exits(self, linestring):
        if isinstance(linestring, Iterable):
            for ls in linestring:
                self._set_exit(ls)
        else:
            self._set_exit(linestring)


class ConfigAgent:
    def __init__(self, field, size, body, model):
        self.field = field
        self.load = Load()
        self.agent = None

        # Number of agents placed
        self.i = 0

        # Surfaces that is occupied by obstacles, exits, or other agents
        self.occupied = cascaded_union((field.obstacles, field.exits))

        self.set_body(size, body)
        self.set_model(model)

    @staticmethod
    def truncnorm(loc, abs_scale, size, std=3.0):
        """Scaled symmetrical truncated normal distribution."""
        scale = abs_scale / std
        return truncnorm.rvs(-std, std, loc=loc, scale=scale, size=size)

    @staticmethod
    def random_vector(size, orient=(0.0, 2.0 * np.pi), mag=1.0):
        orientation = np.random.uniform(orient[0], orient[1], size=size)
        return mag * np.stack((np.cos(orientation), np.sin(orientation)),
                              axis=1)

    def set_body(self, size, body):
        logging.info("In: {}, {}".format(size, body))

        # Load tabular values
        bodies = self.load.csv("body")
        try:
            body = bodies[body]
        except:
            raise KeyError(
                "Body \"{}\" is not in bodies {}.".format(body, bodies))
        values = self.load.csv("agent")["value"]

        # Arguments for Agent
        mass = self.truncnorm(body["mass"], body["mass_scale"], size)
        radius = self.truncnorm(body["radius"], body["dr"], size)
        radius_torso = body["k_t"] * radius
        radius_shoulder = body["k_s"] * radius
        torso_shoulder = body["k_ts"] * radius
        target_velocity = self.truncnorm(body['v'], body['dv'], size)
        inertia_rot = eval(values["inertia_rot"]) * np.ones(size)
        target_angular_velocity = eval(values["target_angular_velocity"]) * \
                                  np.ones(size)

        # Agent class
        self.agent = Agent(size, mass, radius, radius_torso, radius_shoulder,
                           torso_shoulder, inertia_rot, target_velocity,
                           target_angular_velocity)

    def set_model(self, model):
        logging.info("{}".format(model))
        if model == "circular":
            self.agent.set_circular()
        elif model == "three_circle":
            self.agent.set_three_circle()
        else:
            logging.warning("")
            raise ValueError()
        logging.info("Out")

    def set_motion(self, i, target_direction, target_angle, velocity,
                   orientation):
        if target_direction is None:
            pass
        elif isinstance(target_direction, np.ndarray):
            self.agent.target_direction[i] = target_direction
        elif target_direction == "random":
            self.agent.target_direction[i] = self.random_vector(1)

        if velocity is None:
            pass
        elif isinstance(velocity, np.ndarray):
            self.agent.velocity[i] = velocity
        elif velocity == "random":
            self.agent.velocity[i] = self.random_vector(1)
        elif velocity == "auto":
            self.agent.velocity[i] = self.agent.target_direction[i]
            self.agent.velocity[i] *= self.agent.target_velocity[i]

        if target_angle is None:
            pass
        elif isinstance(target_angle, np.ndarray):
            self.agent.target_angle[i] = target_angle
        elif target_angle == "random":
            self.agent.target_angle[i] = self.random_vector(1)
        elif velocity == "auto":
            self.agent.target_angle[i] = angle(self.agent.target_direction[i])

        if orientation is None:
            pass
        elif isinstance(orientation, Number):
            self.agent.orientation[i] = orientation
        elif orientation == "random":
            self.agent.angle[i] = np.random.random()
        elif velocity == "auto":
            self.agent.orientation[i] = angle(self.agent.velocity[i])

    def set(self, **kwargs):
        """Set spatial and rotational parameters.

        ================== ==========================
        **Kwargs:**
        *size*             Integer: Number of agent to be placed. \n
                           None: Places all agents
        *surface*          surface: Custom value \n
                           None: Domain
        *position*         ndarray: Custom values \n
                           "random": Uses Monte Carlo method to place agent
                           without overlapping with obstacles or other agents.
        *target_direction* ndarray: Custom value \n
                           "random": Uniformly distributed random value \n
                           "auto":
                           None: Default value
        *velocity*         ndarray: Custom value  \n
                           "random: Uniformly distributed random value, \n
                           "auto":
                           None: Default value
        *target_angle*     ndarray: Custom value  \n
                           "random": Uniformly distributed random value, \n
                           "auto":
                           None: Default value
        *orientation*      float: Custom value  \n
                           "random": Uniformly distributed random value, \n
                           "auto":
                           None: Default value
        ================== ==========================
        """
        logging.info("")

        size = kwargs.get("size")
        surface = kwargs.get("surface", self.field.domain)
        position = kwargs.get("position", "random")
        velocity = kwargs.get("velocity", None)
        orientation = kwargs.get("orientation", None)
        target_direction = kwargs.get("target_direction", "auto")
        target_angle = kwargs.get("target_angle", "auto")

        iterations = 0  # Number of iterations
        area_filled = 0  # Total area filled by agents
        random_sample = PolygonSample(surface)

        while self.i < size:
            # Random point inside spawn surface. Center of mass for an agent.
            if position == "random":
                point = random_sample.draw()
                self.agent.position[self.i] = np.asarray(point)
            elif isinstance(position, np.ndarray):
                # self.agent.position[self.i] = position[self.i]
                # point = Point(position[self.i])
                raise NotImplemented

            self.set_motion(self.i, target_direction, target_angle, velocity,
                            orientation)

            if self.agent.three_circle:
                self.agent.update_shoulder_position(self.i)
                point_ls = Point(self.agent.position_ls[self.i])
                point_rs = Point(self.agent.position_rs[self.i])
                agent = cascaded_union((
                    point.buffer(self.agent.r_t[self.i]),
                    point_ls.buffer(self.agent.r_s[self.i]),
                    point_rs.buffer(self.agent.r_s[self.i]),
                ))
            else:
                agent = point.buffer(self.agent.radius[self.i])

            if not agent.intersects(self.occupied):
                density = area_filled / surface.area
                logging.debug("Agent {} | Density {}".format(self.i, density))
                self.occupied = cascaded_union(self.occupied, agent)
                area_filled += agent.area
                self.agent.active[self.i] = True
                self.i += 1
            else:
                # Reset
                self.agent.position[self.i] = 0
                self.agent.position_ls[self.i] = 0
                self.agent.position_rs[self.i] = 0
                self.agent.velocity[self.i] = 0
                self.agent.orientation[self.i] = 0
                self.agent.target_angle[self.i] = 0
                self.agent.target_direction[self.i] = 0
                self.agent.front[self.i] = 0

            iterations += 1

        logging.info("Density: {}".format(area_filled / surface.area))
