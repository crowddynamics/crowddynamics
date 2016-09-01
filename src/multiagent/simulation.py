import logging
from collections import Iterable
from copy import deepcopy
from multiprocessing import Process, Event, Queue
from numbers import Number

import numpy as np
from scipy.spatial.qhull import Delaunay
from scipy.stats import truncnorm
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union

from src.config import Load
from src.core.interactions import agent_agent, agent_wall
from src.core.motion import force_adjust, force_fluctuation, \
    torque_adjust, torque_fluctuation
from src.core.motion import integrator
from src.core.navigation import Navigation, Orientation
from src.core.vector2d import angle, length
from src.io.hdfstore import HDFStore
from src.multiagent.agent import Agent
from src.multiagent.field import LinearObstacle

try:
    from shapely import speedups

    speedups.enable()
except ImportError():
    pass


class PolygonSample:
    """
    Uniform sampling of a polygon
    -----------------------------
    Generates random uniform point from inside of polygon. [1]_

    - Delaunay triangulation to break the polygon into triangular mesh. [2]_
    - Draw random uniform triangle weighted by its area.
    - Draw random uniform sample from inside the triangle.

    .. [1] http://gis.stackexchange.com/questions/6412/generate-points-that-lie-inside-polygon
    .. [2] https://en.wikipedia.org/wiki/Delaunay_triangulation
    """

    def __init__(self, polygon: Polygon):
        """

        :param polygon: Polygon to be sampled.
        """
        self.polygon = polygon
        self.nodes = np.asarray(self.polygon.exterior)

        # Triangular mesh
        self.delaunay = Delaunay(self.nodes)
        self.mesh = self.nodes[self.delaunay.simplices]

        # Cumulative sum of areas of the triangles
        self.weights = self._weights(self.mesh)

    @staticmethod
    def _weights(mesh):
        """Computes cumulative sum of the areas of the triangle mesh.

        :param mesh: Trianle mesh.
        :return: Cumulative sum the area of the triangle mesh
        """
        area_tot = 0
        weigths = np.zeros(mesh.shape[0])
        # TODO: Speedup
        for i, (a, b, c) in enumerate(mesh):
            # Area of triangle
            area_tot += np.abs(a[0] * (b[1] - c[1]) +
                               b[0] * (c[1] - a[1]) +
                               c[0] * (a[1] - b[1])) / 2
            weigths[i] = area_tot
        return weigths

    @staticmethod
    def random_sample_triangle(x):
        """
        Uniform sampling of a triangle
        ------------------------------
        Generate uniform random sample from a triangle defined by points A, B
        and C [1]_, [2]_. Point inside the triangle is given

        .. math::
           P = (1 - \sqrt{r_1}) A + (\sqrt{r_1} (1 - r_2))  B + (r_2 \sqrt{r_1}) C,

        where random variables are

        .. math::
           r_1, r_2 \sim \mathcal{U}(0, 1)

        References
        ----------
        .. [1] http://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
        .. [2] http://mathworld.wolfram.com/TrianglePointPicking.html

        :param x: Three points defining a triangle (A, B, C).
        :return: Uniformly distributed random point P.
        """
        r = np.random.uniform(size=2)  # Random variables
        p = (1 - np.sqrt(r[0])) * x[0] + \
            (np.sqrt(r[0]) * (1 - r[1])) * x[1] + \
            r[1] * np.sqrt(r[0]) * x[2]
        return p

    def draw(self):
        # Draw random triangle weighted by the area of the triangle
        x = np.random.uniform(high=self.weights[-1])
        i = np.searchsorted(self.weights, x)

        # Random sample from the triangle
        sample = self.random_sample_triangle(self.mesh[i])
        return Point(sample)


class Configuration:
    """Set initial configuration for multi-agent simulation"""
    def __init__(self):
        self.domain = None  # Shapely.Polygon
        self.goals = []
        self.obstacles = []  # shapely.LineString
        self.exits = []  # shapely.LineString
        self.agent = None

        # Angle and direction update algorithms
        self.navigation = None
        self.orientation = None

        # Current index of agent to be placed
        self._index = 0

        # Surfaces that is occupied by obstacles, exits, or other agents
        self._occupied = cascaded_union(())  # Empty initially

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

    def set_domain(self, polygon):
        logging.info("")
        if polygon.is_valid and polygon.is_simple:
            self.domain = polygon
        else:
            logging.warning("Domain not valid.")

    def _set_goal(self, polygon):
        if polygon.is_valid and polygon.is_simple:
            self.goals.append(polygon)
        else:
            logging.warning("Goal not valid.")

    def _set_obstacle(self, linestring):
        if linestring.is_valid and linestring.is_simple:
            self.obstacles.append(linestring)
        else:
            logging.warning("Obstacle not valid.")

    def _set_exit(self, linestring):
        if linestring.is_valid and linestring.is_simple:
            self.exits.append(linestring)
        else:
            logging.warning("Exit not valid.")

    def set_goals(self, polygon):
        logging.info("")
        if isinstance(polygon, Iterable):
            for poly in polygon:
                self._set_goal(poly)
        else:
            self._set_goal(polygon)

    def set_obstacles(self, linestring):
        logging.info("")
        if isinstance(linestring, Iterable):
            for ls in linestring:
                self._set_obstacle(ls)
        else:
            self._set_obstacle(linestring)

    def set_exits(self, linestring):
        logging.info("")
        if isinstance(linestring, Iterable):
            for ls in linestring:
                self._set_exit(ls)
        else:
            self._set_exit(linestring)

    def set_navigation(self, custom=None):
        """Default navigation algorithm"""
        logging.info("")
        if custom is None:
            self.navigation = Navigation(self.agent, self.domain,
                                         self.obstacles, self.exits)
        else:
            self.navigation = custom

    def set_orientation(self, custom=None):
        """Default orientation algorithm"""
        logging.info("")
        if custom is None:
            self.orientation = Orientation(self.agent)
        else:
            self.orientation = custom

    def set_body(self, size, body):
        logging.info("In: {}, {}".format(size, body))

        # noinspection PyUnusedLocal
        pi = np.pi

        load = Load()
        # Load tabular values
        bodies = load.csv("body")
        try:
            body = bodies[body]
        except:
            raise KeyError(
                "Body \"{}\" is not in bodies {}.".format(body, bodies))
        values = load.csv("agent")["value"]

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
            self.agent.angle[i] = orientation
        elif orientation == "random":
            self.agent.angle[i] = np.random.random()
        elif velocity == "auto":
            self.agent.angle[i] = angle(self.agent.velocity[i])

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
        surface = kwargs.get("surface", self.domain)
        position = kwargs.get("position", "random")
        velocity = kwargs.get("velocity", None)
        orientation = kwargs.get("orientation", None)
        target_direction = kwargs.get("target_direction", "auto")
        target_angle = kwargs.get("target_angle", "auto")

        iterations = 0  # Number of iterations
        area_filled = 0  # Total area filled by agents
        random_sample = PolygonSample(surface)

        self._occupied = cascaded_union(self.obstacles + self.exits)

        limit = self._index + size
        iter_limit = size * 5
        # TODO: Speedup
        while self._index < limit and iterations < iter_limit:
            # Random point inside spawn surface. Center of mass for an agent.
            if position == "random":
                point = random_sample.draw()
                self.agent.position[self._index] = np.asarray(point)
            elif isinstance(position, np.ndarray):
                point = Point(position[self._index])
                self.agent.position[self._index] = position[self._index]
            else:
                raise ValueError()

            self.set_motion(self._index, target_direction, target_angle,
                            velocity, orientation)

            # Geometry of the agent
            if self.agent.three_circle:
                self.agent.update_shoulder_position(self._index)

                point_ls = Point(self.agent.position_ls[self._index])
                point_rs = Point(self.agent.position_rs[self._index])
                agent = cascaded_union((
                    point.buffer(self.agent.r_t[self._index]),
                    point_ls.buffer(self.agent.r_s[self._index]),
                    point_rs.buffer(self.agent.r_s[self._index]),
                ))
            else:
                d = self.agent.position[size:self._index-1] - \
                    self.agent.position[self._index]
                d = length(d) - \
                    self.agent.radius[size:self._index-1] - \
                    self.agent.radius[self._index]

                agent = point.buffer(self.agent.radius[self._index])

            # Check if agent intersects with other agents or obstacles
            if not agent.intersects(self._occupied):
                density = area_filled / surface.area
                logging.debug(
                    "Agent {} | Density {}".format(self._index, density))
                self._occupied = cascaded_union((self._occupied, agent))
                area_filled += agent.area
                self.agent.active[self._index] = True
                self._index += 1
            else:
                # Reset
                self.agent.position[self._index] = 0
                self.agent.position_ls[self._index] = 0
                self.agent.position_rs[self._index] = 0
                self.agent.velocity[self._index] = 0
                self.agent.angle[self._index] = 0
                self.agent.target_angle[self._index] = 0
                self.agent.target_direction[self._index] = 0
                self.agent.front[self._index] = 0

            iterations += 1

        logging.info("Density: {}".format(area_filled / surface.area))


class QueueDict:
    def __init__(self, producer):
        self.producer = producer
        self.dict = {}

    def set(self, args):
        self.dict.clear()
        for key, attrs in args:
            self.dict[key] = {}
            for attr in attrs:
                self.dict[key][attr] = None

    def fill(self, d):
        for key, attrs in d.items():
            item = getattr(self.producer, key)
            for attr in attrs.keys():
                d[key][attr] = np.copy(getattr(item, attr))

    def get(self):
        d = deepcopy(self.dict)
        self.fill(d)
        return d


class MultiAgentSimulation(Process, Configuration):
    """
    Class that calls numerical algorithms of the multi-agent simulation.
    """
    structures = ("domain", "goals", "exits", "walls", "agent")
    parameters = ("dt_min", "dt_max", "time_tot", "in_goal", "dt_prev")

    def __init__(self, queue: Queue = None):
        super(MultiAgentSimulation, self).__init__()  # Multiprocessing
        Configuration.__init__(self)

        self.queue = queue
        self.exit = Event()

        self.walls = None  # LinearWalls

        # Additional models
        self.game = None

        # Integrator timestep
        self.dt_min = 0.001
        self.dt_max = 0.01

        # State of the simulation
        self.iterations = 0  # Integer
        self.time_tot = 0.0  # Float (types matter for saving to a file)
        self.in_goal = 0  # Integer TODO: Move to area?
        self.dt_prev = 0.1  # Float. Last used time step.

        # Data flow
        self.hdfstore = None  # Sends data to hdf5 file
        self.queue_items = None  # Sends data to graphics

    @property
    def name(self):
        return self.__class__.__name__

    def stop(self):
        """Sets event to true in order to stop the simulation process."""
        logging.info("MultiAgent Exit...")
        self.exit.set()

    def run(self):
        """Runs simulation process until is called. This calls the update method
        repeatedly. Finally at stop it puts poison pill (None) into the queue to
        denote last generated value."""
        logging.info("MultiAgent Starting")
        while not self.exit.is_set():
            self.update()
        self.queue.put(None)  # Poison pill. Ends simulation
        logging.info("MultiAgent Stopping")

    def set_obstacles_to_linear_walls(self):
        points = []
        for obstacle in self.obstacles:
            a = np.asarray(obstacle)
            for i in range(len(a) - 1):
                points.append((a[i], a[i + 1]))
        if points:
            params = np.array(points)
            self.walls = LinearObstacle(params)

    def configure_hdfstore(self):
        if self.hdfstore is None:
            logging.info("")

            # Configure hdfstore file
            self.hdfstore = HDFStore(self.name)

            # Add dataset
            load = Load()
            parameters = load.yaml('parameters')

            args = self.agent, parameters['agent']
            self.hdfstore.add_dataset(*args)
            self.hdfstore.add_buffers(*args)

            args = self, parameters['simulation']
            self.hdfstore.add_dataset(*args)
            self.hdfstore.add_buffers(*args)

            logging.info("")
        else:
            logging.info("Already configured.")

    def configure_queuing(self, args):
        """

        :param args: Example [("agent", ["position", "active", "position_ls", "position_rs"])]
        :return:
        """
        # FIXME
        if self.queue is not None:
            logging.info("")
            self.queue_items = QueueDict(self)
            self.queue_items.set(args)
        else:
            logging.info("Queue is not defined.")

    def update(self):
        logging.debug("")

        # Path finding and rotation planning
        if self.navigation is not None:
            self.navigation.update()

        if self.orientation is not None and self.agent.orientable:
            self.orientation.update()

        # Computing motion (forces and torques) for the system
        self.agent.reset_motion()
        self.agent.reset_neighbor()

        force_adjust(self.agent)
        force_fluctuation(self.agent)
        if self.agent.orientable:
            torque_adjust(self.agent)
            torque_fluctuation(self.agent)
        agent_agent(self.agent)
        if self.walls is not None:
            agent_wall(self.agent, self.walls)

        # Integration of the system
        self.dt_prev = integrator(self.agent, self.dt_min, self.dt_max)
        self.time_tot += self.dt_prev

        # Game theoretical model
        if self.game is not None:
            self.game.update(self.time_tot, self.dt_prev)

        # Check which agent are inside the domain aka active
        # if self.domain is not None:
        #     self.agent.active &= self.domain.contains(self.agent.position)

        # Check which agent have reached their desired goals
        # for goal in self.goals:
        #     num = -np.sum(self.agent.goal_reached)
        #     self.agent.goal_reached |= goal.contains(self.agent.position)
        #     num += np.sum(self.agent.goal_reached)
        #     self.in_goal += num

        # Raise iteration count
        self.iterations += 1

        # Stores the simulation data into buffers and dumps buffer into file
        if self.hdfstore is not None:
            self.hdfstore.update_buffers()
            if self.iterations % 100 == 0:
                self.hdfstore.dump_buffers()

        data = self.queue_items.get()
        self.queue.put(data)
