import logging
from copy import deepcopy
from multiprocessing import Process, Event, Queue
from numbers import Number

import numba
import numpy as np
from matplotlib.path import Path
from scipy.stats import truncnorm
from shapely.geometry import LineString
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union

from crowddynamics.core.distance import distance_three_circle
from crowddynamics.core.vector2D import angle, length
from crowddynamics.functions import Timed, load_config, public
from crowddynamics.geometry import check_shapes
from crowddynamics.io.hdfstore import HDFStore
from crowddynamics.multiagent.agent import Agent
from crowddynamics.sampling import PolygonSample


class QueueDict:
    """Queue dict"""

    def __init__(self, producer):
        """

        Args:
            producer:
        """
        self.producer = producer
        self.dict = {}
        self.args = None

    def set(self, args):
        self.args = args

        self.dict.clear()
        for (key, key2), attrs in self.args:
            self.dict[key2] = {}
            for attr in attrs:
                self.dict[key2][attr] = None

    def fill(self, d):
        for (key, key2), attrs in self.args:
            item = getattr(self.producer, key)
            for attr in attrs:
                d[key2][attr] = np.copy(getattr(item, attr))

    def get(self):
        d = deepcopy(self.dict)
        self.fill(d)
        return d


class Configuration:
    """
    Set initial configuration for multi-agent simulation.

    .. csv-table::
       :file: configs/configuration.csv

    """

    def __init__(self):
        """Configuration"""
        self.logger = logging.getLogger("crowddynamics.configuration")

        # Field
        self.domain = None
        self.obstacles = []
        self.exits = []

        # Numpy + Numba. More computationally efficient forms
        self.agent = None
        self.omega = None

        # Angle and direction update algorithms
        # TODO: Add root node
        self.task_graph = None

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

    def set_field(self, domain=None, obstacles=None, exits=None):
        """
        Shapely BaseGeometry types

        - Point
        - LineString
        - LinearRing
        - Polygon
        - MultiPoint
        - MultiLineString
        - MultiPolygon
        - GeometryCollection

        =========== ===========================================================
        **Kwargs**

        *domain*    Polygon which contains all the other objects.

        *goals*     --

        *obstacles* Collection of polygons and LineStrings.

        *exits*     Collection of polygons and LineStrings.

        =========== ===========================================================
        """
        self.logger.info("")

        # TODO: Conditions: is_valid, is_simple, ...
        self.obstacles = check_shapes(obstacles, (Polygon, LineString))
        self.exits = check_shapes(exits, (Polygon, LineString))

        if isinstance(domain, Polygon):
            self.domain = domain
        elif domain is None:
            # Construct polygon that bounds other objects
            raise NotImplemented("")
        else:
            raise Exception("")

        # TODO: Move to own task node
        self.omega = Path(np.asarray(self.domain.exterior))

    def set_body(self, size, body):
        self.logger.info("In: {}, {}".format(size, body))

        # noinspection PyUnusedLocal
        pi = np.pi

        # Load tabular values
        bodies = load_config("body.csv")
        try:
            body = bodies[body]
        except:
            raise KeyError(
                "Body \"{}\" is not in bodies {}.".format(body, bodies))
        values = load_config("agent.csv")["value"]

        # Arguments for Agent
        # TODO: Scaling inertia_rot
        mass = self.truncnorm(body["mass"], body["mass_scale"], size)
        radius = self.truncnorm(body["radius"], body["radius_scale"], size)
        ratio_rt = body["ratio_rt"]
        ratio_rs = body["ratio_rs"]
        ratio_ts = body["ratio_ts"]
        target_velocity = self.truncnorm(body['velocity'],
                                         body['velocity_scale'], size)
        inertia_rot = eval(values["inertia_rot"]) * np.ones(size)
        target_angular_velocity = eval(
            values["target_angular_velocity"]) * np.ones(size)

        # Agent class
        self.agent = Agent(size, mass, radius, ratio_rt, ratio_rs, ratio_ts,
                           inertia_rot, target_velocity,
                           target_angular_velocity)

    def set_model(self, model):
        self.logger.info("{}".format(model))
        if model == "circular":
            self.agent.set_circular()
        elif model == "three_circle":
            self.agent.set_three_circle()
        else:
            self.logger.warning("")
            raise ValueError()

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

    def set_agents(self, size=None, surface=None, position="random",
                   velocity=None, orientation=None, target_direction="auto",
                   target_angle="auto"):
        """Set spatial and rotational parameters.

        ================== ==========================
        **kwargs**

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
        self.logger.info("")

        if surface is None:
            surface = self.domain

        iterations = 0  # Number of iterations
        area_filled = 0  # Total area filled by agents
        random_sample = PolygonSample(surface)

        self._occupied = cascaded_union(self.obstacles + self.exits)

        start_index = self._index
        limit = start_index + size
        iter_limit = size * 5

        # FIXME
        @numba.jit(nopython=True)
        def agent_distance_condition(agent, start_index, i):
            """Test function for determining if agents are overlapping."""
            if agent.three_circle:
                condition = True
                for j in range(start_index, i):
                    if condition:
                        t = distance_three_circle(
                            (agent.position[i], agent.position_ls[i],
                             agent.position_rs[i]),
                            (agent.r_t[i], agent.r_s[i], agent.r_s[i]),
                            (agent.position[j], agent.position_ls[j],
                             agent.position_rs[j]),
                            (agent.r_t[j], agent.r_s[j], agent.r_s[j])
                        )
                        # FIXME: cannot unify bool and array(bool, 1d, C) for 'condition'
                        condition &= t[1] > 0
                    else:
                        break
                return condition
            else:
                condition2 = True
                for j in range(start_index, i):
                    if condition2:
                        d = agent.position[i] - agent.position[j]
                        s = length(d) - agent.radius[i] - agent.radius[j]
                        condition2 &= s > 0
                    else:
                        break
                return condition2

        while self._index < limit and iterations < iter_limit:
            # Random point inside spawn surface. Center of mass for an agent.
            if position == "random":
                point = Point(random_sample.draw())
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
                agent = point.buffer(self.agent.radius[self._index])

            # Check if agent intersects with other agents or obstacles
            if agent_distance_condition(self.agent, start_index, self._index) \
                    and not agent.intersects(self._occupied):
                density = area_filled / surface.area
                self.logger.debug(
                    "Agent {} | Density {}".format(self._index, density))
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

        self.logger.info("Density: {}".format(area_filled / surface.area))


class MultiAgentSimulation(Process, Configuration):
    """
    Class that calls numerical algorithms of the multi-agent simulation.
    """

    def __init__(self, queue: Queue = None):
        """
        MultiAgentSimulation

        Args:
            queue:
        """
        super(MultiAgentSimulation, self).__init__()  # Multiprocessing
        Configuration.__init__(self)

        # Logger
        self.logger = logging.getLogger("crowddynamics.simulation")

        # Multiprocessing
        self.queue = queue
        self.exit = Event()

        # State of the simulation (types matter when saving to a file)
        self.iterations = int(0)
        self.time_tot = float(0)
        self.in_goal = int(0)
        self.dt_prev = float(0)

        # Data flow
        self.hdfstore = None  # Sends data to hdf5 file
        self.queue_items = None  # Sends data to graphics

    @property
    def name(self):
        return self.__class__.__name__

    def parameters(self):
        params = (
            "time_tot",
            "in_goal",
        )
        for p in params:
            assert hasattr(self,
                           p), "{cls} doesn't have attribute {attr}".format(
                cls=p, attr=p)
        return params

    def stop(self):
        """Sets event to true in order to stop the simulation process."""
        self.logger.info("MultiAgent Exit...")
        # self.queue.put(None)  # Poison pill. Ends simulation
        self.exit.set()

    def run(self):
        """Runs simulation process until is called. This calls the update method
        repeatedly. Finally at stop it puts poison pill (None) into the queue to
        denote last generated value."""
        self.logger.info("MultiAgent Starting")
        while not self.exit.is_set():
            self.update()

        self.queue.put(None)  # Poison pill. Ends simulation
        self.logger.info("MultiAgent Stopping")

    def configure_hdfstore(self):
        if self.hdfstore is None:
            self.logger.info("")

            # Configure hdfstore file
            self.hdfstore = HDFStore(self.name)

            # Add dataset
            parameters = load_config('parameters.yaml')

            args = self.agent, parameters['agent']
            self.hdfstore.add_dataset(*args)
            self.hdfstore.add_buffers(*args)

            args = self, parameters['simulation']
            self.hdfstore.add_dataset(*args)
            self.hdfstore.add_buffers(*args)

            self.logger.info("")
        else:
            self.logger.info("Already configured.")

    def configure_queuing(self, args):
        """

        :param args: Example [("agent", ["position", "active", "position_ls", "position_rs"])]
        :return:
        """
        # FIXME
        if self.queue is not None:
            self.logger.info("")
            self.queue_items = QueueDict(self)
            self.queue_items.set(args)
        else:
            self.logger.info("Queue is not defined.")

    @Timed("Total Simulation Time")
    def update(self):
        """Update"""
        self.agent.reset_motion()
        self.agent.reset_neighbor()
        self.task_graph.evaluate()

        # TODO: TaskNode
        # Check which agent are inside the domain
        if self.domain is not None:
            num = np.sum(self.agent.active)
            self.agent.active &= self.omega.contains_points(self.agent.position)
            num -= np.sum(self.agent.active)
            self.in_goal += num

        # Raise iteration count
        self.iterations += 1

        # Stores the simulation data into buffers and dumps buffer into file
        if self.hdfstore is not None:
            self.hdfstore.update_buffers()
            if self.iterations % 100 == 0:
                self.hdfstore.dump_buffers()

        if self.queue is not None:
            data = self.queue_items.get()
            self.queue.put(data)

    # To measure JIT compilation time of numba decorated functions.
    initial_update = Timed("Jit:")(deepcopy(update))

    try:
        # If using line_profiler decorate function.
        update = profile(update)
    except NameError:
        pass
