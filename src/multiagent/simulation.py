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

from src.config import Load
from src.core.geometry import check_shapes, shapes_to_point_pairs
from src.core.sampling import PolygonSample
from src.core.interactions import agent_agent, agent_wall, \
    agent_agent_distance_three_circle
from src.core.motion import force_adjust, force_fluctuation, \
    torque_adjust, torque_fluctuation
from src.core.motion import integrator
from src.core.navigation import Navigation, Orientation
from src.core.vector2D import angle, length
from src.io.hdfstore import HDFStore
from src.multiagent.agent import Agent
from src.multiagent.field import LinearObstacle


class Configuration:
    """Set initial configuration for multi-agent simulation.

    ==  ==========  ==============  =========  ====================================================================================
     0
     1  Field
     2              domain
     3
     4              goals
     5
     6              obstacles
     7
     8              exits
     9
    10  Algorithms
    11              navigation      None       Agent follow the initial target directions and do not update their target directions
    12                              callable   Custom class that has callable `update` function
    13                              “static”
    14                              “dynamic”
    15
    16              orientation     None
    17
    18
    19
    20              exit_selection  None
    21
    22  Agent
    23              size
    24              body
    25              model
    26              …
    ==  ==========  ==============  =========  ====================================================================================
    """

    def __init__(self):
        # Field
        self.domain = None
        self.obstacles = []
        self.exits = []

        # Numpy + Numba. More computationally efficient forms
        self.agent = None
        self.walls = None  # LinearWalls

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
        logging.info("")

        # TODO: Conditions: is_valid, is_simple, ...
        self.obstacles = check_shapes(obstacles, (Polygon, LineString))
        self.exits = check_shapes(exits, (Polygon, LineString))

        if isinstance(domain, Polygon):
            self.domain = domain
        elif domain is None:
            # Construct polygon that bounds other objects
            raise NotImplemented("")
        else:
            raise ValueError("")

        points = shapes_to_point_pairs(self.obstacles)
        if len(points) != 0:
            self.walls = LinearObstacle(points)

    def set_algorithms(self, navigation=None, orientation=None, exit_selection=None):
        logging.info("")

        # Navigation
        # TODO: Navigation to different exits
        if isinstance(navigation, str):
            if navigation == "static":
                self.navigation = Navigation(self)
                self.navigation.static_potential()
            elif navigation == "dynamic":
                self.navigation = Navigation(self)
                self.navigation.dynamic_potential()
            else:
                raise ValueError("")
        elif hasattr(navigation, "update") and callable(navigation.update):
            pass
        else:
            self.navigation = None

        # Orientation
        if orientation is None:
            self.orientation = Orientation(self)
        else:
            self.orientation = orientation

        # TODO: Exit Selection
        pass

    def set_body(self, size, body):
        """

        ========================= ============================================
        **Load values form csv**
        *mass*                    --
        *radius*                  --
        *dr*                      --
        *k_t*                     --
        *k_s*                     --
        *v*                       --
        *dv*                      --
        *inertia_rot*             --
        *target_angular_velocity* --
        ========================= ============================================
        """
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
        target_angular_velocity = eval(values["target_angular_velocity"]) * np.ones(size)

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

    def set_motion(self, i, target_direction, target_angle, velocity, orientation):
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
        logging.info("")

        if surface is None:
            surface = self.domain

        iterations = 0  # Number of iterations
        area_filled = 0  # Total area filled by agents
        random_sample = PolygonSample(surface)

        self._occupied = cascaded_union(self.obstacles + self.exits)

        start_index = self._index
        limit = start_index + size
        iter_limit = size * 5

        @numba.jit(nopython=True)
        def agent_distance_condition(agent, start_index, i):
            """Test function for determining if agents are overlapping."""
            condition = True
            if agent.three_circle:
                for j in range(start_index, i):
                    if condition:
                        t = agent_agent_distance_three_circle(agent, i, j)
                        condition &= t[1] > 0
                    else:
                        break
            else:
                for j in range(start_index, i):
                    if condition:
                        d = agent.position[i] - agent.position[j]
                        s = length(d) - agent.radius[i] - agent.radius[j]
                        condition &= s > 0
                    else:
                        break
            return condition

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
                agent = point.buffer(self.agent.radius[self._index])

            # Check if agent intersects with other agents or obstacles
            if agent_distance_condition(self.agent, start_index, self._index) \
                    and not agent.intersects(self._occupied):
                density = area_filled / surface.area
                logging.debug(
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
    structures = ("domain", "exits", "walls", "agent")
    parameters = ("dt_min", "dt_max", "time_tot", "in_goal", "dt_prev")

    def __init__(self, queue: Queue = None):
        super(MultiAgentSimulation, self).__init__()  # Multiprocessing
        Configuration.__init__(self)

        self.queue = queue
        self.exit = Event()

        # Additional models
        self.game = None

        # Integrator timestep
        self.dt_min = 0.001
        self.dt_max = 0.01

        # State of the simulation
        self.iterations = 0  # Integer
        self.time_tot = 0.0  # Float (types matter when saving to a file)
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

        # Path finding
        if self.navigation is not None:
            self.navigation.update()

        # Rotation planning
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
        if self.domain is not None:
            domain = Path(np.asarray(self.domain.exterior))
            num = -np.sum(self.agent.active)
            self.agent.active &= domain.contains_points(self.agent.position)
            num += np.sum(self.agent.active)
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
