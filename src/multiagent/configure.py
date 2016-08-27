import logging
import numpy as np
from scipy.stats import truncnorm

from shapely.ops import cascaded_union

from src.config import Load
from src.multiagent.agent import Agent

pi = np.pi


class ConfigField:
    def __init__(self):
        self.domain = None
        self.obstacles = None
        self.exits = None


class ConfigAgent:
    """
    0. Field
    1. Body
    2. Model
    3. - Orientation
       - Velocity
       - Position
    """
    def __init__(self, field):
        self.field = field
        self.load = Load()
        self.agent = None

        # Surfaces that is occupied by obstacles, exits, or other agents
        self.occupied = cascaded_union((field.obstacles, field.exits))

        # self.set_body()
        # self.set_model()

    @staticmethod
    def truncnorm(loc, abs_scale, size, std=3.0):
        """Scaled symmetrical truncated normal distribution."""
        scale = abs_scale / std
        return truncnorm.rvs(-std, std, loc=loc, scale=scale, size=size)

    @staticmethod
    def random_vector(size, orient=(0.0, 2.0*pi), mag=1.0):
        orientation = np.random.uniform(orient[0], orient[1], size=size)
        return mag * np.stack((np.cos(orientation), np.sin(orientation)), axis=1)

    def set_body(self, size, body):
        logging.info("In: {}, {}".format(size, body))

        # Load tabular values
        bodies = self.load.csv("body")
        try:
            body = bodies[body]
        except:
            raise KeyError("Body \"{}\" is not in bodies {}.".format(body, bodies))
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

    def set(self, **kwargs):
        """
        Kwargs
        ------

        size: Integer > 0

        surface: surface, "default": Domain

        target_direction: ndarray

        target_angle: ndarray

        velocity: ndarray

        orientation: float

        :param kwargs:
        :return:
        """
        size = kwargs.get("size")
        surface = kwargs.get("surface", self.field.domain)
        velocity = kwargs.get("velocity", None)
        orientation = kwargs.get("orientation", None)
        target_direction = kwargs.get("target_direction", None)
        target_angle = kwargs.get("target_angle", None)

        i = 0            # Number of agents placed
        iterations = 0   # Number of iterations
        area_filled = 0  # Total area filled by agents

        while i < size:
            iterations += 1

            # Random point inside spawn surface. Center of mass for an agent.
            point = surface.random()  # shapely.Point

            if self.agent.three_circle:
                agent = cascaded_union((point.buffer(self.agent.r_t[i]), ))
            else:
                agent = point.buffer(self.agent.radius[i])

            if not agent.intersects(self.occupied):
                self.occupied = cascaded_union(self.occupied, agent)
                self.agent.position[i] = point.as_array()
                area_filled += agent.area
                i += 1

        logging.info("Density: {}".format(area_filled / surface.area))
