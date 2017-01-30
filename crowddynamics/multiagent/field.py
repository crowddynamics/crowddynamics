import logging

import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon, GeometryCollection
from shapely.ops import cascaded_union

from crowddynamics.core.interactions import overlapping_circle_circle, \
    overlapping_three_circle
from crowddynamics.core.random.sampling import PolygonSample
from crowddynamics.logging import log_with
from crowddynamics.multiagent import Agent
from crowddynamics.multiagent.agent import positions
from crowddynamics.multiagent.parameters import Parameters


def agent_polygon(position, radius):
    if isinstance(position, tuple):
        return cascaded_union((
            Point(position[0]).buffer(radius[0]),
            Point(position[1]).buffer(radius[1]),
            Point(position[2]).buffer(radius[2]),
        ))
    else:
        return Point(position).buffer(radius)


class Field:
    r"""
    MultiAgent simulation setup

    1) Set the Field

       - Domain
       - Obstacles
       - Targets (aka exits)

    2) Initialise Agents

       - Set maximum number of agents. This is the limit of the size of array
         inside ``Agent`` class.
       - Select Agent model.

    3) Place Agents into any surface that is contained by the domain.

       - Body type
       - Number of agents that is placed into the surface

    """
    logger = logging.getLogger(__name__)

    def __init__(self):
        # Field
        self.domain = Polygon()
        self.obstacles = GeometryCollection()
        self.targets = GeometryCollection()
        # self.agents = dict()
        self.agent = None
        # Currently occupied surface by Agents and Obstacles
        self._occupied = Polygon()

    @log_with(logger)
    def init_domain(self, domain):
        """
        Initialize domain

        Args:
            domain (Polygon, optional):
                - ``Polygon``: Subset of real domain
                  :math:`\Omega \subset \mathbb{R}^{2}`.
                - ``None``: Real domain :math:`\Omega = \mathbb{R}^{2}`.
        """
        self.logger.info("")
        self.domain = domain

    @log_with(logger)
    def init_agents(self, max_size, model):
        """
        Initialize agents

        Args:
            max_size (int, optional):
                - ``int``: Maximum number of agents.
                - ``None``: Dynamically increase the size when adding new agents

            model (str):
                Choice from:
                - ``circular``
                - ``three_circle``
        """
        self.logger.info("")
        if max_size is None:
            raise NotImplemented
        self.agent = Agent(max_size)
        if model == 'three_circle':
            self.agent.set_three_circle()
        else:
            self.agent.set_circular()

    @log_with(logger)
    def add_obstacle(self, geom):
        """
        Add new ``obstacle`` to the Field

        Args:
            geom (BaseGeometry):
        """
        self.logger.info("")
        self.obstacles |= geom
        self._occupied |= geom

    @log_with(logger)
    def remove_obstacle(self, geom):
        """Remove obstacle"""
        self.logger.info("")
        self.obstacles -= geom
        self._occupied -= geom

    @log_with(logger)
    def add_target(self, geom):
        """
        Add new ``target`` to the Field

        Args:
            geom (BaseGeometry):
        """
        self.logger.info("")
        self.targets |= geom

    @log_with(logger)
    def remove_target(self, geom):
        """Remove target"""
        self.logger.info("")
        self.targets -= geom

    @log_with(logger)
    def add_agents(self, num, spawn, body_type, iterations_limit=100):
        r"""
        Add multiple agents at once.

        1) Sample new position from ``PolygonSample``
        2) Check if agent in new position is overlapping with existing ones
        3) Add new agent if there is no overlapping

        Args:
            num (int, optional):
                - Number of agents to be placed into the ``surface``. If given
                  amount of agents does not fit into the ``surface`` only the
                  amount that fits will be placed.
                - ``None``: Places maximum size of agents

            spawn (Polygon, optional):
                - ``Polygon``: Custom polygon that is contained inside the
                  domain
                - ``None``: Domain

            body_type (str):
                Choice from ``Parameter.body_types``:
                - 'adult'
                - 'male'
                - 'female'
                - 'child'
                - 'eldery'

            iterations_limit (int):
                Limits iterations to ``max_iter = iterations_limit * num``.

        Yields:
            int: Index of agent that was placed.

        """
        # Draw random uniformly distributed points from the set on points
        # that belong to the surface. These are used as possible new position
        # for an agents (if it does not overlap other agents).
        iterations = 0
        sampling = PolygonSample(spawn)
        parameters = Parameters(body_type=body_type)

        while num > 0 and iterations <= iterations_limit * num:
            # Parameters
            position = sampling.draw()
            mass = parameters.mass.default()
            radius = parameters.radius.default()
            ratio_rt = parameters.radius_torso.default()
            ratio_rs = parameters.radius_shoulder.default()
            ratio_ts = parameters.radius_torso_shoulder.default()
            inertia_rot = parameters.moment_of_inertia.default()
            max_velocity = parameters.maximum_velocity.default()
            max_angular_velocity = parameters.maximum_angular_velocity.default()

            # Polygon of the agent
            overlapping_agents = False
            overlapping_obstacles = False
            num_active_agents = np.sum(self.agent.active)
            if num_active_agents > 0:
                # Conditions
                if self.agent.three_circle:
                    r_t = ratio_rt * radius
                    r_s = ratio_rs * radius
                    orientation = 0.0
                    poly = agent_polygon(
                        positions(position, orientation, ratio_rt * radius),
                        (r_t, r_s, r_s)
                    )
                    overlapping_agents = overlapping_three_circle(
                        self.agent, self.agent.indices(),
                        positions(position, orientation, ratio_rt * radius),
                        (r_t, r_s, r_s),
                    )
                else:
                    poly = agent_polygon(position, radius)
                    overlapping_agents = overlapping_circle_circle(
                        self.agent, self.agent.indices(),
                        position,
                        radius
                    )
                overlapping_obstacles = self.obstacles.intersects(poly)

            if not overlapping_agents and not overlapping_obstacles:
                # Add new agent
                index = self.agent.add(
                    position, mass, radius, ratio_rt, ratio_rs, ratio_ts,
                    inertia_rot, max_velocity, max_angular_velocity
                )
                if index >= 0:
                    # Yield index of an agent that was successfully placed.
                    num -= 1
                    # self.agents[index] = poly
                    yield index
                else:
                    break
            iterations += 1

    def remove_agents(self, indices):
        pass
