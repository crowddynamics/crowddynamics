from collections import Callable, Collection, Generator

import numba
import numpy as np
from configobj import ConfigObj
from numba import typeof, void, boolean, float64
from numba.types import UniTuple
from traitlets.traitlets import HasTraits, Float, default, Unicode, \
    observe, Bool, Int, Type, Instance, TraitError, Union, List
from traittypes import Array

from crowddynamics.config import load_config, BODY_TYPES_CFG, \
    BODY_TYPES_CFG_SPEC
from crowddynamics.core.block_list import MutableBlockList
from crowddynamics.core.distance import distance_circles, \
    distance_circle_line, distance_three_circle_line
from crowddynamics.core.distance import distance_three_circles
from crowddynamics.core.rand import truncnorm
from crowddynamics.core.structures import obstacle_type_linear
from crowddynamics.core.vector2D import unit_vector, rotate270
from crowddynamics.exceptions import CrowdDynamicsException
from crowddynamics.simulation.base import AgentsBase
from crowddynamics.traits import shape_validator, length_validator, \
    table_of_traits, \
    class_to_struct_dtype
from crowddynamics.utils import interpolate_docstring

NO_TARGET = -1
NO_LEADER = -1


class States(HasTraits):
    active = Bool(
        default_value=True,
        help='Denotes if agent is currently active')

    target_reached = Bool(
        default_value=False,
        help='Denotes if agent has reached its target')

    # Navigation
    target = Int(
        default_value=NO_TARGET,
        min=NO_TARGET,
        help='Positive integer for target index, -1 for agent that do not have '
             'a target.')
    is_leader = Bool(
        default_value=False,
        help='Boolean indicating if agent is leader')
    is_follower = Bool(
        default_value=False,
        help='Boolean indicating if agent is herding (following average '
             'direction of other agent).')
    index_leader = Int(
        default_value=NO_LEADER,
        help='Index of the agent that is the leader of this agent.')
    familiar_exit = Int(
        default_value=NO_TARGET,
        min=NO_TARGET,
        help='Target that is familiar to a follower agent.')


class Body(HasTraits):
    radius = Float(
        min=0,
        help='Radius',
        symbol='r')
    r_t = Float(
        min=0,
        help='Torso radius',
        symbol='r_t')
    r_s = Float(
        min=0,
        help='Shoulder radius',
        symbol='r_s')
    r_ts = Float(
        min=0,
        help='Distance from torso to shoulder',
        symbol='r_{ts}')
    mass = Float(
        min=0,
        help='Mass',
        symbol='m')
    inertia_rot = Float(
        min=0,
        help='Rotational moment',
        symbol='I_{rot}')
    target_velocity = Float(
        min=0,
        help='Target velocity',
        symbol='v_0')
    target_angular_velocity = Float(
        min=0,
        help='Target angular velocity',
        symbol=r'\omega_0')


class BodyType(Body):
    body_type = Unicode(
        help='Selected body type')
    body_types = Instance(
        ConfigObj,
        help='Mapping of body type names to values')

    # Ratios of radii for shoulders and torso
    ratio_rt = Float(
        default_value=0, min=0, max=1,
        help='Ratio between total radius and torso radius')
    ratio_rs = Float(
        default_value=0, min=0, max=1,
        help='Ratio between total radius and shoulder radius')
    ratio_ts = Float(
        default_value=0, min=0, max=1,
        help='Ratio between total radius and distance from torso to shoulder')

    # Scales for settings values from truncated normal distribution
    # TODO: Distributions class as instance traits
    radius_mean = Float(
        default_value=0, min=0)
    radius_scale = Float(
        default_value=0, min=0)
    target_velocity_mean = Float(
        default_value=0, min=0)
    target_velocity_scale = Float(
        default_value=0, min=0)
    mass_mean = Float(
        default_value=0, min=0)
    mass_scale = Float(
        default_value=0, min=0)

    @staticmethod
    def _truncnorm(mean, abs_scale):
        """Individual value from truncnorm"""
        return np.asscalar(truncnorm(-3.0, 3.0, loc=mean, abs_scale=abs_scale))

    @default('body_types')
    def _default_body_types(self):
        return load_config(BODY_TYPES_CFG, BODY_TYPES_CFG_SPEC)

    @observe('body_type')
    def _observe_body_type(self, change):
        if change['old'] == '':
            new = change['new']
            for k, v in self.body_types[new].items():
                setattr(self, k, v)
        else:
            raise TraitError('Body type can only be set once.')

    @observe('radius_mean', 'radius_scale')
    def _observe_radius_truncnorm(self, change):
        if self.radius == 0 and self.radius_mean > 0 and self.radius_scale > 0:
            self.radius = self._truncnorm(self.radius_mean, self.radius_scale)

    @observe('radius', 'ratio_rt', 'ratio_rs', 'ratio_ts')
    def _observe_radius(self, change):
        """Set torso radius if ratio_rt changes and radius is defined or if
        radius changes and ratio_rt is defined."""
        name = change['name']
        if name == 'radius':
            if self.ratio_rt > 0:
                self.r_t = self.ratio_rt * self.radius
            if self.ratio_rs > 0:
                self.r_s = self.ratio_rs * self.radius
            if self.ratio_ts > 0:
                self.r_ts = self.ratio_ts * self.radius
        elif self.radius > 0:
            if name == 'ratio_rt':
                self.r_t = self.ratio_rt * self.radius
            elif name == 'ratio_rs':
                self.r_s = self.ratio_rs * self.radius
            elif name == 'ratio_ts':
                self.r_ts = self.ratio_ts * self.radius

    @observe('mass_mean', 'mass_scale')
    def _observe_mass_truncnorm(self, change):
        if self.mass == 0 and self.mass_mean > 0 and self.mass_scale > 0:
            self.mass = self._truncnorm(self.mass_mean, self.mass_scale)

    @observe('target_velocity_mean', 'target_velocity_scale')
    def _observe_target_velocity_truncnorm(self, change):
        if self.target_velocity == 0 and self.target_velocity_mean > 0 and self.target_velocity_scale > 0:
            self.target_velocity = self._truncnorm(self.target_velocity_mean,
                                                   self.target_velocity_scale)

    @observe('mass', 'radius')
    def _observe_inertia_rot(self, change):
        if self.inertia_rot == 0 and self.mass > 0 and self.radius > 0:
            inertia = 4.0 * np.pi
            mass = 80.0
            radius = 0.27
            self.inertia_rot = inertia * (self.mass / mass) * (
                                                                  self.radius / radius) ** 2


class TranslationalMotion(HasTraits):
    position = Array(
        default_value=(0, 0),
        dtype=np.float64,
        help='Position',
        symbol=r'\mathbf{x}').valid(shape_validator(2))
    velocity = Array(
        default_value=(0, 0),
        dtype=np.float64,
        help='Velocity',
        symbol=r'\mathbf{v}').valid(shape_validator(2))
    target_direction = Array(
        default_value=(0, 0),
        dtype=np.float64,
        help='Target direction',
        symbol='\mathbf{\hat{e}}_0').valid(shape_validator(2),
                                           length_validator(0, 1))
    force = Array(
        default_value=(0, 0),
        dtype=np.float64,
        help='Force',
        symbol='\mathbf{f}').valid(shape_validator(2))
    force_prev = Array(
        default_value=(0, 0),
        dtype=np.float64,
        help='Previous force',
        symbol='\mathbf{f}_{prev}').valid(shape_validator(2))

    tau_adj = Float(
        default_value=0.5,
        min=0,
        help='Characteristic time for agent adjusting its movement',
        symbol=r'\tau_{adj}')
    k_soc = Float(
        default_value=1.5,
        min=0,
        help='Social force scaling constant',
        symbol=r'k_{soc}')
    tau_0 = Float(
        default_value=3.0,
        min=0,
        help='Interaction time horizon',
        symbol=r'\tau_{0}')
    mu = Float(
        default_value=1.2e5,
        min=0,
        help='Compression counteraction constant',
        symbol=r'\mu')
    kappa = Float(
        default_value=4e4,
        min=0,
        help='Sliding friction constant',
        symbol=r'\kappa')
    damping = Float(
        default_value=500,
        min=0,
        help='Damping coefficient for contact force',
        symbol=r'c_{d}')
    std_rand_force = Float(
        default_value=0.1,
        min=0,
        help='Standard deviation for fluctuation force',
        symbol=r'\xi / m')


class RotationalMotion(HasTraits):
    orientation = Float(
        default_value=0.0,
        min=-np.pi, max=np.pi,
        help='Orientation',
        symbol=r'\varphi')
    angular_velocity = Float(
        default_value=0.0,
        help='Angular velocity',
        symbol=r'\omega')
    target_orientation = Float(
        default_value=0.0,
        help='Target orientation',
        symbol=r'\varphi_0')
    torque = Float(
        default_value=0.0,
        help='Torque',
        symbol=r'M')
    torque_prev = Float(
        default_value=0.0,
        help='Previous torque',
        symbol=r'M_{prev}')

    tau_rot = Float(
        default_value=0.2,
        min=0,
        help='Characteristic time for agent adjusting its rotational movement',
        symbol=r'\tau_{adjrot}')
    std_rand_torque = Float(
        default_value=0.1,
        min=0,
        help='Standard deviation for fluctuation torque',
        symbol=r'\eta / I{rot}')


class AgentType(HasTraits):
    """Mixin for different agent types. Implements some common methods."""
    __slots__ = ()

    @classmethod
    def dtype(cls):
        """Structured numpy.dtype for forming an array of the value of agent
        type.

        Returns:
            numpy.dtype: Numpy structured dtype for the agent type
        """
        return class_to_struct_dtype(cls, None, lambda c: c is BodyType)

    def __array__(self):
        """Array interface for using ``numpy.array`` on the agent type.

        Returns:
            numpy.ndarray:
        """
        dtype = self.dtype()
        values = tuple(getattr(self, field) for field in dtype.fields)
        return np.array([values], dtype=dtype)

    array = __array__

    def overlapping(self, others) -> bool:
        """Determines if agent is overlapping with any of the agent supplied
        in other argument.

        Args:
            others:

        Returns:
            bool:
        """
        raise NotImplementedError

    def overlapping_obstacles(self, obstacles) -> bool:
        raise NotImplementedError

    def from_array(self, array):
        """Set values from array."""
        if len(array) != 1:
            raise ValueError('Array should be length 1')
        for field, value in zip(array.dtype.fields, array.item()):
            setattr(self, field, value)

    def __str__(self):
        return self.__class__.__name__


@interpolate_docstring(**{'table_of_traits': table_of_traits})
class Circular(AgentType, States, BodyType, TranslationalMotion):
    r"""Circular agent type

    .. tikz:: Circular agent
       :include: ../tikz/circular_agent.tex

    **Circular** agents are modelled as a disk with radius :math:`r > 0`
    from the center of mass :math:`\mathbf{x}`. This type of agents do not
    have orientation. This is the simplest model for an agent and works
    quite well for sparse and medium density crowds, but modelling higher
    density crowds with this model can be unrealistic because circular
    model is too wide in the  perpendicular width compared to three-circle
    or capsule representations  and lacks the ability change orientation to
    fit through smaller spaces. [Helbing2000a]_

    %(table_of_traits)s
    """
    def overlapping(self, others):
        return overlapping_circles(others, self.position, self.radius)

    def overlapping_obstacles(self, obstacles):
        return overlapping_circle_line(np.array(self), obstacles)


@interpolate_docstring(**{'table_of_traits': table_of_traits})
class ThreeCircle(AgentType, States, BodyType, TranslationalMotion,
                  RotationalMotion):
    r"""Three-circle agent type

    .. tikz:: Three circle agent
       :include: ../tikz/three_circle_agent.tex

    **Three-circle** agents are modelled as three disks representing the
    torso and two shoulders of an average human. Torso is a disk with radius
    :math:`r_t > 0` from the center of mass :math:`\mathbf{x}`. Two
    shoulders are disks with radius :math:`r_s` located at along the
    tangents at distance :math:`r_{ts}` from the center of mass
    :math:`\mathbf{x} \pm r_{ts} \mathbf{\hat{e}_t}`, where
    :math:`\mathbf{\hat{e}_t} = [\sin(\varphi), -\cos(\varphi)]`. Three
    circle type has orientation of :math:`\varphi`. Model was proposed
    *Crowd dynamics discrete element multi-circle model* [Langston2006]_ and
    has been used for example in FDS+EVAC [Korhonen2008b]_.

    %(table_of_traits)s
    """
    position_ls = Array(
        default_value=(0, 0),
        dtype=np.float64).valid(shape_validator(2))

    position_rs = Array(
        default_value=(0, 0),
        dtype=np.float64).valid(shape_validator(2))

    @default('position_ls')
    def _default_position_ls(self):
        return self.position - self.r_ts * rotate270(
            unit_vector(self.orientation))

    @default('position_rs')
    def _default_position_rs(self):
        return self.position + self.r_ts * rotate270(
            unit_vector(self.orientation))

    def overlapping(self, others):
        return overlapping_three_circles(
            others,
            (self.position, self.position_ls, self.position_rs),
            (self.r_t, self.r_s, self.r_s))

    def overlapping_obstacles(self, obstacles) -> bool:
        return overlapping_three_circle_line(np.array(self), obstacles)


@interpolate_docstring(**{'table_of_traits': table_of_traits})
class Capsule(AgentType, States, BodyType, TranslationalMotion,
              RotationalMotion):
    r"""Capsule

    .. tikz:: Capsule agent
       :include: ../tikz/capsule_agent.tex

    **Capsule** shaped model used in *Dense Crowds of Virtual Humans*
    [Stuvel2016]_ and *Simulating competitive egress of noncircular
    pedestrians* [Hidalgo2017]_.

    .. math::
       r &= T / 2 \\
       w &= W - 2 r

    where

    - :math:`T` is the thickness of the chest
    - :math:`W` is the width of the chest

    %(table_of_traits)s
    """
    pass


agent_type_circular = Circular.dtype()
agent_type_three_circle = ThreeCircle.dtype()
# agent_type_capsule = Capsule.dtype()
AgentTypes = [
    Circular,
    ThreeCircle,
]
AgentModelToType = {
    'circular': agent_type_circular,
    'three_circle': agent_type_three_circle,
}


def is_model(agents, model):
    """Test if agent if type same type as model name

    Args:
        agents (numpy.ndarray):
        model (str):

    Returns:
        bool:
    """
    return hash(agents.dtype) == hash(AgentModelToType[model])


@numba.jit(void(typeof(agent_type_three_circle)[:]),
           nopython=True, nogil=True, cache=True)
def shoulders(agents):
    """Positions of the center of mass, left- and right shoulders.

    Args:
        agents (ndarray):
            Numpy array of datatype ``dtype=agent_type_three_circle``.
    """
    for agent in agents:
        tangent = rotate270(unit_vector(agent['orientation']))
        offset = tangent * agent['r_ts']
        agent['position_ls'][:] = agent['position'] - offset
        agent['position_rs'][:] = agent['position'] + offset


@numba.jit([boolean(typeof(agent_type_circular)[:], float64[:], float64)],
           nopython=True, nogil=True, cache=True)
def overlapping_circles(agents, x, r):
    """Test if two circles are overlapping.

    Args:
        agents:
        x: Position of agent that is tested
        r: Radius of agent that is tested

    Returns:
        bool:
    """
    for agent in agents:
        h, _ = distance_circles(agent['position'], agent['radius'], x, r)
        if h < 0.0:
            return True
    return False


@numba.jit([boolean(typeof(agent_type_three_circle)[:],
                    UniTuple(float64[:], 3), UniTuple(float64, 3))],
           nopython=True, nogil=True, cache=True)
def overlapping_three_circles(agents, x, r):
    """Test if two three-circle models are overlapping.

    Args:
        x1: Positions of other agents
        r1: Radii of other agents
        x: Position of agent that is tested
        r: Radius of agent that is tested

    Returns:
        bool:

    """
    for agent in agents:
        h, _, _, _ = distance_three_circles(
            (agent['position'], agent['position_ls'], agent['position_rs']),
            (agent['r_t'], agent['r_s'], agent['r_s']),
            x, r
        )
        if h < 0:
            return True
    return False


@numba.jit([boolean(typeof(agent_type_circular)[:],
                    typeof(obstacle_type_linear)[:])],
           nopython=True, nogil=True, cache=True)
def overlapping_circle_line(agents, obstacles):
    for agent in agents:
        for obstacle in obstacles:
            h, _ = distance_circle_line(agent['position'], agent['radius'],
                                        obstacle['p0'], obstacle['p1'])
            if h < 0.0:
                return True
    return False


@numba.jit([boolean(typeof(agent_type_three_circle)[:],
                    typeof(obstacle_type_linear)[:])],
           nopython=True, nogil=True, cache=True)
def overlapping_three_circle_line(agents, obstacles):
    for agent in agents:
        for obstacle in obstacles:
            h, _, _ = distance_three_circle_line(
                (agent['position'], agent['position_ls'], agent['position_rs']),
                (agent['r_t'], agent['r_s'], agent['r_s']),
                obstacle['p0'], obstacle['p1']
            )
            if h < 0.0:
                return True
    return False


class AgentGroup(HasTraits):
    """Group of agents

    Examples:
        >>> group = AgentGroup(
        >>>             size=10,
        >>>             agent_type=Circular,
        >>>             attributes=...,
        >>>         )

    """
    agent_type = Type(
        AgentType,
        allow_none=True,
        help='AgentType for generating agent from attributes.')
    size = Int(
        default_value=0,
        help='Size of the agent group. Optional is attributes are instance of '
             'collection')
    attributes = Union(
        (Instance(Collection), Instance(Generator), Instance(Callable)),
        allow_none=True,
        help='Attributes of the chosen agent type.')
    members = List(
        Instance(AgentType),
        help='')

    @observe('size', 'agent_type', 'attributes')
    def _observe_members(self, change):
        if self.size > 0 and self.attributes is not None and self.agent_type is not None:
            if isinstance(self.attributes, Collection):
                self.members = [self.agent_type(**a) for a in self.attributes]
            elif isinstance(self.attributes, Generator):
                self.members = [self.agent_type(**next(self.attributes)) for _ in range(self.size)]
            elif isinstance(self.attributes, Callable):
                self.members = [self.agent_type(**self.attributes()) for _ in range(self.size)]
            else:
                raise TraitError


class Agents(AgentsBase):
    """Set groups of agents

    Examples:
        >>> agent = Agents(agent_type=Circular)
        >>> agent.add_non_overlapping_group(...)

    """
    agent_type = Type(
        klass=AgentType,
        help='Instance of AgentType. This will be used to create attributes '
             'for the agent.')
    size_max = Int(
        allow_none=None,
        help='Maximum number of agents that can be created.None allows the '
             'size grow dynamically.')
    cell_size = Float(
        default_value=0.6, min=0,
        help='Cell size for block list. Value should be little over the '
             'maximum of agent radii')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = 0
        self.array = np.zeros(0, dtype=self.agent_type.dtype())
        # Block list for speeding up overlapping checks
        self._neighbours = MutableBlockList(cell_size=self.cell_size)

    def add_non_overlapping_group(self, group, position_gen, obstacles=None):
        """Add group of agents

        Args:
            group (AgentGroup):
            position_gen (Generator|Callable):
            obstacles (numpy.ndarray):
        """
        if self.agent_type is not group.agent_type:
            raise CrowdDynamicsException

        # resize self.array to fit new agents
        array = np.zeros(group.size, dtype=group.agent_type.dtype())
        self.array = np.concatenate((self.array, array))

        index = 0
        overlaps = 0
        overlaps_max = 10 * group.size

        while index < group.size and overlaps < overlaps_max:
            new_agent = group.members[index]
            new_agent.position = position_gen() if callable(position_gen) \
                else next(position_gen)

            # Overlapping check
            neighbours = self._neighbours.nearest(new_agent.position, radius=1)
            if new_agent.overlapping(self.array[neighbours]):
                # Agent is overlapping other agent.
                overlaps += 1
                continue

            if obstacles is not None and new_agent.overlapping_obstacles(obstacles):
                # Agent is overlapping with an obstacle.
                overlaps += 1
                continue

            # Agent can be successfully placed
            self.array[self.index] = np.array(new_agent)
            self._neighbours[new_agent.position] = self.index
            self.index += 1
            index += 1

        # TODO: remove agents that didn't fit from self.array
        if self.index + 1 < self.array.size:
            pass

        # Array should remain contiguous
        assert self.array.flags.c_contiguous
