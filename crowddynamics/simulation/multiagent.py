import json
import logging
import multiprocessing
import os
from collections import OrderedDict, Iterable
from datetime import datetime
from functools import lru_cache
from multiprocessing import Process, Event
from typing import Optional

import bokeh.io
import bokeh.plotting
import numpy as np
from anytree.iterators import PostOrderIter
from loggingtools import log_with
from matplotlib.path import Path
from shapely.geometry import Polygon
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from sortedcontainers import SortedSet
from traitlets import HasTraits, Unicode, Instance, default, validate, \
    List

from crowddynamics.config import load_config, MULTIAGENT_CFG, \
    MULTIAGENT_CFG_SPEC, AGENT_CFG, AGENT_CFG_SPEC
from crowddynamics.core.geometry import geom_to_linear_obstacles, union
from crowddynamics.core.integrator import velocity_verlet_integrator
from crowddynamics.core.interactions.block_list import MutableBlockList
from crowddynamics.core.interactions.interactions import \
    agent_agent_block_list, agent_obstacle
from crowddynamics.core.motion.adjusting import force_adjust_agents, \
    torque_adjust_agents
from crowddynamics.core.motion.fluctuation import force_fluctuation, \
    torque_fluctuation
from crowddynamics.core.rand.functions import truncnorm
from crowddynamics.core.rand.sampling import polygon_sample
from crowddynamics.core.steering.navigation import navigation, static_potential, \
    herding
from crowddynamics.core.steering.orientation import \
    orient_towards_target_direction
from crowddynamics.core.structures.agents import is_model, reset_motion, \
    AgentModelToType, shoulders, overlapping_circles, overlapping_three_circles
from crowddynamics.core.tree import Node
from crowddynamics.exceptions import CrowdDynamicsException, InvalidType, \
    AgentStructureFull, OverlappingError, ValidationError
from crowddynamics.io import save_npy, save_csv
from crowddynamics.visualization.bokeh import set_aspect, plot_geom, \
    plot_distance_map, plot_direction_map

Domain = Optional[Polygon]
Obstacles = Optional[BaseGeometry]
Spawn = Polygon
Targets = BaseGeometry


def _truncnorm(mean, abs_scale):
    """Individual value from truncnorm"""
    return np.asscalar(truncnorm(-3.0, 3.0, loc=mean, abs_scale=abs_scale))


def call(obj):
    """Iterate, call or return the value."""
    if hasattr(obj, '__next__'):
        return next(obj)
    elif callable(obj):
        return obj()
    else:
        return obj


class Field(HasTraits):
    r"""Multi-Agent simulation Field consists of

    .. tikz:: Example of Field

       \draw[color=gray!20] (-2, -1) grid (12, 7);
       % Domain
       \fill[gray!20] (0, 0) rectangle (10, 6);
       \node[] () at (5, 3) {$ \Omega $};
       % Spawn 0
       \fill[blue!20] (0, 3) -- ++(2, 0) -- ++(1, 1) -- ++(0, 2) 
                      -- ++(-3, 0) -- ++(0, -3);
       \node[] () at (1.5, 4.5) {$ \mathcal{S}_0 $};
       % Spawn 1
       \fill[blue!20] (3, 0) -- ++(0, 1) -- ++(1, 1) -- ++(2, 0) -- ++(1, -1)
                      -- ++(0, -1);
       \node[] () at (5, 0.5) {$ \mathcal{S}_1 $};
       % Obstacles
       \draw[thick] (0, 0) rectangle (10, 6);
       \draw[fill=black] (9, 2) circle (0.5);
       \draw[fill=black] (9, 4) circle (0.5);
       % Room 1
       \draw[thick] (0, 3) -- ++(2, 0) -- ++(1, 1);
       \draw[thick] (3, 5) -- ++(0, 1);
       % Target 0
       \draw[thick, white] (4, 6) -- ++(2, 0);
       \draw[thick, dashed] (4, 6) -- node[above] {$ \mathcal{E}_0 $} ++(2, 0);
       % Target 1
       \draw[thick, white] (10, 2) -- ++(0, 2);
       \draw[thick, dashed] (10, 2) -- node[right] {$ \mathcal{E}_1 $} ++(0, 2);
       

    Domain
        :tikz:`\draw[black, fill=gray!20] (0, 0) rectangle (0.4, 0.4);`
        **Domain** :math:`\Omega \subset \mathbb{R}^{2}` is a plane that
        contains  all the other objects in the simulation such as agents and
        obstacles. Agents that move outside the domain will be marked as
        inactive and not used to compute any of the simulation logic.

    Obstacles
        :tikz:`\draw[black, fill=black] (0, 0) rectangle (0.4, 0.4);`
        **Obstacles** :math:`\mathcal{O} \subset \Omega` are impassable regions
        of the domain. Agents have have psychological tendency to try to avoid
        colliding with an obstacle, but if they do, for example being pushed by
        other agents, there will be friction force between the agent and the
        obstacles. Obstacles avoidance is handled by a navigation algorithm.
        
    Targets
        :tikz:`\draw[thick, dashed, black] (0, 0) rectangle (0.4, 0.4);`
        **Targets** :math:`\mathcal{E}_i \subset \Omega` for
        :math:`i \in \{0, ..., m-1\}` are passable regions of the domain. Agents
        can have a psychological tendency  to try to reach one or more of these
        regions. This psycological tendency is also handled by a navigation
        algorithm.

    Spawns
        :tikz:`\draw[fill=blue!20] (0, 0) rectangle (0.4, 0.4);`
        **Spawns** :math:`\mathcal{S}_j \subset \Omega` for
        :math:`j \in \{0, ..., n-1\}` are passable regions of the domain. These
        are the regions where new agents can be placed in the beginning or
        during the simulation. Polygon sampling algorithm handles the sampling
        of new potential points for  placing the agent and then algorithm test
        that the agent does not  overlap with other agents of obstacles. If it
        doesn't new agent is  placed here.

    """
    # TODO: invalidate caches if field changes?
    # TODO: validation, spawn should be convex
    name = Unicode()
    domain = Instance(Polygon, allow_none=True)
    obstacles = Instance(BaseGeometry, allow_none=True)
    targets = List(Instance(BaseGeometry))
    spawns = List(Instance(BaseGeometry))

    @default('name')
    def _default_name(self):
        return self.__class__.__name__

    @validate('domain')
    def _valid_domain(self, proposal):
        value = proposal['value']
        if not value.is_valid:
            raise ValidationError('{} should not be invalid'.format(value))
        if value.is_empty:
            raise ValidationError('{} should not empty'.format(value))
        return value

    @validate('obstacles')
    def _valid_obstacles(self, proposal):
        value = proposal['value']
        if not value.is_valid:
            raise ValidationError('{} should not be invalid'.format(value))
        if value.is_empty:
            raise ValidationError('{} should not empty'.format(value))
        return value

    def add_spawns(self, *spawns: Spawn):
        """Add new spawns"""
        for spawn in spawns:
            self.spawns.append(spawn)

    def add_targets(self, *targets: Targets):
        """Add new targets"""
        for target in targets:
            self.targets.append(target)

    def remove_spawn(self, index):
        return self.spawns.pop(index)

    def remove_target(self, index):
        return self.targets.pop(index)

    def convex_hull(self):
        """Convex hull of union of all objects in the field."""
        field = self.obstacles | union(*self.targets) | union(*self.spawns)
        return field.convex_hull

    @staticmethod
    def _samples(spawn, obstacles, radius=0.3):
        """Generates positions for agents"""
        geom = spawn - obstacles.buffer(radius) if obstacles else spawn
        vertices = np.asarray(geom.convex_hull.exterior)
        return polygon_sample(vertices)

    def sample_spawn(self, spawn_index: int, radius: float = 0.3):
        """Generator for sampling points inside spawn without overlapping with
        obstacles"""
        return self._samples(self.spawns[spawn_index], self.obstacles, radius)

    @lru_cache()
    def navigation_to_target(self, index, step, radius, strength):
        if not self.targets:
            raise CrowdDynamicsException('No targets are set.')
        if isinstance(index, (int, np.uint8)):
            targets = self.targets[index]
        elif index == 'closest':
            targets = union(*self.targets)
        else:
            raise InvalidType('Index "{0}" should be integer or '
                              '"closest".'.format(index))

        return static_potential(self.domain, targets, self.obstacles, step,
                                radius, strength)

    def dump_json(self, fname: str):
        """Dump field into JSON"""

        def _mapping(geom):
            if isinstance(geom, Iterable):
                return list(map(mapping, geom))
            else:
                return mapping(geom) if geom else geom

        _, ext = os.path.splitext(fname)
        if ext != '.json':
            fname += '.json'

        with open(fname, 'w') as fp:
            obj = {
                'domain': _mapping(self.domain),
                'obstacles': _mapping(self.obstacles),
                'targets': _mapping(self.targets),
                'spawns': _mapping(self.spawns)
            }
            # TODO: maybe compact layout?
            json.dump(obj, fp, indent=2, separators=(', ', ': '))

    def load_json(self, fname: str):
        """Load field from JSON"""

        def _shape(geom):
            if isinstance(geom, Iterable):
                return list(map(shape, geom))
            else:
                return shape(geom) if geom else geom

        with open(fname, 'r') as fp:
            obj = json.load(fp)
            self.domain = _shape(obj['domain'])
            self.obstacles = _shape(obj['obstacles'])
            self.add_targets(*_shape(obj['targets']))
            self.add_spawns(*_shape(obj['spawns']))

    def plot(self, step=0.02, radius=0.3, strength=0.3, **kwargs):
        bokeh.io.output_file(self.name + '.html', self.name)
        p = bokeh.plotting.Figure(**kwargs)

        if self.domain:
            minx, miny, maxx, maxy = self.domain.bounds
        else:
            minx, miny, maxx, maxy = self.convex_hull().bounds

        set_aspect(p, (minx, maxx), (miny, maxy))
        p.grid.minor_grid_line_color = 'navy'
        p.grid.minor_grid_line_alpha = 0.05

        # indices = chain(range(len(self.targets)), ('closest',))
        # for index in indices:
        #     mgrid, distance_map, direction_map = \
        #         self.navigation_to_target(index, step, radius, strength)

        mgrid, distance_map, direction_map = self.navigation_to_target(
            'closest', step, radius, strength)

        # TODO: masked values on distance map
        plot_distance_map(p, mgrid, distance_map.filled(1.0),
                          legend='distance_map')
        plot_direction_map(p, mgrid, direction_map, legend='direction_map')

        plot_geom(p, self.domain,
                  legend='domain',
                  alpha=0.05,
                  )

        for i, spawn in enumerate(self.spawns):
            plot_geom(p, spawn,
                      legend='spawn_{}'.format(i),
                      alpha=0.5,
                      line_width=0,
                      color='green',
                      )

        for i, target in enumerate(self.targets):
            plot_geom(p, target,
                      legend='target_{}'.format(i),
                      alpha=0.8,
                      line_width=3.0,
                      line_dash='dashed',
                      color='olive',
                      )

        plot_geom(p, self.obstacles,
                  legend='obstacles',
                  line_width=3.0,
                  alpha=0.8,
                  )

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        bokeh.io.show(p)


class Agents(object):
    r"""Multi-Agent simulation agent types. Modelled as rigid bodies.
     
    Circular
        .. tikz::
           \begin{scope}[scale=3]
             \draw[color=gray!20] (-2, -1) grid (2, 1);
             \draw[thick] (0, 0) circle (1);
             \fill[color=gray!20, opacity=0.5] (0, 0) circle (1);
             \node[below] () at (0, 0) {$ \mathbf{x} $};
             \fill (0, 0) circle(0.5pt);
             \draw[dashed, <->] (0, 0) -- node[above] {$ r $} ++(135:1);
           \end{scope}

        **Circular** agents are modelled as a disk with radius :math:`r > 0`
        from the center of mass :math:`\mathbf{x}`. This type of agents do not
        have orientation. This is the simplest model for an agent and works
        quite well for sparse and medium density crowds, but modelling higher
        density crowds with this model can be unrealistic because circular
        model is too wide in the  perpendicular width compared to three-circle
        or capsule representations  and lacks the ability change orientation to
        fit through smaller spaces. [Helbing2000a]_

    Three-Circle
        .. tikz:: 
           \begin{scope}[scale=3]
             \draw[color=gray!20] (-2, -1) grid (2, 1);
             \draw[thick] (0, 0) circle (0.59);
             \draw[thick] (-0.63, 0) circle (0.37);
             \draw[thick] (0.63, 0) circle (0.37);
             \fill[gray!20, opacity=0.5] (0, 0) circle (0.59);
             \fill[gray!20, opacity=0.5] (-0.63, 0) circle (0.37);
             \fill[gray!20, opacity=0.5] (0.63, 0) circle (0.37);
             \node[below] () at (0, 0) {$ \mathbf{x} $};
             \fill (0, 0) circle(0.5pt);
             \fill (-0.63, 0) circle(0.5pt);
             \fill (0.63, 0) circle(0.5pt);
             \draw[dashed, <->] (0, 0) -- node[above] {$ r_{t} $} ++(135:0.59);
             \draw[dashed, <->] (-0.63, 0) -- node[above] {$ r_{s} $} ++(135:0.37);
             \draw[dashed, <->] (0.63, 0) -- node[above] {$ r_{s} $} ++(45:0.37);
             \draw[dashed, <->] (0, 0) -- node[above] {$ r_{ts} $} (-0.63, 0);
             \draw[dashed, <->] (0, 0) -- node[above] {$ r_{ts} $} (0.63, 0);
             %\draw[thick, ->] (0, 0) -- node[left] {$ \mathbf{\hat{e}_n} $} (0, 0.3);
             %\draw[thick, ->] (0, 0) -- node[below] {$ \mathbf{\hat{e}_t} $} (0.3, 0);
           \end{scope}

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

    Capsule
        .. note:: Capsule is not implemented yet

        .. tikz::
           \begin{scope}[scale=3]
             \draw[color=gray!20] (-2, -1) grid (2, 1);
             \draw[thick] (0.5, 0.5) arc (90:-90:0.5) 
                          -- ++(-1, 0) arc (270:90:0.5) 
                          -- ++(1, 0);
              \fill[gray!20, opacity=0.5] (0.5, 0.5) arc (90:-90:0.5) 
                          -- ++(-1, 0) arc (270:90:0.5) 
                          -- ++(1, 0);
             \node[below] () at (0, 0) {$ \mathbf{x} $};
             \fill (0, 0) circle(0.5pt);
             \fill (-0.5, 0) circle(0.5pt);
             \fill (0.5, 0) circle(0.5pt);
             \draw[thick] (-0.5, 0) -- (0.5, 0);
             \draw[dashed, <->] (-0.5, 0) -- node[above] {$ r $} ++(135:0.5);
             \draw[dashed, <->] (-0.5, -0.2) -- node[below]{$ w $} (0.5, -0.2);
             %\draw[thick, ->] (0, 0) -- node[left] {$ \mathbf{\hat{e}_n} $} (0, 0.3);
             %\draw[thick, ->] (0, 0) -- node[below] {$ \mathbf{\hat{e}_t} $} (0.3, 0);
           \end{scope}

        **Capsule** shaped model used in *Dense Crowds of Virtual Humans* 
        [Stuvel2016]_ and *Simulating competitive egress of noncircular 
        pedestrians* [Hidalgo2017]_.
        
        .. math::
           r &= T / 2 \\
           w &= W - 2 r
        
        where 
        
        - :math:`T` is the thickness of the chest
        - :math:`W` is the width of the chest

    """
    logger = logging.getLogger(__name__)
    # TODO: remove size -> make it resize with addition of new agents
    # TODO: traits: agent_type, body_types, configuration
    # TODO: set attributes -> attributes to Callable[_, dict]

    def __init__(self,
                 size,
                 agent_type='circular',
                 agent_cfg=AGENT_CFG,
                 agent_cfg_spec=AGENT_CFG_SPEC):
        """Agent manager

        Args:
            size (int):
                Number of agents
            agent_type (str|numpy.dtype):
                AgentModelToType
            agent_cfg:
                Agent configuration filepath
            agent_cfg_spec:
                Agent configuration spec filepath
        """
        if isinstance(agent_type, str):
            dtype = AgentModelToType[agent_type]
        else:
            dtype = agent_type

        # Keeps track of which agents are active and which in active. Stores
        # indices of agents.
        self._active = SortedSet()
        self._inactive = SortedSet(range(size))

        self.array = np.zeros(size, dtype=dtype)
        self.config = load_config(infile=agent_cfg, configspec=agent_cfg_spec)

        # Faster check for neighbouring agents for initializing agents into
        # random positions.
        self._grid = MutableBlockList(
            cell_size=self.config['constants']['cell_size'])

    @staticmethod
    def reset_agent(index, agents):
        """Reset agent"""
        agents[index] = np.zeros(1, agents.dtype)

    @staticmethod
    def body_to_values(body):
        """Body to values"""
        radius = _truncnorm(body['radius'], body['radius_scale'])
        mass = _truncnorm(body['mass'], body['mass_scale'])
        # Rotational inertia of mass 80 kg and radius 0.27 m agent.
        # Should be scaled to correct value for agents.
        inertia_rot = 4.0 * np.pi * (mass / 80.0) * (radius / 0.27) ** 2
        return {
            'r_t': body['ratio_rt'] * radius,
            'r_s': body['ratio_rs'] * radius,
            'r_ts': body['ratio_ts'] * radius,
            'radius': radius,
            'target_velocity': _truncnorm(body['velocity'],
                                          body['velocity_scale']),
            'mass': mass,
            'target_angular_velocity': 4 * np.pi,
            'inertia_rot': inertia_rot
        }

    @staticmethod
    def set_agent_attributes(agents, index, attributes):
        """Set attributes for agent

        Args:
            agents:
            attributes:
                Dictionary values, iterators or callables.
                - Iterator; return value of next is used
                - Callable: return value of __call__ will be used

        """
        for attribute, value in attributes.items():
            try:
                agents[index][attribute] = call(value)
            except ValueError:
                # logger = logging.getLogger(__name__)
                # logger.warning('Agent: {} doesn\'t have attribute {}'.format(
                #     of_model(agents), attribute
                # ))
                pass

    def _add(self, attributes, check_overlapping=True):
        """Add new agent with given attributes

        Args:
            check_overlapping (bool):
            attributes:

                - body_type:
                - position: Initial position of the agent
                - orientation: Initial orientation [-π, π] of the agent.
                - velocity: Initial velocity
                - angular_velocity: Angular velocity
                - target_direction: Unit vector to desired direction
                - target_orientation:

        Raises:
            AgentStructureFull: When no more agents can be added.
            OverlappingError: When two agents overlap each other.
        """
        try:
            index = self._inactive.pop(0)
        except IndexError:
            raise AgentStructureFull

        attrs = dict(**self.config['defaults'])
        attrs.update(attributes)

        try:
            body_type = call(attrs['body_type'])
            body = self.config['body_types'][body_type]
            attrs.update(self.body_to_values(body))
            del attrs['body_type']
        except KeyError:
            pass

        self.set_agent_attributes(self.array, index, attrs)

        # Update shoulder positions for three circle agents
        if is_model(self.array, 'three_circle'):
            shoulders(self.array)

        agent = self.array[index]
        radius = agent['radius']
        position = agent['position']

        if check_overlapping:
            neighbours = self._grid.nearest(position, radius=1)
            agents = self.array[neighbours]

            if is_model(self.array, 'circular'):
                if overlapping_circles(agents, position, radius):
                    self.reset_agent(index, self.array)
                    self._inactive.add(index)
                    raise OverlappingError
            elif is_model(self.array, 'three_circle'):
                if overlapping_three_circles(
                        agents,
                        (agent['position'], agent['position_ls'],
                         agent['position_rs']),
                        (agent['r_t'], agent['r_s'], agent['r_s'])):
                    self.reset_agent(index, self.array)
                    self._inactive.add(index)
                    raise OverlappingError
            else:
                raise CrowdDynamicsException

        self._grid[position] = index
        self._active.add(index)
        self.array[index]['active'] = True
        return index

    @log_with(qualname=True, timed=True, ignore=('self',))
    def add_group(self, size, attributes, check_overlapping=True):
        """Add group of agents

        Args:
            check_overlapping:
            size (int):
            attributes (dict|Callable[dict]):
        """
        overlaps = 0
        iteration = 0
        iterations_max = 10 * size

        # Progress bar for displaying number of agents placed
        # FIXME: tqdm does not work with tests
        # success = tqdm(desc='Agents placed: ', total=amount)
        # overlapping = tqdm(desc='Agent overlaps: ', total=iterations_max)
        while iteration < size and overlaps < iterations_max:
            a = attributes() if callable(attributes) else attributes
            try:
                index = self._add(a, check_overlapping=check_overlapping)
                # success.update(1)
            except OverlappingError:
                overlaps += 1
                # overlapping.update(1)
                continue
            except AgentStructureFull:
                break
            iteration += 1
        # success.close()
        # overlapping.close()

        if is_model(self.array, 'three_circle'):
            shoulders(self.array)

    @log_with(qualname=True, timed=True, ignore=('self',))
    def remove(self, index):
        """Remove agent"""
        if index in self._active:
            self._active.remove(index)
            self._inactive.add(index)
            self.array[index]['active'] = False
            self.reset_agent(index, self.array)
            return True
        else:
            return False


class LogicNode(Node):
    """Simulation logic is programmed as a tree of dependencies of the order of
    the execution. For example simulation's logic tree could look like::

        Reset
        └── Integrator
            ├── Fluctuation
            ├── Adjusting
            │   ├── Navigation
            │   └── Orientation
            ├── AgentAgentInteractions
            └── AgentObstacleInteractions

    In this tree we can notice the dependencies. For example before using
    updating `Adjusting` node we need to update `Navigation` and `Orientation`
    nodes.
    """

    def __init__(self, simulation):
        super(LogicNode, self).__init__()
        self.simulation = simulation


class MultiAgentSimulation(HasTraits):
    r"""Constructing a multi-agent simulation

    Field
        Instance of :class:`Field`.

    Agents
        Instance of :class:`Agents`.

    Logic
        **Logic** of the simulation consists of tree of :class:`LogicNode`.
        Simulation is updated by calling the update function of  each logic node
        using *post-order* traversal.

    """
    name = Unicode()
    field = Instance(Field)
    agents = Instance(Agents)
    tasks = Instance(LogicNode)

    def __init__(self, configfile=MULTIAGENT_CFG, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configs = load_config(configfile, MULTIAGENT_CFG_SPEC)

        # Generated simulation data that is shared between task nodes.
        # This should be data that can be updated and should be saved on
        # every iteration.
        self.data = OrderedDict()
        self.data['iterations'] = 0
        self.data['time_tot'] = 0.0
        self.data['dt'] = 0.0
        self.data['goal_reached'] = 0

    @default('name')
    def _default_name(self):
        return self.__class__.__name__

    @lru_cache()
    def name_with_timestamp(self):
        """Simulation name with timestamp. First call is cached."""
        return self.name + '_' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')

    def update(self):
        """Execute new iteration cycle of the simulation."""
        for node in PostOrderIter(self.logic.root):
            node.update()
        self.data['iterations'] += 1

    @log_with(qualname=True, ignore={'self'})
    def run(self, exit_condition=lambda _: False):
        """Updates simulation until exit condition is met (returns True).

        Args:
            exit_condition (Callable[MultiAgentSimulation, bool]):
        """
        while not exit_condition(self):
            self.update()

    def dump_config(self):
        raise NotImplementedError

    def plot_logic(self, fname):
        from anytree.dotexport import RenderTreeGraph
        name, ext = os.path.splitext(fname)
        if ext != '.png':
            fname += '.png'
        RenderTreeGraph(self.logic.root).to_picture(fname)


class MultiAgentProcess(Process):
    """Class for running MultiAgentSimulation in a new process."""
    logger = logging.getLogger(__name__)

    class EndProcess(object):
        """Marker for end of simulation"""

    def __init__(self, simulation, queue):
        """MultiAgentProcess

        Examples:
            >>> process = MultiAgentProcess(simulation, queue)
            >>> process.start()  # Starts the simulation
            >>> ...
            >>> process.stop()  # Stops the simulation

        Args:
            simulation (MultiAgentSimulation):
            queue (multiprocessing.Queue):
        """
        super(MultiAgentProcess, self).__init__()
        self.simulation = simulation
        self.exit = Event()
        self.queue = queue

    @log_with(qualname=True)
    def run(self):
        """Runs simulation process by calling update method repeatedly until
        stop is called. This method is called automatically by Process class
        when start is called."""
        try:
            self.simulation.run(exit_condition=lambda _: self.exit.is_set())
        except CrowdDynamicsException as error:
            self.logger.error(
                'Simulation stopped to error: {}'.format(error))
            self.stop()
        self.queue.put(self.EndProcess)

    @log_with(qualname=True)
    def stop(self):
        """Sets event to true in order to stop the simulation process."""
        self.exit.set()


class Reset(LogicNode):
    def update(self):
        reset_motion(self.simulation.agents.array)
        # TODO: reset agent neighbor


class Integrator(LogicNode):
    def update(self):
        dt_min = self.simulation.configs['integrator']['dt_min']
        dt_max = self.simulation.configs['integrator']['dt_max']
        dt = velocity_verlet_integrator(self.simulation.agents.array,
                                        dt_min, dt_max)
        self.simulation.data['dt'] = dt
        self.simulation.data['time_tot'] += dt


class Fluctuation(LogicNode):
    def update(self):
        agents = self.simulation.agents.array
        agents['force'] += force_fluctuation(agents['mass'],
                                             agents['std_rand_force'])
        if is_model(agents, 'three_circle'):
            agents['torque'] += torque_fluctuation(agents['inertia_rot'],
                                                   agents['std_rand_torque'])


class Adjusting(LogicNode):
    def update(self):
        agents = self.simulation.agents.array
        force_adjust_agents(agents)
        if is_model(agents, 'three_circle'):
            torque_adjust_agents(agents)


class AgentAgentInteractions(LogicNode):
    def update(self):
        agent_agent_block_list(self.simulation.agents.array)


class AgentObstacleInteractions(LogicNode):
    def update(self):
        agents = self.simulation.agents.array
        obstacles = geom_to_linear_obstacles(self.simulation.field.obstacles)
        agent_obstacle(agents, obstacles)


class Navigation(LogicNode):
    def update(self):
        step = self.simulation.configs['navigation']['step']
        radius = radius = self.simulation.configs['navigation']['radius']
        value = value = self.simulation.configs['navigation']['strength']
        agents = self.simulation.agents.array
        targets = agents['target']
        for target in set(targets):
            mgrid, distance_map, direction_map = \
                self.simulation.field.navigation_to_target(
                    target, step, radius, value)
            navigation(agents, targets == target, mgrid, direction_map)


class Herding(LogicNode):
    def update(self, *args, **kwargs):
        sight_herding = 3.0
        agents = self.simulation.agents.array
        herding(agents, agents['herding'], sight_herding)


class Orientation(LogicNode):
    def update(self):
        if is_model(self.simulation.agents.array, 'three_circle'):
            orient_towards_target_direction(self.simulation.agents.array)


def _save_condition(simulation, frequency=100):
    return (simulation.data['iterations'] + 1) % frequency == 0


class SaveSimulationData(LogicNode):
    def __init__(self, simulation, directory, save_condition=_save_condition):
        super().__init__(simulation)
        self.save_condition = save_condition
        self.directory = os.path.join(directory,
                                      self.simulation.name_with_timestamp())
        os.makedirs(self.directory)

        self.simulation.field.dump_geometry(
            os.path.join(self.directory, 'geometry.json'))

        self.save_agent_npy = save_npy(self.directory, 'agents')
        self.save_agent_npy.send(None)

        self.save_data_csv = save_csv(self.directory, 'data')
        self.save_data_csv.send(None)

    def update(self):
        save = self.save_condition(self.simulation)

        self.save_agent_npy.send(self.simulation.agents.array)
        self.save_agent_npy.send(save)

        self.save_data_csv.send(self.simulation.data)
        self.save_data_csv.send(save)


def save_simulation_data(simulation, directory):
    node = SaveSimulationData(simulation, directory)
    simulation.logic['Reset'].inject_before(node)


def contains(simulation, vertices, state):
    """Contains

    Args:
        simulation (MultiAgentSimulation):
        vertices (numpy.ndarray): Vertices of a polygon
        state (str):

    Yields:
        int: Number of states that changed
    """
    geom = Path(vertices)
    old_state = simulation.agents.array[state]
    while True:
        position = simulation.agents.array['position']
        new_state = geom.contains_points(position)
        simulation.agents.array[state][:] = new_state
        changed = old_state ^ new_state
        old_state = new_state
        yield np.sum(changed)


class InsideDomain(LogicNode):
    def __init__(self, simulation):
        super().__init__(simulation)
        # TODO: handle domain is None
        self.gen = contains(simulation,
                            np.asarray(self.simulation.field.domain.exterior),
                            'active')

    def update(self, *args, **kwargs):
        self.simulation.data['goal_reached'] += next(self.gen)
