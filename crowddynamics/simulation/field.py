from functools import lru_cache

import bokeh.io
import bokeh.plotting
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from traitlets import Instance, List, validate

from crowddynamics.core.geometry import union
from crowddynamics.core.rand.sampling import polygon_sample
from crowddynamics.core.steering.navigation import static_potential
from crowddynamics.exceptions import ValidationError, CrowdDynamicsException, \
    InvalidType
from crowddynamics.simulation.base import FieldBase
from crowddynamics.visualizations import set_aspect, add_geom, add_distance_map, \
    add_direction_map


class Domain(object):
    pass


class Obstacles(object):
    pass


class Target(object):
    pass


class Spawn(object):
    pass


class Field(FieldBase):
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
    domain = Instance(
        Polygon,
        allow_none=True,
        help='Domain')
    obstacles = Instance(
        BaseGeometry,
        allow_none=True,
        help='Obstacles')
    targets = List(
        Instance(BaseGeometry),
        help='List of targets')
    spawns = List(
        Instance(BaseGeometry),
        help='List of spawns')

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

    # TODO: invalidate caches if field changes?
    @lru_cache()
    def navigation_to_target(self, index, step, radius, strength):
        if not self.targets:
            raise CrowdDynamicsException('No targets are set.')
        if isinstance(index, (int, np.int64)):
            targets = self.targets[index]
        elif index == 'closest':
            targets = union(*self.targets)
        else:
            raise InvalidType('Index "{0}" should be integer or '
                              '"closest".'.format(index))

        return static_potential(self.domain, targets, self.obstacles, step,
                                radius, strength)

    # TODO: Move to visualizations
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
        add_distance_map(p, mgrid, distance_map.filled(1.0),
                         legend='distance_map')
        add_direction_map(p, mgrid, direction_map, legend='direction_map')

        add_geom(p, self.domain,
                 legend='domain',
                 alpha=0.05,
                 )

        for i, spawn in enumerate(self.spawns):
            add_geom(p, spawn,
                     legend='spawn_{}'.format(i),
                     alpha=0.5,
                     line_width=0,
                     color='green',
                     )

        for i, target in enumerate(self.targets):
            add_geom(p, target,
                     legend='target_{}'.format(i),
                     alpha=0.8,
                     line_width=3.0,
                     line_dash='dashed',
                     color='olive',
                     )

        add_geom(p, self.obstacles,
                 legend='obstacles',
                 line_width=3.0,
                 alpha=0.8,
                 )

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        bokeh.io.show(p)
