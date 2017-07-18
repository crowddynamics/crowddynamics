from functools import lru_cache

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from traitlets import Instance, List, validate

from crowddynamics.core.geometry import union
from crowddynamics.core.sampling import polygon_sample
from crowddynamics.core.steering.obstacle_handling import \
    direction_map_obstacles, obstacle_handling
from crowddynamics.core.steering.quickest_path import meshgrid, shortest_path
from crowddynamics.exceptions import ValidationError, CrowdDynamicsException, \
    InvalidType
from crowddynamics.simulation.base import FieldBase


class Field(FieldBase):
    r"""Field is a collection of static geometric objects that can
    exist in crowd dynamics simulations. This module uses geometric types
    from Shapely_ module. Shapely's fundamental geometric types are *points*,
    *curves* and *surfaces*, which correspond to shapely objects

    - ``Point``: Individual point
    - ``LineString``: Curve created by set of points connected by line
      segments. Closed ``LineString`` has special type ``LinearRing``.
    - ``Polygon``: Surface created by set of points covered by closed curve
      excluding points that belong to holes that the surface may have.

    .. _shapely: http://toblerity.org/shapely/manual.html
    .. _spatial data model: http://toblerity.org/shapely/manual.html#spatial-data-model

    .. tikz:: Example of Field
        :include: ../docs/tikz/field_example.tex

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
        **Targets** :math:`\mathcal{T}_i \subset \Omega` for
        :math:`i \in T = \{0, ..., m-1\}` are passable regions of the domain. Agents
        can have a psychological tendency  to try to reach one or more of these
        regions. This psycological tendency is also handled by a navigation
        algorithm. Targets can for examples be exit doors denoted
        :math:`\mathcal{E} \subset \mathcal{T}_{i \in T}`.

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
    # TODO: classes?
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

    # TODO: invalidate caches if field changes?
    # TODO: implement direction and distance map as lazy properties

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
        field = BaseGeometry()
        if self.obstacles:
            field |= self.obstacles
        if self.targets:
            field |= union(*self.targets)
        if self.spawns:
            field |= union(*self.spawns)
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
    def meshgrid(self, step):
        if self.domain is None:
            raise CrowdDynamicsException(
                'Domain cannot be dicretized if it is None.')
        return meshgrid(step, *self.domain.bounds)

    @lru_cache()
    def shortest_path_target(self, step, index, radius):
        if isinstance(index, (int, np.int64)):
            targets = self.targets[index]
        elif index == 'closest':
            targets = union(*self.targets)
        else:
            raise InvalidType('Index "{0}" should be integer or '
                              '"closest".'.format(index))

        return shortest_path(self.meshgrid(step), self.domain, targets,
                             self.obstacles, radius)

    @lru_cache()
    def direction_map_obstacles(self, step):
        return direction_map_obstacles(self.meshgrid(step), self.obstacles)

    @lru_cache()
    def navigation_to_target(self, index, step, radius, strength):
        if not self.targets:
            raise CrowdDynamicsException('No targets are set.')

        dir_map_targets, dmap_targets = self.shortest_path_target(step, index, radius)
        dir_map_obs, dmap_obs = self.direction_map_obstacles(step)
        dir_map = obstacle_handling(dmap_obs, dir_map_obs, dir_map_targets,
                                    radius, strength)
        return self.meshgrid(step), dmap_targets, dir_map
