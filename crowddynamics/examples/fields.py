from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.base import BaseGeometry
from traitlets.traitlets import Float, observe, default, Enum

from crowddynamics.simulation.field import Field


def rectangle(x, y, width, height):
    return Polygon(
        [(x, y), (x + width, y), (x + width, y + height), (x, y + height)])


class OutdoorField(Field):
    width = Float(
        default_value=10.0,
        min=0)
    height = Float(
        default_value=10.0,
        min=0)

    @default('domain')
    def _default_domain(self):
        return Polygon([
            (0, 0), (0, self.height),
            (self.width, self.height), (self.width, 0)
        ])

    @default('spawns')
    def _default_spawns(self):
        return [self.domain]

    @observe('width', 'height')
    def _observe_field(self, change):
        self.domain = Polygon([
            (0, 0), (0, self.height),
            (self.width, self.height), (self.width, 0)
        ])
        self.spawns = [self.domain]


class HallwayField(Field):
    width = Float(
        default_value=10.0,
        min=0)
    height = Float(
        default_value=10.0,
        min=0)
    ratio = Float(
        default_value=1 / 3,
        min=0, max=1)

    @default('obstacles')
    def _default_obstacles(self):
        return LineString([(0, 0), (self.width, 0)]) | \
               LineString([(0, self.height), (self.width, self.height)])

    @default('targets')
    def _default_targets(self):
        return [LineString([(0, 0), (0, self.height)]),
                LineString([(self.width, 0), (self.width, self.height)])]

    @default('spawns')
    def _default_spawns(self):
        return [rectangle(0, 0, self.ratio * self.width, self.height),
                rectangle((1 - self.ratio) * self.width, 0, self.ratio *
                          self.width, self.height)]

    @default('domain')
    def _default_domain(self):
        return self.convex_hull()

    @observe('width', 'height', 'ratio')
    def _observe_field(self, change):
        obstacles = LineString([(0, 0), (self.width, 0)]) | \
                    LineString([(0, self.height), (self.width, self.height)])
        spawn0 = rectangle(0, 0, self.ratio * self.width, self.height)
        spawn1 = rectangle((1 - self.ratio) * self.width, 0, self.ratio *
                           self.width, self.height)

        target0 = LineString([(0, 0), (0, self.height)])
        target1 = LineString([(self.width, 0), (self.width, self.height)])

        self.obstacles = obstacles
        self.spawns = [spawn0, spawn1]
        self.targets = [target0, target1]
        self.domain = self.convex_hull()


class Rounding(Field):
    width = Float(
        default_value=10.0,
        min=0)
    height = Float(
        default_value=10.0,
        min=0)
    ratio = Float(
        default_value=0.6,
        min=0, max=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        height = self.height
        width = self.width

        domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
        target = LineString([(0, height / 2), (0, height)])
        spawn = Polygon([(0, 0),
                         (0, height / 2),
                         (width / 2, height / 2),
                         (width / 2, 0)])

        obstacles = LineString([(0, height / 2),
                                (width * self.ratio, height / 2)]) | \
                    domain.exterior - target

        self.obstacles = obstacles
        self.targets = [target]
        self.spawns = [spawn]
        self.domain = domain


class AvoidObstacle(Field):
    width = Float(
        default_value=10.0,
        min=0)
    height = Float(
        default_value=10.0,
        min=0)
    exit_width = Float(
        default_value=3.0,
        min=1.0)
    ratio_obs = Float(
        default_value=0.6,
        min=0, max=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        domain = rectangle(0, 0, self.width, self.height)
        spawn_follower = rectangle(0, 0, self.ratio_obs * self.width, self.height)
        spawn_leader = rectangle(0, 0.5 * self.height,
                                 self.ratio_obs * self.width, 0.5 * self.height)
        target = LineString([(self.width, 0), (self.width, self.exit_width)])

        self.obstacles = (domain.exterior - target |
                          LineString([(self.ratio_obs * self.width, 0),
                                      (self.ratio_obs * self.width, 0.5 * self.height)]))
        self.targets = [target]
        self.spawns = [spawn_leader, spawn_follower]
        self.domain = domain


class ClosedRoom(Field):
    width = Float(
        default_value=10.0,
        min=0)
    height = Float(
        default_value=10.0,
        min=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        domain = Polygon([
            (0, 0), (0, self.height),
            (self.width, self.height), (self.width, 0)
        ])
        obstacles = domain.exterior
        spawn = domain

        self.obstacles = obstacles
        self.spawns = [spawn]
        self.domain = domain


class RoomWithOneExit(Field):
    width = Float(
        default_value=10.0,
        min=0)
    height = Float(
        default_value=10.0,
        min=0)
    exit_width = Float(
        default_value=1.25,
        min=0, max=10)
    exit_hall_width = Float(
        default_value=2.0,
        min=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.width
        height = self.height
        exit_width = self.exit_width
        exit_hall_width = self.exit_hall_width

        self.room = LineString(
            [(width, 0), (0, 0), (0, height), (width, height)])
        self.hall = rectangle(width, (height + exit_width) / 2,
                              exit_hall_width, (height - exit_width) / 2) | \
                    rectangle(width, 0,
                              exit_hall_width, (height - exit_width) / 2)
        target = LineString(
            [(width + exit_hall_width, (height - exit_width) / 2),
             (width + exit_hall_width, (height + exit_width) / 2)])

        # Field attributes
        self.obstacles = self.room | self.hall
        self.targets = [target]
        self.spawns = [self.room.convex_hull]
        self.domain = self.convex_hull()


class FourExitsField(Field):
    exit_width = Float(
        default_value=1.25,
        min=0, max=10)

    def __init__(self, *args, **kwargs):
        super(FourExitsField, self).__init__(*args, **kwargs)

        A = (0, 0)
        B = (87, 0)
        C = (0, 10)
        D = (20, 40)
        E = (20, 0)
        F = (20, 20)
        G = (40, 20)
        H = (60, 0)
        I = (60, 20)

        J = (0, 10 + self.exit_width)
        K = (0, 80)
        L = (6, 80)
        M = (0, 60)
        N = (40, 60)

        O = (6 + self.exit_width, 80)
        P = (100, 80)
        Q = (100, 50 + self.exit_width)
        R = (60, 60)
        S = (100, 60)
        T = (70, 60)
        U = (70, 40)

        V = (87 + self.exit_width, 0)
        W = (100, 0)
        Z = (100, 50)

        obstacles = BaseGeometry()

        obstacles |= LineString([C, A, B])
        obstacles |= LineString([E, D])
        obstacles |= LineString([F, G])
        obstacles |= LineString([H, I])
        obstacles |= LineString([J, K, L])
        obstacles |= LineString([M, N])
        obstacles |= LineString([O, P, Q])
        obstacles |= LineString([S, R])
        obstacles |= LineString([T, U])
        obstacles |= LineString([Z, W, V])

        targets = [LineString([C, J]),
                   LineString([L, O]),
                   LineString([Q, Z]),
                   LineString([B, V])]

        # h1 = LineString([(J[0] - 2, J[1]), J]) | LineString([(C[0] - 2, C[1]), C])
        # obstacles |= h1
        #
        # targets = [
        #     h1.convex_hull,
        # ]

        self.obstacles = obstacles
        self.targets = targets

        spawn = obstacles.convex_hull
        self.spawns = [spawn]
        self.domain = self.convex_hull()


class PillarInTheMiddle(Field):
    width = Float(
        default_value=20.0,
        min=0)
    height = Float(
        default_value=20.0,
        min=0)
    radius_pillar = Float(
        default_value=1.0,
        min=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        domain = rectangle(self.width / 2, self.height / 2,
                           self.width / 2, self.height / 2)
        obstacles = Point(0, 0).buffer(self.radius_pillar)

        self.domain = domain
        self.obstacles = obstacles
        self.spawns = [domain]


class AvoidPillar(Field):
    width = Float(
        default_value=10.0,
        min=0)
    height = Float(
        default_value=20.0,
        min=0)
    pillar_type = Enum(
        values=('ellipse', 'rectangle'),
        default_value='ellipse')
    width_pillar = Float(
        default_value=2.0,
        min=0)
    height_pillar = Float(
        default_value=2.0,
        min=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
