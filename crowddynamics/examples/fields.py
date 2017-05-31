from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.base import BaseGeometry
from traitlets.traitlets import Float, observe, default

from crowddynamics.simulation.field import Field


def _rectangle(x, y, width, height):
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
        default_value=1/3,
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
        return [_rectangle(0, 0, self.ratio * self.width, self.height),
                _rectangle((1 - self.ratio) * self.width, 0, self.ratio *
                           self.width, self.height)]

    @default('domain')
    def _default_domain(self):
        return self.convex_hull()

    @observe('width', 'height', 'ratio')
    def _observe_field(self, change):
        obstacles = LineString([(0, 0), (self.width, 0)]) | \
                    LineString([(0, self.height), (self.width, self.height)])
        spawn0 = _rectangle(0, 0, self.ratio * self.width, self.height)
        spawn1 = _rectangle((1 - self.ratio) * self.width, 0, self.ratio *
                   self.width, self.height)

        target0 = LineString([(0, 0), (0, self.height)])
        target1 = LineString([(self.width, 0), (self.width, self.height)])

        self.obstacles = obstacles
        self.spawns = [spawn0, spawn1]
        self.targets = [target0, target1]
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

        self.obstacles = obstacles
        self.targets = targets

        domain = self.convex_hull()
        self.spawns = [domain]
        self.domain = domain
