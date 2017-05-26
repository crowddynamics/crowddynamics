from shapely.geometry import Polygon, LineString, Point
from traitlets.traitlets import Float, observe, default

from crowddynamics.simulation.field import Field


def _rectangle(x, y, width, height):
    return Polygon(
        [(x, y), (x + width, y), (x + width, y + height), (x, y + height)])


class Outdoor(Field):
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


class Hallway(Field):
    width = Float(
        default_value=10.0,
        min=0)
    height = Float(
        default_value=10.0,
        min=0)
    ratio = Float(
        default_value=1/3,
        min=0, max=1
    )

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
