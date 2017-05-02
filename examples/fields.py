from shapely.geometry import Polygon, LineString

from crowddynamics.simulation.multiagent import Field


def _rectangle(x, y, width, height):
    return Polygon(
        [(x, y), (x + width, y), (x + width, y + height), (x, y + height)])


def outdoor(width, height):
    field = Field('Outdoor')
    field.domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    field.add_spawns(field.domain)
    return field


def hallway(width, height):
    field = Field('Hallway')
    field.obstacles = LineString([(0, 0), (width, 0)]) | \
                      LineString([(0, height), (width, height)])
    k = 1 / 3
    spawn0 = _rectangle(0, 0, k * width, height)
    spawn1 = _rectangle((1 - k) * width, 0, k * width, height)
    target0 = LineString([(0, 0), (0, height)])
    target1 = LineString([(width, 0), (width, height)])
    field.add_spawns(spawn0, spawn1)
    field.add_targets(target0, target1)
    field.set_domain_convex_hull()
    return field


def rounding(width, height):
    field = Field('Rounding')
    field.domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    target = LineString([(0, height / 2), (0, height)])
    spawn = Polygon([(0, 0),
                     (0, height / 2),
                     (width / 2, height / 2),
                     (width / 2, 0)])
    field.obstacles = LineString([(0, height / 2),
                                  (width * 3 / 4, height / 2)]) | \
                      field.domain.exterior - target
    field.add_targets(target)
    field.add_spawns(spawn)
    return field


def uturn(width, height):
    field = Field('U-turn')
    field.domain = Polygon([(0, -height / 2), (0, height / 2),
                            (width, height / 2), (width, -height / 2)])
    b = 0.9 * height / 2
    b2 = 0.2 * height / 2
    target = LineString([(0.0, b2), (0.0, b)])
    spawn = Polygon([(0, -b2), (0, -b), (width / 4, -b), (width / 4, -b2)])
    field.obstacles = field.domain - \
                      LineString([(0, 0), (0.95 * (width - b), 0)]).buffer(b) | \
                      LineString([(0, 0), (0.95 * (width - b), 0)]).buffer(b2) & \
                      field.domain
    field.add_targets(target)
    field.add_spawns(spawn)
    return field


def room_with_exit(width, height, door_width, exit_hall_width):
    field = Field('RoomWithExit')
    room = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    hall = Polygon([(width, (height - door_width) / 2),
                    (width, (height + door_width) / 2),
                    (width + exit_hall_width, (height + door_width) / 2),
                    (width + exit_hall_width, (height - door_width) / 2)])
    targets = LineString([(width + exit_hall_width, (height - door_width) / 2),
                          (width + exit_hall_width, (height + door_width) / 2)])
    field.domain = room | hall
    field.obstacles = (room | hall).exterior - targets
    field.add_targets(targets)
    field.add_spawns(room)
    return field


def room_with_two_exits():
    pass


def crossing(l, d, u, v, k=1/3):
    field = Field('Crossing')
    field.obstacles = _rectangle(0, 0, l, d) | \
                      _rectangle(l + u, 0, l, d) | \
                      _rectangle(l + u, d + v, l, d) | \
                      _rectangle(0, d + v, l, d)
    rects = (_rectangle(0, d, k * l, v),
             _rectangle(l, 0, u, k * d),
             _rectangle(l + u + (1 - k) * l, d, k * l, v),
             _rectangle(l, d + v + (1 - k) * d, u, k * d))
    field.add_spawns(*rects)
    field.add_targets(*rects)
    field.set_domain_convex_hull()
    return field


def bottleneck(width, height, d, l, h):
    field = Field('RoomWithExit')
    a = (width - 2 * h - l) / 2
    field.obstacles = LineString([(0, 0), (width, 0)]) | \
                      LineString([(0, height), (width, height)]) | \
                      Polygon([(a, 0), (a+h, d), (a+h+l, d), (a+h+l+h, 0)]) | \
                      Polygon([(a, height), (a+h, height-d),
                               (a+h+l, height-d), (a+h+l+h, height)])
    k = 1 / 3
    rects = (_rectangle(0, 0, k*a, height),
             _rectangle(width-k*a, 0, k*a, height))
    field.add_spawns(*rects)
    field.add_targets(*rects)
    field.set_domain_convex_hull()
    return field
