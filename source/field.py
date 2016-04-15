import numpy as np

from source.core.positions import set_positions


def construct_round_wall(round_params):
    """

    :param round_params: Iterable of center point and radius.
    :return: Round wall parameters in array.
    """
    # TODO: Arc, start - end angles
    rows = len(round_params)
    cols = 3
    wall = np.zeros((rows, cols), dtype=np.float64)
    for i, (p_0, r_0) in enumerate(round_params):
        p_0 = np.array(p_0)
        r_0 = np.array(r_0)
        wall[i, :] = np.hstack([p_0, r_0])
    return wall


def construct_linear_wall(linear_params):
    """

    :param linear_params: Iterable of start and end points.
    :return: Linear wall parameters in array.
    """
    # 90 degree counterclockwise rotation
    rot90 = np.array([[0, -1], [1,  0]])

    rows = len(linear_params)
    cols = 9

    wall = np.zeros((rows, cols), dtype=np.float64)
    for i, p in enumerate(linear_params):
        p = np.array(p)
        d = p[1] - p[0]
        l_w = np.hypot(d[1], d[0])  # Length of the wall
        if l_w == 0:
            raise ValueError(
                "P_0 = {} must not be equal to P-1 = {}".format(p[0], p[1]))
        t_w = d / l_w                # Tangential (unit)vector
        n_w = np.dot(rot90, t_w)     # Normal (unit)vector
        l_w = np.array([l_w])
        wall[i, :] = np.hstack([p[0], p[1], t_w, n_w, l_w])
    return wall


def set_walls(round_params, linear_params):
    wall = {
        'round_wall': construct_round_wall(round_params),
        'linear_wall': construct_linear_wall(linear_params)
    }
    return wall


def set_velocities(amount):
    """
    Set velocities.
    """
    orientation = np.random.uniform(0, 2 * np.pi, amount)
    velocity = np.stack((np.cos(orientation), np.sin(orientation)), axis=1)
    return velocity


def set_uniform(arg, amount):
    if isinstance(arg, (list, tuple)):
        if len(arg) == 1:
            return float(arg[0])
        elif len(arg) == 2:
            return np.random.uniform(arg[0], arg[1], size=amount)
        else:
            raise ValueError("Too many elements in iterable.")
    elif isinstance(arg, (int, float)):
        return float(arg)
    elif isinstance(arg, np.ndarray):
        return arg
    else:
        raise ValueError("Argument is wrong type.")


def set_agents(mass, radius, goal_velocity, amount, x_dims, y_dims, walls):
    agent = dict()
    agent['mass'] = set_uniform(mass, amount)
    agent['radius'] = set_uniform(radius, amount)
    agent['position'] = set_positions(amount, x_dims, y_dims, agent['radius'],
                                      walls)
    agent['velocity'] = set_velocities(amount)
    agent['goal_velocity'] = goal_velocity
    agent['goal_direction'] = set_velocities(amount)
    return agent


def set_field(field_params, wall_params, agent_params):
    """
    Set Walls and agents.
    """
    walls = set_walls(**wall_params)
    kwargs = dict(agent_params, **field_params)
    agents = set_agents(walls=walls, **kwargs)
    return agents, walls
