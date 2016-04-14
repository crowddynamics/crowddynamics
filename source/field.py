import numpy as np

from source.core.positions import set_positions


def round_wall(p_0, r):
    # TODO: Arc
    p_0 = np.array(p_0)
    return p_0, r


def linear_wall(p):
    p_0, p_1 = p
    if p_0 == p_1:
        raise ValueError("{} must not be equal to {}".format(p_0, p_1))
    rot90 = np.array([[0, -1], [1,  0]])  # 90 degree counterclockwise rotation
    p_0 = np.array(p_0)
    p_1 = np.array(p_1)
    d = p_1 - p_0
    l_w = np.sqrt(np.dot(d, d))  # Length of the wall
    t_w = d / l_w                # Tangential (unit)vector
    n_w = np.dot(rot90, t_w)     # Normal (unit)vector
    return p_0, p_1, t_w, n_w, l_w


def set_walls(round_params, linear_params):
    wall = {
        'round': list(map(round_wall, round_params)),
        'linear': list(map(linear_wall, linear_params))
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


def set_agents(mass, radius, goal_velocity, amount, x_dims, y_dims, walls=None):
    agent = dict()
    agent['mass'] = set_uniform(mass, amount)
    agent['radius'] = set_uniform(radius, amount)
    agent['position'] = set_positions(amount, x_dims, y_dims, radius, walls)
    agent['velocity'] = set_velocities(amount)
    agent['goal_velocity'] = goal_velocity
    return agent


def set_field(amount, x_dims, y_dims, agent_params, wall_params, seed=None):
    """
    Set Walls and agents.
    """
    np.random.seed(seed)
    wall = set_walls()
    agent = set_agents()
