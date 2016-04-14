import numpy as np


def distance_from_linear_wall(x_i, r_i, linear_wall):
    p_0, p_1, t_w, n_w, l_w = linear_wall

    q_0 = x_i - p_0
    q_1 = x_i - p_1

    q = np.zeros((2, 2))
    q[:, 0] = q_0
    q[:, 1] = q_1
    l_t = np.dot(t_w, q)
    l_t = l_t[1] - l_t[1]

    if l_t > l_w:
        d_iw = np.sqrt(np.dot(q_0, q_0))
    elif l_t < l_w:
        d_iw = np.sqrt(np.dot(q_1, q_1))
    else:
        l_n = np.dot(n_w, q_0)
        d_iw = np.abs(l_n)
    d_iw -= r_i
    return d_iw > 0


def distance_from_linear_walls(x_i, r_i, linear_wall):
    """

    :param x_i:
    :param r_i:
    :param linear_wall:
    :return: True if agent is not overlapping with walls otherwise false.
    """
    cond = 1
    for i in range(len(linear_wall)):
        d = distance_from_linear_wall(x_i, r_i, linear_wall[i])
        cond *= d
        if cond == 0:
            break
    return cond


def distance_from_agents(agent: np.ndarray, others: np.ndarray,
                         radius: np.ndarray, radii: np.ndarray):
    """
    :param agent:
    :param others:
    :param radius:
    :param radii:
    :return: True if agent is not overlapping with others otherwise false.
    """
    if len(others) == 0:
        return True
    d = (agent - others)**2
    d = np.sum(d, axis=1)
    d = np.sqrt(d) - (radius + radii)
    return np.all(d > 0)


def set_positions(amount: int, x_dims, y_dims,
                  radius, linear_walls, seed: int=None):
    """
    Populate the positions of the agents in to the field so that they don't
    overlap each others or the walls.
    """
    np.random.seed(seed)
    position = np.zeros((amount, 2))
    i = 0
    while i < amount:
        # Random uniform position inside x and y dimensions
        agent = np.zeros(2)
        agent[0] = np.random.uniform(x_dims[0], x_dims[1])
        agent[1] = np.random.uniform(y_dims[0], y_dims[1])
        others = position[:i]

        if isinstance(radius, np.ndarray):
            rad = radius[i]
            radii = radius[:i]
        else:
            rad = radius
            radii = radius

        # Test overlapping with other agents
        c = distance_from_agents(agent, others, rad, radii)
        if not c:
            continue

        # Test overlapping with walls
        c = distance_from_linear_walls(agent, rad, linear_walls)
        if not c:
            continue

        # print(i)
        position[i, :] = agent
        i += 1
    return position
