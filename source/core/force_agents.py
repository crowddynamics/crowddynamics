import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def f_soc_ij(xi, xj, vi, vj, ri, rj, tau_0, sight, f_max):
    r"""
    About
    -----
    Social interaction force between two agents `i` and `j`. [1]

    Params
    ------
    :param xi: Position (center of mass) of agent i.
    :param xj: Position (center of mass) of agent j.
    :param vi: Velocity of agent i.
    :param vj: Velocity of agent j.
    :param ri: Radius of agent i.
    :param rj: Radius of agent j.
    :param tau_0: Max interaction range 2 - 4, aka interaction time horizon
    :param sight: Max distance between agents for interaction to occur
    :param f_max: Maximum magnitude of force. Forces greater than this are scaled to force max.
    :return: Vector of length 2 containing `x` and `y` components of force
             on agent i.

    References
    ----------
    [1] http://motion.cs.umn.edu/PowerLaw/
    """
    # Init output values.
    force = np.zeros(2)

    # Variables
    x_ij = xi - xj  # position
    v_ij = vi - vj  # velocity
    r_ij = ri + rj  # radius

    x_dot = np.dot(x_ij, x_ij)
    dist = np.sqrt(x_dot)
    # No force if another agent is not in range of sight
    if dist > sight:
        return force

    # TODO: Update overlapping to f_c_ij
    # If two agents are overlapping reduce r
    if r_ij > dist:
        r_ij = 0.50 * dist

    a = np.dot(v_ij, v_ij)
    b = - np.dot(x_ij, v_ij)
    c = x_dot - r_ij ** 2
    d = b ** 2 - a * c

    if (d < 0) or (- 0.001 < a < 0.001):
        return force

    d = np.sqrt(d)
    tau = (b - d) / a  # Time-to-collision

    k = 1.5  # Constant for setting units for interaction force. Scale with mass
    m = 2.0  # Exponent in power law
    maxt = 999.0

    if tau < 0 or tau > maxt:
        return force

    # Force is returned negative as repulsive force
    force -= k / (a * tau ** m) * np.exp(-tau / tau_0) * \
             (m / tau + 1 / tau_0) * (v_ij - (v_ij * b + x_ij * a) / d)

    mag = np.sqrt(np.dot(force, force))
    if mag > f_max:
        # Scales magnitude of force to force max
        force *= f_max / mag

    return force


@numba.jit(nopython=True, nogil=True)
def f_c_ij(h_ij, n_ij, v_ij, t_ij, mu, kappa):
    force = h_ij * (mu * n_ij - kappa * np.dot(v_ij, t_ij) * t_ij)
    return force


@numba.jit(nopython=True, nogil=True)
def f_ij(i, x, v, r, tau_0, sight, f_max, mu, kappa):
    force = np.zeros(2)
    rot270 = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    for j in range(len(x)):
        if i == j:
            continue
        x_ij = x[i] - x[j]
        v_ij = v[i] - v[j]
        x_dot = np.dot(x_ij, x_ij)
        d_ij = np.sqrt(x_dot)
        r_ij = r[i] + r[j]
        h_ij = d_ij - r_ij
        n_ij = x_ij / d_ij
        t_ij = np.dot(rot270, n_ij)

        force += f_soc_ij(x[i], x[j], v[i], v[j], r[i], r[j],
                          tau_0, sight, f_max)

        force += f_c_ij(h_ij, n_ij, v_ij, t_ij, mu, kappa)
    return force
