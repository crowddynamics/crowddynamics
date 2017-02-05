r"""Universal power law governing pedestrian

[Karamouzas2014b]_
"""
import numba
import numpy as np
from numba import f8

from crowddynamics.core.vector2D import dot2d, truncate, rotate90


@numba.jit(nopython=True, nogil=True)
def potential(k, tau, tau_0):
    r"""
    Psychological force based on human anticipatory behaviour. Interaction potential
    between two agents is defined

    .. math::
       E(\tau) = m \frac{k}{\tau^{2}} \exp \left( -\frac{\tau}{\tau_{0}} \right)

    Args:
        k (float):
            Scaling parameter for the magnitude. Reference value of
            :math:`k=1.5`.

        tau (float):
            Time-to-collision :math:`\tau > 0` is obtained by linearly extrapolating
            current trajectories and finding where or if agents collide i.e
            skin-to-skin distance :math:`h` equals zero.

        tau_0 (float): Interaction time horizon :math:`\tau_{0} > 0`.

    Returns:
        float: Interaction potential

    """
    return k / tau**2 * np.exp(-tau / tau_0)


@numba.jit(f8(f8, f8), nopython=True, nogil=True)
def magnitude(tau, tau_0):
    r"""
    Force affecting agent can be derived by taking spatial gradient of the energy,
    where time-to-collision :math:`\tau` is function of the relative displacement
    of the mass centers :math:`\tilde{\mathbf{x}} = \mathbf{x}_i - \mathbf{x}_j`

    .. math::
       \mathbf{f}^{soc} &= -\nabla_{\tilde{\mathbf{x}}} E(\tau) \\
       &= - k \cdot m \left(\frac{1}{\tau^{2}}\right) \left(\frac{2}{\tau} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right ) \nabla_{\tilde{\mathbf{x}}} \tau

    If :math:`\tau < 0` or :math:`\tau` is undefined trajectories are not
    colliding and social force is :math:`\mathbf{0}`.

    Magnitude of social force.

    .. math::
       \left(\frac{1}{\tau^{2}}\right) \left(\frac{2}{\tau} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right )

    Args:
        tau (float):
            Time-to-collision :math:`\tau > 0` is obtained by linearly extrapolating
            current trajectories and finding where or if agents collide i.e
            skin-to-skin distance :math:`h` equals zero.

        tau_0 (float): Interaction time horizon :math:`\tau_{0} > 0`.

    Returns:
        float: Magnitude
    """
    return (2.0 / tau + 1.0 / tau_0) * np.exp(-tau / tau_0) / tau ** 2


@numba.jit(f8[:](f8[:], f8[:], f8, f8, f8), nopython=True, nogil=True)
def gradient_circle_circle(x_rel, v_rel, a, b, d):
    r"""
    Gradient of :math:`\tau` between two circles.

    Args:
        x_rel (numpy.ndarray): Relative position of the centers of mass.
            :math:`\tilde{x} = x_i - x_j`
        v_rel (numpy.ndarray): Relative velocity between two agents.
            :math:`\tilde{v} = v_i - v_j`
        a (float):
        b (float):
        d (float):

    Returns:
        numpy.ndarray:

    """
    return (v_rel - (v_rel * b + x_rel * a) / d) / a


@numba.jit(f8[:](f8[:], f8[:], f8[:], f8, f8, f8), nopython=True, nogil=True)
def gradient_three_circle(x_rel, v_rel, r_off, a, b, d):
    r"""
    Gradient of :math:`\tau` between two three-circle representations.

    Args:
        x_rel (numpy.ndarray):
        v_rel (numpy.ndarray):
        r_off (numpy.ndarray):
        a (float):
        b (float):
        d (float):

    Returns:
        numpy.ndarray:

    """
    return (v_rel - (a * (x_rel + 2 * r_off) + b * v_rel) / d) / a


@numba.jit(f8[:](f8[:], f8[:]), nopython=True, nogil=True)
def gradient_circle_line(v, n):
    """
    Gradient circle line

    Args:
        v:
        n:

    Returns:

    """
    return n / dot2d(v, n)


@numba.jit(numba.types.Tuple((f8, f8[:]))(f8[:], f8[:], f8),
           nopython=True, nogil=True)
def time_to_collision_circle_circle(x_rel, v_rel, r_tot):
    r"""
    Time-to-collision of two circles. From *skin-to-skin* distance

    .. math::
       h(\tau) = \| \tau \tilde{\mathbf{v}} + \mathbf{c} \| - \tilde{r}.

    Solve for root

    .. math::
       h(\tau) &= 0 \\
       \| \mathbf{c} + \tau \tilde{\mathbf{v}} \| &= \tilde{r} \\
       \| \mathbf{c} + \tau \tilde{\mathbf{v}} \|^2 &= \tilde{r}^2

    Quadratic equation is obtained

    .. math::
       \tau^2 (\tilde{\mathbf{v}} \cdot \tilde{\mathbf{v}}) + 2 \tau (\mathbf{c} \cdot \tilde{\mathbf{v}}) + \mathbf{c} \cdot \mathbf{c} - \tilde{r}^2 =0

    Solution with quadratic formula gives us

    .. math::
       a &= \tilde{\mathbf{v}} \cdot \tilde{\mathbf{v}} \\
       b &= -\mathbf{c} \cdot \tilde{\mathbf{v}} \\
       c &= \mathbf{c} \cdot \mathbf{c} - \tilde{r}^{2}\\
       d &= \sqrt{b^{2} - a c} \\
       \tau &= \frac{b - d}{a}.

    Args:
        x_rel (numpy.ndarray):
            Relative center of mass :math:`\mathbf{c}`.

        v_rel (numpy.ndarray):
            Relative velocity :math:`\mathbf{\tilde{v}}`.

        r_tot (float):
            Total radius of :math:`\tilde{r} = \mathrm{constant}`.

    Returns:
        (float, numpy.ndarray):

    """
    a = dot2d(v_rel, v_rel)
    b = -dot2d(x_rel, v_rel)
    c = dot2d(x_rel, x_rel) - r_tot ** 2
    d = np.sqrt(b ** 2 - a * c)

    # No interaction if tau cannot be defined.
    if np.isnan(d) or d == 0 or a == 0:
        return np.nan, np.zeros(2)

    tau = (b - d) / a  # Time-to-collision. In seconds

    if tau <= 0:
        return np.nan, np.zeros(2)

    grad = gradient_circle_circle(x_rel, v_rel, a, b, d)
    return tau, grad


@numba.jit(nopython=True, nogil=True)
def time_to_collision_circle_line(x_rel, v_rel, r_tot, n):
    r"""
    Time-to-collision of circle with line.

    Args:
        x_rel:
        v_rel:
        r_tot:
        n:

    Returns:

    """
    dot_vn = dot2d(v_rel, n)
    if dot_vn == 0:
        return np.nan, np.zeros(2)

    g0 = -dot2d(x_rel, n) / dot_vn
    g1 = r_tot / dot_vn
    tau0 = g0 + g1
    tau1 = g0 - g1
    grad = gradient_circle_line(v_rel, n)
    if tau0 > g0 > 0:
        return tau0, grad
    elif 0 < tau1 <= g0:
        return tau1, grad
    else:
        return np.nan, np.zeros(2)


@numba.jit(nopython=True, nogil=True)
def force_social_circular(agent, i, j):
    """Social force based on human anticipatory behaviour.

    Args:
        agent:
        i:
        j:

    Returns:

    """
    x_rel = agent.position[i] - agent.position[j]
    v_rel = agent.velocity[i] - agent.velocity[j]
    r_tot = agent.radius[i] + agent.radius[j]

    force = np.zeros(2), np.zeros(2)

    a = dot2d(v_rel, v_rel)
    b = -dot2d(x_rel, v_rel)
    c = dot2d(x_rel, x_rel) - r_tot ** 2
    d = np.sqrt(b ** 2 - a * c)

    # No interaction if tau cannot be defined.
    if np.isnan(d) or d == 0 or a == 0:
        return force

    tau = (b - d) / a  # Time-to-collision. In seconds
    tau_max = 30.0  # Maximum time for interaction.

    if tau <= 0 or tau > tau_max:
        return force

    # Force is returned negative as repulsive force
    grad = gradient_circle_circle(x_rel, v_rel, a, b, d)
    force[0][:] += - agent.mass[i] * agent.k_soc[i] * grad * \
                   magnitude(tau, agent.tau_0[i])
    force[1][:] -= - agent.mass[j] * agent.k_soc[j] * grad * \
                   magnitude(tau, agent.tau_0[j])

    # Truncation for small tau
    truncate(force[0], agent.f_soc_ij_max)
    truncate(force[1], agent.f_soc_ij_max)

    return force


@numba.jit(nopython=True, nogil=True)
def force_social_three_circle(agent, i, j):
    """
    Minimium time-to-collision for two circles of relative displacements.

    Args:
        agent:
        i:
        j:

    Returns:

    """
    # Forces for agent i and j
    force = np.zeros(2), np.zeros(2)

    v_rel = agent.velocity[i] - agent.velocity[j]
    a = dot2d(v_rel, v_rel)

    # Agents are not moving relative to each other.
    if a == 0:
        return force

    # Meaning of indexes for tuples of three
    # 0 = torso
    # 1 = left shoulder
    # 2 = right shoulder

    # Positions: center, left, right
    # x_i = (agent.position[i], agent.position_ls[i], agent.position_rs[i])
    # x_j = (agent.position[j], agent.position_ls[j], agent.position_rs[j])
    x_i = agent.positions(i)
    x_j = agent.positions(j)

    # Radii of torso and shoulders
    # r_i = (agent.r_t[i], agent.r_s[i], agent.r_s[i])
    # r_j = (agent.r_t[j], agent.r_s[j], agent.r_s[j])
    r_i = agent.radii(i)
    r_j = agent.radii(j)

    # Parts that will be first in contact for agents i and j if colliding.
    contact_i = np.int64(0)
    contact_j = np.int64(0)

    # Find smallest time-to-collision. In seconds.
    tau = np.nan
    b_min = np.nan
    d_min = np.nan

    for part_i, (xi, ri) in enumerate(zip(x_i, r_i)):
        for part_j, (xj, rj) in enumerate(zip(x_j, r_j)):
            # Relative position and total radius
            x_rel = xi - xj
            r_tot = ri + rj

            # Coefficients for time-to-collision
            b = -dot2d(x_rel, v_rel)
            c = dot2d(x_rel, x_rel) - r_tot ** 2
            d = np.sqrt(b ** 2 - a * c)

            # No interaction if tau cannot be defined.
            if np.isnan(d) or d == 0:
                continue

            tau_new = (b - d) / a
            if np.isnan(tau) or 0 < tau_new < tau:
                contact_i, contact_j = part_i, part_j
                tau = tau_new
                b_min = b
                d_min = d

    if np.isnan(tau) or tau <= 0:
        return force

    # Shoulder displacement vectors
    r_off_i = np.zeros(2)
    r_off_j = np.zeros(2)

    # TODO: Fix signs
    if contact_i == 1:
        phi = agent.orientation[i]
        r_off_i += agent.r_ts[i] * np.array((np.sin(phi), -np.cos(phi)))
    elif contact_i == 2:
        phi = agent.orientation[i]
        r_off_i -= agent.r_ts[i] * np.array((np.sin(phi), -np.cos(phi)))

    if contact_j == 1:
        phi = agent.orientation[j]
        r_off_j += agent.r_ts[j] * np.array((np.sin(phi), -np.cos(phi)))
    elif contact_j == 2:
        phi = agent.orientation[j]
        r_off_j -= agent.r_ts[j] * np.array((np.sin(phi), -np.cos(phi)))

    x_rel = agent.position[i] - agent.position[j]
    r_off = r_off_i - r_off_j

    # Force
    grad = gradient_three_circle(x_rel, v_rel, r_off, a, b_min, d_min)
    force[0][:] += - agent.mass[i] * agent.k_soc[i] * grad * \
                   magnitude(tau, agent.tau_0[i])
    force[1][:] -= - agent.mass[j] * agent.k_soc[j] * grad * \
                   magnitude(tau, agent.tau_0[j])

    truncate(force[0], agent.f_soc_ij_max)
    truncate(force[1], agent.f_soc_ij_max)

    return force


@numba.jit(nopython=True, nogil=True)
def force_social_linear_wall(i, w, agent, wall):
    """
    Force social linear wall

    Args:
        i:
        w:
        agent:
        wall:

    Returns:

    """
    force = np.zeros(2)
    tau = np.zeros(3)
    grad = np.zeros((3, 2))

    # p_0, p_1, t_w, n_w, l_w = wall.deconstruct(w)

    p_0 = wall[w, 0, :]
    p_1 = wall[w, 1, :]
    d = p_1 - p_0  # Vector from p_0 to p_1
    l_w = np.hypot(d[1], d[0])  # Length of the wall
    t_w = d / l_w  # Tangential unit-vector
    n_w = rotate90(t_w)  # Normal unit-vector

    x_rel0 = agent.position[i] - p_0
    x_rel1 = agent.position[i] - p_1
    v_rel = agent.velocity[i]
    r_tot = agent.radius[i]

    dot_vt = dot2d(v_rel, t_w)
    if dot_vt == 0:
        tau_t0 = np.nan
        tau_t1 = np.nan
    else:
        tau_t0 = -dot2d(x_rel0, t_w) / dot_vt
        tau_t1 = -dot2d(x_rel1, t_w) / dot_vt

    tau[0], grad[0] = time_to_collision_circle_circle(x_rel0, v_rel, r_tot)
    tau[1], grad[1] = time_to_collision_circle_circle(x_rel1, v_rel, r_tot)
    tau[2], grad[2] = time_to_collision_circle_line(x_rel0, v_rel, r_tot, n_w)

    if not np.isnan(tau[0]) and tau[0] <= tau_t0:
        mag = magnitude(tau[0], agent.tau_0[i])
        force[:] = - agent.mass[i] * agent.k_soc * mag * grad[0]
    elif not np.isnan(tau[1]) and tau[1] > tau_t1:
        mag = magnitude(tau[1], agent.tau_0[i])
        force[:] = - agent.mass[i] * agent.k_soc * mag * grad[1]
    elif not np.isnan(tau[2]):
        mag = magnitude(tau[2], agent.tau_0[i])
        force[:] = - agent.mass[i] * agent.k_soc * mag * grad[2]

    truncate(force, agent.f_soc_iw_max)

    return force
