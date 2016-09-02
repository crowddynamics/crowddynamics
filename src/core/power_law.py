import numba
import numpy as np
from numba import f8

from .vector2D import dot2d, truncate


@numba.jit(f8(f8, f8), nopython=True, nogil=True)
def magnitude(tau, tau_0):
    """Magnitude of social force."""
    return (2.0 / tau + 1.0 / tau_0) * np.exp(-tau / tau_0) / tau ** 2.0


@numba.jit(f8[:](f8[:], f8[:], f8, f8, f8), nopython=True, nogil=True)
def gradient_circle_circle(x_rel, v_rel, a, b, d):
    return (v_rel - (v_rel * b + x_rel * a) / d) / a


@numba.jit(f8[:](f8[:], f8[:], f8[:], f8, f8, f8), nopython=True, nogil=True)
def gradient_three_circle(x_rel, v_rel, r_off, a, b, d):
    """Gradient of tau."""
    return (v_rel - (a * (x_rel + 2 * r_off) + b * v_rel) / d) / a


@numba.jit(f8[:](f8[:], f8[:]), nopython=True, nogil=True)
def gradient_circle_line(v, n):
    return n / dot2d(v, n)


@numba.jit(numba.types.Tuple((f8, f8[:]))(f8[:], f8[:], f8),
           nopython=True, nogil=True)
def time_to_collision_circle_circle(x_rel, v_rel, r_tot):
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
    """Social force based on human anticipatory behaviour."""
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
    mag = magnitude(tau, agent.tau_0)
    grad = gradient_circle_circle(x_rel, v_rel, a, b, d)
    coeff = agent.k_soc * mag * grad
    force[0][:] += - agent.mass[i] * coeff
    force[1][:] -= - agent.mass[j] * coeff

    # Truncation for small tau
    truncate(force[0], agent.f_soc_ij_max)
    truncate(force[1], agent.f_soc_ij_max)

    return force


@numba.jit(nopython=True, nogil=True)
def force_social_three_circle(agent, i, j):
    """Minimium time-to-collision for two circles of relative displacements."""
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
    x_i = (agent.position[i], agent.position_ls[i], agent.position_rs[i])
    x_j = (agent.position[j], agent.position_ls[j], agent.position_rs[j])

    # Radii of torso and shoulders
    r_i = (agent.r_t[i], agent.r_s[i], agent.r_s[i])
    r_j = (agent.r_t[j], agent.r_s[j], agent.r_s[j])

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
        phi = agent.angle[i]
        r_off_i += agent.r_ts[i] * np.array((np.sin(phi), -np.cos(phi)))
    elif contact_i == 2:
        phi = agent.angle[i]
        r_off_i -= agent.r_ts[i] * np.array((np.sin(phi), -np.cos(phi)))

    if contact_j == 1:
        phi = agent.angle[j]
        r_off_j += agent.r_ts[j] * np.array((np.sin(phi), -np.cos(phi)))
    elif contact_j == 2:
        phi = agent.angle[j]
        r_off_j -= agent.r_ts[j] * np.array((np.sin(phi), -np.cos(phi)))

    x_rel = agent.position[i] - agent.position[j]
    r_off = r_off_i - r_off_j

    # Force
    mag = magnitude(tau, agent.tau_0)
    grad = gradient_three_circle(x_rel, v_rel, r_off, a, b_min, d_min)
    f = -agent.k_soc * mag * grad

    # Truncation for small tau
    force[0][:] += agent.mass[i] * f
    force[1][:] -= agent.mass[j] * f

    truncate(force[0], agent.f_soc_ij_max)
    truncate(force[1], agent.f_soc_ij_max)

    return force


@numba.jit(nopython=True, nogil=True)
def force_social_linear_wall(i, w, agent, wall):
    force = np.zeros(2)
    tau = np.zeros(3)
    grad = np.zeros((3, 2))

    p_0, p_1, t_w, n_w, l_w = wall.deconstruct(w)
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
        mag = magnitude(tau[0], agent.tau_0)
        force[:] = - agent.mass[i] * agent.k_soc * mag * grad[0]
    elif not np.isnan(tau[1]) and tau[1] > tau_t1:
        mag = magnitude(tau[1], agent.tau_0)
        force[:] = - agent.mass[i] * agent.k_soc * mag * grad[1]
    elif not np.isnan(tau[2]):
        mag = magnitude(tau[2], agent.tau_0)
        force[:] = - agent.mass[i] * agent.k_soc * mag * grad[2]

    truncate(force, agent.f_soc_iw_max)

    return force
