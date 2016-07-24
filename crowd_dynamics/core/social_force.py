import numba
import numpy as np

from .vector2d import dot2d, truncate


@numba.jit(nopython=True, nogil=True)
def magnitude_soc(tau, tau_0):
    """Magnitude of social force."""
    return 1 / (tau ** 2.0) * np.exp(-tau / tau_0) * (2.0 / tau + 1.0 / tau_0)


@numba.jit(nopython=True, nogil=True)
def gradient_soc_circular(x_rel, v_rel, a, b, d):
    return (v_rel - (v_rel * b + x_rel * a) / d) / a


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

    # Avoid zero division.
    # No interaction if tau cannot be defined.
    if np.isnan(d) or d == 0 or a == 0:
        return force

    tau = (b - d) / a  # Time-to-collision. In seconds
    tau_max = 30.0  # Maximum time for interaction.

    if tau <= 0 or tau > tau_max:
        return force

    # Force is returned negative as repulsive force
    mag = magnitude_soc(tau, agent.tau_0)
    grad = gradient_soc_circular(x_rel, v_rel, a, b, d)
    coeff = agent.k_soc * mag * grad
    force[0][:] = - agent.mass[i] * coeff
    force[1][:] = - agent.mass[j] * coeff

    # Truncation for small tau
    truncate(force[0], agent.f_soc_ij_max)
    truncate(force[1], agent.f_soc_ij_max)

    return force


@numba.jit(nopython=True, nogil=True)
def gradient_soc_three_circle(x_rel, v_rel, r_off_this, r_off_other, a, b, d):
    """Gradient of tau."""
    s = np.zeros(2)
    for n in range(len(r_off_this)):
        r_off = r_off_this[n] - r_off_other
        s += (a * (x_rel + 2 * r_off) + b[n] * v_rel) / d[n]
    return (3 * v_rel - s) / a


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

    # Parts that will be in contact for agents i and j if colliding.
    contact = 0, 0

    # For saving values to later use
    # rows = i, columns = j
    #                  | torso_j | left_shoulder_j | right_shoulder_j
    # torso_i          |
    # left_shoulder_i  |
    # right_shoulder_i |
    b_matrix = np.zeros((3, 3))
    d_matrix = np.zeros((3, 3))

    # Find smallest time-to-collision. In seconds.
    tau = np.nan

    for part_i, (xi, ri) in enumerate(zip(x_i, r_i)):
        for part_j, (xj, rj) in enumerate(zip(x_j, r_j)):
            # Relative position and total radius
            x_rel = xi - xj
            r_tot = ri + rj

            # Coefficients for time-to-collision
            b = -dot2d(x_rel, v_rel)
            c = dot2d(x_rel, x_rel) - r_tot ** 2
            d = np.sqrt(b ** 2 - a * c)

            b[part_i, part_j] = b
            d[part_i, part_j] = d

            # Avoid zero division. No interaction if tau cannot be defined.
            if np.isnan(d) or d == 0:
                continue

            tau_new = (b - d) / a
            if np.isnan(tau) or tau_new < tau:
                contact = part_i, part_j
                tau = tau_new

    if np.isnan(tau) or tau <= 0:
        return force

    # Shoulder displacement vectors
    r_ti = agent.r_ts[i] * np.array((-np.sin(agent.angle[i]),
                                     np.cos(agent.angle[i])))
    r_tj = agent.r_ts[j] * np.array((-np.sin(agent.angle[j]),
                                     np.cos(agent.angle[j])))
    r_off_i = (np.zeros(2), -r_ti, r_ti)
    r_off_j = (np.zeros(2), -r_tj, r_tj)

    # Gradient vectors
    x_rel = agent.position[i] - agent.position[j]

    part_j = contact[1]
    b = b_matrix[:, part_j]
    d = d_matrix[:, part_j]
    r_off1 = r_off_i
    r_off2 = r_off_j[part_j]
    grad_i = gradient_soc_three_circle(x_rel, v_rel, r_off1, r_off2, a, b, d)

    part_i = contact[0]
    b = b_matrix[part_i, :]
    d = d_matrix[part_i, :]
    r_off1 = r_off_j
    r_off2 = r_off_i[part_i]
    grad_j = -gradient_soc_three_circle(x_rel, v_rel, r_off1, r_off2, a, b, d)

    # Magnitude from tau
    mag = magnitude_soc(tau, agent.tau_0)

    # Forces
    force[0][:] = - agent.mass[i] * agent.k_soc * mag * grad_i
    force[1][:] = - agent.mass[j] * agent.k_soc * mag * grad_j

    # Truncation for small tau
    truncate(force[0], agent.f_soc_ij_max)
    truncate(force[1], agent.f_soc_ij_max)

    return force
