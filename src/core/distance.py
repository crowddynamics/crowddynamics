import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def agent_distance(agent, i, j, x_rel, dist_rel):
    t_i = np.array((-np.sin(agent.angle[i]), np.cos(agent.angle[i])))
    t_j = np.array((-np.sin(agent.angle[j]), np.cos(agent.angle[j])))

    tr_i = agent.radius_torso_shoulder[i] * t_i
    tr_j = agent.radius_torso_shoulder[j] * t_j

    r_i = (agent.radius_torso[i],
           agent.radius_shoulder[i],
           agent.radius_shoulder[i])
    r_j = (agent.radius_torso[j],
           agent.radius_shoulder[j],
           agent.radius_shoulder[j])

    coeffs = (0, 1, -1)
    h_min = dist_rel
    r_min = (0.0, 0.0)
    c_min = (np.zeros(2), np.zeros(2))
    direction = np.zeros(2)

    for ri, k in zip(r_i, coeffs):
        for rj, k2 in zip(r_j, coeffs):
            ci = k * tr_i
            cj = k2 * tr_j
            vec = ci - cj + x_rel
            hypot = np.hypot(vec[0], vec[1])
            d = hypot - (ri + rj)
            if d < h_min:
                h_min = d
                r_min = ri, rj
                c_min = ci, cj
                direction = vec / hypot

    # Torque
    if h_min < 0:
        # Physical contact
        pass

    return h_min
