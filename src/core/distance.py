import numpy as np


def agent_distance(agent, i, j):
    r_c = np.zeros(2)
    r_soc = np.zeros(2)

    x = agent.position[i] - agent.position[j]
    r_tot = agent.radius[i] + agent.radius[j]
    d = np.hypot(x[0], x[1])

    threshold = 7.0
    if not agent.three_circles_flag or d > threshold:
        # Approximates agent circular
        return r_tot - d

    t_i = np.array((-np.sin(agent.angle[i]), np.cos(agent.angle[i])))
    t_j = np.array((-np.sin(agent.angle[j]), np.cos(agent.angle[j])))

    v_i = agent.radius_torso_shoulder[i] * t_i
    v_j = agent.radius_torso_shoulder[j] * t_j

    return

