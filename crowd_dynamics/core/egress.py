import numba
import numpy as np

from crowd_dynamics.core.vector2d import length_nx2, length


class ExitDoor(object):
    def __init__(self, p0, p1, agent_radius):
        """Exit door / Bottleneck"""
        self.p = np.array((p0, p1))
        self.mid = (self.p[0] + self.p[1]) / 2.0
        self.capacity_coeff = 1.0  # Agents per second
        self.radius = length(self.p[1] - self.p[0]) / 2.0
        self.capacity = self.capacity_coeff * self.radius // agent_radius


@numba.jit(nopython=True)
def clock(interval, dt):
    """Probabilistic clock with expected frequency of update interval.
    :return: Boolean whether strategy should be updated of not.
    """
    if dt >= interval:
        return True
    else:
        return np.random.random() < (dt / interval)


@numba.jit(nopython=True)
def payoff(s_our, s_neighbor, time_aset, t_evac_i, t_evac_j):
    """Payout from game matrix.
    :param s_our: Our strategy
    :param s_neighbor: Neighbor strategy
    """
    if s_neighbor == 0:
        if s_our == 0:
            average = (t_evac_i + t_evac_j) / 2
            return time_aset / average
        elif s_our == 1:
            return 1
    elif s_neighbor == 1:
        if s_our == 0:
            return -1
        elif s_our == 1:
            return 0
    else:
        raise ValueError("Not valid strategy.")


@numba.jit(nopython=True)
def best_response_strategy(agent, strategy, strategies, time_aset, time_evac,
                           interval, dt):
    # Best response strategy. Minimizes loss
    # TODO: Shuffle indices
    # values: loss, indices: strategy
    loss = np.zeros(2)
    for i in range(agent.size):
        if clock(interval, dt):
            for j in agent.neighbors[i]:
                if j < 0:
                    continue
                for s_our in strategies:
                    loss[s_our] += payoff(s_our, strategy[j], time_aset,
                                          time_evac[i], time_evac[j])
            strategy[i] = np.argmin(loss)  # Update strategy
            loss[:] = 0  # Reset loss array
            # TODO: Update agents parameters by the new strategy


class EgressGame(object):
    def __init__(self, agent, exit_door, time_aset_0, interval):
        """Patient and impatient pedestrians in a spatial game for egress
        congestion. Strategies are denoted: {0: "Impatient", 1: "Patient"}.
        :param agent: Agent class
        :param exit_door: Exit door class
        :param time_aset_0: Initial available safe egress time.
        :param interval: Interval for updating strategies
        """
        self.agent = agent
        self.agent.neighbor_radius = 0.4
        self.agent.reset_neighbor()
        self.exit_door = exit_door

        self.time_aset_0 = time_aset_0
        self.strategies = np.array((0, 1), dtype=np.int64)
        self.strategy = np.zeros(self.agent.size, dtype=np.int64)
        self.interval = interval

    def agent_closer_to_exit(self):
        # values: distances, indices: agents
        tmp = length_nx2(self.exit_door.mid - self.agent.position)
        # values: agents, indices: number of agents closer to exit
        tmp = np.argsort(tmp)
        # values: number of agents closer to exit, indices: agents
        tmp = np.argsort(tmp)
        return tmp

    def update(self, time, dt):
        """Update strategies for all agents.
        :param dt: Timestep used by integrator to update simulation.
        """
        # TODO: Mask agent that are not playing the game (anymore).
        # Number of agents closer to the exit than agent i
        time_evac = self.agent_closer_to_exit() / self.exit_door.capacity
        time_aset = self.time_aset_0 - time
        # Loop over agents and update strategies
        best_response_strategy(self.agent, self.strategy, self.strategies,
                               time_aset, time_evac, self.interval, dt)
