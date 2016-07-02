import numpy as np

from crowd_dynamics.core.vector2d import length_nx2


class EgressGame(object):
    def __init__(self, agent, exit_door, time_aset, update_interval=0.1):
        """Patient and impatient pedestrians in a spatial game for egress congestion.

        +-----------+--------------------------+-----------+
        |           |                Impatient |   Patient |
        +===========+==========================+===========+
        | Impatient | T_ASET/T_ij, T_ASET/T_ij |     -1, 1 |
        +-----------+--------------------------+-----------+
        |   Patient |                    1, -1 |      0, 0 |
        +-----------+--------------------------+-----------+

        :param agent:
        :param exit_door:
        :param time_aset:
        :param update_interval:
        :param neighbor_radius:
        :param neighborhood_size:
        """

        self.agent = agent.size
        self.exit_door = exit_door
        self.update_interval = update_interval

        self.time_evac = np.zeros(self.agent.size)
        self.time_aset = time_aset

        # Number of agents closer to the exit than agent i
        self.agent_closer_to_exit = np.zeros(self.agent.size)

        # Strategies in {Impatient, Patient} = {0, 1}
        self.strategies = (0, 1)

        # Current state of the game
        self.current_strategy = np.zeros(self.agent.size)

    def clock(self, dt):
        """
        Probabilistic clock with expected frequency of update interval.
        http://preshing.com/20111007/how-to-generate-random-timings-for-a-poisson-process/

        :return: Boolean whether strategy should be updated of not.
        """
        prob = dt / self.update_interval  # Probability of updating
        return np.random.random() < prob

    def payoff(self, s_our, s_neighbor, avg_evac_time):
        """Payout from game matrix.
        :param s_our: Our strategy
        :param s_neighbor: Neighbor strategy
        """
        s = (s_our, s_neighbor)
        if s == (0, 0):
            return self.time_aset / avg_evac_time
        elif s == (0, 1):
            return -1
        elif s == (1, 0):
            return 1
        elif s == (1, 1):
            return 0
        return 0

    def best_response_strategy(self, i):
        """Update single strategy."""
        gain_loss = np.zeros(2)
        for j in self.agent.neighbors[i]:
            if j < 0:
                continue
            avg_evac_time = (self.time_evac[i] + self.time_evac[j]) / 2
            for strategy in self.strategies:
                gain_loss[strategy] += self.payoff(
                    strategy, self.current_strategy[j], avg_evac_time)

        optimal_strategy = np.argmin(gain_loss)
        self.current_strategy[i] = optimal_strategy

    def update_strategies(self, dt):
        """Update strategies for all agents.
        :param dt: Timestep used by integrator to update simulation.
        """
        # TODO: Mask agent that are not playing the game (anymore).

        # Updates agents closer to the exit
        self.agent_closer_to_exit = np.argsort(np.argsort(
            length_nx2(self.exit_door.mid - self.agent.position)))

        # Update evacution times
        self.time_evac = self.agent_closer_to_exit / self.exit_door.capacity

        # Loop over agents and update strategies
        for i in range(self.agent.size):
            if self.clock(dt):
                self.best_response_strategy(i)

    def update_agent_parameters(self):
        pass
