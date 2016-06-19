import numpy as np


class EgressGame(object):
    def __init__(self,
                 agent_size,
                 update_interval,
                 neighbor_radius,
                 neighborhood_size=8):
        """
        Patient and impatient pedestrians in a spatial game for egress congestion.

        +-----------+--------------------------+-----------+
        |           |                Impatient |   Patient |
        +===========+==========================+===========+
        | Impatient | T_ASET/T_ij, T_ASET/T_ij |     -1, 1 |
        +-----------+--------------------------+-----------+
        |   Patient |                    1, -1 |      0, 0 |
        +-----------+--------------------------+-----------+

        :param agent_size:
        :param update_interval:
        :param neighbor_radius:
        :param neighborhood_size:
        """
        self.update_interval = update_interval
        self.agent_size = agent_size
        self.neighbor_radius = neighbor_radius
        self.neighborhood_size = neighborhood_size
        self.current_strategy = np.zeros(agent_size)
        self.neighbors = np.zeros((agent_size, neighborhood_size))
        self.strategies = (0, 1)  # Strategies in {Impatient, Patient} = {0, 1}

    def time_aset(self):
        """
        :return:  Available safe egress time
        """
        return 0

    def time_evac(self, i):
        """
        :param agent_closer_to_exit: Agent closer to exit
        :param exit_capacity: Exit capacity
        :return: Estimated evacuation time
        """
        agent_closer_to_exit = 0
        exit_capacity = 0
        return agent_closer_to_exit / exit_capacity

    def clock(self, dt):
        """
        Probabilistic clock with expected frequency of update interval.
        http://preshing.com/20111007/how-to-generate-random-timings-for-a-poisson-process/

        :return: Boolean whether strategy should be updated of not.
        """
        prob = dt / self.update_interval  # Probability of updating
        return np.random.random() < prob

    def game_matrix(self, strategy_p1, strategy_p2, t_aset, t_ij):
        """
        :param strategy_p1: Our strategy
        :param strategy_p2: Neighbor strategy
        """
        if strategy_p1 == 0 and strategy_p2 == 0:
            return t_aset / t_ij
        elif strategy_p1 == 0 and strategy_p2 == 1:
            return -1
        elif strategy_p1 == 1 and strategy_p2 == 0:
            return 1
        elif strategy_p1 == 1 and strategy_p2 == 1:
            return 0
        else:
            raise ValueError()

    def best_response_strategy(self, i):
        """Update single strategy."""
        gain_loss = np.zeros(2)
        for strategy in self.strategies:
            for j in range(self.neighborhood_size):
                mean_time = (self.time_evac(i) + self.time_evac(j)) / 2
                t_aset = self.time_aset()
                gain_loss[strategy] += self.game_matrix(
                    self.current_strategy[i],
                    self.current_strategy[j],
                    t_aset, mean_time)

        strategy = np.argmin(gain_loss)
        self.current_strategy[i] = strategy

    def update_strategies(self, dt):
        """Update strategies for all agents."""
        for i in range(self.agent_size):
            if self.clock(dt):
                self.best_response_strategy(i)
