import numba
import numpy as np
from matplotlib.path import Path
from numba.types import deferred_type
from shapely.geometry import Polygon

from crowddynamics.core.vector2D import length_nx2, length


@numba.jit(nopython=True)
def poisson_clock(interval, dt):
    """Probabilistic clock with expected frequency of update interval.

    :param interval: Expected frequency of update.
    :param dt: Discrete timestep
    :return: Boolean whether strategy should be updated of not.
    """
    if dt >= interval:
        return True
    else:
        return np.random.random() < (dt / interval)


@numba.jit(nopython=True)
def payoff(s_our, s_neighbor, t_aset, t_evac_i, t_evac_j):
    """Payout from game matrix.

    :param s_our: Our strategy
    :param s_neighbor: Neighbor strategy
    """
    if s_neighbor == 0:
        if s_our == 0:
            average = (t_evac_i + t_evac_j) / 2
            if average == 0:
                return np.inf
            return t_aset / average
        elif s_our == 1:
            return 1
    elif s_neighbor == 1:
        if s_our == 0:
            return -1
        elif s_our == 1:
            return 0
    else:
        raise Exception("Not valid strategy.")


@numba.jit(nopython=True)
def agent_closer_to_exit(points, position):
    mid = (points[0] + points[1]) / 2.0
    dist = length_nx2(mid - position)
    # players[values] = agents, indices = number of agents closer to exit
    num = np.argsort(dist)
    # values = number of agents closer to exit, players[indices] = agents
    num = np.argsort(num)
    return num


@numba.jit(nopython=True)
def exit_capacity(points, agent_radius):
    """Capacity of narrow exit."""
    door_radius = length(points[1] - points[0]) / 2.0
    capacity = door_radius // agent_radius
    return capacity


@numba.jit(nopython=True)
def best_response_strategy(agent, players, door, radius_max, strategy,
                           strategies, t_aset, interval, dt):
    """Best response strategy. Minimizes loss."""
    x = agent.position[players]
    t_evac = agent_closer_to_exit(door, x) / exit_capacity(door, radius_max)

    loss = np.zeros(2)  # values: loss, indices: strategy
    np.random.shuffle(players)
    for i in players:
        if poisson_clock(interval, dt):
            for j in agent.neighbors[i]:
                if j < 0:
                    continue
                for s_our in strategies:
                    loss[s_our] += payoff(s_our, strategy[j], t_aset, t_evac[i],
                                          t_evac[j])
            strategy[i] = np.argmin(loss)  # Update strategy
            loss[:] = 0  # Reset loss array


class EgressGame(object):
    """
    Patient and impatient pedestrians in a spatial game for egress congestion
    -------------------------------------------------------------------------
    Strategies: {0: "Impatient", 1: "Patient"}.

    .. [1] Heli??vaara, S., Ehtamo, H., Helbing, D., & Korhonen, T. (2013). Patient and impatient pedestrians in a spatial game for egress congestion. Physical Review E - Statistical, Nonlinear, and Soft Matter Physics. http://doi.org/10.1103/PhysRevE.87.012802
    """

    def __init__(self, simulation, door, room, t_aset_0,
                 interval=0.1, neighbor_radius=0.4, neighborhood_size=8):
        """
        Parameters
        ----------
        :param simulation: MultiAgent Simulation
        :param room:
        :param door: numpy.ndarray
        :param t_aset_0: Initial available safe egress time.
        :param interval: Interval for updating strategies
        :param neighbor_radius:
        :param neighborhood_size:
        """
        super().__init__()
        # TODO: Not include agent that have reached their goals
        # TODO: check if j not in players:
        # TODO: Update agents parameters by the new strategy

        self.simulation = simulation

        self.door = door
        if isinstance(room, Polygon):
            vertices = np.asarray(room.exterior)
        elif isinstance(room, np.ndarray):
            vertices = room
        else:
            raise Exception()

        self.room = Path(vertices)  # Polygon vertices

        # Game properties
        self.strategies = np.array((0, 1), dtype=np.int64)
        self.interval = interval
        self.t_aset_0 = t_aset_0
        self.t_aset = t_aset_0
        self.t_evac = np.zeros(self.simulation.agent.size, dtype=np.float64)
        self.radius = np.max(self.simulation.agent.radius)

        # Agent states
        self.strategy = np.ones(self.simulation.agent.size, dtype=np.int64)

        # Set neigbourhood
        self.simulation.agent.neighbor_radius = neighbor_radius
        self.simulation.agent.neighborhood_size = neighborhood_size
        self.simulation.agent.reset_neighbor()

    def reset(self):
        self.t_evac[:] = 0

    def update(self):
        """Update strategies for all agents.

        :param dt: Timestep used by integrator to update simulation.
        """
        self.reset()

        # Indices of agents that are playing the game
        # Loop over agents and update strategies
        agent = self.simulation.agent
        mask = agent.active & self.room.contains_points(agent.position)
        indices = np.arange(agent.size)[mask]

        # Agents that are not playing anymore will be patient again
        self.strategy[mask ^ True] = 1

        self.t_aset = self.t_aset_0 - self.simulation.time_tot
        best_response_strategy(self.simulation.agent, indices, self.door,
                               self.radius, self.strategy, self.strategies,
                               self.t_aset, self.interval, self.simulation.dt_prev)
