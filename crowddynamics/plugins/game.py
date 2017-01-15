import numba
import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon

from crowddynamics.core.vector2D import length_nx2, length
from crowddynamics.functions import public
from crowddynamics.task_graph import TaskNode


@numba.jit(nopython=True)
def clock(interval, dt):
    """
    Probabilistic clock with expected frequency of update interval.

    Args:
        interval: Expected frequency of update
        dt: Discrete timestep

    Returns:
        Boolean: Whether strategy should be updated of not.
    """
    if dt >= interval:
        return True
    else:
        return np.random.random() < (dt / interval)


@numba.jit(nopython=True)
def poisson_clock(interval, dt):
    """
    Poisson process that simulates when agents will change their strategies.

    Args:
        interval (float):
            Expected frequency of update

        dt (float):
            Discrete timestep aka time window

    Returns:
        list:
            Moments in the time window when the strategies should be updated.

    References:
    .. [#] https://en.wikipedia.org/wiki/Poisson_distribution
    .. [#] https://en.wikipedia.org/wiki/Poisson_point_process
    .. [#] http://preshing.com/20111007/how-to-generate-random-timings-for-a-poisson-process/
    """
    lamda = dt / interval
    times = []
    tsum = 0.0
    while True:
        time = np.random.poisson(lamda)
        tsum += time
        if tsum < dt:
            times.append(time)
        else:
            break
    return times


@numba.jit(nopython=True)
def payoff(s_our, s_neighbor, t_aset, t_evac_i, t_evac_j):
    """
    Payout from game matrix.

    Args:
        s_our (int): Our strategy
        s_neighbor (int): Neighbor strategy
        t_aset (float): Available safe egress time.
        t_evac_i (float): Time to evacuate for agent i.
        t_evac_j (float): Time to evacuate for agent j.

    Returns:
        float:
    """
    if s_neighbor == 0:
        if s_our == 0:
            average = (t_evac_i + t_evac_j) / 2
            if average == 0:
                return np.inf
            return t_aset / average
        elif s_our == 1:
            return 1.0
    elif s_neighbor == 1:
        if s_our == 0:
            return -1.0
        elif s_our == 1:
            return 0.0
    else:
        raise Exception("Not valid strategy.")


@numba.jit(nopython=True)
def agent_closer_to_exit(points, position):
    """
    Args:
        points (numpy.ndarray):
        position (numpy.ndarray):

    Returns:
        numpy.ndarray:
            Array where indices denote agents and value how many agents are
            closer to the exit.
    """
    mid = (points[0] + points[1]) / 2.0
    dist = length_nx2(mid - position)
    # players[values] = agents, indices = number of agents closer to exit
    num = np.argsort(dist)
    # values = number of agents closer to exit, players[indices] = agents
    num = np.argsort(num)
    return num


@numba.jit(nopython=True)
def exit_capacity(points, agent_radius):
    """
    Capacity of narrow exit. Narrow exit means :math:`3\,\mathrm{m}` wide.

    Args:
        points (numpy.ndarray):
        agent_radius (float):

    Returns:
        float:
    """
    return length(points[1] - points[0]) // (2 * agent_radius)


@numba.jit(nopython=True)
def best_response_strategy(agent, players, door, radius_max, strategy,
                           strategies, t_aset, interval, dt):
    """
    Best response strategy. Minimizes loss.

    Args:
        agent:
        players:
        door:
        radius_max:
        strategy:
        strategies:
        t_aset:
        interval:
        dt:

    Returns:
        None:
    """
    x = agent.position[players]
    t_evac = agent_closer_to_exit(door, x) / exit_capacity(door, radius_max)

    loss = np.zeros(2)  # values: loss, indices: strategy
    np.random.shuffle(players)
    for i in players:
        if clock(interval, dt):
            for j in agent.neighbors[i]:
                if j < 0:
                    continue
                for s_our in strategies:
                    loss[s_our] += payoff(s_our, strategy[j], t_aset, t_evac[i],
                                          t_evac[j])
            strategy[i] = np.argmin(loss)  # Update strategy
            loss[:] = 0  # Reset loss array


@public
class EgressGame(TaskNode):
    def __init__(self, simulation, door, room, t_aset_0,
                 interval=0.1, neighbor_radius=0.4, neighborhood_size=8):
        """
        Patient and impatient pedestrians in a spatial game for egress congestion

        Strategies: {
            0: "Impatient",
            1: "Patient"
        }

        Args:
            simulation: MultiAgent Simulation
            door:
            room (numpy.ndarray):
            t_aset_0: Initial available safe egress time.
            interval: Interval for updating strategies
            neighbor_radius:
            neighborhood_size:

        .. [1] Heli??vaara, S., Ehtamo, H., Helbing, D., & Korhonen, T. (2013).
           Patient and impatient pedestrians in a spatial game for egress
           congestion. Physical Review E - Statistical, Nonlinear, and Soft
           Matter Physics. http://doi.org/10.1103/PhysRevE.87.012802
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

    def parameters(self):
        """Parameters that can be saved or plotted."""
        params = (
            "strategies",
            "strategy",
            "t_aset_0",
            "t_evac",
            "interval",
        )
        for p in params:
            assert hasattr(self, p),  "{cls} doesn't have attribute {attr}".format(cls=p, attr=p)
        return params

    def reset(self):
        """Reset"""
        self.t_evac[:] = 0

    def update(self):
        """
        Update strategies for all agents.

        Returns:
            None:
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
