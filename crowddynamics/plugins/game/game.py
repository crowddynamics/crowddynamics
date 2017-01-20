import numba
import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon

from crowddynamics.core.vector2D import length_nx2, length
from crowddynamics.task_graph import TaskNode


@numba.jit(nopython=True)
def poisson_clock(interval, dt):
    r"""
    Poisson process that simulates when agents will change their strategies.
    Time between updates are independent random variables defined

    .. math::

       \tau_i \sim \operatorname{Exp}(\lambda)

    Where the ``rate`` parameter is

    .. math::

       \operatorname{E}(\tau) = \frac{1}{\lambda} = interval

    Times when agent updates its strategy
    
    .. math::
    
       \begin{split}\begin{cases}
       T_n = \tau_1 + \ldots + \tau_n, & n \geq 1 \\
       T_0 = 0
       \end{cases}\end{split}

    Sequence of update times

    .. math::

       \{T_1, T_2, T_3, \ldots, T_n\}, \quad T_n < \Delta t

    Number of arrivals by the :math:`s`
    
    .. math::
    
       N(s) &= \max\{n : T_n \leq s\} \\
       N(s) &\sim \operatorname{Poi}(\lambda)

    Args:
        interval (float):
            Expected frequency of update.

        dt (float):
            Discrete timestep :math:`\Delta t` aka time window.

    Yields:
        float:
            Moments in the time window when the strategies should be updated.

    """
    t_tot = 0.0
    while True:
        # Numpy exponential distribution's scale parameter is equal to
        # 1/lambda which is why we can supply interval directly into the
        # function.
        time = np.random.exponential(scale=interval)
        t_tot += time
        if t_tot < dt:
            yield t_tot
        else:
            break


@numba.jit(nopython=True)
def poisson_timings(players, interval, dt):
    r"""
    Update times for all agent in the game using Poisson clock.

    Args:
        players (numpy.ndarray):
            Indices of the players.

        interval (float):
        dt (float):

    Returns:
        list: List of indices of agents sorted by their update times.
    """
    times = []
    indices = []
    indices_sorted = []

    # Mix the agents
    np.random.shuffle(players)

    # Compute update times for all agents
    for i in players:
        for t in poisson_clock(interval, dt):
            times.append(t)
            indices.append(i)

    # Sort the indices by the update times
    for j in np.argsort(np.array(times)):
        indices_sorted.append(indices[j])

    # Return indices sorted by the update times
    return indices_sorted


@numba.jit(nopython=True)
def payoff(s_i, s_j, t_aset, t_evac_i, t_evac_j):
    r"""
    Payoff matrix of the game.

    Args:
        s_i (int): Our strategy
        s_j (int): Neighbor strategy
        t_aset (float): Available safe egress time.
        t_evac_i (float): Time to evacuate for agent i.
        t_evac_j (float): Time to evacuate for agent j.

    Returns:
        float:

    """
    if s_j == 0:
        if s_i == 0:
            average = (t_evac_i + t_evac_j) / 2
            if average == 0:
                return np.inf
            return t_aset / average
        elif s_i == 1:
            return 1.0
    elif s_j == 1:
        if s_i == 0:
            return -1.0
        elif s_i == 1:
            return 0.0
    else:
        raise Exception("Not valid strategy.")


@numba.jit(nopython=True)
def agent_closer_to_exit(points, position):
    r"""
    Agent closer to exit

    Function mapping players to number of agents that are closer to the exit is denoted

    .. math::
       \lambda : P \mapsto [0, | P | - 1].

    Ordering is defined as the distances between the exit and an agent

    .. math::
       d(\mathcal{E}_i, \mathbf{x}_{i})

    where

    - :math:`\mathcal{E}_i` is the exit the agent is trying to reach
    - :math:`\mathbf{x}_{i}` is the center of the mass of an agent

    For narrow bottlenecks we can approximate the distance

    .. math::
       d(\mathcal{E}_i, \mathbf{x}_{i}) \approx \| \mathbf{c} - \mathbf{x}_{i} \|

    where

    - :math:`\| \cdot \|` is euclidean `metric`_
    - :math:`\mathbf{c}` is the center of the exit.

    .. _metric: https://en.wikipedia.org/wiki/Metric_(mathematics)

    .. Then we sort the distances by indices to get the order of agent indices from closest to the exit door to farthest, sorting by indices again gives us number of agents closer to the exit door

    Algorithm

    #) Sort by distances to map number of closer agents to player

       .. math::
           \boldsymbol{\lambda}^{-1} = \underset{i \in P}{\operatorname{arg\,sort}} \left( d(\mathcal{E}_i, \mathbf{x}_{i}) \right)

    #) Sort by players to map player to number of closer agents

    .. math::
       \boldsymbol{\lambda} = \operatorname{arg\,sort} (\boldsymbol{\lambda}^{-1})


    Args:
        points (numpy.ndarray):
            Array [[x1, y2], [x2, y2]]

        position (numpy.ndarray):
            Positions of the agents.

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
    r"""
    Estimation of the capacity of narrow exit. Narrow exit is defined max
    :math:`3\,\mathrm{m}` wide.

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
    r"""
    New strategy is selected using best response dynamics which finds the
    strategy that minimizes loss using the payoff function

    .. math::
       s_{i} = \underset{s \in S}{\arg \min} \sum_{j \in N_{i}^{neigh}} f(s, s_{j})

    where

    - :math:`N_{i}^{neigh} \subset P \setminus \{P_{i}\}` set is eight closes
      agents excluding the agent itself at maximum skin-to-skin distance of
      :math:`0.40 \ \mathrm{m}` from agent :math:`i`.

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
    for i in poisson_timings(players, interval, dt):
        for j in agent.neighbors[i]:
            if j < 0:
                continue
            for s_our in strategies:
                loss[s_our] += payoff(s_our, strategy[j], t_aset, t_evac[i],
                                      t_evac[j])
        strategy[i] = np.argmin(loss)  # Update strategy
        loss[:] = 0  # Reset loss array


class EgressGame(TaskNode):
    r"""
    Patient and impatient pedestrians in a spatial game :math:`(S, f)` for
    egress congestion between players :math:`P \subset A` with set of strategies 
    :math:`S` and payoff function :math:`f : S \times S \mapsto \mathbb{R}`.

    Set of strategies

    .. math::
       S &= \{ \text{Impatient}, \text{Patient} \} \\
         &= \{ 0, 1 \}

    Payoff matrix / function

    .. math::
       f(s_i, s_j) = \left[\begin{matrix}\left ( \frac{T_{aset}}{T_{ij}}, \quad \frac{T_{aset}}{T_{ij}}\right ) & \left ( -1, \quad 1\right )\\\left ( 1, \quad -1\right ) & \left ( 0, \quad 0\right )\end{matrix}\right]

    Estimated evacuation time for an agent

    .. math::
       T_i = \frac{\lambda_i}{\beta},

    where

    - :math:`\beta` is the capacity of the exit door
    - :math:`\lambda_i` number of other agents closer to the exit.

    Average evacuation time

    .. math::
       T_{ij} = \frac{(T_i + T_j)}{2}.

    Effect on agents

    - :math:`k`
    - :math:`\tau_{adj}`
    - :math:`\sigma_{force}`

    References

    .. [game2013] Heli??vaara, S., Ehtamo, H., Helbing, D., & Korhonen, T. (2013). Patient and impatient pedestrians in a spatial game for egress congestion. Physical Review E - Statistical, Nonlinear, and Soft Matter Physics. http://doi.org/10.1103/PhysRevE.87.012802
    .. [game2014] Von Schantz, A., & Ehtamo, H. (2014). Cellular automaton evacuation model coupled with a spatial game. In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics). http://doi.org/10.1007/978-3-319-09912-5_31

    """

    def __init__(self, simulation, door, room, t_aset_0,
                 interval=0.1, neighbor_radius=0.4, neighborhood_size=8):
        r"""
        EgressGame

        Args:
            simulation: MultiAgent Simulation
            door:
            room (numpy.ndarray):
            t_aset_0: Initial available safe egress time.
            interval: Interval for updating strategies
            neighbor_radius:
            neighborhood_size:

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
        r"""Parameters that can be saved or plotted."""
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
        r"""Reset"""
        self.t_evac[:] = 0

    def update(self):
        r"""Update strategies for all agents."""
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
