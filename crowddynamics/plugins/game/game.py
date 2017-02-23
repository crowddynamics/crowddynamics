r"""Patient and impatient pedestrians in a spatial game.

Patient and impatient pedestrians in a spatial game :math:`(S, f)` for
egress congestion between players :math:`P \subset A` with set of strategies
:math:`S` and payoff function :math:`f : S \times S \mapsto \mathbb{R}`.

[Heliovaara2013]_, [VonSchantz2014]_

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

"""
import numba
import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon

from crowddynamics.core.evacuation.evacuation import narrow_exit_capacity, \
    agent_closer_to_exit
from crowddynamics.core.random.functions import poisson_timings
from crowddynamics.core.vector.vector2D import length
from crowddynamics.taskgraph import TaskNode


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
def best_response_strategy(agent, players, door, radius_max, strategy,
                           strategies, t_aset, interval, dt):
    r"""Best Response Strategy

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
    d_layer = 0.45
    d_door = length(door[1] - door[0])
    d_agent = 2 * radius_max
    coeff = 1.0
    c_door = (door[0] + door[1]) / 2.0
    t_evac = agent_closer_to_exit(c_door, x) / \
             narrow_exit_capacity(d_door, d_agent, d_layer, coeff)

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
    r"""EgressGame

    Todo:
        - Not include agent that have reached their goals
        - check if j not in players:
        - Update agents parameters by the new strategy
        - Fix neighborhood

    """
    # Parameters that can be saved or plotted.
    parameters = [
        "strategies",
        "strategy",
        "t_aset_0",
        "t_evac",
        "interval",
    ]

    def __init__(self, simulation, door, room, t_aset_0,
                 interval=0.1, neighbor_radius=0.4, neighborhood_size=8):
        r"""Init EgressGame

        Args:
            simulation (MultiAgentSimulation)
            door (numpy.ndarray):
            room (numpy.ndarray|Polygon):
            t_aset_0: Initial available safe egress time.
            interval: Interval for updating strategies
            neighbor_radius:
            neighborhood_size:

        """
        super().__init__()
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
        # self.simulation.agent.neighbor_radius = neighbor_radius
        # self.simulation.agent.neighborhood_size = neighborhood_size
        # self.simulation.agent.reset_neighbor()

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
