from crowddynamics.multiagent.tasks import Integrator, Adjusting, Orientation, \
    Navigation, AgentAgentInteractions, AgentObstacleInteractions, Fluctuation
from crowddynamics.multiagent.examples import RoomEvacuation
from crowddynamics.plugins.game import EgressGame


class RoomEvacuationGame(RoomEvacuation):
    r"""
    Room Evacuation Game
    """
    def __init__(self, queue, size, width, height, model, body_type, spawn_shape,
                 door_width, exit_hall_width, t_aset_0, interval,
                 neighbor_radius, neighborhood_size):
        super(RoomEvacuationGame, self).__init__(
            queue, size, width, height, model, body_type, spawn_shape, door_width,
            exit_hall_width)

        # FIXME
        game = EgressGame(self, self.door, self.room, t_aset_0, interval,
                          neighbor_radius, neighborhood_size)

        self.task_graph = Integrator(self, (0.001, 0.01))
        adjusting = Adjusting(self)
        adjusting += Orientation(self)
        adjusting += Navigation(self)
        self.task_graph += adjusting
        agent_agent_interactions = AgentAgentInteractions(self)
        agent_agent_interactions += game
        self.task_graph += agent_agent_interactions
        self.task_graph += AgentObstacleInteractions(self)
        self.task_graph += Fluctuation(self)
