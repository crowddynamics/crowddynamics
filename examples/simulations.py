import numpy as np
from traitlets.traitlets import Float, Int, observe, Enum, default

import examples.fields as fields
from crowddynamics.core.vector2D import unit_vector
from crowddynamics.simulation.agents import Agents, AgentGroup, Circular, \
    ThreeCircle
from crowddynamics.simulation.logic import Reset, Integrator, Fluctuation, \
    Adjusting, AgentAgentInteractions, AgentObstacleInteractions, \
    Orientation, Navigation, InsideDomain
from crowddynamics.simulation.multiagent import MultiAgentSimulation


class Outdoor(MultiAgentSimulation):
    r"""Simulation for visualizing collision avoidance."""
    size = Int(
        default_value=100,
        min=1)
    width = Float(
        default_value=10.0,
        min=0)
    height = Float(
        default_value=10.0,
        min=0)
    agent_type = Enum(
        default_value=Circular,
        values=(Circular, ThreeCircle))
    body_type = Enum(
        default_value='adult',
        values=('adult',))

    @default('logic')
    def _default_logic(self):
        return Reset(self) << (
                Integrator(self) << (
                    Fluctuation(self),
                    Adjusting(self) << Orientation(self),
                    AgentAgentInteractions(self),
                )
            )

    @default('field')
    def _default_field(self):
        return fields.Outdoor(width=self.width, height=self.height)

    @default('agents')
    def _default_agents(self):
        agents = Agents(agent_type=self.agent_type)

        def attributes():
            orientation = np.random.uniform(-np.pi, np.pi)
            return dict(
                body_type=self.body_type,
                orientation=orientation,
                velocity=np.zeros(2),
                angular_velocity=0.0,
                target_direction=unit_vector(orientation),
                target_orientation=orientation)

        group = AgentGroup(
            agent_type=self.agent_type,
            size=self.size,
            attributes=attributes)

        agents.add_non_overlapping_group(group, position_gen=self.field.sample_spawn(0))
        return agents

    # @observe('width', 'height')
    # def _observe_field(self, change):
    #     self.field = fields.Outdoor(
    #         width=self.width,
    #         height=self.height)
    #
    # @observe('size', 'agent_type')
    # def _observe_agents(self, change):
    #     self.agents = Agents(agent_type=self.agent_type)
    #
    #     def attributes():
    #         orientation = np.random.uniform(-np.pi, np.pi)
    #         return dict(
    #             body_type=self.body_type,
    #             orientation=orientation,
    #             velocity=np.zeros(2),
    #             angular_velocity=0.0,
    #             target_direction=unit_vector(orientation),
    #             target_orientation=orientation)
    #
    #     group = AgentGroup(
    #         agent_type=self.agent_type,
    #         size=self.size,
    #         attributes=attributes)
    #
    #     self.agents.add_non_overlapping_group(
    #         group, position_gen=self.field.sample_spawn(0))


class Hallway(MultiAgentSimulation):
    r"""Hallway

    - Low / medium / high crowd density
    - Overtaking
    - Counterflow
    - Bidirectional flow
    - Unidirectional flow

    """
    size = Int(
        default_value=50,
        min=2)
    width = Float(
        default_value=40.0,
        min=0)
    height = Float(
        default_value=5.0,
        min=0)
    ratio = Float(
        default_value=1 / 3,
        min=0, max=1
    )
    agent_type = Enum(
        default_value=Circular,
        values=(Circular, ThreeCircle))
    body_type = Enum(
        default_value='adult',
        values=('adult',))

    @default('logic')
    def _default_logic(self):
        return Reset(self) << \
            InsideDomain(self) << (
                Integrator(self) << (
                    Fluctuation(self),
                    Adjusting(self) << (
                        Navigation(self),
                        Orientation(self)
                    ),
                    AgentAgentInteractions(self),
                    AgentObstacleInteractions(self)
                )
            )

    @default('field')
    def _default_field(self):
        return fields.Hallway(
            width=self.width,
            height=self.height,
            ratio=self.ratio)

    @default('agents')
    def _default_agents(self):
        agents = Agents(agent_type=self.agent_type)

        def attributes1():
            orientation = 0.0
            return dict(body_type=self.body_type,
                        orientation=orientation,
                        velocity=np.zeros(2),
                        angular_velocity=0.0,
                        target_direction=unit_vector(orientation),
                        target_orientation=orientation,
                        target=1)

        def attributes2():
            orientation = np.pi
            return dict(body_type=self.body_type,
                        orientation=orientation,
                        velocity=np.zeros(2),
                        angular_velocity=0.0,
                        target_direction=unit_vector(orientation),
                        target_orientation=orientation,
                        target=0)

        group1 = AgentGroup(size=self.size // 2,
                            agent_type=self.agent_type,
                            attributes=attributes1)
        group2 = AgentGroup(size=self.size // 2,
                            agent_type=self.agent_type,
                            attributes=attributes2)

        agents.add_non_overlapping_group(
            group=group1,
            position_gen=self.field.sample_spawn(0))
        agents.add_non_overlapping_group(
            group=group2,
            position_gen=self.field.sample_spawn(1))
        return agents

    # @observe('width', 'height', 'ratio')
    # def _observe_field(self, change):
    #     self.field = fields.Hallway(
    #         width=self.width,
    #         height=self.height,
    #         ratio=self.ratio)
    #
    # @observe('size', 'agent_type')
    # def _observe_agents(self, change):
    #     agents = Agents(agent_type=self.agent_type)
    #
    #     def attributes1():
    #         orientation = 0.0
    #         return dict(body_type=self.body_type,
    #                     orientation=orientation,
    #                     velocity=np.zeros(2),
    #                     angular_velocity=0.0,
    #                     target_direction=unit_vector(orientation),
    #                     target_orientation=orientation,
    #                     target=1)
    #
    #     def attributes2():
    #         orientation = np.pi
    #         return dict(body_type=self.body_type,
    #                     orientation=orientation,
    #                     velocity=np.zeros(2),
    #                     angular_velocity=0.0,
    #                     target_direction=unit_vector(orientation),
    #                     target_orientation=orientation,
    #                     target=0)
    #
    #     group1 = AgentGroup(size=self.size // 2,
    #                         agent_type=self.agent_type,
    #                         attributes=attributes1)
    #     group2 = AgentGroup(size=self.size // 2,
    #                         agent_type=self.agent_type,
    #                         attributes=attributes2)
    #
    #     agents.add_non_overlapping_group(
    #         group=group1,
    #         position_gen=self.field.sample_spawn(0))
    #     agents.add_non_overlapping_group(
    #         group=group2,
    #         position_gen=self.field.sample_spawn(1))
    #     self.agents = agents
