import numpy as np
from traitlets.traitlets import Float, Int, observe, Enum, default

from crowddynamics.core.geometry import geom_to_linear_obstacles
from crowddynamics.core.vector2D import unit_vector
from crowddynamics.examples import fields
from crowddynamics.simulation.agents import Agents, AgentGroup, Circular, \
    ThreeCircle
from crowddynamics.simulation.logic import Reset, Integrator, Fluctuation, \
    Adjusting, AgentAgentInteractions, AgentObstacleInteractions, \
    Orientation, Navigation, InsideDomain, Herding
from crowddynamics.simulation.multiagent import MultiAgentSimulation


# TODO: agent_type traits should accept classes or class.__name__
# TODO: do not use observe patterns


class Outdoor(MultiAgentSimulation):
    r"""Simulation for visualizing collision avoidance."""
    size = Int(
        default_value=100,
        min=1)
    width = Float(
        default_value=20.0,
        min=0)
    height = Float(
        default_value=20.0,
        min=0)
    agent_type = Enum(
        default_value=Circular,
        values=(Circular, ThreeCircle))
    body_type = Enum(
        default_value='adult',
        values=('adult',))

    def attributes(self):
        orientation = np.random.uniform(-np.pi, np.pi)
        d = dict(
            body_type=self.body_type,
            orientation=orientation,
            velocity=1.0 * unit_vector(orientation),
            angular_velocity=0.0,
            target_direction=unit_vector(orientation),
            target_orientation=orientation)
        return d

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
        return fields.OutdoorField(width=self.width, height=self.height)

    @default('agents')
    def _default_agents(self):
        agents = Agents(agent_type=self.agent_type)

        group = AgentGroup(
            agent_type=self.agent_type,
            size=self.size,
            attributes=self.attributes)

        agents.add_non_overlapping_group(
            group, position_gen=self.field.sample_spawn(0))

        return agents


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
        min=0, max=1)
    agent_type = Enum(
        default_value=Circular,
        values=(Circular, ThreeCircle),)
    body_type = Enum(
        default_value='adult',
        values=('adult',))

    def attributes1(self):
        orientation = 0.0
        return dict(body_type=self.body_type,
                    orientation=orientation,
                    velocity=unit_vector(orientation),
                    angular_velocity=0.0,
                    target_direction=unit_vector(orientation),
                    target_orientation=orientation,
                    target=1)

    def attributes2(self):
        orientation = np.pi
        return dict(body_type=self.body_type,
                    orientation=orientation,
                    velocity=unit_vector(orientation),
                    angular_velocity=0.0,
                    target_direction=unit_vector(orientation),
                    target_orientation=orientation,
                    target=0)

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
        return fields.HallwayField(
            width=self.width,
            height=self.height,
            ratio=self.ratio)

    @default('agents')
    def _default_agents(self):
        agents = Agents(agent_type=self.agent_type)

        group1 = AgentGroup(size=self.size // 2,
                            agent_type=self.agent_type,
                            attributes=self.attributes1)
        group2 = AgentGroup(size=self.size // 2,
                            agent_type=self.agent_type,
                            attributes=self.attributes2)

        agents.add_non_overlapping_group(
            group=group1,
            position_gen=self.field.sample_spawn(0))
        agents.add_non_overlapping_group(
            group=group2,
            position_gen=self.field.sample_spawn(1))

        return agents


class FourExits(MultiAgentSimulation):
    size_active = Int(
        default_value=1,
        min=1,
        help='Amount of active agents')
    size_herding = Int(
        default_value=99,
        min=0,
        help='Amount of herding agents')
    agent_type = Enum(
        default_value=Circular,
        values=(Circular, ThreeCircle))
    body_type = Enum(
        default_value='adult',
        values=('adult',))
    exit_width = Float(
        default_value=1.25,
        min=0, max=10)

    def attributes(self, has_target: bool=True, herding: bool=False):
        def wrapper():
            target = np.random.randint(0, 4) if has_target else -1
            orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                target=target,
                herding=herding,
                body_type=self.body_type,
                orientation=orientation,
                velocity=np.zeros(2),
                angular_velocity=0.0,
                target_direction=np.zeros(2),
                target_orientation=orientation)
            return d
        return wrapper

    @default('logic')
    def _default_logic(self):
        return Reset(self) << \
            InsideDomain(self) << (
                Integrator(self) << (
                    Fluctuation(self),
                    Adjusting(self) << (
                        Navigation(self),
                        Herding(self),
                        Orientation(self)
                    ),
                    AgentAgentInteractions(self),
                    AgentObstacleInteractions(self)
                )
            )

    @default('field')
    def _default_field(self):
        return fields.FourExitsField(exit_width=self.exit_width)

    @default('agents')
    def _default_agents(self):
        agents = Agents(agent_type=self.agent_type)

        group_active = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_active,
            attributes=self.attributes(has_target=True, herding=False))

        group_herding = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_herding,
            attributes=self.attributes(has_target=False, herding=True))

        for group in (group_active, group_herding):
            agents.add_non_overlapping_group(
                group,
                position_gen=self.field.sample_spawn(0),
                obstacles=geom_to_linear_obstacles(self.field.obstacles))

        return agents
