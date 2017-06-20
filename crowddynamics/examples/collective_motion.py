import numpy as np
from traitlets import Int, Enum, Float, default

from crowddynamics.core.geometry import geom_to_linear_obstacles
from crowddynamics.core.vector2D import unit_vector
from crowddynamics.examples import fields
from crowddynamics.simulation.agents import Circular, ThreeCircle, NO_TARGET, \
    Agents, AgentGroup
from crowddynamics.simulation.logic import Reset, InsideDomain, Integrator, \
    Fluctuation, Adjusting, Navigation, ExitDetection, LeaderFollower, \
    Orientation, AgentAgentInteractions, AgentObstacleInteractions
from crowddynamics.simulation.multiagent import MultiAgentSimulation


class Outdoor(MultiAgentSimulation):
    r"""Simulation for visualizing collision avoidance."""
    size_leaders = Int(
        default_value=1,
        min=0, max=1,
        help='Amount of active agents')
    size_herding = Int(
        default_value=150,
        min=0,
        help='Amount of herding agents')
    agent_type = Enum(
        default_value=Circular,
        values=(Circular, ThreeCircle))
    body_type = Enum(
        default_value='adult',
        values=('adult',))
    width = Float(
        default_value=40.0,
        min=0)
    height = Float(
        default_value=40.0,
        min=0)

    def attributes(self, is_leader):
        def wrapper():
            if is_leader:
                orientation = 0.0
            else:
                orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                is_leader=is_leader,
                is_herding=not is_leader,
                body_type=self.body_type,
                orientation=orientation,
                velocity=1.0 * unit_vector(orientation),
                angular_velocity=0.0,
                target_direction=unit_vector(orientation),
                target_orientation=orientation)
            return d
        return wrapper

    @default('logic')
    def _default_logic(self):
        return Reset(self) << (
                Integrator(self) << (
                    Fluctuation(self),
                    Adjusting(self) << (
                        LeaderFollower(self),
                        Orientation(self),),
                    AgentAgentInteractions(self),))

    @default('field')
    def _default_field(self):
        return fields.OutdoorField(width=self.width, height=self.height)

    @default('agents')
    def _default_agents(self):
        agents = Agents(agent_type=self.agent_type)

        group_leader = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_leaders,
            attributes=self.attributes(is_leader=True))

        agents.add_non_overlapping_group(
            group_leader,
            position_gen=lambda: np.array((0.8 * self.width, 0.5 * self.height)))

        group_herding = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_herding,
            attributes=self.attributes(is_leader=False))

        agents.add_non_overlapping_group(
            group_herding, position_gen=self.field.sample_spawn(0))

        return agents


class Rounding(MultiAgentSimulation):
    size_leaders = Int(
        default_value=1,
        min=0,
        help='Amount of active agents')
    size_herding = Int(
        default_value=50,
        min=0,
        help='Amount of herding agents')
    agent_type = Enum(
        default_value=Circular,
        values=(Circular, ThreeCircle))
    body_type = Enum(
        default_value='adult',
        values=('adult',))
    width = Float(
        default_value=30.0,
        min=0)
    height = Float(
        default_value=20.0,
        min=0)

    def attributes(self, has_target: bool=True, is_herding: bool=False):
        def wrapper():
            rand_target = np.random.randint(0, len(self.field.targets))
            target = rand_target if has_target else NO_TARGET
            orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                target=target,
                is_leader=not is_herding,
                is_herding=is_herding,
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
                        LeaderFollower(self),
                        Orientation(self)),
                    AgentAgentInteractions(self),
                    AgentObstacleInteractions(self)))

    @default('field')
    def _default_field(self):
        return fields.Rounding(width=self.width, height=self.height)

    @default('agents')
    def _default_agents(self):
        agents = Agents(agent_type=self.agent_type)

        group_active = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_leaders,
            attributes=self.attributes(has_target=True, is_herding=False))

        group_herding = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_herding,
            attributes=self.attributes(has_target=False, is_herding=True))

        for group in (group_active, group_herding):
            agents.add_non_overlapping_group(
                group,
                position_gen=self.field.sample_spawn(0),
                obstacles=geom_to_linear_obstacles(self.field.obstacles))

        return agents


class ClosedRoom(MultiAgentSimulation):
    r"""Simulation for visualizing collision avoidance."""
    size_leaders = Int(
        default_value=0,
        min=0,
        help='Amount of active agents')
    size_herding = Int(
        default_value=150,
        min=0,
        help='Amount of herding agents')
    agent_type = Enum(
        default_value=Circular,
        values=(Circular, ThreeCircle))
    body_type = Enum(
        default_value='adult',
        values=('adult',))
    width = Float(
        default_value=40.0,
        min=0)
    height = Float(
        default_value=40.0,
        min=0)

    def attributes(self, is_leader):
        def wrapper():
            orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                is_leader=is_leader,
                is_herding=not is_leader,
                body_type=self.body_type,
                orientation=orientation,
                velocity=1.0 * unit_vector(orientation),
                angular_velocity=0.0,
                target_direction=unit_vector(orientation),
                target_orientation=orientation)
            return d
        return wrapper

    @default('logic')
    def _default_logic(self):
        return Reset(self) << (
                Integrator(self) << (
                    Fluctuation(self),
                    Adjusting(self) << (
                        LeaderFollower(self),
                        Orientation(self),),
                    AgentAgentInteractions(self),
                    AgentObstacleInteractions(self)))

    @default('field')
    def _default_field(self):
        return fields.ClosedRoom(width=self.width, height=self.height)

    @default('agents')
    def _default_agents(self):
        agents = Agents(agent_type=self.agent_type)

        group_leader = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_leaders,
            attributes=self.attributes(is_leader=True))

        agents.add_non_overlapping_group(
            group_leader, position_gen=self.field.sample_spawn(0))

        group_herding = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_herding,
            attributes=self.attributes(is_leader=False))

        agents.add_non_overlapping_group(
            group_herding, position_gen=self.field.sample_spawn(0))

        return agents


class FourExits(MultiAgentSimulation):
    size_leaders = Int(
        default_value=4,
        min=0,
        help='Amount of active agents')
    size_herding = Int(
        default_value=150,
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

    def attributes(self, has_target: bool=True, is_herding: bool=False):
        def wrapper():
            rand_target = np.random.randint(0, len(self.field.targets))
            target = rand_target if has_target else NO_TARGET
            orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                target=target,
                is_leader=not is_herding,
                is_herding=is_herding,
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
                        Navigation(self) << ExitDetection(self),
                        LeaderFollower(self),
                        Orientation(self)),
                    AgentAgentInteractions(self),
                    AgentObstacleInteractions(self)))

    @default('field')
    def _default_field(self):
        return fields.FourExitsField(exit_width=self.exit_width)

    @default('agents')
    def _default_agents(self):
        agents = Agents(agent_type=self.agent_type)

        group_active = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_leaders,
            attributes=self.attributes(has_target=True, is_herding=False))

        group_herding = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_herding,
            attributes=self.attributes(has_target=False, is_herding=True))

        for group in (group_active, group_herding):
            agents.add_non_overlapping_group(
                group,
                position_gen=self.field.sample_spawn(0),
                obstacles=geom_to_linear_obstacles(self.field.obstacles))

        return agents
