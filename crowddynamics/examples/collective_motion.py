import numpy as np
from traitlets import Int, Enum, Float, default

from crowddynamics.core.geometry import geom_to_linear_obstacles
from crowddynamics.core.vector2D import unit_vector
from crowddynamics.examples import fields
from crowddynamics.examples.fields import rectangle
from crowddynamics.simulation.agents import Circular, ThreeCircle, NO_TARGET, \
    Agents, AgentGroup
from crowddynamics.simulation.logic import Reset, InsideDomain, Integrator, \
    Fluctuation, Adjusting, Navigation, ExitDetection, \
    LeaderFollowerWithHerding, \
    Orientation, AgentAgentInteractions, AgentObstacleInteractions, \
    LeaderFollower, TargetReached
from crowddynamics.simulation.multiagent import MultiAgentSimulation


class Outdoor(MultiAgentSimulation):
    r"""Simulation for visualizing collision avoidance."""
    size_leaders = Int(
        default_value=1,
        min=0, max=1,
        help='Amount of active agents')
    size_herding = Int(
        default_value=100,
        min=0,
        help='Amount of herding agents')
    agent_type = Enum(
        default_value=Circular,
        values=(Circular, ThreeCircle))
    body_type = Enum(
        default_value='adult',
        values=('adult',))
    width = Float(
        default_value=20.0,
        min=0)
    height = Float(
        default_value=20.0,
        min=0)
    vertical_ratio =  Float(
        default_value=0.5,
        min=-0.5, max=1.5)
    horizontal_ratio = Float(
        default_value=0.7,
        min=-0.5, max=1.5)

    def attributes(self, is_leader):
        def wrapper():
            if is_leader:
                orientation = 0.0
            else:
                orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                is_leader=is_leader,
                is_follower=not is_leader,
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
                        LeaderFollowerWithHerding(self),
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
            position_gen=lambda: np.array((self.horizontal_ratio * self.width,
                                           self.vertical_ratio * self.height)))

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

    def attributes(self, has_target: bool=True, is_follower: bool=False):
        def wrapper():
            rand_target = np.random.randint(0, len(self.field.targets))
            target = rand_target if has_target else NO_TARGET
            orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                target=target,
                is_leader=not is_follower,
                is_follower=is_follower,
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
                        Navigation(self) << LeaderFollowerWithHerding(self),
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
            attributes=self.attributes(has_target=True, is_follower=False))

        group_herding = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_herding,
            attributes=self.attributes(has_target=False, is_follower=True))

        for group in (group_active, group_herding):
            agents.add_non_overlapping_group(
                group,
                position_gen=self.field.sample_spawn(0),
                obstacles=geom_to_linear_obstacles(self.field.obstacles))

        return agents


class AvoidObstacle(MultiAgentSimulation):
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
        default_value=10.0,
        min=0)
    height = Float(
        default_value=10.0,
        min=0)
    exit_width = Float(
        default_value=1.25,
        min=0, max=10)
    ratio_obs = Float(
        default_value=0.6,
        min=0, max=1)

    def attributes(self, has_target: bool=True, is_follower: bool=False):
        def wrapper():
            rand_target = np.random.randint(0, len(self.field.targets))
            target = rand_target if has_target else NO_TARGET
            orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                target=target,
                is_leader=not is_follower,
                is_follower=is_follower,
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
                        Navigation(self) << ExitDetection(
                            self, detection_range=(1 - self.ratio_obs) * self.width) << LeaderFollowerWithHerding(self),
                        Orientation(self)),
                    AgentAgentInteractions(self),
                    AgentObstacleInteractions(self)))

    @default('field')
    def _default_field(self):
        return fields.AvoidObstacle(
            width=self.width, height=self.height, exit_width=self.exit_width,
            ratio_obs=self.ratio_obs)

    @default('agents')
    def _default_agents(self):
        agents = Agents(agent_type=self.agent_type)

        group_leader = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_leaders,
            attributes=self.attributes(has_target=True, is_follower=False))

        group_follower = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_herding,
            attributes=self.attributes(has_target=False, is_follower=True))

        for group, spawn in zip((group_leader, group_follower), (0, 1)):
            agents.add_non_overlapping_group(
                group,
                position_gen=self.field.sample_spawn(spawn),
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
        default_value=20.0,
        min=0)
    height = Float(
        default_value=20.0,
        min=0)

    def attributes(self, is_leader):
        def wrapper():
            orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                is_leader=is_leader,
                is_follower=not is_leader,
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
                        LeaderFollowerWithHerding(self),
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


class AroundCircle(MultiAgentSimulation):
    width = Float(
        default_value=20.0,
        min=0)
    height = Float(
        default_value=20.0,
        min=0)
    radius_pillar = Float(
        default_value=1.0,
        min=0)
    radius_trajectory = Float(
        default_value=5.0,
        min=0)

    @default('logic')
    def _default_logic(self):
        return Reset(self) << (
            Integrator(self) << (
                Fluctuation(self),
                Adjusting(self) << (
                    LeaderFollowerWithHerding(self),
                    Orientation(self),),
                AgentAgentInteractions(self),
                AgentObstacleInteractions(self)))

    @default('field')
    def _default_field(self):
        return fields.PillarInTheMiddle(
            width=self.width, height=self.height,
            radius_pillar=self.radius_pillar)

    @default('agents')
    def _default_agents(self):
        return


class AvoidPillar(MultiAgentSimulation):
    pass


class FourExitsRandomPlacing(MultiAgentSimulation):
    size_leaders = Int(
        default_value=4,
        min=0,
        help='Amount of active agents')
    size_herding = Int(
        default_value=100,
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

    def attributes(self, has_target: bool=True, is_follower: bool=False):
        def wrapper():
            rand_target = np.random.randint(0, len(self.field.targets))
            target = rand_target if has_target else NO_TARGET
            orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                target=target,
                is_leader=not is_follower,
                is_follower=is_follower,
                body_type=self.body_type,
                orientation=orientation,
                velocity=np.zeros(2),
                angular_velocity=0.0,
                target_direction=np.zeros(2),
                target_orientation=orientation,
                familiar_exit=np.random.randint(0, len(self.field.targets)))
            return d
        return wrapper

    @default('logic')
    def _default_logic(self):
        return Reset(self) << \
            InsideDomain(self) << (
                Integrator(self) << (
                    Fluctuation(self),
                    Adjusting(self) << (
                        Navigation(self) << ExitDetection(self) << LeaderFollower(self),
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
            attributes=self.attributes(has_target=True, is_follower=False))

        group_herding = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_herding,
            attributes=self.attributes(has_target=False, is_follower=True))

        for group in (group_active, group_herding):
            agents.add_non_overlapping_group(
                group,
                position_gen=self.field.sample_spawn(0),
                obstacles=geom_to_linear_obstacles(self.field.obstacles))

        return agents


class FourExitsFixedPlacing(MultiAgentSimulation):
    size_leaders = Int(
        default_value=4,
        min=4, max=4,
        help='Amount of active agents')
    size_herding = Int(
        default_value=100,
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

    def attributes(self, target: int = NO_TARGET, is_follower: bool=False):
        def wrapper():
            orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                target=target,
                is_leader=not is_follower,
                is_follower=is_follower,
                body_type=self.body_type,
                orientation=orientation,
                velocity=np.zeros(2),
                angular_velocity=0.0,
                target_direction=np.zeros(2),
                target_orientation=orientation,
                familiar_exit=np.random.randint(0, len(self.field.targets)))
            return d
        return wrapper

    @default('logic')
    def _default_logic(self):
        return Reset(self) << \
            InsideDomain(self) << TargetReached(self) << (
                Integrator(self) << (
                    Fluctuation(self),
                    Adjusting(self) << (
                        Navigation(self) << ExitDetection(self) << LeaderFollower(self),
                        Orientation(self)),
                    AgentAgentInteractions(self),
                    AgentObstacleInteractions(self)))

    @default('field')
    def _default_field(self):
        return fields.FourExitsField(exit_width=self.exit_width)

    @default('agents')
    def _default_agents(self):
        agents = Agents(agent_type=self.agent_type)
        obstacles = geom_to_linear_obstacles(self.field.obstacles)

        # Add new spawns to the field for the leaders
        self.field.spawns.extend([
            rectangle(25, 45, 10, 10),
            rectangle(80, 65, 10, 10),
            rectangle(75, 35, 10, 10),
            rectangle(35, 5, 10, 10),
        ])

        for i in range(self.size_leaders):
            group_leader = AgentGroup(
                agent_type=self.agent_type,
                size=1,
                attributes=self.attributes(target=i, is_follower=False))

            agents.add_non_overlapping_group(
                group_leader,
                position_gen=self.field.sample_spawn(i + 1),
                obstacles=obstacles)

        group_herding = AgentGroup(
            agent_type=self.agent_type,
            size=self.size_herding,
            attributes=self.attributes(target=NO_TARGET, is_follower=True))

        agents.add_non_overlapping_group(
            group_herding,
            position_gen=self.field.sample_spawn(0),
            obstacles=obstacles)

        return agents
