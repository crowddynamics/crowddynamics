from functools import partial

import numpy as np
from traitlets.traitlets import Float, Int, observe, Enum, default

from crowddynamics.core.geometry import geom_to_linear_obstacles
from crowddynamics.core.vector2D import unit_vector
from crowddynamics.examples import fields
from crowddynamics.simulation.agents import Agents, AgentGroup, Circular, \
    ThreeCircle
from crowddynamics.simulation.logic import Reset, Integrator, Fluctuation, \
    Adjusting, AgentAgentInteractions, AgentObstacleInteractions, \
    Orientation, Navigation, InsideDomain, LeaderFollowerWithHerding
from crowddynamics.simulation.multiagent import MultiAgentSimulation


class TestMovement(MultiAgentSimulation):
    """Test simulation for testing agent movement."""
    size = 1
    width = 20.0
    height = 10.0
    agent_type = Enum(
        values=(Circular, ThreeCircle),
        allow_none=True)

    def attributes(self, orientation=0.0):
        d = dict(
            ratio_rt=0.5882,
            ratio_rs=0.3725,
            ratio_ts=0.6275,
            radius=0.255,
            target_velocity=1.0,
            mass=73.5,
            orientation=orientation,
            velocity=1.0 * unit_vector(orientation),
            angular_velocity=0.0,
            target_direction=unit_vector(orientation),
            target_orientation=orientation)
        return d

    @default('logic')
    def _default_logic(self):
        return Reset(self) << Integrator(self) << Adjusting(
            self) << Orientation(self)

    @default('field')
    def _default_field(self):
        return fields.OutdoorField(width=self.width, height=self.height)

    @observe('agent_type')
    def _observe_agents(self, change):
        if self.agent_type is not None:
            agents = Agents(agent_type=self.agent_type)

            group = AgentGroup(
                agent_type=self.agent_type,
                size=self.size,
                attributes=self.attributes)

            agents.add_non_overlapping_group(
                group, position_gen=lambda: np.array((0, 0)))

            self.agents = agents


class TestAgentInteraction(MultiAgentSimulation):
    start_pos1 = (0, 0)
    start_pos2 = (5, 0)

    width = 20.0
    height = 10.0
    agent_type = Enum(
        values=(Circular, ThreeCircle),
        allow_none=True)

    def attributes(self, orientation=0.0):
        d = dict(
            ratio_rt=0.5882,
            ratio_rs=0.3725,
            ratio_ts=0.6275,
            radius=0.255,
            target_velocity=1.0,
            mass=73.5,
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

    @observe('agent_type')
    def _observe_agents(self, change):
        if self.agent_type is not None:
            agents = Agents(agent_type=self.agent_type)

            group = AgentGroup(
                agent_type=self.agent_type,
                size=1,
                attributes=partial(self.attributes, orientation=0.0))

            agents.add_non_overlapping_group(
                group, position_gen=lambda: self.start_pos1)

            group = AgentGroup(
                agent_type=self.agent_type,
                size=1,
                attributes=partial(self.attributes, orientation=np.pi))

            agents.add_non_overlapping_group(
                group, position_gen=lambda: self.start_pos2)

            self.agents = agents
