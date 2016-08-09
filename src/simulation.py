import datetime
import logging as log
import os

import numpy as np

from .core.egress import egress_game_attrs
from .core.motion import motion, integrator
from .core.navigation import direction_to_target_angle, navigator
from .functions import filter_none, timed_execution
from .io.attributes import Intervals, Attrs, Attr
from .io.hdfstore import HDFStore
from .structure.agent import agent_attr_names
from .structure.obstacle import wall_attr_names


class Simulation:
    """Class for initialising and running a crowd simulation."""

    def __init__(self,
                 agent,
                 wall=None,
                 goals=None,
                 name=None,
                 dirpath=None,
                 angle_update=direction_to_target_angle,
                 direction_update=None,
                 egress_model=None,
                 domain=None,
                 dt_min=0.001,
                 dt_max=0.01):

        if name is None or len(name) == 0:
            self.name = str(datetime.datetime.now())
        else:
            self.name = name

        self.result_attr_names = (
            "dt_min",
            "dt_max",
            "iterations",
            "time",
            "time_steps",
            "in_goal_time",
        )
        self.attrs_result = Attrs(self.result_attr_names)

        # Integrator timestep
        self.dt_min = dt_min
        self.dt_max = dt_max

        self.iterations = 0
        self.time = 0
        self.time_steps = [0]

        self.in_goal = 0
        self.in_goal_time = []

        # Structures
        self.domain = domain
        self.agent = agent
        self.wall = filter_none(wall)
        self.goals = filter_none(goals)
        self.egress_model = egress_model

        # Angle and direction update algorithms
        self.angle_update = angle_update
        self.direction_update = direction_update

        # Saving data to HDF5 file for analysis and resuming a simulation.
        self.hdfstore = None
        self.configure_save(dirpath)

    def configure_save(self, dirpath):
        log.info("Simulation: Configuring save")

        if dirpath is None or len(dirpath) == 0:
            filepath = self.name
        else:
            os.makedirs(dirpath, exist_ok=True)
            filepath = os.path.join(dirpath, self.name)

        self.hdfstore = HDFStore(filepath)

        attrs_agent = Attrs(agent_attr_names, Intervals(1.0))
        attrs_wall = Attrs(wall_attr_names)

        # TODO: Gui selection for saveable attributes
        attrs = ("position", "velocity", "force", "angle", "angular_velocity",
                 "torque")
        for attr in attrs:
            attrs_agent[attr] = Attr(attr, True, True)

        self.hdfstore.save(self.agent, attrs_agent)
        for w in self.wall:
            self.hdfstore.save(w, attrs_wall)

        if self.egress_model is not None:
            attrs_egress = Attrs(egress_game_attrs, Intervals(1.0))
            attrs_egress["strategy"] = Attr("strategy", True, True)
            attrs_egress["time_evac"] = Attr("time_evac", True, True)
            self.hdfstore.save(self.egress_model, attrs_egress)

    def load(self):
        pass

    def save(self):
        if self.hdfstore is not None:
            self.hdfstore.save(self, self.attrs_result, overwrite=True)
            self.hdfstore.record(brute=True)

    def exit(self):
        log.info("Simulation exit")
        self.save()

    @timed_execution
    def advance(self):
        """
        :return: True -> continues, False -> ends.
        """
        navigator(self.agent, self.angle_update, self.direction_update)
        motion(self.agent, self.wall)
        dt = integrator(self.agent, self.dt_min, self.dt_max)

        self.iterations += 1
        self.time += dt
        self.time_steps.append(dt)

        if self.egress_model is not None:
            self.egress_model.update(self.time, dt)

        self.agent.reset_neighbor()

        if self.domain is not None:
            self.agent.active &= self.domain.contains(self.agent.position)

        # Goals
        for goal in self.goals:
            num = -np.sum(self.agent.goal_reached)
            self.agent.goal_reached |= goal.contains(self.agent.position)
            num += np.sum(self.agent.goal_reached)
            self.in_goal += num
            self.in_goal_time.extend(num * (self.time,))

        # Save
        if self.hdfstore is not None:
            self.hdfstore.record()

        return True

    def run(self, max_iter=np.inf, max_time=np.inf):
        """

        :param max_iter: Iteration limit
        :param max_time: Time (simulation not real time) limit
        :return: None
        """
        while self.advance() and \
              self.iterations < max_iter and \
              self.time < max_time:
            pass
        self.exit()
