import datetime
import os
import numpy as np
import logging as log

from .core.egress import egress_game_attrs
from .core.motion import motion, integrator
from .core.navigation import direction_to_target_angle, navigator
from .functions import filter_none, timed_execution
from .io.attributes import Intervals, Attrs, Attr
from .io.hdfstore import HDFStore
from .structure.agent import agent_attr_names
from .structure.result import Result, result_attr_names
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

        # Integrator timestep
        self.dt_min = dt_min
        self.dt_max = dt_max

        # Structures
        self.domain = domain
        self.agent = agent
        self.wall = filter_none(wall)
        self.goals = filter_none(goals)
        self.egress_model = egress_model

        # Simulation results storage structure
        self.result = Result()

        # Angle and direction update algorithms
        self.angle_update = angle_update
        self.direction_update = direction_update

        # Interval for printing the values in result during the simulation.
        self.interval = Intervals(1.0)
        self.interval2 = Intervals(1.0)

        # Saving data to HDF5 file for analysis and resuming a simulation.
        self.hdfstore = None
        self.attrs_result = None

        self.configure_save(dirpath)

    def configure_save(self, dirpath):
        log.info("HDFStore configuring...")

        if dirpath is None or len(dirpath) == 0:
            filepath = self.name
        else:
            os.makedirs(dirpath, exist_ok=True)
            filepath = os.path.join(dirpath, self.name)

        self.hdfstore = HDFStore(filepath)

        self.attrs_result = Attrs(result_attr_names)
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

        log.info("HDFStore configured.")

    def load(self):
        pass

    def save(self):
        if self.hdfstore is not None:
            self.hdfstore.save(self.result, self.attrs_result, overwrite=True)
            self.hdfstore.record(brute=True)

    @timed_execution
    def advance(self):
        """
        :return: False is simulation ends otherwise True.
        """
        navigator(self.agent, self.angle_update, self.direction_update)
        motion(self.agent, self.wall)
        dt = integrator(self.agent, self.dt_min, self.dt_max)

        if self.egress_model is not None:
            self.egress_model.update(self.result.simulation_time, dt)

        self.agent.reset_neighbor()
        self.result.increment_simulation_time(dt)

        if self.domain is not None:
            self.agent.active &= self.domain.contains(self.agent.position)

        # Goals
        for goal in self.goals:
            num = -np.sum(self.agent.goal_reached)
            self.agent.goal_reached |= goal.contains(self.agent.position)
            num += np.sum(self.agent.goal_reached)
            self.result.increment_in_goal_time(num)

        if self.interval():
            print(self.result)

        # Save
        if self.hdfstore is not None:
            if self.interval2():
                self.hdfstore.save(self.result, self.attrs_result,
                                   overwrite=True)
            self.hdfstore.record()

        if len(self.result.in_goal_time) == self.agent.size:
            self.exit()
            return False

        return True

    def exit(self):
        """Exit simulation."""
        print(self.result)
        self.save()

    def run(self, iter_limit=None, simu_time_limit=None):
        """
        :param iter_limit: Execute simulation until number of iterations has been reached or if None run until simulation ends.
        :return: None
        """
        if iter_limit is None:
            iter_limit = np.inf

        if simu_time_limit is None:
            simu_time_limit = np.inf

        while self.advance() and \
                (iter_limit > 0) and \
                (self.result.simulation_time < simu_time_limit):
            iter_limit -= 1
