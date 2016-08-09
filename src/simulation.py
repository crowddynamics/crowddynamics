import numpy as np
import logging as log

from .core.egress import egress_game_attrs
from .core.motion import motion, integrator
from .core.navigation import direction_to_target_angle, navigator
from .functions import filter_none, timed_execution
from .io.attributes import Intervals, Attrs, Attr
from .io.save import Save
from .structure.agent import agent_attr_names
from .structure.result import Result, result_attr_names
from .structure.obstacle import wall_attr_names


class Simulation:
    """
    Class for initialising and running a crowd simulation.
    """
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
        self.saver = None
        self.savers = None
        self.attrs_result = None

        self.configure_save(dirpath, name)

    def configure_save(self, dirpath, name):
        self.saver = Save(dirpath, name)
        args = []

        self.attrs_result = Attrs(result_attr_names)
        attrs_agent = Attrs(agent_attr_names, Intervals(1.0))
        attrs_wall = Attrs(wall_attr_names)

        attrs = ("position", "velocity", "force", "angle", "angular_velocity",
                 "torque")
        for attr in attrs:
            attrs_agent[attr] = Attr(attr, True, True)

        args.append(self.saver.hdf(self.agent, attrs_agent))
        for w in self.wall:
            args.append(self.saver.hdf(w, attrs_wall))

        if self.egress_model is not None:
            attrs_egress = Attrs(egress_game_attrs, Intervals(1.0))
            attrs_egress["strategy"] = Attr("strategy", True, True)
            attrs_egress["time_evac"] = Attr("time_evac", True, True)
            args.append(self.saver.hdf(self.egress_model, attrs_egress))

        self.savers = filter_none(args)

    def load(self):
        pass

    def save(self):
        pass

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

        if self.interval2():
            self.saver.hdf(self.result, self.attrs_result, overwrite=True)

        # Save
        for saver in self.savers:
            saver()

        if len(self.result.in_goal_time) == self.agent.size:
            self.exit()
            return False

        return True

    def exit(self):
        """Exit simulation."""
        print(self.result)
        self.saver.hdf(self.result, self.attrs_result, overwrite=True)
        for save in self.savers:
            save(brute=True)

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