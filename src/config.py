import os
from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache

import pandas as pd
import ruamel.yaml as yaml


root = os.path.abspath(__file__)
root = os.path.split(root)[0]
folder = "configs"


class Create:
    def parameters(self):
        from src.multiagent.agent import spec_agent

        ext = ".yaml"
        name = "parameters"
        filepath = os.path.join(root, folder, name + ext)

        # TODO: Add Comments
        default = OrderedDict([("resizable", False),
                               ("graphics", False)])

        data = OrderedDict([('simulation', OrderedDict()),
                            ('agent', OrderedDict()), ])

        for item in spec_agent:
            data['agent'][item[0]] = deepcopy(default)

        # Mutable values that are stored
        resizable = ("position", "angle", "active")
        for item in resizable:
            data['agent'][item]['resizable'] = True

        # Values to be updated in graphics
        graphics = ("position", "angle", "active")
        for item in graphics:
            data['agent'][item]['graphics'] = True

        parameters = ("dt_min", "dt_max", "time_tot", "in_goal")
        for item in parameters:
            data['simulation'][item] = deepcopy(default)

        resizable = ("time_tot", "in_goal")
        for item in resizable:
            data['simulation'][item]['resizable'] = True

        game = (
            "strategies",
            "strategy",
            "t_aset_0",
            "t_evac",
            "interval",
        )
        for item in game:
            # data['game'][item]['resizable'] = True
            pass

        with open(filepath, "w") as file:
            yaml.dump(data,
                      stream=file,
                      Dumper=yaml.RoundTripDumper,
                      default_flow_style=False)


class Load:
    @lru_cache()
    def csv(self, name):
        """Load csv with pandas."""
        ext = ".csv"
        path = os.path.join(root, folder, name + ext)
        return pd.read_csv(path, index_col=[0])

    @lru_cache()
    def yaml(self, name):
        """Load yaml with ordered loader."""
        ext = ".yaml"
        path = os.path.join(root, folder, name + ext)
        with open(path) as f:
            return yaml.load(f, Loader=yaml.Loader)
