import os
from collections import OrderedDict
from functools import lru_cache

import pandas as pd
import ruamel.yaml as yaml


root = os.path.abspath(__file__)
root = os.path.split(root)[0]
folder = "configs"


class Create:
    def attributes(self):
        from src.structure.agent import spec_agent

        ext = ".yaml"
        name = "attributes"
        filepath = os.path.join(root, folder, name + ext)

        # TODO: Add Comments
        data = OrderedDict([('agent', OrderedDict())])

        for item in spec_agent:
            data['agent'][item[0]] = OrderedDict([("resizable", False)])

        data['agent']['position']['resizable'] = True
        data['agent']['angle']['resizable'] = True
        data['agent']['active']['resizable'] = True

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
            return yaml.safe_load(f)
