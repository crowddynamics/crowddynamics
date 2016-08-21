import os
from functools import lru_cache

import pandas as pd

from collections import OrderedDict

try:
    from ruamel import yaml
except ImportError:
    import yaml


class Config:
    root = os.path.abspath(__file__)
    root = os.path.split(root)[0]

    def structure(self):
        from src.structure.agent import spec_agent
        from src.structure.obstacle import spec_linear
        ext = ".yaml"
        name = "structure"
        filepath = os.path.join(self.root, name + ext)

        d = {
            "agent": spec_agent,
            "walls": spec_linear
        }

        data = {}
        for name, spec in d.items():
            data[name] = [item[0] for item in spec]

        with open(filepath, "w") as f:
            yaml.safe_dump(data, stream=f, default_flow_style=False)


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """
    http://stackoverflow.com/questions/5121931/in-python-how-can-you-load- yaml-mappings-as-ordereddicts
    """

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


class Load:
    root = os.path.abspath(__file__)
    # TODO: converters. Evaluate to values.

    root = os.path.split(root)[0]

    @lru_cache()
    def csv(self, name):
        """Load csv with pandas."""
        ext = ".csv"
        path = os.path.join(self.root, name + ext)
        return pd.read_csv(path, index_col=[0])

    @lru_cache()
    def yaml(self, name):
        """Load yaml with ordered loader."""
        ext = ".yaml"
        path = os.path.join(self.root, name + ext)
        with open(path) as f:
            return yaml.safe_load(f)
