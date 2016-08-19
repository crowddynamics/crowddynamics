import os
from functools import lru_cache

import yaml
import pandas as pd

from collections import OrderedDict


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
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
    root = os.path.split(root)[0]
    filenames = ("agent", "body")

    @lru_cache()
    def csv(self, name):
        ext = ".csv"
        # TODO: converters. Evaluate to values.
        if name in self.filenames:
            path = os.path.join(self.root, name + ext)
            return pd.read_csv(path, index_col=[0])

    @lru_cache()
    def yaml(self, filepath):
        """Ordered yaml loader
        http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
        """
        ext = ".yaml"
        with open(filepath) as f:
            return ordered_load(f, yaml.SafeLoader)
