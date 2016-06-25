import os

import pandas

"""
Tables of anthropometric (human measure) data. These can be used to generate
agents.
"""

root = os.path.abspath(__file__)
root, _ = os.path.split(root)

path1 = os.path.join(root, "body_types.csv")
path2 = os.path.join(root, "agent_table.csv")

# TODO: Values converters.
body_types = pandas.read_csv(path1, index_col=[0])
agent_table = pandas.read_csv(path2, index_col=[0])
