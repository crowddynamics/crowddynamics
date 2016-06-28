import os
import numpy as np
import pandas as pd


csv_ext = ".csv"
tex_ext = ".tex"
folder = "/home/jaan/Dropbox/Projects/Crowd-Dynamics/crowd_dynamics/"
table_folder = "tables"

os.makedirs(table_folder, exist_ok=True)

# Don't truncate string in DataFrames
pd.set_option('display.max_colwidth', -1)


def convert_symbol(expr):
    from sympy import latex, sympify
    if expr is not np.nan:
        return r"%s" % latex(sympify(expr), mode="inline",
                             fold_func_brackets=True)


def convert_units(expr):
    from astropy.units import Unit
    if expr is not np.nan:
        return r"%s" % Unit(expr).to_string("latex")


def convert_verb(expr):
    # TODO: Change to custom column type
    if expr is not np.nan:
        return r"\verb|%s|" % str(expr)


def to_vector(expr, unit=False):
    if expr is not np.nan:
        if unit:
            return r"\hat{\mathbf{%s}}" % str(expr)
        else:
            return r"\mathbf{%s}" % str(expr)


def agent(filename="agent_table"):
    formatter = {
        "name": convert_verb,
        "symbol": convert_symbol,
        "value": convert_symbol,
        "unit": convert_units,
        "type": str,
        "source": str,
        "explanation": str,
    }

    load = os.path.join(folder, filename + csv_ext)
    save = os.path.join(table_folder, filename + tex_ext)
    df = pd.read_csv(load)
    del df['type']  # Don't include type
    with open(save, "w") as file:
        df.to_latex(file, escape=False, longtable=False, formatters=formatter,
                    na_rep="")


def body_types(filename="body_types"):
    formatter = {
        "name": convert_verb,
        "symbol": convert_symbol,
        "adult": convert_symbol,
        "male": convert_symbol,
        "female": convert_symbol,
        "child": convert_symbol,
        "eldery": convert_symbol,
        "explanation": str,
    }
    load = os.path.join(folder, filename + csv_ext)
    save = os.path.join(table_folder, filename + tex_ext)
    df = pd.read_csv(load)
    with open(save, "w") as file:
        df.to_latex(file, escape=False, longtable=False, formatters=formatter,
                    na_rep="")

agent()
body_types()
