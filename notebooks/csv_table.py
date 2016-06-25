import numpy as np
import pandas as pd

from sympy import latex, sympify
from astropy.units import Unit


def convert(expr):
    if expr is np.nan:
        return ""
    else:
        return r"%s" % latex(sympify(expr), mode="inline", fold_func_brackets=True)


def convert_units(expr):
    if expr is np.nan:
        return ""
    else:
        return r"%s" % Unit(expr).to_string("latex")


def convert_verb(expr):
    if expr is np.nan:
        return ""
    else:
        return r"\verb|%s|" % str(expr)

converters = {
    "name": convert_verb,
    "symbol": convert,
    "value": convert,
    "unit": convert_units,
    "type": str,
    "source": str,
    "explanation": str,
}
df = pd.read_csv("backup/agent.csv")

df["name"] = df["name"].apply(convert_verb)
df["symbol"] = df["symbol"].apply(convert)
df["value"] = df["value"].apply(convert)
df["unit"] = df["unit"].apply(convert_units)
df.fillna("", inplace=True)

with open("agent.tex", "w") as f:
    df.to_latex(f, escape=False)
