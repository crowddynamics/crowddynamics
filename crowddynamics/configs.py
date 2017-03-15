import os
import csv

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CFG_DIR = os.path.join(BASE_DIR, 'crowddynamics')


def load_csv(filename: str) -> dict:
    """Load csv configuration

    http://stackoverflow.com/questions/6740918/creating-a-dictionary-from-a-csv-file

    Args:
        filename (str):

    Returns:
        dict:
    """
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)

        return


BODIES = pd.read_csv(os.path.join(CFG_DIR, 'body.csv'), index_col=[0])
