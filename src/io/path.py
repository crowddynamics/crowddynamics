import os
import pandas as pd


class SimulationPath(object):
    def __init__(self, dirpath):
        self.root = dirpath

    def generic(self, *folders, fname=None, exists_ok=True):
        folder_path = os.path.join(self.root, *folders)
        os.makedirs(folder_path, exist_ok=True)
        if fname is None:
            fname = str(pd.Timestamp)
        file_path = os.path.join(folder_path, fname)
        if not exists_ok and os.path.exists(file_path):
            raise FileExistsError("File: {}".format(file_path))
        else:
            return file_path

    def result(self, fname=None):
        folder = "results"
        return self.generic(folder, fname=fname, exists_ok=True)

    def animation(self, fname=None):
        folder = "animations"
        return self.generic(folder, fname=fname, exists_ok=True)

    def figure(self, fname=None):
        folder = "figures"
        return self.generic(folder, fname=fname, exists_ok=True)
