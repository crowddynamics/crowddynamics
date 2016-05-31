import os
import numpy as np
import pandas as pd
import h5py
from collections import Iterable


class Save(object):
    """
    Create/Open HDF5 File
    Create Group

        Save metadata:
        - Timestamp
        - Size (Number of agents)

        Create dataset

        Save data in certain intervals and at end:
        - Position

        Save data at the end:
        - Results

    Close
    """
    HDF5 = ".hdf5"

    def __init__(self, dirpath, name):
        self.root = dirpath
        self.name = name
        os.makedirs(self.root)
        # HDF5
        self.hdf_filepath = os.path.join(self.root, self.name + self.HDF5)
        self.group_name = None
        self.new_hdf_group()

    def new_hdf_group(self):
        with h5py.File(self.hdf_filepath, mode='a') as file:
            # Group Name
            groups = (name for name in file if name.isdigit())
            try:
                self.group_name = str(int(max(groups)) + 1)
            except ValueError:
                # If generator is empty
                self.group_name = '0'
            # Create Group
            group = file.create_group(self.group_name)
            # Metadata
            group.attrs["timestamp"] = str(pd.Timestamp)
            group.attrs["size"] = 0  # Agents.size

    def jitclass_to_hdf(self, struct, attrs):
        if isinstance(attrs, Iterable):
            attrs = tuple(attr for attr in attrs if hasattr(struct, attr))
        else:
            attrs = tuple(attr for attr in (attrs,) if hasattr(struct, attr))

        if len(attrs) == 0:
            raise ValueError("Struct doesn't contain any of given attributes.")

        # HDF5 File
        resizables = []
        struct_name = struct.__class__.__name__.lower()
        with h5py.File(self.hdf_filepath, mode='a') as file:
            base = file[self.group_name]
            # New group for struct
            # TODO: HDF attributes?
            group = base.create_group(struct_name)
            # Create new datasets
            for attr in attrs:
                value = getattr(struct, attr.name)
                if attr.is_resizable:  # Resizable?
                    # Resizable
                    resizables.append(attr)
                    value = np.array(value)
                    maxshape = (None,) + value.shape
                    value = np.expand_dims(value, axis=0)
                    group.create_dataset(attr, data=value, maxshape=maxshape)
                else:
                    # Not Resizable
                    group.create_dataset(attr, data=value)

        if len(resizables) == 0:
            return None

        buffer = {attr: [] for attr in resizables}
        index = 1
        while True:
            for attr in resizables:
                # Append value to buffer
                value = getattr(struct, attr.name)
                buffer[attr].append(value)

                # Save values to hdf
                wall_time = NotImplemented
                if attr.interval is not None or attr.interval == 0 or \
                   wall_time % attr.interval == 0:
                    values = np.array(buffer[attr])
                    with h5py.File(self.hdf_filepath, mode='a') as file:
                        dset = file[self.group_name][struct_name][attr]
                        new_shape = (index+1,) + values.shape[1:]
                        dset.resize(new_shape)
                        dset[index:] = values
                    buffer[attr].clear()
            index += 1
            yield

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

    def animation(self, fname=None):
        folder = "animations"
        return self.generic(folder, fname=fname, exists_ok=True)

    def figure(self, fname=None):
        folder = "figures"
        return self.generic(folder, fname=fname, exists_ok=True)

    def result(self, dataframe):
        key = "results"
        return dataframe.to_hdf(self.hdf_filepath, self.group_name + "/" + key)
