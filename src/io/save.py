import datetime
import os
from collections import Iterable

import h5py
import numpy as np


class Save(object):
    HDF5 = ".hdf5"

    def __init__(self, dirpath, name):
        # Path to root directory and filename
        self.root = dirpath
        self.name = name

        os.makedirs(self.root, exist_ok=True)

        # HDF5
        self.hdf_filepath = os.path.join(self.root, self.name + self.HDF5)
        self.group_name = None
        # New HDF5 File
        with h5py.File(self.hdf_filepath, mode='a') as file:
            # Group Name
            groups = (int(name) for name in file if name.isdigit())
            try:
                self.group_name = "{:04d}".format(max(groups) + 1)
            except ValueError:
                # If generator is empty
                self.group_name = '0'
            # Create Group
            group = file.create_group(self.group_name)
            # Metadata
            group.attrs["timestamp"] = str(datetime.datetime.now())

        # TODO: Bytes saved, memory consumption

    def to_hdf(self, struct, attrs):
        """

        :param struct: numba.jitclass
        :param attrs: attributes of the jitclass
        :return: Generator that saves records and dumps
        """
        struct_name = struct.__class__.__name__.lower()

        if isinstance(attrs, Iterable):
            attrs = tuple(attr for attr in attrs if hasattr(struct, attr.name))
        else:
            attrs = tuple(attr for attr in (attrs,) if hasattr(struct, attr.name))

        if len(attrs) == 0:
            raise ValueError("Struct \"{}\" doesn't contain any of given "
                             "attributes.".format(struct_name))

        # HDF5 File
        # TODO: Attributes for new group?
        with h5py.File(self.hdf_filepath, mode='a') as file:
            base = file[self.group_name]
            # New group for struct
            group = base.create_group(struct_name)
            # Create new datasets
            for attr in attrs:
                value = np.copy(getattr(struct, attr.name))
                if attr.is_resizable:  # Resizable?
                    # Resizable
                    value = np.array(value)
                    maxshape = (None,) + value.shape
                    value = np.expand_dims(value, axis=0)
                    group.create_dataset(attr.name, data=value,
                                         maxshape=maxshape)
                else:
                    # Not Resizable
                    group.create_dataset(attr.name, data=value)

        # TODO: Saving when finished
        recordable = list(filter(lambda attr: attr.save_func is not None, attrs))
        if len(recordable):
            def gen():
                buffer = {attr.name: [] for attr in recordable}
                i = 1
                j = 1
                while True:
                    for attr in recordable:
                        # Append value to buffer
                        value = np.copy(getattr(struct, attr.name))
                        buffer[attr.name].append(value)
                        # Save values to hdf
                        if True:  # TODO: Fix attr.save_func()
                            values = np.array(buffer[attr.name])
                            with h5py.File(self.hdf_filepath, mode='a') as file:
                                dset = file[self.group_name][struct_name][attr.name]
                                new_shape = (i+1,) + values.shape[1:]
                                dset.resize(new_shape)
                                dset[j:] = values
                            buffer[attr.name].clear()
                            j = i
                    i += 1
                    yield
            return gen()
        else:
            return None

    def generic(self, *folders, fname=None, exists_ok=True):
        folder_path = os.path.join(self.root, *folders)
        os.makedirs(folder_path, exist_ok=True)
        if fname is None:
            fname = str(datetime.datetime.now()).replace(" ", "_")
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
