import datetime
import os

import h5py
import numpy as np

from .attributes import Attrs


def timestamp():
    return str(datetime.datetime.now())


class HDFSaver:
    def __init__(self, struct, struct_name, recordable, hdf_filepath, group_name):
        self.hdf_filepath = hdf_filepath
        self.group_name = group_name
        self.struct_name = struct_name
        self.struct = struct
        self.recordable = recordable
        self.buffer = {attr.name: [] for attr in self.recordable}
        self.start = 0
        self.end = 1

    def __call__(self, brute=False):
        """

        :param brute: If true forces saving
        :return:
        """
        is_save = self.recordable.save_func
        if callable(is_save):
            is_save = is_save()
        for attr in self.recordable:
            # Append value to buffer
            value = np.copy(getattr(self.struct, attr.name))
            self.buffer[attr.name].append(value)

            # Save values to hdf
            if is_save or brute:
                values = np.array(self.buffer[attr.name])
                with h5py.File(self.hdf_filepath, mode='a') as file:
                    dset = file[self.group_name][self.struct_name][attr.name]
                    new_shape = (self.end + 1,) + values.shape[1:]
                    dset.resize(new_shape)
                    dset[self.start + 1:] = values
                self.buffer[attr.name].clear()
        if is_save:
            self.start = self.end
        self.end += 1


class HDFStore(object):
    HDF5 = ".hdf5"

    def __init__(self, path='', name='simulation'):
        """
        :param path: Path to directory
        :param name: Filename
        """
        self.path = path
        self.name = name

        # Make the directory if it doesn't exist
        if len(path) > 0:
            os.makedirs(self.path, exist_ok=True)

        # Path to the HDF5 file
        self.hdf_filepath = os.path.join(self.path, self.name + self.HDF5)
        self.group_name = None
        self.hdf_savers = None
        self.configure_file()

    def configure_file(self):
        # Make new HDF5 File
        with h5py.File(self.hdf_filepath, mode='a') as file:
            # Group Name
            groups = (int(name) for name in file if name.isdigit())
            try:
                num = max(groups) + 1
            except ValueError:
                num = 0  # If generator is empty
            self.group_name = "{:04d}".format(num)

            # Create Group
            group = file.create_group(self.group_name)

            # Metadata
            group.attrs["timestamp"] = timestamp()

    def save(self, struct, attrs: Attrs, overwrite=False):
        """

        :param overwrite:
        :param struct: numba.jitclass
        :param attrs: attributes of the jitclass
        :return: Function that saves records and dumps data into hdf5
        """
        struct_name = struct.__class__.__name__.lower()
        attrs.check_hasattr(struct)

        # HDF5 File
        # TODO: Attributes for new group
        with h5py.File(self.hdf_filepath, mode='a') as file:
            base = file[self.group_name]

            # New group for structure
            if overwrite and (struct_name in base):
                del base[struct_name]
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
                    # if overwrite and (attr.name in group):
                    #     del group[attr.name]
                    group.create_dataset(attr.name, data=value)

        recordable = Attrs(filter(lambda attr: attr.is_recordable, attrs),
                           attrs.save_func)
        if len(recordable) and recordable.save_func is not None:
            return HDFSaver(struct, struct_name, recordable, self.hdf_filepath,
                            self.group_name)
        else:
            return None
