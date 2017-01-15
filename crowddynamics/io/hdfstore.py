import datetime
import logging
import os

import h5py
import numpy as np


class ListBuffer(list):
    """List that tracks start and end indices of added items."""

    def __init__(self, start=0, end=0):
        """
        Args:
            start (int):
            end (int):
        """
        super(ListBuffer, self).__init__()
        self.start = start
        self.end = end

    def append(self, p_object):
        """
        Append

        Args:
            p_object (object):
        """
        super(ListBuffer, self).append(p_object)
        self.end += 1

    def clear(self):
        """Clear"""
        super(ListBuffer, self).clear()
        self.start = self.end


class HDFStore(object):
    """
    Class for saving object's array or scalar data in hdf5 file. Data can be
    saved once or made bufferable so that new data points can be added and
    dumped into the hdf5 file.
    """
    ext = ".hdf5"

    def __init__(self, filepath):
        """
        Args:
            filepath (str):
        """
        self.logger = logging.getLogger("crowddynamics.io")

        # Path to the HDF5 file
        self.filepath = os.path.splitext(filepath)[0] + self.ext
        self.group_name = None

        # Appending data
        self.buffers = []  # (struct, buffers)

        # Configuration
        self.configure_file()

    def configure_file(self):
        """Configure and creates new HDF5 File."""
        self.logger.info("")

        timestamp = str(datetime.datetime.now())
        with h5py.File(self.filepath, mode='a') as file:
            self.group_name = timestamp.replace(" ", "_")  # HDF group name
            file.create_group(self.group_name)  # Create Group

        self.logger.info(self.filepath)
        self.logger.info(self.group_name)

    def create_dataset(self, group, name, values, resizable=False):
        """
        Create dataset

        Args:
            group (h5py.Group):
            name (str):
                Name
            values (numpy.ndarray):
                Values to be stored. Goes through np.array(value).
            resizable (Boolean):
                If true values can be added to the dataset.
        """
        self.logger.info("")
        values = np.array(values)
        kw = {}
        if resizable:
            values = np.array(values)
            maxshape = (None,) + values.shape
            kw.update(maxshape=maxshape)
            values = np.expand_dims(values, axis=0)
        group.create_dataset(name, data=values, **kw)

    def append_buffer_to_dataset(self, dset, buffer):
        """
        Append values to resizable h5py dataset.

        Args:
            dset (h5py.Dataset):
            buffer (ListBuffer):
        """
        if len(buffer):  # Buffer is not empty
            self.logger.info("")
            values = np.array(buffer)
            new_shape = (buffer.end,) + values.shape[1:]
            dset.resize(new_shape)
            dset[buffer.start:] = values
        else:
            self.logger.warning("Buffer is empty.")

    def add_dataset(self, struct, attributes, overwrite=False):
        """

        Args:
            struct:
            attributes:
            overwrite (Boolean):
        """
        self.logger.info("")

        with h5py.File(self.filepath, mode='a') as file:
            name = struct.__class__.__name__.lower()
            base = file[self.group_name]  # New group for structure
            if overwrite and (name in base):
                del base[name]  # Delete existing dataset
            group = base.create_group(name)

            # Create new datasets
            for name, settings in attributes.items():
                value = np.copy(getattr(struct, name))
                self.create_dataset(group, name, value, settings["resizable"])

        self.logger.info("")

    def add_buffers(self, struct, attributes):
        """
        struct
        buffers:
          attr.name: buffer1
          attr.name: buffer2
          ...

        Args:
            struct:
            attributes:
        """
        buffers = {}
        for name, settings in attributes.items():
            if settings["resizable"]:
                buffers[name] = ListBuffer(start=1, end=1)

        self.buffers.append((struct, buffers))

    def update_buffers(self):
        """Update buffers"""
        for struct, buffers in self.buffers:
            for attr_name, buffer in buffers.items():
                value = getattr(struct, attr_name)
                value = np.copy(value)
                buffer.append(value)

    def dump_buffers(self):
        """Dump buffers"""
        with h5py.File(self.filepath, mode='a') as file:
            grp = file[self.group_name]
            for struct, buffers in self.buffers:
                struct_name = struct.__class__.__name__.lower()
                for attr_name, buffer in buffers.items():
                    dset = grp[struct_name][attr_name]
                    self.append_buffer_to_dataset(dset, buffer)
                    buffer.clear()
