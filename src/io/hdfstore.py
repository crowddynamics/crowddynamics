import datetime
import logging
import os

import h5py
import numpy as np


class ListBuffer(list):
    """
    List that tracks start and end indices of added items.
    """

    def __init__(self, start=0, end=0):
        super(ListBuffer, self).__init__()
        self.start = start
        self.end = end

    def append(self, p_object):
        super(ListBuffer, self).append(p_object)
        self.end += 1

    def clear(self):
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
        # Path to the HDF5 file
        self.filepath, _ = os.path.splitext(filepath)  # Remove extension
        self.filepath += self.ext  # Set extension
        self.group_name = None

        # Appending data
        self.buffers = []  # (struct, buffers)

        # Configuration
        self.configure_file()

    @staticmethod
    def create_dataset(group: h5py.Group, name, values, resizable=False):
        """

        :param group: h5py.Group
        :param name: Name
        :param values: Values to be stored. Goes through np.array(value).
        :param resizable: If true values can be added to the dataset.
        :return:
        """
        logging.info("")
        values = np.array(values)
        kw = {}
        if resizable:
            values = np.array(values)
            maxshape = (None,) + values.shape
            kw.update(maxshape=maxshape)
            values = np.expand_dims(values, axis=0)
        group.create_dataset(name, data=values, **kw)

    @staticmethod
    def append_buffer_to_dataset(dset: h5py.Dataset, buffer: ListBuffer):
        """Append values to resizable h5py dataset."""
        if len(buffer):  # Buffer is not empty
            logging.info("")
            values = np.array(buffer)
            new_shape = (buffer.end,) + values.shape[1:]
            dset.resize(new_shape)
            dset[buffer.start:] = values
        else:
            logging.warning("Buffer is empty.")

    def configure_file(self):
        """Configure and creates new HDF5 File."""
        logging.info("")

        timestamp = str(datetime.datetime.now())
        with h5py.File(self.filepath, mode='a') as file:
            self.group_name = timestamp.replace(" ", "_")  # HDF group name
            file.create_group(self.group_name)  # Create Group

        logging.info(self.filepath)
        logging.info(self.group_name)

    def add_dataset(self, struct, attributes, overwrite=False):
        logging.info("")

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

        logging.info("")

    def add_buffers(self, struct, attributes):
        """
        struct
        buffers:
          attr.name: buffer1
          attr.name: buffer2
          ...
        """
        buffers = {}
        for name, settings in attributes.items():
            if settings["resizable"]:
                buffers[name] = ListBuffer(start=1, end=1)

        self.buffers.append((struct, buffers))

    def update_buffers(self):
        for struct, buffers in self.buffers:
            for attr_name, buffer in buffers.items():
                value = getattr(struct, attr_name)
                value = np.copy(value)
                buffer.append(value)

    def dump_buffers(self):
        with h5py.File(self.filepath, mode='a') as file:
            grp = file[self.group_name]
            for struct, buffers in self.buffers:
                struct_name = struct.__class__.__name__.lower()
                for attr_name, buffer in buffers.items():
                    dset = grp[struct_name][attr_name]
                    self.append_buffer_to_dataset(dset, buffer)
                    buffer.clear()
