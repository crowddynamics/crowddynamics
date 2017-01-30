import datetime
import logging
import os

import h5py
import numpy as np


def struct_name(struct):
    """
    Get the name of the structure.

    Args:
        struct:

    Returns:
        str: Name of the structure
    """
    return struct.__class__.__name__.lower()


class ListBuffer(list):
    """
    List that tracks start and end indices of added items.
    """

    def __init__(self, start=0, end=0):
        """
        Initialise list buffer by setting start and end indices.

        Args:
            start (int):
            end (int):

        """
        super(ListBuffer, self).__init__()
        self.start = start
        self.end = end

    def append(self, p_object):
        """
        Appends the ``p_object`` to the end of the list and increments end index
        by one.

        Args:
            p_object (object):
        """
        super(ListBuffer, self).append(p_object)
        self.end += 1

    def clear(self):
        """
        Clears the list buffer and sets start index equal to end.
        """
        super(ListBuffer, self).clear()
        self.start = self.end


class HDFStore(object):
    """
    Class for saving object's array or scalar data in ``hdf5`` file. Data can be
    saved once or made bufferable so that new data points can be added and
    dumped into the ``hdf5`` file.
    """
    ext = ".hdf5"

    def __init__(self, filepath):
        """
        HDFStore

        Args:
            filepath (str):
                Filepath to the ``hdf5`` file where data should be saved.

        """
        self.logger = logging.getLogger(__name__)

        # Time
        self.timestamp = datetime.datetime.now()

        # Path to the HDF5 file
        self.filepath = os.path.splitext(filepath)[0] + self.ext
        self.group_name = self.timestamp.strftime('%Y-%m-%d_%H:%M:%S%f')

        # Appending data
        self.buffers = []  # (struct, buffers)

        # Configuration
        with h5py.File(self.filepath, mode='a') as file:
            file.create_group(self.group_name)

        self.logger.info(self.filepath)
        self.logger.info(self.group_name)

    def _create_dataset(self, group, name, values, resizable=False):
        """
        Create dataset

        Args:
            group (h5py.Group):
                Group in which the dataset is created to.

            name (str):
                Name of the dataset.

            values (numpy.ndarray):
                Values to be stored. Goes through np.array(value).

            resizable (bool):
                If true new values can be added to the dataset.

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

    def _append_buffer_to_dataset(self, dset, buffer):
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
        Add new dataset

        Args:
            struct (object):
                Python class that has attributes ``attributes.keys()``

            attributes (dict[str, dict]):
                Dictionary of ``attribute_name: settings``

            overwrite (bool):
                If True allows to overwrite existing dataset.

        Raises:
            AttributeError:
                If struct doesn't have attribute.

        """
        self.logger.info("")

        with h5py.File(self.filepath, mode='a') as file:
            name = struct_name(struct)
            base = file[self.group_name]  # New group for structure
            if overwrite and (name in base):
                del base[name]  # Delete existing dataset
            group = base.create_group(name)

            # Create new datasets
            for name, settings in attributes.items():
                value = np.copy(getattr(struct, name))
                self._create_dataset(group, name, value, settings["resizable"])

    def add_buffers(self, struct, attributes):
        """
        Add new buffers to the hdfstore.

        Args:
            struct (object):
            attributes (dict[str, dict]):

        """
        buffers = {
            name: ListBuffer(start=1, end=1)
            for name, settings in attributes.items() if settings["resizable"]
        }
        self.buffers.append((struct, buffers))

    def update_buffers(self):
        """Update new values to the buffers from the structs."""
        for struct, buffers in self.buffers:
            for attr_name, buffer in buffers.items():
                value = getattr(struct, attr_name)
                value = np.copy(value)
                buffer.append(value)

    def dump_buffers(self):
        """Dump values in the buffers into hdf5 file."""
        with h5py.File(self.filepath, mode='a') as file:
            grp = file[self.group_name]
            for struct, buffers in self.buffers:
                name = struct_name(struct)
                for attr_name, buffer in buffers.items():
                    dset = grp[name][attr_name]
                    self._append_buffer_to_dataset(dset, buffer)
                    buffer.clear()
