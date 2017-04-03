"""HDFStore for saving simulation data into hdf5 file.

::

    Record:
        object:
            name: array|number
        attributes:
            Attribute:
                name:
                resizable:

    Buffer:
        object:
        list_buffer:
            name:
            start:
            end:

"""
import datetime
import logging
import os
from collections import namedtuple

import h5py
import numpy as np

from loggingtools import log_with


Attribute = namedtuple('Attribute', ('name', 'resizable'))
Record = namedtuple('Record', ['object', 'attributes'])
Buffer = namedtuple('Buffer', ['object', 'list_buffers'])


def struct_name(struct):
    """Get the name of the structure.

    Args:
        struct:

    Returns:
        str: Name of the structure
    """
    return struct.__class__.__name__.lower()


class ListBuffer(list):
    """List that tracks start and end indices of added items."""

    def __init__(self, name, start=0, end=0):
        """Initialise list buffer by setting start and end indices.

        Args:
            name (str):
            start (int):
            end (int):

        """
        super(ListBuffer, self).__init__()
        self.name = name
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
    """HDFStore

    Class for saving object's array or scalar data in ``hdf5`` file. Data can be
    saved once or made bufferable so that new data points can be added and
    dumped into the ``hdf5`` file.

    Attributes:
        timestamp:
        filepath: Filepath to the ``hdf5`` file where data should be saved.
        group_name:
        buffers:

    Todo:
        - set loglevel to log_with decorators
    """
    logger = logging.getLogger(__name__)
    ext = ".hdf5"

    @log_with(logger)
    def __init__(self, filepath):
        self.timestamp = datetime.datetime.now()
        self.filepath = os.path.splitext(filepath)[0] + self.ext
        self.group_name = self.timestamp.strftime('%Y-%m-%d_%H:%M:%S%f')
        self.buffers = []

        with h5py.File(self.filepath, mode='a') as file:
            file.create_group(self.group_name)

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
        values = np.array(values)
        kw = {}
        if resizable:
            values = np.array(values)
            maxshape = (None,) + values.shape
            kw.update(maxshape=maxshape)
            values = np.expand_dims(values, axis=0)
        group.create_dataset(name, data=values, **kw)

    def _append_buffer_to_dataset(self, dset, list_buffer):
        """Append values to resizable h5py dataset.

        Args:
            dset (h5py.Dataset):
            list_buffer (ListBuffer):
        """
        if list_buffer:
            values = np.array(list_buffer)
            new_shape = (list_buffer.end,) + values.shape[1:]
            dset.resize(new_shape)
            dset[list_buffer.start:] = values

    @log_with(logger)
    def add_dataset(self, record, overwrite=False):
        """Add new dataset

        Args:
            record (Record):

            overwrite (bool):
                If True allows to overwrite existing dataset.

        Raises:
            AttributeError:
                If struct doesn't have attribute.

        """
        with h5py.File(self.filepath, mode='a') as file:
            grp_name = struct_name(record.object)
            base = file[self.group_name]

            # Delete existing dataset
            if overwrite and (grp_name in base):
                del base[grp_name]

            group = base.create_group(grp_name)

            # Create new datasets
            list_buffers = []  # TODO: grp_name
            for attr in record.attributes:
                value = np.copy(getattr(record.object, attr.name))
                self._create_dataset(group=group,
                                     name=attr.name,
                                     values=value,
                                     resizable=attr.resizable)

                if attr.resizable:
                    list_buffers.append(ListBuffer(attr.name, start=1, end=1))

            if list_buffers:
                self.buffers.append(Buffer(record.object, list_buffers))

    def update_buffers(self):
        """Update new values to the buffers from the structs."""
        for buffer in self.buffers:
            for list_buffer in buffer.list_buffers:
                value = np.copy(getattr(buffer.object, list_buffer.name))
                list_buffer.append(value)

    @log_with(logger)
    def dump_buffers(self):
        """Dump values in the buffers into hdf5 file."""
        if not self.buffers:
            return
        with h5py.File(self.filepath, mode='a') as file:
            grp = file[self.group_name]
            for buffer in self.buffers:
                grp_name = struct_name(buffer.object)
                for list_buffer in buffer.list_buffers:
                    dset = grp[grp_name][list_buffer.name]
                    self._append_buffer_to_dataset(dset, list_buffer)
                    list_buffer.clear()
