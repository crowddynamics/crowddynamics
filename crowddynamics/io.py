"""Crowddynamics IO operations for storing simulation data into disk

Data flow ::

    data (ndarray) -> buffer (list) -> file (.npy)

"""
import os
from collections import Iterable, Mapping
from io import StringIO
import csv
import json

import numpy as np


# NPY : Array data

def save_npy(directory, basename):
    """Save simulation data

    Args:
        directory (str|Path):
        basename (str):

    Examples:
        >>> storage = save_npy('.', 'basename')
        >>> storage.send(None)  # Initialise coroutine
        >>> storage.send(data)  # Send some data (ndarray)
        >>> storage.send(False)  # False dumps data into buffers
        >>> storage.send(data)
        >>> storage.send(True)  # True dumps data into file
    """
    filepath = os.path.join(directory, basename + '_{index}.npy')
    buffer = []
    index = 0
    while True:
        data = yield  # numpy.ndarray
        buffer.append(data)

        dump = yield  # bool
        if dump:
            np.save(filepath.format(index=index), np.vstack(buffer))
            buffer.clear()
            index += 1


def find_npy_files(directory, basename):
    """Find data files

    Args:
        directory (str|Path):
        basename (str):

    Yields:
        (int, numpy.ndarray): Tuple containing (index, data)
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(basename) and file.endswith('.npy'):
                index = int(file.lstrip(basename + '_').rstrip('.npy'))
                yield index, os.path.join(root, file)


def load_npy(directory, basename):
    """Load simulation data files in order

    Args:
        directory (str|Path):
        basename (str):

    Yields:
        numpy.ndarray:

    Examples:
        >>> for data in load_npy('.', 'basename'):
        >>>     ...

    """
    values = list(find_npy_files(directory, basename))
    for index, filepath in sorted(values, key=lambda x: x[0]):
        yield np.load(filepath)


def load_npy_concatenated(directory, basename):
    """Load simulation data files concatenated into one

    Args:
        directory (str|Path):
        basename (str):

    Returns:
        numpy.ndarray:

    Examples:
        >>> load_npy_concatenated('.', 'basename')
    """
    return np.vstack(list(load_npy(directory, basename)))


# CSV : Simulation data

def save_csv(directory, basename):
    """Save dictionary data into csv file.

    Args:
        directory (str|Path):
        basename (str):

    Examples:
        >>> storage = save_csv('.', 'basename')
        >>> storage.send(None)
        >>> storage.send(data)  # Send some data (dict)
        >>> storage.send(False)  # False dumps data into buffers
        >>> storage.send(data)
        >>> storage.send(True)  # True dumps data into file
    """
    filepath = os.path.join(directory, basename + '.csv')
    with StringIO() as buffer:
        writer = csv.writer(buffer)

        def dumper():
            with open(filepath, 'a') as fp:
                fp.write(buffer.getvalue())
                # clear stringIO
                buffer.truncate(0)
                buffer.seek(0)

        # Initial data
        data = yield  # dict
        writer.writerow(data.keys())
        writer.writerow(data.values())
        dump = yield  # bool
        if dump:
            dumper()

        while True:
            data = yield  # dict
            writer.writerow(data.values())
            dump = yield  # bool
            if dump:
                dumper()


# JSON : Simulation metadata and Shapely geometries

def geometry_mapping(geom):
    from shapely.geometry import mapping

    if isinstance(geom, Iterable):
        return [mapping(geo) for geo in geom]
    elif isinstance(geom, Mapping):
        return {name: mapping(geo) for name, geo in geom.items()}
    else:
        return mapping(geom) if geom else geom


def geometry_shape(geom):
    from shapely.geometry import shape

    if isinstance(geom, Iterable):
        return [shape(geo) for geo in geom]
    elif isinstance(geom, Mapping):
        return {name: shape(geo) for name, geo in geom.items()}
    else:
        return shape(geom) if geom else geom


def save_geometry_json(filename, geometries):
    """Save dictonary of geometries to filename (.json).

    Args:
        filename:
        geometries (dict):

    """
    with open(filename, 'a') as fp:
        obj = {name: geometry_mapping(geom) for name, geom in
               geometries.items()}
        json.dump(obj, fp, indent=2, separators=(', ', ': '))


def load_geometry_json(filename):
    """Load dictonary of geometries from filename (.json).

        Args:
            filename:

        """
    with open(filename, 'r') as fp:
        obj = json.load(fp)
        return {name: geometry_shape(geo) for name, geo in obj.items()}
