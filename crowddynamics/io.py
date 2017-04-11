"""Crowddynamics IO operations

- Loading configurations
- Saving simulation data

Data flow

data (ndarray) -> buffer (list) -> file (.npy)

Todo:
    - https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html
    - json/yaml for metadata
    - config files
    - serializing geometry (shapely) objects (.shp), Fiona

References:
    - http://www.dabeaz.com/coroutines/Coroutines.pdf
"""
import os

import numpy as np
from configobj import ConfigObj
from crowddynamics.exceptions import InvalidConfigurationError
from validate import Validator


def load_config(infile, configspec=None):
    """Load configuration from INI file."""
    config = ConfigObj(infile=infile, configspec=configspec)
    if configspec and not config.validate(Validator()):
        raise InvalidConfigurationError
    return config


def save_data(directory, basename):
    """Save simulation data
      
    Args:
        directory (str|Path): 
        basename (str): 
        
    Examples:
        >>> storage = save_data('.', 'basename')
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


def find_files(directory, basename):
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


def load_data(directory, basename):
    """Load simulation data files in order
    
    Args:
        directory (str|Path): 
        basename (str):
    
    Yields:
        numpy.ndarray:
    
    Examples:
        >>> for data in load_data('.', 'basename'):
        >>>     ...

    """
    values = list(find_files(directory, basename))
    for index, filepath in sorted(values, key=lambda x: x[0]):
        yield np.load(filepath)


def load_data_concatenated(directory, basename):
    """Load simulation data files concatenated into one 
    
    Args:
        directory (str|Path): 
        basename (str):
    
    Returns:
        numpy.ndarray:
    
    Examples:
        >>> load_data_concatenated('.', 'basename')
    """
    return np.vstack(list(load_data(directory, basename)))
