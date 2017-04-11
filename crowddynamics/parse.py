import contextlib
import os
import importlib.util
import inspect
import logging
from collections import namedtuple

from crowddynamics.exceptions import InvalidArgument
from crowddynamics.io import load_config

ArgSpec = namedtuple('ArgSpec', 'name default annotation')


@contextlib.contextmanager
def remember_cwd(directory):
    curdir = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(curdir)


def empty_to_none(value):
    """Convert parameter.empty to none"""
    return None if value is inspect._empty else value


def mkspec(parameter):
    if isinstance(parameter.default, inspect.Parameter.empty):
        raise InvalidArgument('Default argument should not be empty.')
    return ArgSpec(name=parameter.name,
                   default=empty_to_none(parameter.default),
                   annotation=empty_to_none(parameter.annotation))


def parse_signature(function):
    """Parse signature

    .. list-table::
       :header-rows: 1

       * - Type
         - Validation
         - Click option
         - Qt widget
       * - int
         - interval
         - IntRange
         - QSpinBox
       * - float
         - interval
         - float with callback
         - QDoubleSpinBox
       * - bool
         - flag
         - Boolean flag
         - QRadioButton
       * - str
         - choice
         - Choice
         - QComboBox

    Args:
        function:

    Yields:
        ArgSpec:

    """
    sig = inspect.signature(function)
    for name, p in sig.parameters.items():
        if name != 'self':
            yield mkspec(p)


def import_simulation_callables(confpath):
    """Import simulations callables
    
    https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

    Args:
        confpath (str|Path):

    Yields:
        (str, typing.Callable[[...], MultiAgentSimulation]):
    """
    logger = logging.getLogger(__name__)
    base, _ = os.path.split(confpath)
    config = load_config(confpath)
    conf = config.get('simulations', [])
    if conf:
        for modulename, configs in conf.items():
            try:
                with remember_cwd(base):
                    spec = importlib.util.spec_from_file_location(
                        modulename, configs['module'])
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    try:
                        yield modulename, getattr(module, configs['function'])
                    except AttributeError as e:
                        logger.warning(e)
            except ImportError as e:
                logger.warning(e)
