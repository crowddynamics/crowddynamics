"""Utility functions that have no better place for them"""
import contextlib
import importlib.util
import inspect
import os
from collections import namedtuple, OrderedDict

from crowddynamics.exceptions import InvalidType


@contextlib.contextmanager
def remember_cwd(directory):
    """Change current working directory inside the scope of an context 
    manager.
    
    Examples:
        >>> with remember_cwd(directory):
        >>>     ... # Inside directory
        >>> ... # Back to directory where we were before
    """
    curdir = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(curdir)


def interpolate_docstring(**substitutions):
    r"""Class decorator for interpolating docstring

    Args:
        **substitutions: 
            Mapping of keys and values to be interpolated. Value can be *string*
            or *callable* that takes *cls* (decorated class) as argument and 
            returns *string*.
    
    Examples:
        >>> @interpolate_docstring(name=lambda cls: cls.__name__)
        >>> class Foo(object):
        >>>     '''%(name)s'''
        >>>     pass

    """

    def wrapper(cls):
        def _(value): return value(cls) if callable(value) else value

        cls.__doc__ %= {name: _(value) for name, value in substitutions.items()}
        return cls

    return wrapper


# Parse functions signatures

ArgSpec = namedtuple('ArgSpec', 'name default annotation')


def empty_to_none(value):
    """Convert parameter.empty to none"""
    return None if value is inspect._empty else value


def mkspec(parameter):
    if isinstance(parameter.default, inspect.Parameter.empty):
        raise InvalidType('Default argument should not be empty.')
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


# Programmatic importing

def import_module(module_path):
    """Import module from modulepath

    Args:
        module_path: 

    Returns:
        object: Module that was imported. 
    """
    base, ext = os.path.splitext(module_path)
    _, name = os.path.split(base)
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def filter_cls(cls, obj):
    """Filter out objects that are not subclasses of cls or cls itself."""
    try:
        return issubclass(obj, cls) and obj is not cls
    except:
        return False


def import_subclasses(module_path, cls):
    """Import subclasses of class from given module path.
    
    Args:
        module_path (str|Path): 
        cls (type): 

    Returns:
        OrderedDict: Dictionary of name, object pairs.
    """
    mod = import_module(module_path)
    return OrderedDict([(name, obj) for name, obj in vars(mod).items() if
                        filter_cls(cls, obj)])
