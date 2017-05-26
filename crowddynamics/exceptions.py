"""CrowdDynamics exceptions"""
import functools
import warnings

import traitlets


# Exceptions

class CrowdDynamicsException(Exception):
    """CrowdDynamics base exception."""
    pass


class InvalidType(CrowdDynamicsException, TypeError):
    """Used to indicate that the arguments to a CrowdDynamics function were 
    invalid type."""


class InvalidValue(CrowdDynamicsException, ValueError):
    """Used to indicate that the arguments to a CrowdDynamics function had
    incorrect value."""


class ValidationError(CrowdDynamicsException, traitlets.TraitError):
    """Argument is invalid"""


class OverlappingError(CrowdDynamicsException):
    """Two agents are overlapping."""


class AgentStructureFull(CrowdDynamicsException):
    """Agent structure is full."""


class NotACrowdDynamicsDirectory(CrowdDynamicsException):
    """Directory is not recognized as a crowddynamics simulation directory"""


class DirectoryIsAlreadyCrowdDynamicsDirectory(CrowdDynamicsException):
    """Directory is already recognized as a crowddynamics simulation 
    directory"""


# Warnings


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    https://wiki.python.org/moin/PythonDecoratorLibrary#Generating_Deprecation_Warnings
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            filename=func.func_code.co_filename,
            lineno=func.func_code.co_firstlineno + 1
        )
        return func(*args, **kwargs)
    return new_func
