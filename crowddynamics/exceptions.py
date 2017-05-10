"""CrowdDynamics exceptions"""
from traitlets import TraitError


class CrowdDynamicsException(Exception):
    """CrowdDynamics base exception."""
    pass


class InvalidType(CrowdDynamicsException, TypeError):
    """Used to indicate that the arguments to a CrowdDynamics function were 
    invalid type."""


class InvalidValue(CrowdDynamicsException, ValueError):
    """Used to indicate that the arguments to a CrowdDynamics function had
    incorrect value."""


class ValidationError(CrowdDynamicsException, TraitError):
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
