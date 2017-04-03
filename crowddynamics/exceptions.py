"""CrowdDynamics exceptions"""


class CrowdDynamicsException(Exception):
    """CrowdDynamics base exception."""
    pass


class InvalidArgument(CrowdDynamicsException, TypeError):
    """Used to indicate that the arguments to a CrowdDynamics function were in
    some manner incorrect."""


class ValidationError(CrowdDynamicsException):
    """Argument is not correct type or value"""


class OverlappingError(CrowdDynamicsException):
    """Two agents are overlapping."""


class AgentStructureFull(CrowdDynamicsException):
    """Agent structure is full."""


class NotACrowdDynamicsDirectory(CrowdDynamicsException):
    """Directory is not recognized as a crowddynamics simulation directory"""


class DirectoryIsAlreadyCrowdDynamicsDirectory(CrowdDynamicsException):
    """Directory is already recognized as a crowddynamics simulation 
    directory"""
