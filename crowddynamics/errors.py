"""CrowdDynamics exceptions"""


class CrowdDynamicsException(Exception):
    """CrowdDynamics base exception."""
    pass


class InvalidArgument(CrowdDynamicsException, TypeError):
    """Used to indicate that the arguments to a CrowdDynamics function were in
    some manner incorrect."""


class ValidationError(CrowdDynamicsException):
    """Argument is not correct type or value"""
