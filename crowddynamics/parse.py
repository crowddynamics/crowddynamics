import inspect
from collections import namedtuple

from crowddynamics.exceptions import InvalidType

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
