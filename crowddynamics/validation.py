"""Simulation input validation"""
import functools
import inspect
from collections import namedtuple

from crowddynamics.exceptions import InvalidArgument, ValidationError


def validate(stype, value):
    try:
        stype.validate(value)
    except Exception as error:
        return error


def validates(arg_names, type_dict, *args, **kwargs):
    """Validate

    Args:
        sig:
        type_dict:
        *args:
        **kwargs:

    Raises:
        ValidationError:

    """
    errors = dict()
    for i, name in enumerate(arg_names):
        try:
            value = args[i]
        except IndexError:
            value = kwargs[name]

        # Validate
        stype = type_dict[name]
        error = validate(stype, value)
        if error is not None:
            errors[name] = error

    if errors:
        raise ValidationError(errors)


class validator(object):
    """Validator decorator class

    Examples:
        >>> from schematics.types.base import FloatType, IntType, StringType
        >>>
        >>> @validator(
        >>>     FloatType(),
        >>>     IntType(),
        >>>     StringType(),
        >>>     StringType()
        >>> )
        >>> def function(a, b, c='foo', d=None):
        >>>     return True

    Attributes:
        args tuple[BaseType]:
            Positional arguments corresponding to the function arguments.
        kwargs (dict[str, BaseType]:

    """

    def __init__(self, *args, **kwargs):
        """Decorator arguments"""
        self.args = args
        self.kwargs = kwargs

        if args and not kwargs:
            pass
        elif kwargs and not args:
            pass
        else:
            raise InvalidArgument()

    def __call__(self, function):
        """Decorates the function"""
        sig = inspect.signature(function)
        arg_names = sig.parameters.keys()

        if self.kwargs:
            type_dict = self.kwargs
        else:
            type_dict = dict(zip(arg_names, self.args))

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            validates(arg_names, type_dict, *args, **kwargs)
            result = function(*args, **kwargs)
            return result

        return wrapper


ArgSpec = namedtuple('ArgSpec', ('name', 'default', 'type', 'annotation'))


def mkspec(parameter):
    if isinstance(parameter.default, inspect.Parameter.empty):
        raise InvalidArgument('Default argument should not be empty.')
    return ArgSpec(name=parameter.name,
                   default=parameter.default,
                   type=type(parameter.default),
                   annotation=parameter.annotation)


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
