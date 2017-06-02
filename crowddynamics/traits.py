import csv
import inspect
import io
import textwrap

import click
import numpy as np
from traitlets import Tuple
from traitlets.traitlets import TraitError, TraitType, Int, Float, Complex, \
    Bool, HasTraits, Unicode, Enum, is_trait
from traittypes import Array

from crowddynamics.core.vector2D import length
from crowddynamics.exceptions import InvalidType, InvalidValue


def shape_validator(*dimensions):
    """Validate the shape of :class:`traittypes.Array` trait.

    Args:
        *dimensions (int): Dimensions of the array

    Returns:
        Callable[trait, value]: Validator function
    """

    def validator(trait, value):
        if value.shape == dimensions:
            return value
        else:
            raise TraitError(
                'Expected an of shape %s and got and array with shape %s' % (
                    dimensions, value.shape))

    return validator


def length_validator(*lengths):
    """Validates the length of :class:`traittypes.Array` trait.

    Args:
        *lengths (numbers.Real):

    Returns:
        Callable[trait, value]: Validator function
    """

    def validator(trait, value):
        l = length(value)
        if np.any(np.isclose(l, lengths)):
            return value
        else:
            raise TraitError(
                'Expected an of length %s and got and array with length %s' % (
                    lengths, l))

    return validator


def trait_to_type(trait):
    if is_trait(trait):
        if isinstance(trait, Int):
            return int
        elif isinstance(trait, Float):
            return float
        elif isinstance(trait, Complex):
            return complex
        elif isinstance(trait, Bool):
            return bool
        elif isinstance(trait, Unicode):
            raise str
        else:
            raise InvalidValue('Trait conversion is not supported for: '
                               '{}'.format(trait))
    else:
        raise InvalidType('Trait should be instance of {}'.format(TraitType))


def trait_to_dtype(name, trait):
    """Convert TraitType to numpy.dtype in format (name, dtype, shape)::

        Int -> numpy.int64
        Float -> numpy.float64
        Complex -> numpy.complex128
        Bool -> numpy.bool_
        Array -> Array.dtype

    Args:
        name (str): Name of the trait
        trait (TraitType): Instance of TraitType

    Returns:
        tuple:
            - Scalar: (str, numpy.dtype)
            - Array: (str, numpy.dtype, shape)
    """
    if is_trait(trait):
        if isinstance(trait, Int):
            return name, np.int64
        elif isinstance(trait, Float):
            return name, np.float64
        elif isinstance(trait, Complex):
            return name, np.complex128
        elif isinstance(trait, Bool):
            return name, np.bool_
        elif isinstance(trait, Unicode):
            raise NotImplementedError
        elif isinstance(trait, Enum):
            raise NotImplementedError
        elif isinstance(trait, Array):
            return name, trait.dtype, trait.default_value.shape
        else:
            raise InvalidValue('Trait conversion is not supported for: '
                               '{}'.format(trait))
    else:
        raise InvalidType('Trait should be instance of {}'.format(TraitType))


def trait_to_option(name, trait):
    if is_trait(trait):
        if isinstance(trait, Int):
            return click.Option(param_decls=('--' + name,),
                                default=trait.default_value,
                                type=click.IntRange(trait.min, trait.max))
        elif isinstance(trait, Float):
            return click.Option(param_decls=('--' + name,),
                                default=trait.default_value,
                                type=float)
        elif isinstance(trait, Complex):
            return click.Option(param_decls=('--' + name,),
                                default=trait.default_value,
                                type=complex)
        elif isinstance(trait, Bool):
            return click.Option(param_decls=('--' + name,),
                                default=trait.default_value,
                                is_flag=True)
        elif isinstance(trait, Unicode):
            return click.Option(param_decls=('--' + name,),
                                default=trait.default_value,
                                type=str)
        elif isinstance(trait, Enum):
            # FIXME: trait.values should be strings
            return click.Option(param_decls=('--' + name,),
                                default=str(trait.default_value),
                                type=click.Choice(list(map(str, trait.values))))
        elif isinstance(trait, Tuple):
            return click.Option(param_decls=('--' + name,),
                                default=trait.default_value,
                                type=tuple(
                                    trait_to_type(t) for t in trait._traits))
        else:
            raise InvalidValue('Trait conversion is not supported for: '
                               '{}'.format(trait))
    else:
        raise InvalidType('Trait should be instance of {}'.format(TraitType))


def class_own_traits(cls, exclude_attrs=None):
    """Yield traits directly owned by the class (not by subclasses) in order
    defined in the class.

    Args:
        cls:
        exclude_attrs:
    """
    if issubclass(cls, HasTraits):
        for name, trait in vars(cls).items():
            if callable(exclude_attrs) and exclude_attrs(name):
                continue
            if isinstance(trait, TraitType):
                yield name, trait


def class_traits(cls, exclude_attrs=None, exclude_cls=None):
    """Traverse the class hierarchy and yield traits in order defined in the
    classes.

    Args:
        cls (type):
            Subclass of HasTraits
        exclude_attrs (Callable[str, bool], optional):
            Optional function to exclude class attribute names.
        exclude_cls (Callable[type, bool], optional):
            Optional function to exclude classes

    Yields:
        (str, TraitType): Tuple of the trait's name and type.

    Examples:
        >>> cls = ...  # Class with traits
        >>> # excludes private attributes
        >>> class_traits(cls, exclude_attrs=lambda name: name.startswith('_'))
    """
    for c in inspect.getmro(cls):
        if callable(exclude_cls) and exclude_cls(c):
            continue
        yield from class_own_traits(c, exclude_attrs)


def class_to_struct_dtype(cls, exclude_attrs, exclude_cls):
    """Construct structured numpy.dtype from class with traits.

    Args:
        cls (type):
        exclude_attrs (Callable[str, bool], optional):
            Optional function to exclude class attribute names.
        exclude_cls (Callable[type, bool], optional):
            Optional function to exclude classes

    Returns:
        numpy.dtype: Numpy structured dtype for the class
    """
    return np.dtype([
        trait_to_dtype(name, trait) for name, trait in
        class_traits(
            cls,
            exclude_attrs=exclude_attrs,
            exclude_cls=exclude_cls
        )])


class Rst(object):

    """Name space for ReStructuredText related functions"""

    @staticmethod
    def math(s):
        if s in ('', None):
            return ''
        else:
            return ':math:`%s`' % s
    @staticmethod
    def literal(s):
        if isinstance(s, str) and s == '':
            return ''
        else:
            return '``%s``' % s


def table_of_traits(cls):
    """Generate ReStructuredText table from trait of class::

        .. csv-table::
           :header-rows: 1

           name, help, symbol, default value
           radius, Radius, :math:`r`, ""
           ...

    Args:
        cls: Class that has traits

    Returns:
        str: Table as string
    """

    # TODO: unit, type, truncate long strings, sections (class names)
    indent = 3 * ' '
    header = ('name', 'symbol', 'default value', 'help')
    with io.StringIO() as buffer:
        writer = csv.writer(buffer)
        writer.writerow(header)
        for name, trait in class_traits(cls):
            row = [Rst.literal(name),
                   Rst.math(trait.metadata.get('symbol', '')),
                   Rst.literal(trait.default_value),
                   trait.metadata.get('help', '')]
            writer.writerow(row)
        table = buffer.getvalue()

    return '\n'.join((
        '',
        '.. csv-table:: Traits',
        textwrap.indent(':header-rows: 1', indent),
        '',
        textwrap.indent(table, indent)
    ))
