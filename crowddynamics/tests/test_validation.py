import string

import pytest
from hypothesis import given
import hypothesis.strategies as st
from schematics.exceptions import ValidationError
from schematics.types.base import FloatType, IntType, StringType

from crowddynamics.exceptions import InvalidArgument
from crowddynamics.validation import validator


@validator(
    FloatType(),
    IntType(),
    StringType(),
    StringType()
)
def function(a, b, c='foo', d=None):
    return True


@validator(
    a=FloatType(),
    b=IntType(),
    c=StringType(),
    d=StringType()
)
def function2(a, b, c='foo', d=None):
    return True


@pytest.mark.skip
@given(
    a=st.floats(),
    b=st.integers(),
    c=st.text(string.printable, min_size=1, max_size=100),
    d=st.text(string.printable, min_size=1, max_size=100)
)
def test_validator(a, b, c, d):
    assert function(a, b, c, d)
    assert function(a, b, c=c, d=d)
    assert function(a=a, b=b, c=c, d=c)


@pytest.mark.skip
def test_validator_invalid_decorator():
    with pytest.raises(InvalidArgument):
        @validator(
            FloatType(),
            IntType(),
            b=StringType(),
            c=StringType()
        )
        def function(a, b, c='foo', d=None):
            return True


@pytest.mark.skip
def test_validator_raises():
    with pytest.raises(ValidationError):
        function(1, 0.5, 1, 'bar')
