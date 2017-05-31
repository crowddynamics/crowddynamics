from unittest.mock import Mock
import numpy as np

import pytest
import traitlets
import traittypes

from crowddynamics.traits import shape_validator, length_validator, \
    trait_to_dtype


def test_shape_validator():
    trait = None
    value = Mock()

    correct = (1, 1)
    validator = shape_validator(*correct)
    value.shape = correct
    assert validator(trait, value).shape == value.shape

    incorrect = (1, 2)
    value.shape = incorrect
    pytest.raises(traitlets.TraitError, validator, trait, value)


def test_length_validator():
    trait = None
    value = np.array((1.0, 1.0))

    lengths = np.sqrt(2), 1.0
    validator = length_validator(*lengths)
    assert np.allclose(validator(trait, value), value)

    value = np.array((0.0, 0.0))
    pytest.raises(traitlets.TraitError, validator, trait, value)


class Traits(traitlets.HasTraits):
    int = traitlets.Int()
    float = traitlets.Float()
    complex = traitlets.Complex()
    bool = traitlets.Bool()
    array = traittypes.Array(default_value=(0, 0), dtype=np.float64)


def test_trait_to_primitive_dtype():
    for name, trait in Traits.class_traits().items():
        dtypespec = trait_to_dtype(name, trait)
        assert True
