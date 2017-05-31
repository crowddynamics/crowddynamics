from hypothesis.core import given

from crowddynamics.core.steering.herding import herding_block_list
from crowddynamics.testing import reals


@given(reals(0.0, 1.0, shape=(10, 2)), reals(0.0, 1.0, shape=(10, 2)))
def test_herding(position, direction):
    ans = herding_block_list(position, direction, 0.1)
    assert True
