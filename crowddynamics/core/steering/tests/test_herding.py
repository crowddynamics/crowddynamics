from hypothesis.core import given

from crowddynamics.core.steering.herding import herding_block_list
from crowddynamics.testing import reals


@given(position=reals(0.0, 1.0, shape=(20, 2)),
       direction=reals(0.0, 1.0, shape=(20, 2)))
def test_herding_block_list(position, direction):
    sight = 0.1
    neighborhood_size = 5
    ans = herding_block_list(position, direction, sight, neighborhood_size)
    assert True
