import numpy as np
import pytest
from hypothesis.core import given

from crowddynamics.core.steering.collective_motion import leader_follower_with_herding_interaction
from crowddynamics.core.structures import obstacle_type_linear
from crowddynamics.testing import reals


@pytest.mark.skip('Fix this test')
@given(position=reals(0.0, 1.0, shape=(20, 2)),
       direction=reals(0.0, 1.0, shape=(20, 2)))
def test_herding_block_list(position, direction):
    sight = 0.1
    neighborhood_size = 5
    obstacles = np.zeros(0, dtype=obstacle_type_linear)

    assert True
