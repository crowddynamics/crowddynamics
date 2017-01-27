import pytest
from hypothesis import given
from hypothesis import settings

import crowddynamics.testing.strategies as st


@given(st.polygon(a=-1.0, b=1.0))
@settings(max_examples=1000)
def test_polygon(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    assert polygon.area > 0
    assert -1.0 <= minx < maxx <= 1.0
    assert -1.0 <= miny < maxy <= 1.0


# @pytest.mark.skip
@given(st.field())
@settings(max_examples=200)
def test_field(field):
    domain, targets, obstacles = field
    assert True


@pytest.mark.skip
@given(st.agent(size=10))
@settings(max_examples=200)
def test_agent(agent):
    assert True
