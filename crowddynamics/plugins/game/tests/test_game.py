import hypothesis.strategies as st
import pytest
from hypothesis import given

from crowddynamics.plugins.game.game import payoff
from crowddynamics.testing.strategies import real


@given(
    s_i=st.integers(0, 1),
    s_j=st.integers(0, 1),
    t_aset=real(0),
    t_evac_i=real(0),
    t_evac_j=real(0),
)
def test_payoff(s_i, s_j, t_aset, t_evac_i, t_evac_j):
    value = payoff(s_i, s_j, t_aset, t_evac_i, t_evac_j)
    assert isinstance(value, float)
    assert value >= -1.0


@pytest.mark.skip
def test_best_response_strategy():
    assert True


@pytest.mark.skip
def test_egress_game():
    assert True
