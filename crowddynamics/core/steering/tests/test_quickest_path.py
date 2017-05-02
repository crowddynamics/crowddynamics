import numpy as np
import pytest

from crowddynamics.core.steering.quickest_path import direction_map, distance_map


@pytest.mark.skip
def test_distance_map(mgrid, targets, obstacles):
    dmap = distance_map(mgrid, targets, obstacles)
    assert True


@pytest.mark.skip
def test_travel_time_map():
    assert True


@pytest.mark.skip
def test_direction_map(dmap):
    dir_map = direction_map(dmap)
    assert True


@pytest.mark.skip
def test_merge_dir_maps():
    assert True


@pytest.mark.skip
def test_static_potential():
    assert True


@pytest.mark.skip
def test_dynamic_potential():
    assert True

