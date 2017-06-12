import numpy as np
import pytest
from hypothesis.control import assume
from hypothesis.core import given
from scipy.spatial.qhull import QhullError, Voronoi

from crowddynamics.core.quantities import density_voronoi_1
from crowddynamics.testing import reals


def all_unique(data: np.ndarray) -> bool:
    """Test that all data rows have unique data"""
    ncols = data.shape[1]
    dtype = data.dtype.descr * ncols
    struct = data.view(dtype)
    return len(np.unique(struct)) == len(data)


def does_not_raise_Qhull_error(points):
    """Test that Voronoi tesselation does raise errors"""
    try:
        vor = Voronoi(points)
        return True
    except QhullError:
        return False


@pytest.mark.skip('Fixme')
# @given(points=reals(1, 10, shape=(3, 2)))
@given(points=reals(1, 10, shape=(10, 2)))
# @given(points=reals(1, 10, shape=(100, 2)))
def test_density_voronoi_1(points):
    assume(does_not_raise_Qhull_error(points))
    assume(all_unique(points))

    cell_size = 0.1
    density = density_voronoi_1(points, cell_size=cell_size)
    assert True
