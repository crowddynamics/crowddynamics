import tempfile
import pytest

import numpy as np
import os

from crowddynamics.io import save_npy, load_npy, load_npy_concatenated


@pytest.mark.skip
def test_config():
    pass


def test_io():
    basename = 'basename'
    data = np.zeros(10)
    concatenated = np.vstack((data, data))

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, basename + '_{}.npy')
        storage = save_npy(tmpdir, basename)
        storage.send(None)

        for i in range(2):
            storage.send(data)
            storage.send(False)

            storage.send(data)
            storage.send(True)

            assert os.path.exists(filepath.format(i))
            assert np.all(np.load(filepath.format(i)) == concatenated)

        for data in load_npy(tmpdir, basename):
            assert np.all(data == concatenated)

        assert np.all(load_npy_concatenated(tmpdir, basename) ==
                      np.vstack((concatenated, concatenated)))
