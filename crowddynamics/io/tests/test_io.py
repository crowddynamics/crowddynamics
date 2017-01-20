import os
import tempfile
from collections import namedtuple

import h5py
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays

from crowddynamics.io import ListBuffer, HDFStore, struct_name


@pytest.fixture(scope='module')
def hdfstore(filename='hdfstore'):
    """
    Fixture that yields HDFStore object in temporary directory. Automatically
    removes the temporary directory after usage.

    Yields:
        HDFStore:

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        yield HDFStore(filepath)


@st.composite
def struct(draw, attributes, name='struct', elements=st.floats(),
           dtype=np.float64):
    """
    Strategy that generates ``structs`` which are objects with defined
    attributes.

    Args:
        name:
        draw:
        attributes (typing.Iterable[str]):
        elements (hypothesis.searchstrategy.strategies.SearchStrategy):
        dtype (numpy.dtype):

    Returns:
        hypothesis.searchstrategy.strategies.SearchStrategy:
    """

    def array(size=st.integers(1, 100)):
        dim = draw(st.integers(1, 2))
        shape = draw(size) if dim == 1 else draw(st.tuples(size, st.just(dim)))
        return draw(arrays(dtype, shape, elements))

    structure = namedtuple(name, attributes)
    args = {attr: array() for attr in attributes}
    return structure(**args)


@given(st.integers(), st.lists(st.integers()))
def test_list_buffer(start, elements):
    end = start
    lb = ListBuffer(start=start, end=end)

    for i, e in enumerate(elements):
        lb.append(e)
        assert lb.end == end + (i + 1)

    lb.clear()
    assert lb.start == end + len(elements)


def test_hdfstore_fixture(hdfstore):
    assert os.path.exists(hdfstore.filepath)
    with h5py.File(hdfstore.filepath, mode='r') as file:
        assert hdfstore.group_name in file


@given(structure=struct(attributes=('x', 'y', 'z')))
def test_hdfstore(hdfstore, structure):
    names = tuple(structure._asdict().keys())
    attributes = dict(zip(names, len(names) * ({"resizable": False},)))

    hdfstore.add_dataset(
        struct=structure,
        attributes=attributes,
        overwrite=True
    )
    name = struct_name(structure)

    with h5py.File(hdfstore.filepath, mode='r') as file:
        grp = file[hdfstore.group_name]
        assert name in grp
        subgrp = grp[name]
        for attr in attributes.keys():
            assert attr in subgrp

    hdfstore.add_buffers(struct=structure, attributes=attributes)
    hdfstore.update_buffers()
    hdfstore.dump_buffers()
