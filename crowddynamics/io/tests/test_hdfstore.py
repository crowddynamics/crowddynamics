import os
import tempfile
from collections import namedtuple

import h5py
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays

from crowddynamics.io.hdfstore import Attribute, Record
from crowddynamics.io.hdfstore import ListBuffer, HDFStore, struct_name


@st.composite
def records(draw, attribute_names, name='struct', elements=st.floats(),
            dtype=np.float64):
    """Strategy that generates ``structs`` which are objects with defined
    attributes.

    Args:
        name:
        draw:
        attribute_names (typing.Iterable[str]):
        elements (hypothesis.searchstrategy.strategies.SearchStrategy):
        dtype (numpy.dtype):

    Returns:
        hypothesis.searchstrategy.strategies.SearchStrategy:
    """

    def array(size=st.integers(1, 100)):
        dim = draw(st.integers(1, 2))
        shape = draw(size) if dim == 1 else draw(st.tuples(size, st.just(dim)))
        return draw(arrays(dtype, shape, elements))

    structure = namedtuple(name, attribute_names)
    values = {attr: array() for attr in attribute_names}
    attributes = []
    for name in attribute_names:
        attributes.append(Attribute(name=name, resizable=draw(st.booleans())))

    record = Record(
        object=structure(**values),
        attributes=attributes
    )

    return record


@given(st.integers(), st.lists(st.integers()))
def test_list_buffer(start, elements):
    end = start
    lb = ListBuffer('ListBuffer', start=start, end=end)

    for i, e in enumerate(elements):
        lb.append(e)
        assert lb.end == end + (i + 1)

    lb.clear()
    assert lb.start == end + len(elements)


ATTRIBUTE_NAMES = attribute_names = ('x', 'y', 'z')


@given(record=records(ATTRIBUTE_NAMES))
def test_hdfstore(record):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'hdfstore')
        hdfstore = HDFStore(filepath)
        assert os.path.exists(hdfstore.filepath)

        hdfstore.add_dataset(
            record=record,
            overwrite=True
        )
        name = struct_name(record.object)

        with h5py.File(hdfstore.filepath, mode='r') as file:
            grp = file[hdfstore.group_name]
            assert name in grp
            subgrp = grp[name]
            for attr in ATTRIBUTE_NAMES:
                assert attr in subgrp

        for _ in range(10):
            hdfstore.update_buffers()
        hdfstore.dump_buffers()
