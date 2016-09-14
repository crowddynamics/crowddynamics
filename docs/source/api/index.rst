API
===
.. Source code and Python requirements information.

The aim is to keep the code modular and extensible so that new dynamical models can be easily integrated and tested.

Source is written using ``Python 3``

Required external libraries

#) Core numerical computations are done using

   - ``numpy``
   - ``scipy``
   - ``numba``

#) Setting simulation geometry with linestring and polygons

   - ``shapely``

#) Navigation algorithm

   - ``scikit-fmm`` Fast Marching Method for Eikonal equation
   - ``scikit-image`` Drawing shapes in discrete grid

#) Storing data into hdf5 file format

   - ``h5py``

#) Config files

   - ``yaml``
   - ``ruamel.yaml``

#) Data visualization

   - ``matplotlib`` Static plots
   - ``pyqt4`` Graphical user interface
   - ``pyqtgraph`` Interactive plotting inside gui

#) Data analysis

   - ``pandas``

.. TODO: About architecture

----

Contents

.. toctree::

   interface.rst
