# Multi-Agent Simulation of Crowd Dynamics
## Introduction
Continuous 2-dimensional multi-Agent simulation of crowd dynamics. Motion of the agents is modeled using **social force model** where the agents can thought as particles affected by fictitious forces that attempt to produce realistic human movement.


## Python code
Simulation code is written with [_Python_ (3.4)][python] . Code uses [_Numpy_][numpy] for efficient numerical arrays and [_Numba_][numba] to just-in-time compile Python functions into computationally efficient machine code. This keeps the source code readable and easily understandable. 

The aim is to keep the code highly modular and extendable so that new crowd dynamical models can be easily integrated and tested.

Analysing data uses pandas data tables [_Pandas_][pandas].

Graphical user interface for interactive visualization of the simulation uses [_QtPy4_](qtpy4) and [_pyqtgraph_](pyqtgraph).


<!-- Python references -->
[python]: https://www.python.org/
[numpy]: http://www.numpy.org/
[numba]: http://numba.pydata.org/
[pandas]: http://pandas.pydata.org/
[pyqt4]: https://wiki.python.org/moin/PyQt4
[pyqtgraph]: http://www.pyqtgraph.org/ 
