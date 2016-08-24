Overview
========

Crowd Dynamics
--------------
.. Section of talking why crowd simulation model should be developed and where they can be applied to.

`Crowd dynamics` studies movement of human crowds.

Crowd simulations can be used as a tool for venue design to improve crowd flows and to prevent disasters and stampedes. Algorithms for producing movement can also be used for game and animation technology to produce more realistic looking movement.


Source Code
-----------
.. Source code and Python requirements information.

Source is written using python version 3.4. Code uses *Numpy* for efficient numerical arrays and *Numba* to just-in-time compile Python functions into computationally efficient machine code. This keeps the source code readable and easily understandable.

The aim is to keep the code highly modular and extendable so that new crowd dynamical models can be easily integrated and tested.

Graphical user interface using *PyQt4* and *pyqtgraph* for interactive visualization of the simulation.
