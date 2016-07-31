Overview
========

.. Simulation model introduction

Multi-agent simulation (MAS) of crowd dynamics on continuous 2-dimensional space using classical mechanics, where agents are modeled as particles of rigid bodies affected by fictitious and real forces that attempt to model real crowd motion and phenomena occurring in crowd motion.


.. Section of talking why crowd simulation model should be developed and where they can be applied to.

Crowd simulations can be used as a valuable tool for venue design to improve crowd flows and to prevent disasters and stampedes. They can also be used for game and animation technology to produce realistic looking movement e.g moving large armies.


.. Python

Source is written using python version 3.4. Code uses *Numpy* for efficient numerical arrays and *Numba* to just-in-time compile Python functions into computationally efficient machine code. This keeps the source code readable and easily understandable.

The aim is to keep the code highly modular and extendable so that new crowd dynamical models can be easily integrated and tested.

Graphical user interface for interactive visualization of the simulation uses *QtPy4* and *pyqtgraph*.

*Crowd*

   A crowd is large group of individuals :math:`N \geq 100` within the same space at the same time whose movements are for a prolonged period of time :math:`t \geq 60 \,\mathrm{s}` dependent on predominantly local interactions :math:`\rho \geq 1 \,\mathrm{1/m^{2}}`.

microscopic, mesoscopic, macroscopic

self-driven many-particle systems

empirical data

anthropometry

fundamental diagram

https://en.wikipedia.org/wiki/Fundamental_diagram_of_traffic_flow
