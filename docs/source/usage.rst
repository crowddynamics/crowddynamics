Usage
=====

.. note::
   This section is unfinished

.. todo::
   Video tutorial and demonstration


Basic Architecture
------------------

.. figure:: architecture_simple.png

   *Simple graph of the architecture*

Two main processes are

- Simulation
- Graphical User Interface (GUI)

Simulation generates data which can be stored in file by the I/O module and sent to the GUI to be displayed interactively. Simulation can be run either by

- Starting GUI and selecting simulation and running it
- Running the software from command line

Crowddynamics aims to be modular and extensible so that new models can be easily integrated and tested.

Crowddynamics uses plugin software architecture.


Command-line Interface
----------------------

.. program-output:: crowddynamics --help

Configuration
-------------


