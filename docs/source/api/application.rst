Application
===========

Architecture
------------

.. figure:: figures/architecture_simple.png

Simple graph of the architecture. Two main processes are

- Simulation
- Graphical User Interface (GUI)

Simulation generates data which can be stored in file by the I/O module and sent to the GUI to be displayed interactively. Simulation can be run either by

- Starting GUI and selecting simulation and running it
- Running the software from command line


Configuration Files
-------------------


Command Line Arguments
----------------------
-h, --help  Display help.
-l, --log   Set the logging level.


Logger
------
Logging levels

.. csv-table::

   CRITICAL, 50
   ERROR, 40
   WARNING, 30
   INFO, 20
   DEBUG, 10
   NOTSET, 0


- https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
- https://docs.python.org/2/howto/logging-cookbook.html#logging-cookbook


Tests
-----

Profiling
---------
