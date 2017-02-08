.. Crowd Dynamics documentation master file, created by
   sphinx-quickstart on Sat Mar 26 15:35:19 2016.
   You can adapt this file completely to your liking, but it should at least contain the root `toctree` directive.

.. warning::
   Crowd Dynamics is in very early states of its development and a lot of changes can be made to source code and documentation.


Crowd Dynamics |version|
========================
Crowddynamics is a Python package for crowd simulation. The project was created in summer 2016 for Systems Analysis Laboratory (SAL) in Aalto University in Finland. Initial goal of the project was to study game theoretical model for egress congestion using existing multi-agent simulation models as a base for creating movement.


.. toctree::
   :caption: General
   :maxdepth: 2

   installation
   usage
   contributing


.. toctree::
   :caption: Foundations & Theory
   :maxdepth: 2

   theory


.. toctree::
   :caption: Simulation Models
   :maxdepth: 2

   multiagent/index
   cellular_automata/index
   continuum_flow/index
   optimal_control/index


.. toctree::
   :caption: Api
   :maxdepth: 2

   core/index
   api/taskgraph
   api/io
   api/cli
   api/testing


.. toctree::
   :caption: Simulation Models
   :maxdepth: 2

   plugins/gui
   plugins/game


.. toctree::
   :caption: Research
   :maxdepth: 2

   research/index
   references
   tables
