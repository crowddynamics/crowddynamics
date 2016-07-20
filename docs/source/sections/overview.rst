Overview
========

Introduction
------------
.. Simulation model introduction

Multi-agent simulation (MAS) of crowd dynamics on continuous 2-dimensional space using classical mechanics, where agents are modeled as particles of rigid bodies affected by fictitious and real forces that attempt to model real crowd motion and phenomena occurring in crowd motion.

.. Python introduction

Source is written using python version 3.4. Code uses *Numpy* for efficient numerical arrays and *Numba* to just-in-time compile Python functions into computationally efficient machine code. This keeps the source code readable and easily understandable.

The aim is to keep the code highly modular and extendable so that new crowd dynamical models can be easily integrated and tested.

Graphical user interface for interactive visualization of the simulation uses *QtPy4* and *pyqtgraph*.

Motivation
----------
.. Section of talking why crowd simulation model should be developed and where they can be applied to.

Terminology
-----------
*Multi agent system (MAS)*

   Computerized system composed of multiple interacting intelligent agents within an environment. *Wikipedia*

*Passive agent*

   Obstable like a wall.

*Active agent*

   Individual human pedestrian :math:`\mathrm{P}` that is subject to motion.

*Crowd*

   A crowd is large group of individuals :math:`N \geq 100 \,\mathrm{P}` within the same space at the same time whose movements are for a prolonged period of time :math:`t \geq 60 \,\mathrm{s}` dependent on predominantly local interactions :math:`\rho \geq 1 \,\mathrm{P/m^{2}}`. [stateofart2013]_

*Crowd density*

    Number of agent per unit of area :math:`\rho` of unit :math:`\mathrm{P/m^{2}}`.

*Crowd pressure*

    Pressure that can measured inside a crowd when multiple agents press onto each others.

*Social force model*

   It is suggested that the motion of pedestrians can be described as if they would be subject to *social forces*. These *forces* are not directly exerted by the pedestrians’ personal environment, but they are a measure for the internal motivations of the individuals to perform certain actions (movements). [helbing1995social]_

*Subgroup*

    A subset of a group. It is defined as a number of people that desire to stay together. [Langston2009subgroup]_

*Egress congestion*

   Jamming that happens when multiple agent try to exit through same bottleneck.

*Egress flow*

    Numbers of agents going through a bottleneck per time unit :math:`\mathrm{P/s}`.

*Uni-directional flow*

*Bi-directional/parallel flows*

*Multi-directional flow*

*Orthogonal/crossing flows*

microscopic, mesoscopic, macroscopic

gaskinetic, fluid-dynamics, Navier-Stokes, traffic models

self-driven many-particle systems

empirical data

granular flow within hoppers

flowrate through the door





Literature
----------

Simulation
^^^^^^^^^^


**Social force model for pedestrian dynamics (1995)**

Introduces the concept of *social force model* for pedestrian dynamics.

    *The social force is not excerted by the environment on a pedestrian’s body. It is rather a quantity that describes the concrete motivation to act.*

    *Pedestrian models can provide valuable tools for designing and planning pedestrian areas, subway or railroad stations, big buildings, shopping malls*


**Simulating Dynamical Features of Escape Panic (2000)**

Uses of social force model for modelling room evacuation process. Well formulated social force model. Highly cited paper and equations used here serves as a base for simulation models using social force for crowd dynamics. This model is improved and new features are added in later papers.

* Original social force model


**Crowd dynamics discrete element multi-circle model (2006)**

Introduces *multi-circle model* aka *three circle model* for the shape of the agent instead of circular model. Translational motion and forces are based on the social force model. Rotational motion is introduced and formulated. Also contains table of values used in the simulation.

* Improved agent model

**Morphological and dynamical aspects of the room evacuation process (2007)**

**Modelling subgroup behaviour in crowd dynamics DEM simulation (2009)**

Addresses importance of subgroup modelling, because crowd often do not consist only of pure individual but subgroup e.q friends or families that prefer to stay together.

* Subgroup model, which can model agents that stay together.
* Adds physical damping force to contact force.

Model is validated by comparing simulations to real world data.

**FDS + EVAC**

    **Integration of an agent based evacuation simulation and the state-of-the-art fire simulation (2007)**

    **Fds+ Evac: Evacuation module for fire dynamics simulator (2007)**

    **FDS+ Evac: An Agent Based Fire Evacuation Model (2008)**

    **FDS+Evac: Modelling Social Interactions in Fire Evacuation (2008)**

    **Fire Dynamics Simulator with Evacuation: FDS+Evac (2009)**

Pedestrian model in FDS + EVAC.

**Counterflow model for agent-based simulation of crowd dynamics (2012)**


**A universal power law governing pedestrian interactions (2014)**

Introduces new social force


Experimental research
^^^^^^^^^^^^^^^^^^^^^

**Pedestrian Behavior at Bottlenecks (2005)**


**New insights into pedestrian flow through bottlenecks (2007)**


**Experimental evidence of the “Faster Is Slower” effect (2014)**

Existing models
---------------

`Fire Dynamics Simulator and Smokeview <https://pages.nist.gov/fds-smv/>`_

`PedSim <http://pedsim.silmaril.org/>`_

`Simulex <https://www.iesve.com/software/ve-for-engineers/module/Simulex/480>`_

`Golaem Crowd <http://golaem.com/crowd>`_

`MASSIVE <http://www.massivesoftware.com/>`_

`Legion <http://www.legion.com/>`_

`EXODUS <http://fseg.gre.ac.uk/exodus/>`_

`UNC gamma <http://gamma.cs.unc.edu/research/crowds/>`_

`Crowd Simulation Group <http://www.crowdsimulationgroup.co.uk/>`_


Known crowd phenomena
---------------------

Lane formation
^^^^^^^^^^^^^^


Faster is slower
^^^^^^^^^^^^^^^^


Arching
^^^^^^^


Freezing by heat
^^^^^^^^^^^^^^^^


Turbulence
^^^^^^^^^^


Stop-and-Go waves
^^^^^^^^^^^^^^^^^


Zipper effect
^^^^^^^^^^^^^


Herding
^^^^^^^



References
----------

.. [helbing1995social] Helbing, Dirk, and Peter Molnar. "Social force model for pedestrian dynamics." Physical review E 51, no. 5 (1995): 4282.

.. [helbing2000simulating] Helbing, Dirk, Illés Farkas, and Tamas Vicsek. "Simulating dynamical features of escape panic." Nature 407, no. 6803 (2000): 487-490.

.. [langston2006crowd] Langston, Paul A., Robert Masling, and Basel N. Asmar. "Crowd dynamics discrete element multi-circle model." Safety Science 44, no. 5 (2006): 395-417.

.. [korhonen2009fire] Korhonen, Timo, and Simo Hostikka. "Fire dynamics simulator with evacuation: FDS+ Evac." Technical Reference and User’s Guide. VTT Technical Research Centre of Finland (2009).

.. [Langston2009subgroup] Singh, Harmeet, Robyn Arter, Louise Dodd, Paul Langston, Edward Lester, and John Drury. "Modelling subgroup behaviour in crowd dynamics DEM simulation." Applied Mathematical Modelling 33, no. 12 (2009): 4408-4423.

.. [stateofart2013] Duives, Dorine C., Winnie Daamen, and Serge P. Hoogendoorn. "State-of-the-art crowd motion simulation models." Transportation research part C: emerging technologies 37 (2013): 193-209.

.. [power2014] Karamouzas, Ioannis, Brian Skinner, and Stephen J. Guy. "Universal power law governing pedestrian interactions." Physical review letters 113, no. 23 (2014): 238701.