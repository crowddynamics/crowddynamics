Overview
========

Introduction
------------
.. Simulation model introduction

Multi-agent simulation (MAS) of crowd dynamics on continuous 2-dimensional space using classical mechanics, where agents are modeled as particles of rigid bodies affected by fictitious and real forces that attempt to model real crowd motion and phenomena occurring in crowd motion.


.. Section of talking why crowd simulation model should be developed and where they can be applied to.

Crowd simulations can be used as a valuable tool for venue design to improve crowd flows and to prevent disasters and stampedes. They can also be used for game and animation technology to produce realistic looking movement e.g moving large armies.


.. Python

Source is written using python version 3.4. Code uses *Numpy* for efficient numerical arrays and *Numba* to just-in-time compile Python functions into computationally efficient machine code. This keeps the source code readable and easily understandable.

The aim is to keep the code highly modular and extendable so that new crowd dynamical models can be easily integrated and tested.

Graphical user interface for interactive visualization of the simulation uses *QtPy4* and *pyqtgraph*.


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

granular flow

flowrate through the door




Literature
----------
.. Introduced equations, parameters, crowd phenomena, etc


Simulation
^^^^^^^^^^

**Social force model for pedestrian dynamics (1995)**

Introduces the concept of *social force model* for pedestrian dynamics.

    *The social force is not exerted by the environment on a pedestrian’s body. It is rather a quantity that describes the concrete motivation to act.*

    *Pedestrian models can provide valuable tools for designing and planning pedestrian areas, subway or railroad stations, big buildings, shopping malls*


**Simulating Dynamical Features of Escape Panic (2000)**

Popularization of the use of social force model for crowd dynamics.

Uses of social force model for modelling room evacuation process. Well formulated social force model. Highly cited paper and equations used here serves as a base for simulation models using social force for crowd dynamics. This model is improved and new features are added in later papers.

* Base equation for calculating total force in social force model (not in this exact form in the paper).

  .. math::
      \mathbf{f}_{i}(t) = \mathbf{f}_{i}^{adj} + \sum_{j\neq i}^{} \left(\mathbf{f}_{ij}^{soc} + \mathbf{f}_{ij}^{c}\right) + \sum_{w}^{} \left(\mathbf{f}_{iw}^{soc} + \mathbf{f}_{iw}^{c}\right)

  Formulation of individual forces is improved in later papers.

* Summarizes some characteristic features of escape panic
   1) People move or try to move considerably faster than normal.
   2) Individuals start pushing, and interactions among people become physical in nature.
   3) Moving and, in particular, passing of a bottleneck becomes incoordinated.
   4) At exits, arching and clogging are observed.
   5) Jams are building up.
   6) The physical interactions in the jammed crowd add up and cause dangerous pressures up to :math:`4450` Newtons per meter, which can bend steel barriers or tear down brick walls.
   7) Escape is further slowed down by fallen or injured people turning into “obstacles”.
   8) People show a tendency of mass behaviour, i.e., to do what other people do.
   9) Alternative exits are often overlooked or not efficiently used in escape situations.

* Faster-Is-Slower effect

* Herding equation


**Simulation of Pedestrian Crowds in Normal and Evacuation Situations (2002)**

Quote on chapter *Analogies with gases, fluids, and granular media*

   When the density is low, pedestrians can move freely, and crowd dynamics can be compared with the behavior of gases. At medium and high densities, the motion of pedestrian crowds shows some striking analogies with the motion of fluids and granular flow

   1) Footprints of pedestrians in snow look similar to streamlines of fluids.
   2) At borderlines between opposite directions of walking one can observe *viscous fingering*.
   3) The emergence of pedestrian streams through standing crowds appears analogous to the formation of river beds.
   4) Similar to segregation or stratification phenomena in granular media, pedestrians spontaneously organize in lanes of uniform walking direction, if the pedestrian density is high enough.
   5) At bottlenecks (e.g. corridors, staircases, or doors), the passing direction of pedestrians oscillates. This may be compared to the “saline oscillator” or the granular “ticking hour glass”.
   6) One can find the propagation of shock waves in dense pedestrian crowds pushing forward.
   7) The arching and clogging in panicking crowds is similar to the outflow of rough granular media through small openings.

Freezing by heat is investigated.


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


Crowd disasters
^^^^^^^^^^^^^^^



Existing models
---------------

- `Fire Dynamics Simulator and Smokeview <https://pages.nist.gov/fds-smv/>`_
- `PedSim <http://pedsim.silmaril.org/>`_
- `Simulex <https://www.iesve.com/software/ve-for-engineers/module/Simulex/480>`_
- `Golaem Crowd <http://golaem.com/crowd>`_
- `MASSIVE <http://www.massivesoftware.com/>`_
- `Legion <http://www.legion.com/>`_
- `EXODUS <http://fseg.gre.ac.uk/exodus/>`_
- CrowdDMX (References in papers, couldn't find in the internet.)

Resources
---------

- `UNC gamma <http://gamma.cs.unc.edu/research/crowds/>`_
- `Crowd Simulation Group <http://www.crowdsimulationgroup.co.uk/>`_
- `Crowd Safety and Risk Analysis, Prof. Dr. G. Keith Still <http://www.gkstill.com/index.html>`_

Known crowd phenomena
---------------------

Lane formation
^^^^^^^^^^^^^^
Pedestrians moving into opposite directions organize into lanes.

Herding
^^^^^^^
Herding or mass behaviour is phenomena where agents follow the average movement of their nearest neighbors. In nature similar effect occur for example in large crowds of birds flying where individual bird inside a crowd follows eight of its closest neighbors.


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

