Simulation Models
=================
- #) Microscopic
  #) Mesoscopic
  #) Macroscopic

- #) Continuous space
  #) Discrete lattice

Cellular Automata
-----------------
Discrete grid

----

Multi-Agent
-----------
Microscopic modeling of crowd dynamics.

Social force model approached modelling pedestrian dynamics using classical mechanics. Helbing describes social forces in [helbing1995]_, [helbing2000]_,

   *These forces are not directly exerted by the pedestrians’ personal environment, but they are a measure for the internal motivations of the individuals to perform certain actions (movements).*

Law's of motion are given as `Langevin equations`_ where translational motion and rotational motion

.. _Langevin equations: https://en.wikipedia.org/wiki/Langevin_equation

.. math::
   \begin{cases}
   m \dfrac{d^{2}}{d t^{2}} \mathbf{x}(t) = \mathbf{f}(t) + \boldsymbol{\xi}(t) \\
   I \dfrac{d^{2}}{d t^{2}} \varphi(t) = M(t) + \eta(t)
   \end{cases}


Total force exerted on the agent is the sum of adjusting and interaction forces

.. math::
   \mathbf{f}_{i}(t) = \mathbf{f}_{i}^{adj} + \sum_{j\neq i}^{} \left(\mathbf{f}_{ij}^{soc} + \mathbf{f}_{ij}^{c}\right) + \sum_{w}^{} \left(\mathbf{f}_{iw}^{soc} + \mathbf{f}_{iw}^{c}\right)


Total torque exerted on agent, is the sum of adjusting torque and torque exerted by interaction forces

.. math::
   M_{i}(t) = M_{i}^{adj} + \sum_{j\neq i}^{} \left(M_{ij}^{soc} + M_{ij}^{c}\right) + \sum_{w}^{} \left(M_{iw}^{soc} + M_{iw}^{c}\right)


----

Continuum
---------
Macroscopic modelling of crowds. [cristiani2011]_



----

.. [helbing1995] Helbing, D., & Moln??r, P. (1995). Social force model for pedestrian dynamics. Physical Review E, 51(5), 4282–4286. http://doi.org/10.1103/PhysRevE.51.4282

.. [helbing2000] Helbing, D., Farkas, I., & Vicsek, T. (2000). Simulating dynamical features of escape panic. Nature, 407(6803), 487–490. http://doi.org/10.1038/35035023

.. [fdsevac2009] Korhonen, T., & Hostikka, S. (2009). VTT WORKING PAPERS 119 Fire Dynamics Simulator with Evacuation: FDS+Evac Technical Reference and User’s Guide. Retrieved from http://www.vtt.fi/publications/index.jsp

.. [cristiani2011] Cristiani, E., & Peri, D. (2015). Handling obstacles in pedestrian simulations: Models and optimization. Retrieved from http://arxiv.org/abs/1512.08528

.. [duives2014] Duives, D. C., Daamen, W., & Hoogendoorn, S. P. (2014). State-of-the-art crowd motion simulation models. Transportation Research Part C: Emerging Technologies. http://doi.org/10.1016/j.trc.2013.02.005
