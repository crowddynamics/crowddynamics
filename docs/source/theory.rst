Theory
======
- Time :math:`t \in \mathbb{R}^{+}`
- Domain :math:`\Omega \subset \mathbb{R}^{2}` is surface that contains all objects in the simulation such as agents and obstacles. Agents that outside the domain will turn inactive.
- Area occupied by agent :math:`\mathcal{A}`.
- Area occupied by obstacle :math:`\mathcal{O}`. Unpassable region.
- Area occupied by exit :math:`\mathcal{E}`. Passable region.

Mathematically crowd dynamics or crowd simulation is defined as a function of time

.. math::
   f: \mathcal{A} \times \mathbb{R}^+ \mapsto \mathcal{A}

that maps current state of agent :math:`A` and time :math:`t \in \mathbb{R}^{+}` to new state :math:`A`. Common states to be manipulated are agents *spatial* and *rotational* motion. States are model dependent and more can be defined as needed, keeping in mind the principle of Occam's razor.
