Social Force Model
==================



System of differential equations
--------------------------------
Social force model consists of real contact forces and fictitious adjusting, social forces and random fluctuation to simulate crowd motion. [helbing1995]_, [helbing2000]_

Law's of motion are given as system of differential equations

Position and velocity

.. math::
   m \frac{d^{2}}{d t^{2}} \mathbf{x}(t) = \mathbf{f}(t)

Rotational motion

.. math::
   I \frac{d^{2}}{d t^{2}} \varphi(t) = M(t)

.. [helbing1995] Helbing, Dirk, and Peter Molnar. "Social force model for pedestrian dynamics." Physical review E 51, no. 5 (1995): 4282.

.. [helbing2000] Helbing, Dirk, Illés Farkas, and Tamas Vicsek. "Simulating dynamical features of escape panic." Nature 407, no. 6803 (2000): 487-490.

Total Motion
------------
Total force exerted on the agent is the sum of movement adjusting, social and contact forces between other agents and wall.

.. math::
   \mathbf{f}_{i}(t) = \mathbf{f}_{i}^{adj} + \sum_{j\neq i}^{} \left(\mathbf{f}_{ij}^{soc} + \mathbf{f}_{ij}^{c}\right) + \sum_{w}^{} \left(\mathbf{f}_{iw}^{soc} + \mathbf{f}_{iw}^{c}\right) + \boldsymbol{\xi}_{i}


Total torque exerted on agent, is the sum of adjusting contact and social torques

.. math::
   M_{i}(t) = M_{i}^{adj} + \sum_{j\neq i}^{} \left(M_{ij}^{soc} + M_{ij}^{c}\right) + \sum_{w}^{} \left(M_{iw}^{soc} + M_{iw}^{c}\right) + \eta_{i}


.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: motion

Adjusting Motion
----------------
Force adjusting agent's movement towards desired in some characteristic time

.. math::
   \mathbf{f}^{adj} = \frac{m}{\tau^{adj}} (v_{0} \cdot \hat{\mathbf{e}} - \mathbf{v})


.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_adjust

Torque adjusting agent's rotational motion towards desired

.. math::
   M_{}^{adj} = \frac{I_{rot}}{\tau_{adj}^{rot}} \left( \frac{\varphi_{}(t) - \varphi_{}^{0}}{\pi}  \omega_{}^{0} - \omega_{}(t)\right)

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: torque_adjust

Interactions
------------

Interactions between agent and another agent or wall are modeled using social and contact forces.

Agent-agent
^^^^^^^^^^^
Circular

.. math::
   \tilde{\mathbf{x}} &= \mathbf{x}_{i} - \mathbf{x}_{j} \\
   \tilde{\mathbf{v}} &= \mathbf{v}_{i} - \mathbf{v}_{j} \\
   d &= \left\| \tilde{\mathbf{x}} \right\| \\
   r_{tot} &= r_i + r_j \\
   h &= d - r_{tot} \\
   \hat{\mathbf{n}} &= \tilde{\mathbf{x}} / d \\
   \hat{\mathbf{t}} &= R(-90^{\circ}) \cdot \hat{\mathbf{n}}

Three circles



Agent-wall
^^^^^^^^^^
Linear wall

.. math::
   \tilde{\mathbf{x}} & \\
   \tilde{\mathbf{v}} &= \mathbf{v}_{i} \\
   \mathbf{q}_{0} &= \mathbf{x}_{i} - \mathbf{p}_{0} \\
   \mathbf{q}_{1} &= \mathbf{x}_{i} - \mathbf{p}_{1} \\
   d &= \begin{cases} \left\| \mathbf{q}_{0} \right\| & l_{t} > l_{w} \\
   \left| l_{n} \right| & \text{otherwise} \\
   \left\| \mathbf{q}_{1} \right\| & l_{t} < -l_{w}
   \end{cases} \\
   \hat{\mathbf{n}} &= \begin{cases}
   \hat{\mathbf{q}}_{0} & l_{t} > l_{w} \\
   \operatorname{sign}(l_{n})\hat{\mathbf{n}}_{w} & \text{otherwise} \\
   \hat{\mathbf{q}}_{1} & l_{t} < -l_{w}
   \end{cases} \\
   \hat{\mathbf{t}} &= R(-90^{\circ}) \cdot \hat{\mathbf{n}}


Social
^^^^^^
**Social force model for pedestrian dynamics**

Psychological force for collision avoidance. Distance based algorithm used in the original social force model by Helbing

.. math::
   \mathbf{f}^{soc} = A \exp\left(-\frac{h}{B}\right) \hat{\mathbf{n}}

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_social_velocity_independent

----

**A universal power law governing pedestrian interactions**

.. TODO: Figure on how tau is calculated.

Algorithm based on human anticipatory behaviour

.. math::
   \mathbf{f}^{soc} &= -\nabla_{\tilde{\mathbf{x}}} E(\tau) \\
   &= -\nabla_{\tilde{\mathbf{x}}} \left(\frac{k}{\tau^{2}} \exp \left( -\frac{\tau}{\tau_{0}} \right) \right)

where coefficient :math:`k=1.5 \cdot \langle m_i \rangle` scales the magnitude of the social force. Time-to-collision :math:`\tau` is obtained by linearly extrapolating current trajectories and finding where skin-to-skin distance :math:`h` is zero. If  :math:`\tau < 0` or :math:`\tau` is undefined (complex number or floating point `nan`) trajectories are not colliding and social force is :math:`\mathbf{0}`. [power2014]_

.. [power2014] Karamouzas, Ioannis, Brian Skinner, and Stephen J. Guy. "Universal power law governing pedestrian interactions." Physical review letters 113, no. 23 (2014): 238701.

----

For **circular** agent :math:`(r_i + r_j) = \tilde{r}` is constant. Skin-to-skin distance

.. math::
   h(\tau) &= \| \tau \tilde{\mathbf{v}} + \tilde{\mathbf{x}} \| - (r_i + r_j), \\

where :math:`\tau \tilde{\mathbf{v}} + \tilde{\mathbf{x}}` is trajectory of the center of  mass

.. math::
   h &= 0 \\
   \| \tilde{\mathbf{x}} + \tau \tilde{\mathbf{v}} \| &= r_i + r_j \\
   \| \tilde{\mathbf{x}} + \tau \tilde{\mathbf{v}} \|^2 &= (r_i + r_j)^2 \\
   \tilde{\mathbf{x}} \cdot \tilde{\mathbf{x}} + 2 \tau (\tilde{\mathbf{x}} \cdot \tilde{\mathbf{v}})  + \tau^2 \tilde{\mathbf{v}} \cdot \tilde{\mathbf{v}} &= (r_i + r_j)^2 \\

.. math::
   \tau^2 (\tilde{\mathbf{v}} \cdot \tilde{\mathbf{v}}) + 2 \tau (\tilde{\mathbf{x}} \cdot \tilde{\mathbf{v}}) + \tilde{\mathbf{x}} \cdot \tilde{\mathbf{x}} -  \tilde{r}^2 &= 0

We can solve equation with `quadratic formula <https://en.wikipedia.org/wiki/Quadratic_equation>`_, then we have

.. math::
   a &= \tilde{\mathbf{v}} \cdot \tilde{\mathbf{v}} \\
   b &= -\tilde{\mathbf{x}} \cdot \tilde{\mathbf{v}} \\
   c &= \tilde{\mathbf{x}} \cdot \tilde{\mathbf{x}} - \tilde{r}^{2}\\
   d &= \sqrt{b^{2} - a c} \\
   \tau &= \frac{b - d}{a}.

Social force is now derived by taking spatial gradient of the energy function

.. math::
   \mathbf{f}^{soc} &= - \left(\frac{k}{a \tau^{2}}\right) \left(\frac{2}{\tau} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right ) \left(\tilde{\mathbf{v}} -\frac{a \tilde{\mathbf{x}} + b \tilde{\mathbf{v}}}{d} \right), \\

Source

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_social

----

For **elliptical** agent

.. math::
    \mathbf{\hat{e}}_n &= \operatorname{sin}\left(\varphi\right)\mathbf{\hat{e}_x} + \operatorname{cos}\left(\varphi\right)\mathbf{\hat{e}_y}
    \\
   r &= \| r_t \cos(\phi) \mathbf{\hat{e}_x} + r \sin(\phi) \mathbf{\hat{e}_y} \|, \quad \phi = \angle(\tilde{\mathbf{x}} + \tau\tilde{\mathbf{v}}, \hat{\mathbf{e}}_n) \\
   \cos(\phi) &= \frac{(\tilde{\mathbf{x}} + \tau\tilde{\mathbf{v}})}{\| \tilde{\mathbf{x}} + \tau\tilde{\mathbf{v}} \|} \cdot \hat{\mathbf{e}}_{n}
   \\
   \sin(\phi) &= \left \| \frac{(\tilde{\mathbf{x}} + \tau\tilde{\mathbf{v}})}{\| \tilde{\mathbf{x}} + \tau\tilde{\mathbf{v}} \|} \times \hat{\mathbf{e}}_{n} \right \|

We get radius of form

.. math::
   r &= \sqrt{c_0 + c_1 \tau + c_2 \tau^2}

.. Taylor series approximation?

.. math::
   \tau = ???

----

For **three circle** agent.

Torso-torso

.. math::
   \tilde{\mathbf{x}} &= \mathbf{x}_{i} - \mathbf{x}_{j}

Torso-shoulder

.. math::
   \tilde{\mathbf{x}}_{ts} &= \tilde{\mathbf{x}} \pm r_{ts} \hat{\mathbf{e}}_{t}

Shoulder-shoulder

.. math::
   \tilde{\mathbf{x}}_{ss} &= \tilde{\mathbf{x}} \pm (r_{ts} \hat{\mathbf{e}}_{t})_i \pm (r_{ts} \hat{\mathbf{e}}_{t})_j

----

Contact
^^^^^^^
Physical contact force

.. math::
   \mathbf{f}^{c} = - h \cdot \left(\mu \cdot \hat{\mathbf{n}} - \kappa \cdot (\mathbf{v} \cdot \hat{\mathbf{t}}) \hat{\mathbf{t}}\right) + c_{n} \cdot (\mathbf{v} \cdot \hat{\mathbf{n}}) \hat{\mathbf{n}} , \quad h < 0

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_contact


Torque
^^^^^^

Torque from social forces acting with other agent or wall

.. math::
   \mathbf{M}_{}^{soc} = \mathbf{r}_{}^{soc} \times \mathbf{f}_{}^{soc}

Torque from contact forces acting with other agent or wall

.. math::
   \mathbf{M}_{}^{c} = \mathbf{r}_{}^{c} \times \mathbf{f}_{}^{c}

We can concatenate these because both radii for social and contact forces are the same

.. math::
   \mathbf{r} &= \mathbf{r}_{}^{soc} = \mathbf{r}_{}^{c} \\
   \mathbf{M} &= \mathbf{r} \times (\mathbf{f}_{}^{soc} + \mathbf{f}_{}^{c})

Random fluctuation
------------------
Fluctuation force

.. math::
   \boldsymbol{\xi} &= \xi \cdot \hat{\mathbf{e}}, \quad \xi \in \mathcal{N}(\mu, \sigma^{2}), \\
   \hat{\mathbf{e}}  &= \begin{bmatrix} \cos(\varphi) & \sin(\varphi) \end{bmatrix}, \quad \varphi \in \mathcal{U}(-\pi, \pi)

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_random

Fluctuation torque

.. math::
   \eta \in \mathcal{N}(\mu, \sigma^{2})

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: torque_random


Integrator
----------
System is updated using discrete time step :math:`\Delta t`

.. math::
   t_{0}, t_{1}, \ldots, t_{k} = 0, \Delta t, \ldots, t_{k-1} + \Delta t \\

Adaptive time step :math:`\Delta t` is used when integration is done.

Acceleration on an agent

.. math::
   a_{k} &= \mathbf{f}_{k} / m \\
   \mathbf{x}_{k+1} &= \mathbf{x}_{k} + \mathbf{v}_{k} \Delta t + \frac{1}{2} a_{k} \Delta t^{2} \\
   \mathbf{v}_{k+1} &= \mathbf{v}_{k} + a_{k} \Delta t \\


Angular acceleration

.. math::
   \alpha_{k} &= M_{k} / I \\
   \varphi_{k+1} &= \varphi_{k} + \omega_{k} \Delta t + \frac{1}{2} \alpha_{k} \Delta t^{2} \\
   \omega_{k+1} &= \omega_{k} + \alpha_{k} \Delta t \\


.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: integrator
