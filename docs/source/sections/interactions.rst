Interactions
============

Interactions between agents are modeled using social and contact forces.

Social force
------------

Helbing's social force
^^^^^^^^^^^^^^^^^^^^^^

Psychological force for collision avoidance. Distance based algorithm used in the original social force model by Helbing

.. math::
   \mathbf{f}^{soc} = A \exp\left(-\frac{h}{B}\right) \hat{\mathbf{n}}

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_social_velocity_independent

A universal power law governing pedestrian interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. TODO: Figure on how tau is calculated.

Algorithm based on human anticipatory behaviour. Interaction potential between two agents

.. math::
   E(\tau) &= \frac{k}{\tau^{2}} \exp \left( -\frac{\tau}{\tau_{0}} \right), \quad \tau_{0} > 0, \tau > 0

where coefficient :math:`k=1.5 \cdot \langle m_i \rangle` scales the magnitude and :math:`\tau_{0}` is interaction time horizon. Time-to-collision :math:`\tau` is obtained by linearly extrapolating current trajectories and finding where skin-to-skin distance :math:`h` is zero.

Force affecting agent can be derived by taking spatial gradient of the energy, where time-to-collision :math:`\tau` is function of the relative displacement of the mass centers :math:`\tilde{\mathbf{x}}`


.. math::
   \mathbf{f}^{soc} &= -\nabla_{\tilde{\mathbf{x}}} E(\tau) \\
   &= \left(\frac{k}{\tau^{2}}\right) \left(\frac{2}{\tau} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right ) \nabla_{\tilde{\mathbf{x}}} \tau

If  :math:`\tau < 0` or :math:`\tau` is undefined [#]_ trajectories are not colliding and social force is :math:`\mathbf{0}`. [power2014]_

.. [power2014] Karamouzas, Ioannis, Brian Skinner, and Stephen J. Guy. "Universal power law governing pedestrian interactions." Physical review letters 113, no. 23 (2014): 238701.

.. [#] Complex number or floating point nan

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
   \nabla_{\tilde{\mathbf{x}}} \tau &= \left(\frac{1}{a} \right) \left(\tilde{\mathbf{v}} -\frac{a \tilde{\mathbf{x}} + b \tilde{\mathbf{v}}}{d} \right)

.. math::
   \mathbf{f}^{soc} &= - \left(\frac{k}{\tau^{2}}\right) \left(\frac{2}{\tau} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right ) \left(\frac{1}{a} \right) \left(\tilde{\mathbf{v}} -\frac{a \tilde{\mathbf{x}} + b \tilde{\mathbf{v}}}{d} \right), \\

Source

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_social

----

For **elliptical** agent

.. math::
    \mathbf{\hat{e}}_n &= \operatorname{sin}\left(\varphi\right)\mathbf{\hat{e}_x} + \operatorname{cos}\left(\varphi\right)\mathbf{\hat{e}_y}
    \\
   r &= \| r_t \cos(\phi) \mathbf{\hat{e}_x} + r \sin(\phi) \mathbf{\hat{e}_y} \|, \quad \phi = \angle(\tilde{\mathbf{x}} + \tau\tilde{\mathbf{v}}, \mathbf{\hat{e}_n}) \\
      \cos(\phi) &= \frac{(\tilde{\mathbf{x}} + \tau\tilde{\mathbf{v}})}{\| \tilde{\mathbf{x}} + \tau\tilde{\mathbf{v}} \|} \cdot \mathbf{\hat{e}_{n}}
   \\
   \sin(\phi) &= \left \| \frac{(\tilde{\mathbf{x}} + \tau\tilde{\mathbf{v}})}{\| \tilde{\mathbf{x}} + \tau\tilde{\mathbf{v}} \|} \times \mathbf{\hat{e}_{n}} \right \|

We get radius of form

.. math::
   r &= \sqrt{c_0 + c_1 \tau + c_2 \tau^2}

.. Taylor series approximation?

.. math::
   \tau = ???

----

For **three circle** agent.

.. Relative displacement vector :math:`\mathbf{r}`.

.. math::
   \mathbf{c} &= (\mathbf{x}_{i} + \mathbf{r}_i) - (\mathbf{x}_{j} + \mathbf{r}_j) \\
   &= (\mathbf{x}_{i} + \mathbf{x}_{j}) + (\mathbf{r}_i - \mathbf{r}_j) \\
   &= \tilde{\mathbf{x}} + \mathbf{r}

Torso

.. math::
   \mathbf{r}_i &= \mathbf{0}

Left shoulder

.. math::
   \mathbf{r}_i &= -\mathbf{\hat{e}_{t}}_i

Right shoulder

.. math::
   \mathbf{r}_i &= \mathbf{\hat{e}_{t}}_i

Torso-torso

.. math::
   \tilde{r} &= r_{t, i} + r_{t, j}

Torso-shoulder

.. math::
   \tilde{r} &= r_{t} + r_{s}

Shoulder-shoulder

.. math::
   \tilde{r} &= r_{s, i} + r_{s, j}

Time-to-collision

.. math::
   a &= \tilde{\mathbf{v}} \cdot \tilde{\mathbf{v}} \\
   b &= -(\tilde{\mathbf{x}} + \mathbf{r}) \cdot \tilde{\mathbf{v}} \\
   &= -\tilde{\mathbf{x}} \cdot \tilde{\mathbf{v}} - \mathbf{r} \cdot \tilde{\mathbf{v}} \\
   c &= (\tilde{\mathbf{x}} + \mathbf{r}) \cdot (\tilde{\mathbf{x}} + \mathbf{r}) - \tilde{r}^{2}\\
   &= \| \mathbf{r} \| ^2 + 2 (\mathbf{r} \cdot \tilde{\mathbf{x}}) + \tilde{\mathbf{x}} \cdot \tilde{\mathbf{x}} - \tilde{r}^{2} \\
   d &= \sqrt{b^{2} - a c} \\
   \tau &= \frac{b - d}{a}

Gradient

.. math::
   \nabla_{\tilde{\mathbf{x}}} \tau &= \left(\frac{1}{a} \right) \left(\tilde{\mathbf{v}} -\frac{a (\tilde{\mathbf{x}} + 2 \mathbf{r}) + b \tilde{\mathbf{v}}}{d} \right)

Total gradient

.. math::
   \nabla_{\tilde{\mathbf{x}}} \tau &= \left(\frac{1}{a} \right) \left(3 \tilde{\mathbf{v}} - \sum_{n=1}^{2} \frac{a (\tilde{\mathbf{x}} + 2 \mathbf{r}_n) + b_n \tilde{\mathbf{v}}}{d_n} \right)

Social force for three circle model

.. math::
   \mathbf{f}^{soc} &= \left(\frac{k}{\tau^{2}}\right) \left(\frac{2}{\tau} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right ) \nabla_{\tilde{\mathbf{x}}} \tau
   \\
   &= \left(\frac{k}{\tau^{2}}\right) \left(\frac{2}{\tau} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right ) \left(\frac{1}{a} \right) \left(3 \tilde{\mathbf{v}} - \sum_{n=1}^{2} \frac{a (\tilde{\mathbf{x}} + 2 \mathbf{r}_n) + b_n \tilde{\mathbf{v}}}{d_n} \right)

----

Contact force
-------------
Physical contact force

.. math::
   \mathbf{f}^{c} = - h \cdot \left(\mu \cdot \hat{\mathbf{n}} - \kappa \cdot (\mathbf{v} \cdot \hat{\mathbf{t}}) \hat{\mathbf{t}}\right) + c_{n} \cdot (\mathbf{v} \cdot \hat{\mathbf{n}}) \hat{\mathbf{n}} , \quad h < 0

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_contact


Torque
------

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
