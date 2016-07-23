Interactions
============

Interactions between agents are modeled using social and contact forces.

Social force
------------
**Social force model for pedestrian dynamics**

Psychological force for collision avoidance. Distance based algorithm used in the original social force model by Helbing

.. math::
   \mathbf{f}^{soc} = A \exp\left(-\frac{h}{B}\right) \hat{\mathbf{n}}

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_social_velocity_independent

----

**A universal power law governing pedestrian interactions**

.. TODO: Figure on how tau is calculated.

Algorithm based on human anticipatory behaviour. Interaction potential between two agents

.. math::
   E(\tau) &= \frac{k}{\tau^{2}} \exp \left( -\frac{\tau}{\tau_{0}} \right), \quad \tau_{0} > 0, \tau > 0

Force affecting agent can be derived by taking spatial gradient of the energy

.. math::
   \mathbf{f}^{soc} &= -\nabla_{\tilde{\mathbf{x}}} E(\tau) \\
   &= \left(\frac{k}{\tau^{2}}\right) \left(\frac{2}{\tau} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right ) \nabla_{\tilde{\mathbf{x}}} \tau

where coefficient :math:`k=1.5 \cdot \langle m_i \rangle` scales the magnitude of the social force and :math:`\tau_{0}` is interaction time horizon . Time-to-collision :math:`\tau` is obtained by linearly extrapolating current trajectories and finding where skin-to-skin distance :math:`h` is zero. If  :math:`\tau < 0` or :math:`\tau` is undefined (complex number or floating point `nan`) trajectories are not colliding and social force is :math:`\mathbf{0}`. [power2014]_

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
   \nabla_{\tilde{\mathbf{x}}} \tau &= \left(\frac{1}{a} \right) \left(\tilde{\mathbf{v}} -\frac{a \tilde{\mathbf{x}} + b \tilde{\mathbf{v}}}{d} \right) \\
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

Torso-torso

.. math::
   \tilde{\mathbf{x}} &= \mathbf{x}_{i} - \mathbf{x}_{j} \\
   \tilde{r} &= r_{t, i} + r_{t, j}

.. math::
   a &= \tilde{\mathbf{v}} \cdot \tilde{\mathbf{v}} \\
   b &= -\tilde{\mathbf{x}} \cdot \tilde{\mathbf{v}} \\
   c &= \tilde{\mathbf{x}} \cdot \tilde{\mathbf{x}} - \tilde{r}^{2}\\
   d &= \sqrt{b^{2} - a c} \\
   \tau &= \frac{b - d}{a}.

.. math::
   \mathbf{f}^{soc} &= - \left(\frac{k}{a \tau^{2}}\right) \left(\frac{2}{\tau} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right ) \left(\tilde{\mathbf{v}} -\frac{a \tilde{\mathbf{x}} + b \tilde{\mathbf{v}}}{d} \right), \\

Torso-shoulder

.. math::
   \tilde{\mathbf{x}}_{ts} &= \tilde{\mathbf{x}} \pm r_{ts} \mathbf{\hat{e}_{t}} \\
   \tilde{r} &= r_{t} + r_{s}

.. math::
   a &= \tilde{\mathbf{v}} \cdot \tilde{\mathbf{v}} \\
   b &= -(\tilde{\mathbf{x}} \pm r_{ts} \mathbf{\hat{e}_{t}}) \cdot \tilde{\mathbf{v}} \\
   &= -\tilde{\mathbf{x}} \cdot \tilde{\mathbf{v}} \pm r_{ts} \mathbf{\hat{e}_{t}} \cdot \tilde{\mathbf{v}} \\
   c &= (\tilde{\mathbf{x}} \pm r_{ts} \mathbf{\hat{e}_{t}}) \cdot (\tilde{\mathbf{x}} \pm r_{ts} \mathbf{\hat{e}_{t}}) - \tilde{r}^{2}\\
   &= r_{ts}^2 \mp 2 r_{ts} (\mathbf{\hat{e}_{t}} \cdot \tilde{\mathbf{x}}) + \tilde{\mathbf{x}} \cdot \tilde{\mathbf{x}} - \tilde{r}^{2} \\
   d &= \sqrt{b^{2} - a c} \\
   \tau &= \frac{b - d}{a}.

Shoulder-shoulder

.. math::
   \tilde{\mathbf{x}}_{ss} &= \tilde{\mathbf{x}} \pm (r_{ts} \mathbf{\hat{e}_{t}})_i \pm (r_{ts} \mathbf{\hat{e}_{t}})_j \\
   \tilde{r} &= r_{s, i} + r_{s, j}

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
