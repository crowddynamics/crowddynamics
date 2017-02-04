Social Force Model
==================
Dirk Helbing, a pioneer of social force model, describes social forces

.. epigraph::

   *These forces are not directly exerted by the pedestrians’ personal environment, but they are a measure for the internal motivations of the individuals to perform certain actions (movements).*

Law's of motion are given as coupled *Langevin equations* where translational motion and rotational motion [Helbing2000a]_

.. math::
   \begin{cases}
   m \frac{d^{2}}{d t^{2}} \mathbf{x}(t) = \mathbf{f}(t) + \boldsymbol{\xi}(t) \\
   I \frac{d^{2}}{d t^{2}} \varphi(t) = M(t) + \eta(t)
   \end{cases}

Total force exerted on the agent

.. math::
   \mathbf{f}_{i}(t) = \mathbf{f}_{i}^{adj} + \sum_{j\neq i}^{} \left(\mathbf{f}_{ij}^{soc} + \mathbf{f}_{ij}^{c}\right) + \sum_{w}^{} \left(\mathbf{f}_{iw}^{soc} + \mathbf{f}_{iw}^{c}\right),

Total torque on the agent

.. math::
   M_{i}(t) = M_{i}^{adj} + \sum_{j\neq i}^{} \left(M_{ij}^{soc} + M_{ij}^{c}\right) + \sum_{w}^{} \left(M_{iw}^{soc} + M_{iw}^{c}\right),

Parameters

.. list-table::
   :header-rows: 1

   * - Type
     - Adjusting
     - Social
     - Contact
   * - Force
     - :math:`\mathbf{f}_{i}^{adj}`
     - :math:`\mathbf{f}_{ij}^{soc}` and :math:`\mathbf{f}_{iw}^{soc}`
     - :math:`\mathbf{f}_{ij}^{c}` and :math:`\mathbf{f}_{iw}^{c}`
   * - Torque
     - :math:`M_{i}^{adj}`
     - :math:`M_{ij}^{soc}` and :math:`M_{iw}^{soc}`
     - :math:`M_{ij}^{c}` and :math:`M_{iw}^{c}`
