"""
Crowds in real life often have a number of smaller subgroups. Such as group of
two or three friends walking together or couple with or without children.
Model by Langston for subgroups at most 4 people.
"""


def attractor_point():
    r"""
    Attractor point

    .. math::
       \mathbf{x}_{att} = \mathbf{x}_{n} \pm d \hat{\mathbf{e}} (\theta)

    where

    * :math:`\mathbf{x}_{n}` is the coordinates of relative neighbor
    * :math:`d` is desired distance between two consecutive members
    * :math:`\theta` is desired angular orientation of the subgroup

    Returns:

    """
    return NotImplementedError


def adjusting_force_intra_subgroup():
    r"""
    Intra-subgroup adjusting force.

    Subgroup adjusting force adds additional term to adjusting force in social force model

    .. math::
       \mathbf{f}^{adj} = \frac{m}{\tau^{adj}} (v_{0} \hat{\mathbf{e}}  + \underset{\text{new}}{\underbrace{k \mathbf{e}_{att}}} - \mathbf{v})

    * :math:`\mathbf{e}_{att} = \mathbf{x}_{att} - \mathbf{x}_{i}` relative displacement from the formation attractor point
    * :math:`k` subgroup velocity constant

    Returns:

    """
    return NotImplementedError
