"""
Agent parameters

- docstring
- default value
    - source
- Validation
    - Type
    - Value
- Unit
- Symbol

Parameters (Immutable)
Variables (Mutable)
"""
import numpy as np

from crowddynamics.core.random.random import truncnorm


class Parameter:
    """Base class for agent parameter."""
    values = None
    validator = None
    unit = None
    symbol = None

    def default(self, *args, **kwargs):
        """Default value."""
        return None


class BodyType(Parameter):
    def default(self, *args, **kwargs):
        pass


class Model(Parameter):
    """Model of the agent."""
    values = ['circular', 'three_circle']

    def default(self):
        return self.values[0]


class Mass(Parameter):
    """Mass :math:`m > 0` of the agent."""

    def default(self, mean, mass_scale, size=1):
        return truncnorm(-3.0, 3.0, loc=mean, abs_scale=mass_scale, size=size)


class Radius(Parameter):
    """Total radius :math:`r > 0` of the agent."""

    def default(self, mean, radius_scale, size=1):
        return truncnorm(-3.0, 3.0, loc=mean, abs_scale=radius_scale, size=size)


class RadiusTorso(Parameter):
    """Radius :math:`r_{t} > 0` of the agent's torso."""

    def default(self, *args, **kwargs):
        pass


class RadiusShoulder(Parameter):
    """Radius :math:`r_{s} > 0` of the agent's shoulder."""

    def default(self, *args, **kwargs):
        pass


class RadiusTorsoShoulder(Parameter):
    """Radius :math:`r_{st} > 0` from agent's torso (center of mass) to
    shoulder."""

    def default(self, *args, **kwargs):
        pass


class MomentOfInertia(Parameter):
    """Moment of inertia (aka rotational inertial) :math:`I_{rot}` of agent.
    Default value :math:`4\,kg\,m^{2}` for agent of weight :math:`80` kg and
    radius :math:`0.27` m.
    https://en.wikipedia.org/wiki/Moment_of_inertia
    """
    # TODO: scaling

    def default(self):
        return 4.0


class MaximumVelocity(Parameter):
    """Maximum/Target velocity of the agent."""

    def default(self, mean, velocity_scale, size=1):
        return truncnorm(-3.0, 3.0, loc=mean, abs_scale=velocity_scale, size=size)


class MaximumAngularVelocity(Parameter):
    """Maximum/Target angular velocity :math:`\omega_0 > 0` of the agent.
    Default value :math:`4\pi` """

    def default(self):
        return 4 * np.pi
