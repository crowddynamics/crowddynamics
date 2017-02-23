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

::

   parameter.mass.value
   parameter.mass.unit
   parameter.body_type
   parameter.body
   ...

Parameters (Immutable)
Variables (Mutable)
"""
import numpy as np

from crowddynamics.core.random.functions import truncnorm
from crowddynamics.functions import load_config


class Parameter:
    """Base class for agent parameter."""
    values = None
    validator = None
    unit = None
    symbol = None

    def __init__(self, parameters):
        self.parameters = parameters

    def value(self, *args, **kwargs):
        """Value"""
        return NotImplemented

    def default(self, *args, **kwargs):
        """Default value"""
        return NotImplemented


class Model(Parameter):
    """Model of the agent."""
    values = ['circular', 'three_circle']

    def value(self):
        return self.values[0]


class Mass(Parameter):
    """Mass :math:`m > 0` of the agent."""

    def value(self, mean, mass_scale, size=1):
        v = truncnorm(-3.0, 3.0, loc=mean, abs_scale=mass_scale, size=size)
        if size == 1:
            return np.asscalar(v)
        else:
            return v

    def default(self, size=1):
        body = self.parameters.body()
        return self.value(body['mass'], body['mass_scale'], size=1)


class Radius(Parameter):
    """Total radius :math:`r > 0` of the agent."""

    def value(self, mean, radius_scale, size=1):
        v = truncnorm(-3.0, 3.0, loc=mean, abs_scale=radius_scale, size=size)
        if size == 1:
            return np.asscalar(v)
        else:
            return v

    def default(self, size=1):
        body = self.parameters.body()
        return self.value(body['radius'], body['radius_scale'], size=1)


class RadiusTorso(Parameter):
    """Radius :math:`r_{t} > 0` of the agent's torso."""

    def value(self, *args, **kwargs):
        body = self.parameters.body()
        return body['ratio_rt']

    def default(self, *args, **kwargs):
        return self.value()


class RadiusShoulder(Parameter):
    """Radius :math:`r_{s} > 0` of the agent's shoulder."""

    def value(self, *args, **kwargs):
        body = self.parameters.body()
        return body['ratio_rs']

    def default(self, *args, **kwargs):
        return self.value()


class RadiusTorsoShoulder(Parameter):
    """Radius :math:`r_{st} > 0` from agent's torso (center of mass) to
    shoulder."""

    def value(self, *args, **kwargs):
        body = self.parameters.body()
        return body['ratio_ts']

    def default(self, *args, **kwargs):
        return self.value()


class MomentOfInertia(Parameter):
    """Moment of inertia (aka rotational inertial) :math:`I_{rot}` of agent.
    Default value :math:`4\,kg\,m^{2}` for agent of weight :math:`80` kg and
    radius :math:`0.27` m.
    https://en.wikipedia.org/wiki/Moment_of_inertia
    """

    # TODO: scaling

    def value(self):
        return 4.0

    def default(self, *args, **kwargs):
        return self.value()


class MaximumVelocity(Parameter):
    """Maximum/Target velocity of the agent."""

    def value(self, mean, velocity_scale, size=1):
        v = truncnorm(-3.0, 3.0, loc=mean, abs_scale=velocity_scale, size=size)
        if size == 1:
            return np.asscalar(v)
        else:
            return v

    def default(self, size=1):
        body = self.parameters.body()
        return self.value(body['velocity'], body['velocity_scale'], size=1)


class MaximumAngularVelocity(Parameter):
    """Maximum/Target angular velocity :math:`\omega_0 > 0` of the agent.
    Default value :math:`4\pi` """

    def value(self):
        return 4 * np.pi

    def default(self, *args, **kwargs):
        return self.value()


class Parameters:
    """Agent parameters"""

    def __init__(self, body_type='adult'):
        self.bodies = load_config("body.csv")
        self.body_types = ('adult', 'male', 'female', 'child', 'eldery')
        self.body_type = None
        self.set_body_type(body_type)

        # Parameters
        self.model = Model(self)
        self.mass = Mass(self)
        self.radius = Radius(self)
        self.radius_torso = RadiusTorso(self)
        self.radius_shoulder = RadiusShoulder(self)
        self.radius_torso_shoulder = RadiusTorsoShoulder(self)
        self.moment_of_inertia = MomentOfInertia(self)
        self.maximum_velocity = MaximumVelocity(self)
        self.maximum_angular_velocity = MaximumAngularVelocity(self)

    def set_body_type(self, body_type):
        if body_type in self.body_types:
            self.body_type = body_type
        else:
            raise Exception(
                """Invalid body type.
                Body type: {body_type} not in {body_types}
                """.format(body_type=body_type, body_types=self.body_types))

    def body(self):
        return self.bodies[self.body_type]
