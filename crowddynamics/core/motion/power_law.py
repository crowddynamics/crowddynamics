r"""
Power Law
---------
Anticipatory collision avoidance algorithm introduced in *Universal power law 
governing pedestrian interactions* [Karamouzas2014b]_. Algorithm is derived from
real world data from the behaviour movement of people in crowds.

Force affecting agent can be derived by taking spatial gradient of the energy,
where time-to-collision :math:`\tau` is function of the relative displacement
of the mass centers :math:`\tilde{\mathbf{x}} = \mathbf{x}_i - \mathbf{x}_j`

.. math::
   \mathbf{f}^{soc} &= -\nabla_{\tilde{\mathbf{x}}} E(\tau) \\
   &= - k \cdot m \left(\frac{1}{\tau^{2}}\right) \left(\frac{2}{\tau} + 
   \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right ) 
   \nabla_{\tilde{\mathbf{x}}} \tau

If :math:`\tau < 0` or :math:`\tau` is undefined trajectories are not
colliding and social force is :math:`\mathbf{0}`.

.. tikz::
   
   \draw[color=gray!20] (0, 0) grid (12, 8);
   % Agent i
   \coordinate (c1) at (2, 2);
   \draw[] (c1) circle (1pt) node[below] {$ \mathbf{x}_i $};
   \draw[] (c1) circle (1);
   \draw[dashed, color=gray!80] (c1) -- ++(60:7);
   \draw[thick, ->] (c1) -- node[right] {$ \mathbf{v}_i $} ++(60:1.5);
   % Agent j
   \coordinate (c2) at (10, 4);
   \draw[] (c2) circle (1pt) node[below] {$ \mathbf{x}_i $};
   \draw[] (c2) circle (1);
   \draw[dashed, color=gray!80] (c2) -- ++(155:8);
   \draw[thick, ->] (c2) -- node[left] {$ \mathbf{v}_i $} ++(155:1.5);
   % Distance
   \draw[color=gray!60, dashed] (c1) -- node[below] {$h$} (c2);
   \draw[thick, ->] (c1) -- node[below] {$r_i$} ++(14:1);
   \draw[thick, ->] (c2) -- node[below] {$r_j$} ++(194:1);

"""
import numba
import numpy as np
from numba import f8, i8, typeof
from numba.types import Tuple

from crowddynamics.simulation.agents import agent_type_three_circle, \
    agent_type_circular
from crowddynamics.core.vector2D import dot, truncate


F_SOC_MAX = 2e3  # TODO: load from config


@numba.jit(nopython=True, nogil=True)
def potential(k, tau, tau_0):
    r"""
    Psychological force based on human anticipatory behaviour. Interaction potential
    between two agents is defined

    .. math::
       E(\tau) = m \frac{k}{\tau^{2}} \exp \left( -\frac{\tau}{\tau_{0}} \right)

    Args:
        k (float):
            Scaling parameter for the magnitude. Reference value of
            :math:`k=1.5`.

        tau (float):
            Time-to-collision :math:`\tau > 0` is obtained by linearly extrapolating
            current trajectories and finding where or if agents collide i.e
            skin-to-skin distance :math:`h` equals zero.

        tau_0 (float): Interaction time horizon :math:`\tau_{0} > 0`.

    Returns:
        float: Interaction potential

    """
    return k / tau**2 * np.exp(-tau / tau_0)


@numba.jit(f8(f8, f8), nopython=True, nogil=True, cache=True)
def magnitude(tau, tau_0):
    r"""
    Magnitude of social force.

    .. math::
       \left(\frac{1}{\tau^{2}}\right) \left(\frac{2}{\tau} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right )

    Args:
        tau (float):
            Time-to-collision :math:`\tau > 0` is obtained by linearly extrapolating
            current trajectories and finding where or if agents collide i.e
            skin-to-skin distance :math:`h` equals zero.

        tau_0 (float): Interaction time horizon :math:`\tau_{0} > 0`.

    Returns:
        float: Magnitude
    """
    return (2.0 / tau + 1.0 / tau_0) * np.exp(-tau / tau_0) / tau ** 2


@numba.jit(f8[:](f8[:], f8[:], f8, f8, f8),
           nopython=True, nogil=True, cache=True)
def gradient_circle_circle(x_rel, v_rel, a, b, d):
    r"""Gradient of :math:`\tau` between two circles.
    
    .. math::
       \nabla_{\tilde{\mathbf{x}}} \tau       

    Args:
        x_rel (numpy.ndarray): Relative position of the centers of mass.
            :math:`\tilde{x} = x_i - x_j`
        v_rel (numpy.ndarray): Relative velocity between two agents.
            :math:`\tilde{v} = v_i - v_j`
        a (float):
        b (float):
        d (float):

    Returns:
        numpy.ndarray:

    """
    return (v_rel - (v_rel * b + x_rel * a) / d) / a


@numba.jit(f8[:](f8[:], f8[:], f8[:], f8, f8, f8),
           nopython=True, nogil=True, cache=True)
def gradient_three_circle(x_rel, v_rel, r_off, a, b, d):
    r"""Gradient of :math:`\tau` between two three-circle representations.
    
    .. math::
       \nabla_{\tilde{\mathbf{x}}} \tau

    Args:
        x_rel (numpy.ndarray):
        v_rel (numpy.ndarray):
        r_off (numpy.ndarray):
        a (float):
        b (float):
        d (float):

    Returns:
        numpy.ndarray:

    """
    return (v_rel - (a * (x_rel + 2 * r_off) + b * v_rel) / d) / a


@numba.jit(Tuple((f8, f8[:]))(f8[:], f8[:], f8),
           nopython=True, nogil=True, cache=True)
def time_to_collision_circle_circle(x_rel, v_rel, r_tot):
    r"""Time-to-collision of two circles. From *skin-to-skin* distance

    .. math::
       h(\tau) = \| \tau \tilde{\mathbf{v}} + \mathbf{c} \| - \tilde{r}.

    Solve for root

    .. math::
       h(\tau) &= 0 \\
       \| \mathbf{c} + \tau \tilde{\mathbf{v}} \| &= \tilde{r} \\
       \| \mathbf{c} + \tau \tilde{\mathbf{v}} \|^2 &= \tilde{r}^2

    Quadratic equation is obtained

    .. math::
       \tau^2 (\tilde{\mathbf{v}} \cdot \tilde{\mathbf{v}}) + 2 \tau (\mathbf{c} \cdot \tilde{\mathbf{v}}) + \mathbf{c} \cdot \mathbf{c} - \tilde{r}^2 =0

    Solution with quadratic formula gives us

    .. math::
       a &= \tilde{\mathbf{v}} \cdot \tilde{\mathbf{v}} \\
       b &= -\mathbf{c} \cdot \tilde{\mathbf{v}} \\
       c &= \mathbf{c} \cdot \mathbf{c} - \tilde{r}^{2}\\
       d &= \sqrt{b^{2} - a c} \\
       \tau &= \frac{b - d}{a}.

    Args:
        x_rel (numpy.ndarray):
            Relative center of mass :math:`\mathbf{c}`.

        v_rel (numpy.ndarray):
            Relative velocity :math:`\mathbf{\tilde{v}}`.

        r_tot (float):
            Total radius of :math:`\tilde{r} = \mathrm{constant}`.

    Returns:
        (float, numpy.ndarray):

    """
    a = dot(v_rel, v_rel)
    b = -dot(x_rel, v_rel)
    c = dot(x_rel, x_rel) - r_tot ** 2
    d = np.sqrt(b ** 2 - a * c)

    # No interaction if tau cannot be defined.
    if np.isnan(d) or d == 0 or a == 0:
        return np.nan, np.zeros(2)

    tau = (b - d) / a  # Time-to-collision. In seconds

    if tau <= 0:
        return np.nan, np.zeros(2)

    grad = gradient_circle_circle(x_rel, v_rel, a, b, d)
    return tau, grad


@numba.jit(Tuple((f8[:], f8[:]))(typeof(agent_type_circular)[:], i8, i8),
           nopython=True, nogil=True, cache=True)
def force_social_circular(agent, i, j):
    """Social force based on human anticipatory behaviour.

    Args:
        agent (numpy.ndarray):
        i (int):
        j (int):

    Returns:
        (numpy.ndarray, numpy.ndarray):
    """
    force_i = np.zeros(2)
    force_j = np.zeros(2)

    x_rel = agent[i]['position'] - agent[j]['position']
    v_rel = agent[i]['velocity'] - agent[j]['velocity']
    r_tot = agent[i]['radius'] + agent[j]['radius']

    a = dot(v_rel, v_rel)
    b = -dot(x_rel, v_rel)
    c = dot(x_rel, x_rel) - r_tot ** 2
    d = np.sqrt(b ** 2 - a * c)

    # No interaction if tau cannot be defined.
    if np.isnan(d) or d == 0 or a == 0:
        return force_i, force_j

    tau = (b - d) / a  # Time-to-collision. In seconds
    tau_max = 30.0  # Maximum time for interaction.

    if tau <= 0 or tau > tau_max:
        return force_i, force_j

    # Force is returned negative as repulsive force
    grad = gradient_circle_circle(x_rel, v_rel, a, b, d)
    force_i[:] += - agent[i]['mass'] * agent[i]['k_soc'] * grad * \
                  magnitude(tau, agent[i]['tau_0'])
    force_j[:] -= - agent[j]['mass'] * agent[j]['k_soc'] * grad * \
                  magnitude(tau, agent[j]['tau_0'])

    # Truncation for small tau
    truncate(force_i, F_SOC_MAX)
    truncate(force_j, F_SOC_MAX)

    return force_i, force_j


@numba.jit(Tuple((f8[:], f8[:]))(typeof(agent_type_three_circle)[:], i8, i8),
           nopython=True, nogil=True, cache=True)
def force_social_three_circle(agent, i, j):
    """Social force based on human anticipatory behaviour.

    Args:
        agent (numpy.ndarray):
        i (int):
        j (int):

    Returns:
        (numpy.ndarray, numpy.ndarray):
    """
    # Forces for agent i and j
    force_i = np.zeros(2)
    force_j = np.zeros(2)

    v_rel = agent[i]['velocity'] - agent[j]['velocity']
    a = dot(v_rel, v_rel)

    # Agents are not moving relative to each other.
    if a == 0:
        return force_i, force_j

    # Meaning of indexes for tuples of three
    # 0 = torso
    # 1 = left shoulder
    # 2 = right shoulder

    # Positions: center, left, right
    x_i = (agent[i]['position'], agent[i]['position_ls'], agent[i]['position_rs'])
    x_j = (agent[j]['position'], agent[j]['position_ls'], agent[j]['position_rs'])

    # Radii of torso and shoulders
    r_i = (agent[i]['r_t'], agent[i]['r_s'], agent[i]['r_s'])
    r_j = (agent[j]['r_t'], agent[j]['r_s'], agent[j]['r_s'])

    # Parts that will be first in contact for agents i and j if colliding.
    contact_i = np.int64(0)
    contact_j = np.int64(0)

    # Find smallest time-to-collision. In seconds.
    tau = np.nan
    b_min = np.nan
    d_min = np.nan

    for part_i, (xi, ri) in enumerate(zip(x_i, r_i)):
        for part_j, (xj, rj) in enumerate(zip(x_j, r_j)):
            # Relative position and total radius
            x_rel = xi - xj
            r_tot = ri + rj

            # Coefficients for time-to-collision
            b = -dot(x_rel, v_rel)
            c = dot(x_rel, x_rel) - r_tot ** 2
            d = np.sqrt(b ** 2 - a * c)

            # No interaction if tau cannot be defined.
            if np.isnan(d) or d == 0:
                continue

            tau_new = (b - d) / a
            if np.isnan(tau) or 0 < tau_new < tau:
                contact_i, contact_j = part_i, part_j
                tau = tau_new
                b_min = b
                d_min = d

    if np.isnan(tau) or tau <= 0:
        return force_i, force_j

    # Shoulder displacement vectors
    r_off_i = np.zeros(2)
    r_off_j = np.zeros(2)

    # TODO: Fix signs
    if contact_i == 1:
        phi = agent[i]['orientation']
        r_off_i += agent[i]['r_ts'] * np.array((np.sin(phi), -np.cos(phi)))
    elif contact_i == 2:
        phi = agent[i]['orientation']
        r_off_i -= agent[i]['r_ts'] * np.array((np.sin(phi), -np.cos(phi)))

    if contact_j == 1:
        phi = agent[j]['orientation']
        r_off_j += agent[j]['r_ts'] * np.array((np.sin(phi), -np.cos(phi)))
    elif contact_j == 2:
        phi = agent[j]['orientation']
        r_off_j -= agent[j]['r_ts'] * np.array((np.sin(phi), -np.cos(phi)))

    x_rel = agent[i]['position'] - agent[j]['position']
    r_off = r_off_i - r_off_j

    # Force
    grad = gradient_three_circle(x_rel, v_rel, r_off, a, b_min, d_min)
    force_i[:] += - agent[i]['mass'] * agent[i]['k_soc'] * grad * magnitude(tau, agent[i]['tau_0'])
    force_j[:] -= - agent[j]['mass'] * agent[j]['k_soc'] * grad * magnitude(tau, agent[j]['tau_0'])

    truncate(force_i, F_SOC_MAX)
    truncate(force_j, F_SOC_MAX)

    return force_i, force_j
