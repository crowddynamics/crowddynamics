import numba
from numba import f8

from crowddynamics.core.vector import dot


@numba.jit(f8[:](f8, f8[:], f8[:], f8[:], f8, f8, f8),
           nopython=True, nogil=True, cache=True)
def force_contact(h, n, v, t, mu, kappa, damping):
    r"""Physical contact force with damping. Helbing's original model did not
    include damping, which was added by Langston.

    .. math::
       \mathbf{f}^{c} = - h \cdot \left(\mu \cdot \hat{\mathbf{n}} -
       \kappa \cdot (\mathbf{v} \cdot \hat{\mathbf{t}}) \hat{\mathbf{t}}\right) +
       c_{n} \cdot (\mathbf{v} \cdot \hat{\mathbf{n}}) \hat{\mathbf{n}}

    Args:
        h (float):
            Skin-to-skin distance between agents

        n (numpy.ndarray):
            Normal vector

        v (numpy.ndarray):
            Velocity vector

        t (numpy.ndarray):
            Tangent vector

        mu (float):
            Constant :math:`1.2 \cdot 10^{5}\,\mathrm{kg\,s^{-2}}`

        kappa (float):
            Constant :math:`4.0 \cdot 10^{4}\,\mathrm{kg\,m^{-1}s^{-1}}`

        damping (float):
            Constant :math:`500 \,\mathrm{N}`

    Returns:
        numpy.ndarray: Contact force vector
    """
    return - h * (mu * n - kappa * dot(v, t) * t) + damping * dot(v, n) * n
