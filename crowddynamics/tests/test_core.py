import unittest
from unittest.case import skip

import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from crowddynamics.core.block_list import block_list
from crowddynamics.core.distance import distance_circle_circle, \
    distance_three_circle, distance_circle_line, distance_three_circle_line
from crowddynamics.core.motion import force_fluctuation, torque_fluctuation, \
    force_adjust, torque_adjust, force_social_helbing, force_contact
from crowddynamics.core.vector2D import cross2d, wrap_to_pi, truncate, \
    rotate270, normalize, length, angle, rotate90, dot2d
from crowddynamics.tests.strategies import real, positive, vector, vectors, line, three_vectors, three_positive


# -----
# Tests
# -----


class Vector2DTest(unittest.TestCase):
    @given(phi=real())
    def test_wrap_to_pi(self, phi):
        ans = wrap_to_pi(phi)
        self.assertIsInstance(ans, float)
        self.assertTrue(-np.pi <= ans <= np.pi)
        if (phi + np.pi) % (2 * np.pi) == 0.0:
            if phi > 0:
                self.assertEqual(ans, np.pi)
            else:
                self.assertEqual(ans, -np.pi)

    @given(a=vector())
    def test_rotate90(self, a):
        ans = rotate90(a)
        self.assertIsInstance(ans, np.ndarray)

    @given(a=vector())
    def test_rotate270(self, a):
        ans = rotate270(a)
        self.assertIsInstance(ans, np.ndarray)

    @given(a=vector())
    def test_angle(self, a):
        ans = angle(a)
        self.assertIsInstance(ans, float)
        self.assertTrue(-np.pi <= ans <= np.pi)

    @given(a=vector())
    def test_length(self, a):
        ans = length(a)
        self.assertIsInstance(ans, float)
        self.assertTrue(ans >= 0)

    @given(a=vector(), b=vector())
    def test_dot(self, a, b):
        ans = dot2d(a, b)
        self.assertIsInstance(ans, float)

    @given(a=vector(), b=vector())
    def test_cross(self, a, b):
        ans = cross2d(a, b)
        self.assertIsInstance(ans, float)

    @skip("Fix near zero")
    @given(a=vector())
    def test_normalize(self, a):
        ans = normalize(a)
        self.assertIsInstance(ans, np.ndarray)
        l = length(ans)
        # FIXME: floats close to zero
        if np.all(a == 0.0):
            self.assertAlmostEqual(l, 0.0)
        else:
            self.assertAlmostEqual(l, 1.0)

    @skip("Fix near zero")
    @given(a=vector(), b=real())
    def test_truncate(self, a, b):
        truncate(a, b)
        self.assertTrue(length(a) <= b)


class MotionTest(unittest.TestCase):
    @given(mass=positive(), scale=positive())
    def test_force_fluctuation(self, mass, scale):
        ans = force_fluctuation(mass, scale)
        self.assertIsInstance(ans, np.ndarray)

    @given(inertia_rot=positive(), scale=positive())
    def test_torque_fluctuation(self, inertia_rot, scale):
        ans = torque_fluctuation(inertia_rot, scale)
        self.assertIsInstance(ans, np.ndarray)

    @given(mass=positive(), tau_adj=positive(), v0=positive(), e0=vector(), v=vector())
    def test_force_adjust(self, mass, tau_adj, v0, e0, v):
        if tau_adj == 0.0:
            with self.assertRaises(ZeroDivisionError):
                force_adjust(mass, tau_adj, v0, e0, v)
        else:
            ans = force_adjust(mass, tau_adj, v0, e0, v)
            self.assertIsInstance(ans, np.ndarray)

    @given(inertia_rot=positive(), tau_rot=positive(), phi_0=real(), phi=real(),
           omega_0=real(), omega=real())
    def test_torque_adjust(self, inertia_rot, tau_rot, phi_0, phi, omega_0,
                           omega):
        if tau_rot == 0.0:
            with self.assertRaises(ZeroDivisionError):
                torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0, omega)
        else:
            ans = torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0,
                                omega)
            self.assertIsInstance(ans, float)

    @given(h=real(), n=vector(), a=positive(), b=positive())
    def test_force_social_helbing(self, h, n, a, b):
        if b == 0.0:
            with self.assertRaises(ZeroDivisionError):
                force_social_helbing(h, n, a, b)
        else:
            ans = force_social_helbing(h, n, a, b)
            self.assertIsInstance(ans, np.ndarray)

    @given(h=real(), n=vector(), v=vector(), t=vector(), mu=positive(),
           kappa=positive(), damping=positive())
    def test_force_contact(self, h, n, v, t, mu, kappa, damping):
        ans = force_contact(h, n, v, t, mu, kappa, damping)
        self.assertIsInstance(ans, np.ndarray)


class DistanceTest(unittest.TestCase):
    @given(x0=vector(), r0=positive(), x1=vector(), r1=positive())
    def test_distance_circle_circle(self, x0, r0, x1, r1):
        h, n = distance_circle_circle(x0, r0, x1, r1)

        self.assertIsInstance(h, float)
        self.assertIsInstance(n, np.ndarray)

        x = x0 - x1
        r_tot = r0 + r1

        self.assertGreaterEqual(h, -r_tot)

        if np.all(x == 0.0):
            self.assertAlmostEqual(length(n), 0.0)
        elif np.allclose(x, 0.0):
            # Very small floats cause trouble
            # Don't know if it can be easily fixed
            pass
        else:
            self.assertAlmostEqual(length(n), 1.0)

    @given(x0=three_vectors(), r0=three_positive(),
           x1=three_vectors(), r1=three_positive(), )
    def test_distance_three_circle(self, x0, r0, x1, r1):
        h, n, r_moment0, r_moment1 = distance_three_circle(x0, r0, x1, r1)

        self.assertIsInstance(h, float)
        self.assertIsInstance(n, np.ndarray)
        self.assertIsInstance(r_moment0, np.ndarray)
        self.assertIsInstance(r_moment1, np.ndarray)

    @given(x=vector(), r=positive(), p=line(), )
    def test_distance_circle_line(self, x, r, p):
        h, n = distance_circle_line(x, r, p)
        self.assertIsInstance(h, float)
        self.assertIsInstance(n, np.ndarray)

    @given(x=three_vectors(), r=three_positive(), p=line(), )
    def test_distance_three_circle_line(self, x, r, p):
        h, n, r_moment = distance_three_circle_line(x, r, p)
        self.assertIsInstance(h, float)
        self.assertIsInstance(n, np.ndarray)
        self.assertIsInstance(r_moment, np.ndarray)


class TestBlockList(unittest.TestCase):
    @given(points=vectors(elements=st.floats(0.0, 1.0)),
           cell_width=st.floats(0.1, 1.0))
    def test_block_list(self, points, cell_width):
        index_list, count, offset, x_min, x_max = block_list(points, cell_width)

        self.assertIsInstance(index_list, np.ndarray)
        self.assertIs(index_list.dtype.type, np.int64)

        self.assertIsInstance(count, np.ndarray)
        self.assertIs(count.dtype.type, np.int64)

        self.assertIsInstance(offset, np.ndarray)
        self.assertIs(offset.dtype.type, np.int64)

        self.assertIsInstance(x_min, np.ndarray)
        self.assertIs(x_min.dtype.type, np.int64)

        self.assertIsInstance(x_max, np.ndarray)
        self.assertIs(x_max.dtype.type, np.int64)
