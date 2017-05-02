import numpy as np

from crowddynamics.core.vector2D import rotate90, angle, dot, cross, normalize, \
    unit_vector, rotate270, length, truncate


def test_rotate90(benchmark):
    value = np.random.uniform(-1.0, 1.0, size=2)
    benchmark(rotate90, value)


def test_rotate270(benchmark):
    value = np.random.uniform(-1.0, 1.0, size=2)
    benchmark(rotate270, value)


def test_length(benchmark):
    value = np.random.uniform(-1.0, 1.0, size=2)
    benchmark(length, value)


def test_angle(benchmark):
    value = np.random.uniform(-1.0, 1.0, size=2)
    benchmark(angle, value)


def test_dot(benchmark):
    v0 = np.random.uniform(-1.0, 1.0, size=2)
    v1 = np.random.uniform(-1.0, 1.0, size=2)
    benchmark(dot, v0, v1)


def test_cross(benchmark):
    v0 = np.random.uniform(-1.0, 1.0, size=2)
    v1 = np.random.uniform(-1.0, 1.0, size=2)
    benchmark(cross, v0, v1)


def test_normalize(benchmark):
    value = np.random.uniform(-1.0, 1.0, size=2)
    benchmark(normalize, value)


def test_truncate(benchmark):
    value = np.array((1.0, 1.0))
    benchmark(truncate, value, 0.5)


def test_unit_vector(benchmark):
    value = np.random.uniform(-1.0, 1.0)
    benchmark(unit_vector, value)
