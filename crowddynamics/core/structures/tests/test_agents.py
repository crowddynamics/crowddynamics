import numpy as np

from crowddynamics.core.structures.agents import reset_motion, shoulders, \
    overlapping_circles, overlapping_three_circles

SEED = np.random.randint(0, 100)
np.random.seed(SEED)


def test_circular(agents_circular):
    reset_motion(agents_circular.array)
    assert True


def test_three_circle(agents_three_circle):
    reset_motion(agents_three_circle.array)
    shoulders(agents_three_circle.array)
    assert True


def test_overlapping_circular(agents_circular):
    x = np.random.uniform(-1.0, 1.0, 2)
    r = np.random.uniform(0.0, 1.0)
    overlapping_circles(agents_circular.array, x, r)
    assert True


def test_overlapping_three_circle(agents_three_circle):
    x = (
        np.random.uniform(-1.0, 1.0, 2),
        np.random.uniform(-1.0, 1.0, 2),
        np.random.uniform(-1.0, 1.0, 2)
    )
    r = (
        np.random.uniform(0.0, 1.0),
        np.random.uniform(0.0, 1.0),
        np.random.uniform(0.0, 1.0)
    )
    overlapping_three_circles(agents_three_circle.array, x, r)
    assert True
