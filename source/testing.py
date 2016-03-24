# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from source.social_force import f_soc_ij


def dE(pa, pb, va, vb, ra, rb):
    k = 1.5
    m = 2.0
    t0 = 3
    maxt = 999

    w = pb - pa
    v = va - vb
    r = ra + rb
    dist = np.sqrt(w[0] ** 2 + w[1] ** 2)

    if r > dist:
        r = .99 * dist

    a = v.dot(v)
    b = w.dot(v)
    c = w.dot(w) - r * r
    discr = b * b - a * c

    print("a: {}\n"
          "b: {}\n"
          "c: {}\n"
          "d: {}".format(a, b, c, discr))

    if (discr < 0) or (0.001 > a > - 0.001):
        return np.array([0, 0])

    discr = np.sqrt(discr)
    t1 = (b - discr) / a

    t = t1

    if t < 0:
        return np.array([0, 0])
    if t > maxt:
        return np.array([0, 0])

    d = k * np.exp(-t / t0) * (v - (v * b - w * a) / (discr)) / (a * t ** m) * (
        m / t + 1 / t0)

    return d


size = 4
agents_num = 14
shape = (agents_num, 2)  # rows, cols
angle = np.random.uniform(0, 2 * np.pi, agents_num)

position = np.random.uniform(0, size, shape)
velocity = np.stack((np.cos(angle), np.sin(angle)), axis=1)
rad = 0.2


i = 1
for j in range(agents_num):
    args = (position[i], position[j],
            velocity[i], velocity[j],
            rad, rad)
    # print(dE(*args))
    print(f_soc_ij(*args))