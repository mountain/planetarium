# -*- coding: utf-8 -*-

import numpy as np

from os import environ

xp = np
if environ.get('CUDA_HOME') is not None:
    import cupy as cp
    xp = np


def acceleration_of(unit, m):
    shape = m.shape
    b = shape[0]
    n = shape[1]
    r = xp.zeros([b, n, n], dtype=xp.float64)
    a = xp.zeros([b, n, n, 3], dtype=xp.float64)

    def acceleration(t, x, v):
        for i in range(n):
            for j in range(i):
                dist = xp.linalg.norm(x[:, i] - x[:, j], axis=-1)
                r[:, i, j] = dist
                r[:, j, i] = dist

        for i in range(n):
            for j in range(n):
                if i != j:
                    val = unit.G * m[:, j] / r[:, i, j] / r[:, i, j]
                    a[:, i, j, :] = val[:, np.newaxis] * (x[:, j] - x[:, i]) / r[:, i, j, np.newaxis]
                else:
                    a[:, i, j, :] = xp.zeros([b, 3])

        return xp.sum(a, axis=2)

    return acceleration


def derivative_of(unit, m):
    shape = m.shape
    b = shape[0]
    n = shape[1]
    r = xp.zeros([b, n, n], dtype=xp.float64)
    a = xp.zeros([b, n, n, 3], dtype=xp.float64)

    def derivative(t, phase):
        x = phase[:, 0:n, :]
        v = phase[:, n:2*n, :]

        for i in range(n):
            for j in range(i):
                dist = xp.linalg.norm(x[:, i, :] - x[:, j, :], axis=-1)
                r[:, i, j] = dist
                r[:, j, i] = dist

        for i in range(n):
            for j in range(n):
                if i != j:
                    val = unit.G * m[:, j] / r[:, i, j] / r[:, i, j]
                    a[:, i, j, :] = val[:] * (x[:, j, :] - x[:, i, :]) / r[:, i, j]
                else:
                    a[:, i, j, :] = xp.zeros([b, 3], dtype=xp.float64)

        dv = xp.sum(a, axis=2)

        return xp.concatenate((v, dv), axis=1)

    return derivative
