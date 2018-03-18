# -*- coding: utf-8 -*-

import numpy as np

from os import environ

xp = np
if environ.get('CUDA_HOME') is not None:
    import cupy as cp
    xp = cp


def acceleration_of(unit, m):
    n = len(m)
    r = xp.zeros([n, n])
    a = xp.zeros([n, n, 3])

    def acceleration(t, x, v):
        for i in range(n):
            for j in range(i):
                dist = xp.linalg.norm(x[i] - x[j])
                r[i, j] = dist
                r[j, i] = dist

        for i in range(n):
            for j in range(n):
                if i != j:
                    val = unit.G * m[j] / r[i, j] / r[i, j]
                    a[i, j, :] = val * (x[j] - x[i]) / r[i, j]
                else:
                    a[i, j, :] = xp.zeros([3])

        return xp.sum(a, axis=1)

    return acceleration


def derivative_of(unit, m):
    n = len(m)
    r = xp.zeros([n, n])
    a = xp.zeros([n, n, 3])

    def derivative(t, phase):
        x = phase[:, 0, :]
        v = phase[:, 1::2, :]

        for i in range(n):
            for j in range(i):
                dist = xp.linalg.norm(x[i] - x[j])
                r[i, j] = dist
                r[j, i] = dist

        for i in range(n):
            for j in range(n):
                if i != j:
                    val = unit.G * m[j] / r[i, j] / r[i, j]
                    a[i, j, :] = val * (x[j] - x[i]) / r[i, j]
                else:
                    a[i, j, :] = xp.zeros([3])

        return xp.concatenate((v, xp.sum(a, axis=1, keepdims=True)), axis=1)

    return derivative
