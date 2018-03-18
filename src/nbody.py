# -*- coding: utf-8 -*-

import numpy as np


def acceleration_of(unit, m):
    n = len(m)
    r = np.zeros([n, n])
    a = np.zeros([n, n, 3])

    def acceleration(t, x, v):
        for i in range(n):
            for j in range(i):
                dist = np.linalg.norm(x[i] - x[j])
                r[i, j] = dist
                r[j, i] = dist

        for i in range(n):
            for j in range(n):
                if i != j:
                    val = unit.G * m[j] / r[i, j] / r[i, j]
                    a[i, j, :] = val * (x[j] - x[i]) / r[i, j]
                else:
                    a[i, j, :] = np.zeros([3])

        return np.sum(a, axis=1)

    return acceleration


def derivative_of(unit, m):
    n = len(m)
    r = np.zeros([n, n])
    a = np.zeros([n, n, 3])

    def derivative(t, phase):
        x = phase[:, 0, :]
        v = phase[:, 1::2, :]

        for i in range(n):
            for j in range(i):
                dist = np.linalg.norm(x[i] - x[j])
                r[i, j] = dist
                r[j, i] = dist

        for i in range(n):
            for j in range(n):
                if i != j:
                    val = unit.G * m[j] / r[i, j] / r[i, j]
                    a[i, j, :] = val * (x[j] - x[i]) / r[i, j]
                else:
                    a[i, j, :] = np.zeros([3])

        return np.concatenate((v, np.sum(a, axis=1, keepdims=True)), axis=1)

    return derivative
