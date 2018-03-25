# -*- coding: utf-8 -*-

import numpy as np

"""
Hamiltonian
"""

def hamiltonian(unit, m):
    n = m.shape[0]
    u = np.zeros([n, n], dtype=np.float64)

    def h(x, v):
        s = np.sum(v * v, axis=1)
        k = m * s / 2.0
        u = np.zeros([n, n])

        for i in range(n):
            for j in range(i):
                if i != j:
                    dist = np.linalg.norm(x[i] - x[j])
                    u[i, j] = - unit.G * m[i] / dist
                    u[j, i] = - unit.G * m[j] / dist
                else:
                    u[i, j] = 0.0

        return k + np.sum(u, axis=1)

    return h

