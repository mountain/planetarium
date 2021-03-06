# -*- coding: utf-8 -*-

import numpy as np

"""
Hamiltonian
"""

def hamiltonian(unit, m):
    b = m.shape[0]
    n = m.shape[1]

    def h(x, v, limit=n):
        s = np.sum(v * v, axis=-1)
        k = m * s / 2.0

        u = np.zeros([b, limit, limit])
        for i in range(limit):
            for j in range(i):
                if i != j:
                    dist = np.linalg.norm(x[:, i] - x[:, j], axis=-1)
                    u[:, i, j] = - unit.G * m[:, i] / dist
                    u[:, j, i] = - unit.G * m[:, j] / dist
                else:
                    u[i, j] = 0.0

        return k[:, :limit] + np.sum(u, axis=2)

    return h

