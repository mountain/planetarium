import random

import numpy as np

from scipy.ndimage import gaussian_filter as gf

from flare.dataset.decorators import attributes
from flare.ploter import plotseq


def brusselator(tao, *args):
    a, b = args[0], args[1]
    def evolve(X, Y):
        dX = (a + X * X * Y - b * X - X) * tao
        dY = (b * X - X * X * Y) * tao
        return gf(X + dX, 0.3), gf(Y + dY, 0.3)

    return evolve


def harmonic(tao, *args):
    def evolve(X, Y):
        dX = Y * tao
        dY = - X * tao
        return X + dX, Y + dY

    return evolve


@attributes('ix', 'x', 'y')
def generator(ev, C0, X0, Y0, m, n):
    for _ in range(n):
        for _ in range(m):
            X, Y = ev(X0, Y0)
            X0, Y0 = X, Y
        C0 = C0 + 1
        yield C0, X, Y


if __name__ == '__main__':
    tao = 0.01
    a, b = 0.5, 1.5
    print(a, b)

    SIZE = 50

    ev = brusselator(tao, a, b)
    X = np.random.rand(1, 1, SIZE, SIZE)
    Y = np.random.rand(1, 1, SIZE, SIZE)
    skip = random.randint(1, 10) * 8 * SIZE

    for _ in range(skip):
        X, Y = ev(X, Y)

    seq = list(generator(ev, X, Y))

    plotseq(seq)

