# -*- coding: utf-8 -*-

import numpy as np

import nbody
import ode
import unit.au as au

import torch.nn as nn
import torch.optim as optim

from flare.nn.mlp import MLP

from flare.learner import StandardLearner
from flare.dataset.decorators import attributes, segment, divid, sequential, data


def generator(n, m, yrs):
    n = int(n)
    m = int(m)
    pv = int(n + 1)
    sz = int(n + m + 1)
    szn = int(3 * n)
    szm = int(3 * m)
    m = np.random.rand(sz) * 0.001
    m[0] = 1.0

    x = 100.0 * np.random.rand(sz, 3)
    x[0, :] = np.array([0.0, 0.0, 0.0])

    v = np.sqrt(au.G) * np.random.rand(sz, 3)
    v[0] = np.array([0.0, 0.0, 0.0])

    solver = ode.verlet(nbody.acceleration_of(au, m))

    t = 0
    lastyear = 0
    for epoch in range(366 * (yrs + 1)):
        t, x, v = solver(t, x, v, 1)
        year = t / 365.256363004
        if year != lastyear:
            lastyear = year
            input = x[1:pv].reshape(szn).copy() / 100.0
            output = x[pv:sz].reshape(szm).copy() / 100.0
            yield year, input, output


BATCH = 4
WINDOW = 18
INPUT = 4
OUTPUT = 2


@data
@sequential(['ds.x'], ['ds.y'], layout_in=[WINDOW * INPUT * 3], layout_out=[WINDOW * OUTPUT * 3])
@divid(lengths=[WINDOW], names=['ds'])
@segment(segment_size=WINDOW)
@attributes('yr', 'x', 'y')
def dataset():
    return generator(INPUT, OUTPUT, WINDOW)


mse = nn.MSELoss()


model = MLP(dims=[WINDOW * INPUT * 3, WINDOW * 8 * 3, WINDOW * 6 * 3, WINDOW * 4 * 3, WINDOW * OUTPUT * 3], bsize=1)
optimizer = optim.Adam(model.parameters(), lr=1)


def predict(xs):
    return model(xs)


def loss(result, ys):
    return mse(result, ys)


learner = StandardLearner(model, predict, loss, optimizer, batch=BATCH)


if __name__ == '__main__':
    for epoch in range(1000):
        print('.')
        learner.learn(dataset(), dataset())

    print('--------------------------------')
    errsum = 0.0
    for epoch in range(100):
        err = learner.test(dataset())
        print(err)
        errsum += err

    print('--------------------------------')
    print(errsum / 100)



