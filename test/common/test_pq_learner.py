# -*- coding: utf-8 -*-

import random
import unittest

import torch as th
import torch.nn as nn
import torch.optim as optim

from flare.learner import PQLearner
from flare.dataset.logic_gates import generator, xor
from flare.dataset.decorators import discrete, data


mse = nn.MSELoss()


@data
@discrete(['x1', 'x2'], ['x3'])
def dataset(n):
    return generator(xor, n)


modelp = nn.Sequential(
          nn.Linear(2, 3),
          nn.ReLU(),
          nn.Linear(3, 1),
)

modelq = nn.Sequential(
          nn.Linear(2, 3),
          nn.ReLU(),
          nn.Linear(3, 1),
)


def predictp(xs):
    a, b = xs[:, 0::2], xs[:, 1::2]
    c = th.cat([a, 1 - b], dim=1)
    d = th.cat([1 - a, b], dim=1)
    return modelp(th.cat([modelq(c), modelq(d)], dim=1))


def predictq(xs):
    a, b = xs[:, 0::2], xs[:, 1::2]
    c = th.cat([a, b], dim=1)
    d = th.cat([1 - a, 1 - b], dim=1)
    return modelq(th.cat([modelp(c), modelp(d)], dim=1))


def loss(result, ys):
    return mse(result, ys)


optimizerp = optim.Adam(modelp.parameters(), lr=0.1)
optimizerq = optim.Adam(modelq.parameters(), lr=0.1)

batched_learner = PQLearner(modelp, predictp, optimizerp, modelq, predictq, optimizerq, loss, batch=32)


class TestPQLearner(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_batch_effective(self):
        for epoch in range(10):
            batched_learner.learn(dataset(128), dataset(32))

        self.assertLess(batched_learner.test(dataset(128)), 0.35)
