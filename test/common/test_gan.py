# -*- coding: utf-8 -*-

import random
import unittest

import torch as th
import torch.nn as nn
import torch.optim as optim

from flare.gan import GANLearner
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
          nn.Linear(3, 3),
          nn.ReLU(),
          nn.Linear(3, 1),
)


def predictp(xs):
    return modelp(xs)


def predictq(xs):
    return modelq(xs)


def extrag(xs, ys):
    return xs


def extrad(ys, xs):
    return th.cat([ys, xs], dim=1)


def loss(result, ys):
    return mse(result, ys)


optimizerp = optim.Adam(modelp.parameters(), lr=0.1)
optimizerq = optim.Adam(modelq.parameters(), lr=0.1)

batched_learner = GANLearner(modelp, predictp, optimizerp, modelq, predictq, optimizerq, loss, loss, extrag, extrad, batch=32)


class TestGAN(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_batch_effective(self):
        for epoch in range(10):
            batched_learner.learn(dataset(128), dataset(32))

        self.assertLess(batched_learner.test(dataset(128)), 0.35)
