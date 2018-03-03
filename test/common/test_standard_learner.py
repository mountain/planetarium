# -*- coding: utf-8 -*-

import unittest

import torch as th
import torch.nn as nn
import torch.optim as optim

from flare.dataset.logic_gates import generator, xor
from flare.dataset.decorators import discrete, data
from flare.learner import StandardLearner


BATCH = 32


mse = nn.MSELoss()


@data
@discrete(['x1', 'x2'], ['x3'])
def dataset(n):
    return generator(xor, n)


model = nn.Sequential(
          nn.Linear(2, 3),
          nn.ReLU(),
          nn.Linear(3, 1),
)


def predict(xs):
    return model(xs)


def loss(result, ys):
    return mse(result, ys)


optimizer = optim.Adam(model.parameters(), lr=0.1)

batched_learner = StandardLearner(model, predict, loss, optimizer, batch=32)


class TestStandardLearner(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_batch_effective(self):
        for epoch in range(10):
            batched_learner.learn(dataset(128), dataset(32))

        self.assertLess(batched_learner.test(dataset(128)), 0.35)
