# -*- coding: utf-8 -*-

import unittest
import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as optim

import flare.nn.lstm as fnl

from flare.learner import StandardLearner
from flare.dataset.decorators import attributes, window, discrete, data
from flare.dataset.oscillators import generator, harmonic, brusselator


BATCH = 5
SIZE = 3


mse = nn.MSELoss()


@data
@discrete(['u.x', 'u.y'], ['v.x', 'v.y'], layout=[SIZE, SIZE])
@attributes('u', 'v')
@window(2)
def dataset(n):
    ev = harmonic(0.01)
    #ev = brusselator(0.01, 0.5, 1.5)

    X = np.random.rand(1, 1, SIZE, SIZE)
    Y = np.random.rand(1, 1, SIZE, SIZE)

    return generator(ev, 0, X, Y, 10, n)


model = fnl.StackedConvLSTM(2, 2, 2, 2, 1, bsize=BATCH, padding=0, width=SIZE, height=SIZE)


def predict(xs):
    return model(xs)


def loss(xs, ys, result):
    return mse(result, ys)


optimizer = optim.Adam(model.parameters(), lr=1)

learner = StandardLearner(model, predict, loss, optimizer, batch=BATCH)


class TestOscillator(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_oscillator(self):
        for epoch in range(3):
            learner.learn(dataset(256), dataset(32))

        self.assertLess(learner.test(dataset(64)), 1.5)