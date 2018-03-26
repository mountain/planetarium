# -*- coding: utf-8 -*-

import unittest
import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as optim

import flare.nn.lstm as fnl

from scipy.ndimage import gaussian_filter as gf
from flare.dataset.decorators import segment, divid, sequential, data
from flare.dataset.oscillators import generator, harmonic, brusselator
from flare.learner import StandardLearner


BATCH = 4
WINDOW = 3
SIZE = 5


mse = nn.MSELoss()


@data()
@sequential(['xs.x', 'xs.y'], ['ys.x', 'ys.y'], layout=[WINDOW, SIZE, SIZE])
@divid(lengths=[WINDOW, WINDOW], names=['xs', 'ys'])
@segment(segment_size=2*WINDOW)
def dataset(n):
    ev = harmonic(0.01)
    #ev = brusselator(0.01, 0.5, 1.5)

    X = gf(gf(gf(gf(gf(gf(np.random.rand(1, 1, SIZE, SIZE), sigma=1), sigma=1), sigma=1), sigma=1), sigma=1), sigma=1)
    Y = gf(gf(gf(gf(gf(gf(np.random.rand(1, 1, SIZE, SIZE), sigma=1), sigma=1), sigma=1), sigma=1), sigma=1), sigma=1)

    return generator(ev, 0, X, Y, 10, n)


model = fnl.StackedSquentialConvLSTM(3, WINDOW, 2, 2, 3, bsize=BATCH, padding=1, width=SIZE, height=SIZE)


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
        for epoch in range(2):
            learner.learn(dataset(256), dataset(32))

        self.assertLess(learner.test(dataset(64)), 1.5)


