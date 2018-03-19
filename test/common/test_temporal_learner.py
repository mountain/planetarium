# -*- coding: utf-8 -*-

import unittest
import numpy as np
import random

import torch.nn as nn
import torch.optim as optim

import flare.nn.lstm as fnl

from scipy.ndimage import gaussian_filter as gf
from flare.dataset.decorators import filter, memo, last, feature, segment, divid, sequential, data
from flare.dataset.oscillators import generator, harmonic
from flare.learner import TemporalLearner

BATCH = 2
WINDOW = 3
SIZE = 3

ones = np.ones([SIZE, SIZE])
mse = nn.MSELoss()


@data
@sequential(['xs.dt', 'xs.x', 'xs.y'], ['xs.dt', 'ys.x', 'ys.y'], layout=[WINDOW, SIZE, SIZE])
@divid(lengths=[WINDOW, WINDOW], names=['xs', 'ys'])
@segment(segment_size=2*WINDOW)
@feature(lambda ix, _last_: [(ix - _last_['ix']) / 10.0 * ones] if _last_ else [1.0 * ones], ['ix', '_last_'], ['dt'])
@last
@memo
@filter(lambda: random.random() < 0.1, [])
def dataset(n):
    ev = harmonic(0.01)

    X = gf(gf(gf(gf(gf(gf(np.random.rand(1, 1, SIZE, SIZE), sigma=1), sigma=1), sigma=1), sigma=1), sigma=1), sigma=1)
    Y = gf(gf(gf(gf(gf(gf(np.random.rand(1, 1, SIZE, SIZE), sigma=1), sigma=1), sigma=1), sigma=1), sigma=1), sigma=1)

    return generator(ev, 0, X, Y, 10, n)


model = fnl.StackedResTemporalConvLSTM(3, WINDOW, 2, 2, 2, 2, 3, bsize=BATCH, padding=1, width=SIZE, height=SIZE, temporal_scale=1.0)


def predict(ts, ds, xs):
    return model(ts, ds, xs)


def loss(xs, ys, result):
    return mse(result, ys)


optimizer = optim.Adam(model.parameters(), lr=1)

learner = TemporalLearner(model, predict, loss, optimizer, batch=BATCH)


class TestTemporalOscillator(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_oscillator(self):
        for epoch in range(3):
            learner.learn(dataset(512), dataset(128))

        self.assertLess(learner.test(dataset(128)), 2.5)

