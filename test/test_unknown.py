# -*- coding: utf-8 -*-

import unittest
import numpy as np

import unknown as p
from os import environ

xp = np
if environ.get('CUDA_HOME') is not None:
    import cupy as cp
    xp = np


class TestPlanets(unittest.TestCase):

    def setUp(self):
        p.sun = np.array([[0, 0, 0]])

    def tearDown(self):
        pass

    def test_divergence(self):
        xs = np.array([[0, 0, 1]])
        self.assertEqual(int(xs.shape[0]), 1)
        self.assertEqual(int(xs.shape[1]), 3)

        xs = np.array([[0, 0, 1]])
        self.assertEqual(int(xs.shape[0]), 1)
        self.assertEqual(int(xs.shape[1]), 3)

        xs = np.array([[0, 0, 1]])
        ys = np.array([[0, 0, 1]])
        d = p.divergence(xs, ys)
        self.assertAlmostEqual(float(d), 0.0, 6)

        xs = np.array([[0, 0, 1]])
        ys = np.array([[1, 0, 0]])
        d = p.divergence(xs, ys)
        self.assertAlmostEqual(float(d), 1.0, 6)

        xs = np.array([[0, 0, 1]])
        ys = np.array([[0, 1, 0]])
        d = p.divergence(xs, ys)
        self.assertAlmostEqual(float(d), 1.0, 6)

        xs = np.array([[0, 0, 1]])
        ys = np.array([[0, 0, 2]])
        d = p.divergence(xs, ys)
        self.assertAlmostEqual(float(d), 1.0, 6)
