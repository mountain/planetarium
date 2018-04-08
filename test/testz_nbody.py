# -*- coding: utf-8 -*-

import unittest
import numpy as np

from physics import ode, hamilton, nbody
import unit.au as au
from os import environ

xp = np
if environ.get('CUDA_HOME') is not None:
    xp = np


class TestNbody(unittest.TestCase):

    def setUp(self):
        self.m = xp.array([[1.0, 0.000001]], dtype=xp.float64)
        self.m2 = xp.array([[1.0, 0.000001], [0.000001, 1.0]], dtype=xp.float64)
        self.solver1 = ode.verlet(nbody.acceleration_of(au, self.m))
        self.solver2 = ode.rk4(nbody.derivative_of(au, self.m))
        self.solver3 = ode.verlet(nbody.acceleration_of(au, self.m2))
        self.h = hamilton.hamiltonian(au, self.m)
        self.h2 = hamilton.hamiltonian(au, self.m2)

    def tearDown(self):
        pass

    def test_nbody_verlet(self):
        t = 0
        x = xp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=xp.float64)
        v = xp.array([[[0.0, 0.0, 0.0], [0.0, xp.sqrt(au.G * self.m[0, 0]), 0.0]]], dtype=xp.float64)
        h0 = self.h(x, v)

        for epoch in range(366):
            t, x, v = self.solver1(t, x, v, 1)
            ht = self.h(x, v)

        self.assertLess(xp.linalg.norm(x[0, 0]), 0.001)
        self.assertAlmostEqual(float(xp.linalg.norm(x[0, 1])), 1.0, 6)
        self.assertAlmostEqual(float(x[0, 1][0]), 1.0, 3)
        self.assertAlmostEqual(float(x[0, 1][1]), 0.0, 1)
        self.assertAlmostEqual(float(x[0, 1][2]), 0.0, 3)
        self.assertAlmostEqual(float((ht - h0)[0, 0]), 0.0, 11)
        self.assertAlmostEqual(float((ht - h0)[0, 1]), 0.0, 11)

    def test_nbody_nk4(self):
        t = 0
        x = xp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=xp.float64)
        v = xp.array([[[0.0, 0.0, 0.0], [0.0, xp.sqrt(au.G * self.m[:, 0]), 0.0]]], dtype=xp.float64)
        phase = xp.array([[x[0, 0], x[0, 1], v[0, 0], v[0, 1]]], dtype=xp.float64)
        for epoch in range(366):
            t, phase = self.solver2(t, phase, 1)

        self.assertLess(xp.linalg.norm(x[0, 0]), 0.001)
        self.assertAlmostEqual(float(xp.linalg.norm(x[0, 1])), 1.0, 6)
        self.assertAlmostEqual(float(x[0, 1][0]), 1.0, 3)
        self.assertAlmostEqual(float(x[0, 1][1]), 0.0, 1)
        self.assertAlmostEqual(float(x[0, 1][2]), 0.0, 3)

    def test_nbody_verlet_batch2(self):
        t = 0
        x = xp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=xp.float64)
        v = xp.array([[[0.0, 0.0, 0.0], [0.0, xp.sqrt(au.G * self.m[0, 0]), 0.0]], [[0.0, xp.sqrt(au.G * self.m[0, 0]), 0.0], [0.0, 0.0, 0.0]]], dtype=xp.float64)
        h0 = self.h2(x, v)

        for epoch in range(366):
            t, x, v = self.solver3(t, x, v, 1)
            ht = self.h2(x, v)

        self.assertLess(xp.linalg.norm(x[0, 0]), 0.001)
        self.assertAlmostEqual(float(xp.linalg.norm(x[0, 1])), 1.0, 6)
        self.assertAlmostEqual(float(x[0, 1][0]), 1.0, 3)
        self.assertAlmostEqual(float(x[0, 1][1]), 0.0, 1)
        self.assertAlmostEqual(float(x[0, 1][2]), 0.0, 3)
        self.assertAlmostEqual(float((ht - h0)[0, 0]), 0.0, 11)
        self.assertAlmostEqual(float((ht - h0)[0, 1]), 0.0, 11)

