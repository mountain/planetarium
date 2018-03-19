# -*- coding: utf-8 -*-

import numpy as np

import nbody
import ode
import unit.au as au

from os import environ

xp = np
if environ.get('CUDA_HOME') is not None:
    import cupy as cp
    xp = np

import torch.nn as nn
import torch.optim as optim

from flare.nn.mlp import MLP
from flare.nn.vae import VAE, vae_loss

from flare.learner import StandardLearner
from flare.dataset.decorators import attributes, segment, divid, sequential, data


BATCH = 5
WINDOW = 6
INPUT = 4
OUTPUT = 2

epsilon = 0.00000001

def transform(k, x):
    r = np.sqrt(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1] + x[:, 2] * x[:, 2])
    phi = np.arctan2(x[:, 0], x[:, 1]) / np.pi + 0.5
    theta = np.arccos(x[:, 2] / r) / np.pi
    r = r.reshape([k, 1])
    phi = phi.reshape([k, 1])
    theta = theta.reshape([k, 1])
    return np.concatenate((r / 100.0, theta, phi), axis=1)


def generator(n, m, yrs):
    n = int(n)
    m = int(m)
    pv = int(n + 1)
    sz = int(n + m + 1)
    szn = int(3 * n)
    szm = int(3 * m)
    m = xp.random.rand(sz) * 0.001
    m[0] = 1.0

    x = 75.0 * xp.random.rand(sz, 3)
    x[0, :] = xp.array([0.0, 0.0, 0.0], dtype=xp.float64)

    v = xp.sqrt(au.G) * xp.random.rand(sz, 3)
    v[0] = xp.array([0.0, 0.0, 0.0], dtype=xp.float64)

    solver = ode.verlet(nbody.acceleration_of(au, m))

    t = 0
    lastyear = 0
    for epoch in range(366 * (yrs + 1)):
        t, x, v = solver(t, x, v, 1)
        year = t / 365.256363004
        if year != lastyear:
            lastyear = year
            rtp = transform(sz, x)
            input = rtp[1:pv].reshape(szn)
            output = rtp[pv:sz].reshape(szm)
            yield year, input, output


@data
@sequential(['ds.x'], ['ds.y'], layout_in=[WINDOW * INPUT * 3], layout_out=[WINDOW * OUTPUT * 3])
@divid(lengths=[WINDOW], names=['ds'])
@segment(segment_size=WINDOW)
@attributes('yr', 'x', 'y')
def dataset():
    return generator(INPUT, OUTPUT, WINDOW)


class Model(nn.Module):
    def __init__(self, bsize=1):
        super(Model, self).__init__()
        self.batch = bsize

        self.mlp = MLP(dims=[24, 48, 36, 24, 12])
        self.vaei = VAE(WINDOW * 4 * 3, 48, 24)
        self.vaeo = VAE(WINDOW * 2 * 3, 24, 12)

    def batch_size_changed(self, new_val, orig_val):
        self.batch = new_val
        self.mlp.batch_size_changed(new_val, orig_val)
        self.vae.batch_size_changed(new_val, orig_val)

    def forward(self, x):
        mu, logvar = self.vaei.encode(x)
        z = self.vaei.reparameterize(mu, logvar)
        return self.vaeo.decode(self.mlp(z))


mse = nn.MSELoss()


model = Model()
optimizer = optim.Adam(model.parameters(), lr=1)


def predict(xs):
    return model(xs)


def loss(xs, ys, result):
    rxs, mu, logvar = model.vaei(xs)
    vlossx = vae_loss(model.batch, WINDOW * 4 * 3, rxs, xs, mu, logvar)
    rys, mu, logvar = model.vaeo(ys)
    vlossy = vae_loss(model.batch, WINDOW * 2 * 3, rys, ys, mu, logvar)
    lss = mse(result, ys)

    return lss + vlossx + vlossy


learner = StandardLearner(model, predict, loss, optimizer, batch=BATCH)


if __name__ == '__main__':
    for epoch in range(100000):
        print('.')
        learner.learn(dataset(), dataset())

    print('--------------------------------')
    errsum = 0.0
    for epoch in range(1000):
        err = learner.test(dataset())
        print(err)
        errsum += err

    print('--------------------------------')
    print(errsum / 1000)



