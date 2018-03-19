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

import torch as th
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

from flare.nn.mlp import MLP
from flare.nn.vae import VAE, vae_loss

from flare.learner import StandardLearner
from flare.dataset.decorators import attributes, segment, divid, sequential, data

BATCH = 5
SIZE = 6
WINDOW = 3
INPUT = 4
OUTPUT = 2


def transform(k, x):
    return x / 120.0


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
    v[0] = - np.sum((m[1:, np.newaxis] * v[1:]) / m[0], axis=0)

    solver = ode.verlet(nbody.acceleration_of(au, m))

    t = 0
    lastyear = 0
    for epoch in range(366 * (yrs + 1)):
        t, x, v = solver(t, x, v, 1)
        year = int(t / 365.256363004)
        if year != lastyear:
            lastyear = year
            rtp = transform(sz, x)
            print('----------------------------------------')
            print(year)
            print(np.max(rtp), np.min(rtp), np.average(rtp))
            print(np.max(v), np.min(v), np.average(v))
            print('----------------------------------------')
            input = rtp[1:pv].reshape(szn)
            output = rtp[pv:sz].reshape(szm)
            yield year, input, output


@data
@sequential(['ds.x'], ['ds.y'], layout_in=[SIZE * INPUT * 3], layout_out=[SIZE * OUTPUT * 3])
@divid(lengths=[SIZE], names=['ds'])
@segment(segment_size=SIZE)
@attributes('yr', 'x', 'y')
def dataset():
    return generator(INPUT, OUTPUT, SIZE)


class Model(nn.Module):
    def __init__(self, bsize=1):
        super(Model, self).__init__()
        self.batch = bsize

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.guess = MLP(dims=[WINDOW * INPUT * 3, 72, 63, WINDOW * (INPUT + OUTPUT) * 3])

        self.evolve = MLP(dims=[12, 24, 12])
        self.vae = VAE(WINDOW * (INPUT + OUTPUT) * 3, 48, 12)

        self.zero = th.zeros(1)
        self.zeros = th.zeros([BATCH, 1, SIZE * OUTPUT * 3])
        ones = th.ones(BATCH, 1, 1)
        if th.cuda.is_available():
            ones = ones.cuda()
            self.zero = self.zero.cuda()
            self.zeros = self.zeros.cuda()

        self.error = Variable(self.zero.clone())
        self.divrg = Variable(self.zero.clone())
        self.ratio = Variable(th.cat([ones, ones, ones, ones, ones, ones,
                                      ones / 2.0, ones / 2.0, ones / 2.0, ones / 2.0, ones / 2.0, ones / 2.0,
                                      ones / 3.0, ones / 3.0, ones / 3.0, ones / 3.0, ones / 3.0, ones / 3.0,
                                      ones / 3.0, ones / 3.0, ones / 3.0, ones / 3.0, ones / 3.0, ones / 3.0,
                                      ones / 2.0, ones / 2.0, ones / 2.0, ones / 2.0, ones / 2.0, ones / 2.0,
                                      ones, ones, ones, ones, ones, ones], dim=2), requires_grad=False)

    def batch_size_changed(self, new_val, orig_val):
        self.batch = new_val
        self.guess.batch_size_changed(new_val, orig_val)
        self.evolve.batch_size_changed(new_val, orig_val)
        self.vae.batch_size_changed(new_val, orig_val)

        self.zeros = th.zeros([self.batch, 1, SIZE * OUTPUT * 3])
        ones = th.ones(self.batch, 1, 1)
        if th.cuda.is_available():
            ones = ones.cuda()
            self.zeros = self.zeros.cuda()

        self.ratio = Variable(th.cat([ones, ones, ones, ones, ones, ones,
                                      ones / 2.0, ones / 2.0, ones / 2.0, ones / 2.0, ones / 2.0, ones / 2.0,
                                      ones / 3.0, ones / 3.0, ones / 3.0, ones / 3.0, ones / 3.0, ones / 3.0,
                                      ones / 3.0, ones / 3.0, ones / 3.0, ones / 3.0, ones / 3.0, ones / 3.0,
                                      ones / 2.0, ones / 2.0, ones / 2.0, ones / 2.0, ones / 2.0, ones / 2.0,
                                      ones, ones, ones, ones, ones, ones], dim=2), requires_grad=False)

    def forward(self, x):
        pivot = WINDOW * INPUT * 3
        final = WINDOW * (INPUT + OUTPUT) * 3
        estm = None
        result = Variable(self.zeros.clone())
        self.error = Variable(self.zero.clone())
        self.divrg = Variable(self.zero.clone())
        for i in range(SIZE - WINDOW):
            start = i * INPUT * 3
            end = (i + WINDOW) * INPUT * 3
            input = x[:, :, start:end]
            cur = self.guess(input)
            gss = cur.clone()
            inner_mu, inner_logvar = self.vae.encode(cur)
            inner_cur = self.vae.reparameterize(inner_mu, inner_logvar)
            outer_cur = self.vae.decode(inner_cur)
            self.divrg += vae_loss(self.batch, final, outer_cur, Variable(cur.data, requires_grad=False), inner_mu, inner_logvar)
            
            inner_nxt = self.evolve(inner_cur)
            nxt = self.vae.decode(inner_nxt)
            
            if estm is None:
                output = cur[:, :, pivot:final]
            else:
                output = estm[:, :, pivot:final]
                self.error += mse(estm, Variable(cur.data, requires_grad=False))
            
            start = i * OUTPUT * 3
            end = (i + WINDOW) * OUTPUT * 3
            result[:, :, start:end] = result[:, :, start:end] + output + 2 * gss[:, :, start:end]
            estm = nxt

        return (result * self.ratio) / 3.0


mse = nn.MSELoss()

model = Model(bsize=BATCH)
optimizer = optim.Adam(model.parameters(), lr=1)


def predict(xs):
    return model(xs)


def loss(xs, ys, result):
    lss = mse(result, ys)
    print('-----------------------------')
    print(th.max(lss.data), th.min(lss.data), th.mean(lss.data))
    print(th.max(model.error.data), th.min(model.error.data), th.mean(model.error.data))
    print(th.max(model.divrg.data), th.min(model.divrg.data), th.mean(model.divrg.data))
    print('-----------------------------')
    return lss + model.error + model.divrg


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
