# -*- coding: utf-8 -*-

import sys
import numpy as np

import nbody
import ode
import hamilton
import unit.au as au

from os import environ

xp = np
if environ.get('CUDA_HOME') is not None:
    xp = np

import torch as th
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


from flare.nn.mlp import MLP
from flare.nn.residual import ResidualBlock1D
from flare.nn.vae import VAE, vae_loss

from flare.learner import StandardLearner
from flare.dataset.decorators import attributes, segment, divid, sequential, data


epsilon = 0.00001

SCALE = 50.0
MSCALE = 500.0

BATCH = 5
SIZE = 18
WINDOW = 6
INPUT = 4
OUTPUT = 2


mass = None


def generator(n, m, yrs):
    global mass

    n = int(n)
    m = int(m)
    pv = int(n + 1)
    sz = int(n + m + 1)
    szn = int(3 * n)
    szm = int(3 * m)
    mass = xp.array(xp.random.rand(sz) / MSCALE, dtype=np.float)
    mass[0] = 1.0

    x = SCALE / 2.0 * (2 * xp.random.rand(sz, 3) - 1)
    x[0, :] = xp.array([0.0, 0.0, 0.0], dtype=xp.float64)

    r = xp.sqrt(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1] + x[:, 2] * x[:, 2] + epsilon).reshape([INPUT + OUTPUT + 1, 1])
    v = xp.sqrt(au.G / r) * (2 * xp.random.rand(sz, 3) - 1)
    v[0] = - np.sum((mass[1:, np.newaxis] * v[1:]) / mass[0], axis=0)

    solver = ode.verlet(nbody.acceleration_of(au, mass))
    h = hamilton.hamiltonian(au, mass)
    lasth = h(x, v)

    t = 0
    lastyear = 0
    for epoch in range(366 * (yrs + 1)):
        t, x, v = solver(t, x, v, 1)
        year = int(t / 365.256363004)
        if year != lastyear:
            lastyear = year
            rtp = x / SCALE
            ht = h(x, v)
            hdel = ht - lasth
            input = np.concatenate([rtp[1:pv].reshape([szn]), hdel[1:pv].reshape([n])]).reshape([szn + n])
            output = rtp[pv:sz].reshape(szm)
            yield year, input, output
            lasth = ht


@data
@sequential(['ds.x'], ['ds.y'], layout_in=[SIZE * INPUT * 4], layout_out=[SIZE * OUTPUT * 3])
@divid(lengths=[SIZE], names=['ds'])
@segment(segment_size=SIZE)
@attributes('yr', 'x', 'y')
def dataset():
    return generator(INPUT, OUTPUT, SIZE)


class Permutation(nn.Module):
    def __init__(self):
        super(Permutation, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, bsize=1):
        super(Model, self).__init__()
        self.batch = bsize

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.guess = nn.Sequential(
            MLP(dims=[WINDOW * INPUT * 4, 2187]),
            Permutation(),
            ResidualBlock1D(2187),
            ResidualBlock1D(2187),
            ResidualBlock1D(2187),
            ResidualBlock1D(2187),
            ResidualBlock1D(2187),
            Permutation(),
            MLP(dims=[2187, WINDOW * (INPUT + OUTPUT) * 3 + INPUT + OUTPUT]),
        )

        self.evolve = nn.Sequential(
            MLP(dims=[7 * (INPUT + OUTPUT), 361]),
            Permutation(),
            ResidualBlock1D(361),
            ResidualBlock1D(361),
            ResidualBlock1D(361),
            ResidualBlock1D(361),
            ResidualBlock1D(361),
            Permutation(),
            MLP(dims=[361, 6 * (INPUT + OUTPUT)]),
        )
        self.vae = VAE(WINDOW * (INPUT + OUTPUT) * 3, 90, 6 * (INPUT + OUTPUT))

        self.zero = th.zeros(1)
        self.zeros = th.zeros([BATCH, 1, SIZE * OUTPUT * 3])
        ones = th.ones(BATCH, 1, 1)
        if th.cuda.is_available():
            ones = ones.cuda()
            self.zero = self.zero.cuda()
            self.zeros = self.zeros.cuda()

        self.error = Variable(self.zero.clone())
        self.divrg = Variable(self.zero.clone())

    def batch_size_changed(self, new_val, orig_val):
        self.batch = new_val
        self.vae.batch_size_changed(new_val, orig_val)

        self.zeros = th.zeros([self.batch, 1, SIZE * OUTPUT * 3])
        ones = th.ones(self.batch, 1, 1)
        if th.cuda.is_available():
            ones = ones.cuda()
            self.zeros = self.zeros.cuda()

    def forward(self, x):
        pivot = WINDOW * INPUT * 3
        final = WINDOW * (INPUT + OUTPUT) * 3
        estm = None

        result_p = Variable(self.zeros.clone())
        self.merror = Variable(self.zero.clone())
        self.error = Variable(self.zero.clone())
        self.divrg = Variable(self.zero.clone())
        for i in range(SIZE - WINDOW):
            start = i * INPUT * 3
            end = (i + WINDOW) * INPUT * 3
            input = x[:, :, start:(end + INPUT * WINDOW)]

            guess = self.guess(input)
            gmass = guess[:, :, 0:(INPUT + OUTPUT)] / MSCALE
            cur = guess[:, :, (INPUT + OUTPUT):WINDOW * (INPUT + OUTPUT) * 4]
            self.gmass = gmass

            inner_mu, inner_logvar = self.vae.encode((cur + 1) / 2)
            inner_cur = self.vae.reparameterize(inner_mu, inner_logvar)
            outer_cur = 2 * self.vae.decode(inner_cur) - 1
            self.divrg += vae_loss(self.batch, final, (outer_cur + 1) / 2, (Variable(cur.data, requires_grad=False) + 1) / 2, inner_mu,
                                   inner_logvar)

            inner_nxt = self.evolve(th.cat([gmass, inner_cur], dim=2))
            nxt = 2 * self.vae.decode(inner_nxt) - 1

            if estm is None:
                output = cur[:, :, pivot:final]
            else:
                output = estm[:, :, pivot:final]
                self.error += mse(estm, Variable(cur.data, requires_grad=False))
                self.merror += mse(gmass[0, 0], Variable(th.from_numpy(mass[1:]).float(), requires_grad=False)) * MSCALE * MSCALE

            start = i * OUTPUT * 3
            end = (i + WINDOW) * OUTPUT * 3
            result_p[:, :, start:end] = (output + cur[:, :, start:end]) / 2
            estm = nxt

        return result_p


mse = nn.MSELoss()

model = Model(bsize=BATCH)
optimizer = optim.Adam(model.parameters(), lr=1)


def predict(xs):
    result = model(xs)
    return result


counter = 0


def loss(xs, ys, result):
    global counter
    counter = counter + 1

    lss = mse(result, ys)
    print('-----------------------------')
    print('loss:', th.max(lss.data))
    print('error:', th.max(model.error.data))
    print('divrg:', th.max(model.divrg.data))
    print('merror:', th.max(model.merror.data))
    print('-----------------------------')
    sys.stdout.flush()

    if counter % 1 == 0:
        input = xs.data.numpy().reshape([SIZE, INPUT, 4])
        truth = ys.data.numpy().reshape([SIZE, OUTPUT, 3])
        guess = result.data.numpy().reshape([SIZE, OUTPUT, 3])
        gmass = model.gmass[0, 0, :].data.numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(input[:, 0, 0], input[:, 0, 1], input[:, 0, 2], 'o', markersize=mass[1] * 3000)
        ax.plot(input[:, 1, 0], input[:, 1, 1], input[:, 1, 2], 'o', markersize=mass[2] * 3000)
        ax.plot(input[:, 2, 0], input[:, 2, 1], input[:, 2, 2], 'o', markersize=mass[3] * 3000)
        ax.plot(input[:, 3, 0], input[:, 3, 1], input[:, 3, 2], 'o', markersize=mass[4] * 3000)
        plt.savefig('data/obsv.png')
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(truth[:, 0, 0], truth[:, 0, 1], truth[:, 0, 2], 'ro', markersize=mass[5] * 3000)
        ax.plot(truth[:, 1, 0], truth[:, 1, 1], truth[:, 1, 2], 'bo', markersize=mass[6] * 3000)
        ax.plot(guess[:, 0, 0], guess[:, 0, 1], guess[:, 0, 2], 'r+', markersize=gmass[4] * 3000)
        ax.plot(guess[:, 1, 0], guess[:, 1, 1], guess[:, 1, 2], 'b+', markersize=gmass[5] * 3000)
        plt.savefig('data/pred.png')
        plt.close()

    return lss + model.error + model.divrg + model.merror


learner = StandardLearner(model, predict, loss, optimizer, batch=BATCH)

if __name__ == '__main__':
    for epoch in range(10000):
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
