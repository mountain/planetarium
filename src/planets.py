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

from flare.nn.residual import ResidualBlock2D
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
            dh = ht - lasth
            inputp = rtp[1:pv].reshape([szn])
            inputdh = dh[1:pv].reshape([n])
            output = rtp[pv:sz].reshape(szm)
            yield year, inputp, inputdh, output
            lasth = ht


@data
@sequential(['ds.p', 'ds.dh'], ['ds.y'], layout_in=[[SIZE, INPUT, 3], [SIZE, INPUT, 1]], layout_out=[[SIZE, OUTPUT, 3]])
@divid(lengths=[SIZE], names=['ds'])
@segment(segment_size=SIZE)
@attributes('yr', 'p', 'dh', 'y')
def dataset(n):
    return generator(INPUT, OUTPUT, SIZE * n)


class Permutation(nn.Module):
    def __init__(self):
        super(Permutation, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, bsize=1, omass=None):
        super(Model, self).__init__()
        self.batch = bsize
        self.omass = omass

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.guess = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=3, padding=1),
            ResidualBlock2D(256),
            ResidualBlock2D(256),
            ResidualBlock2D(256),
            nn.Conv2d(256, 6, kernel_size=3, padding=1),
        )

        self.evolve = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=3, padding=1),
            ResidualBlock2D(256),
            ResidualBlock2D(256),
            ResidualBlock2D(256),
            nn.Conv2d(256, 4, kernel_size=3, padding=1),
        )
        self.vae = VAE(4, 256, 2)

    def batch_size_changed(self, new_val, orig_val):
        self.batch = new_val

    def forward(self, p, dh):
        windowi = WINDOW * INPUT * self.batch
        window = WINDOW * (INPUT + OUTPUT) * self.batch

        result = Variable(th.zeros(self.batch, 3, SIZE, (INPUT + OUTPUT)))
        self.merror = Variable(th.zeros(self.batch, 1, 1, 1))
        self.error = Variable(th.zeros(self.batch, 1, 1, 1))
        self.divrg = Variable(th.zeros(self.batch, 1, 1, 1))

        init_p = p[:, :, 0:WINDOW, :]
        init_dh = dh[:, :, 0:WINDOW, :]
        init = th.cat((init_p, init_dh), dim=1).view(windowi * 4)

        guess = self.guess(init).view(self.batch, 6, WINDOW, OUTPUT)
        gmass = guess[:, 0:1, :, :]
        gposn = guess[:, 1:4, :, :]
        gdelh = guess[:, 4:6, :, :]

        self.mass = th.cat((self.omass, gmass), dim=1)
        self.posn = th.cat((init_p, gposn), dim=1)
        self.delh = th.cat((init_dh, gdelh), dim=1)
        self.state = th.cat((self.posn, self.delh), dim=1)

        for i in range(SIZE):
            if i < WINDOW:
                result[:, :, i, :] = init_p[:, :, i, :]
            else:
                state = self.state.view(window * 4)
                mu, logvar = self.vae.encode(state)
                inner = self.vae.reparameterize(mu, logvar)
                outer = self.vae.decode(inner)
                self.divrg += vae_loss(self.batch, window * 5, outer, Variable(state.data, requires_grad=False), mu, logvar)

                inner_nxt = self.evolve(inner, dim=2)
                outer_nxt = self.vae.decode(inner_nxt)
                self.state = outer_nxt.view(WINDOW, (INPUT + OUTPUT), 4)
                result[:, :, i, :] = self.state[:, 0:3, i, :]

        return result


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
        learner.learn(dataset(100), dataset(10))

    print('--------------------------------')
    errsum = 0.0
    for epoch in range(1000):
        err = learner.test(dataset(100))
        print(err)
        errsum += err

    print('--------------------------------')
    print(errsum / 1000)
