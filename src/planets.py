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
from flare.nn.lstm import ConvLSTM
from flare.dataset.decorators import attributes, segment, divid, sequential, data


epsilon = 0.00001

SCALE = 50.0
MSCALE = 500.0

BATCH = 3
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
            inputm = mass[1:INPUT+1].reshape([n, 1]) * MSCALE
            inputp = rtp[1:pv].reshape([n, 3])
            inputdh = dh[1:pv].reshape([n, 1])
            input = np.concatenate([inputm, inputdh, inputp], axis=1).reshape([n * 5])

            outputm = mass[INPUT+1:].reshape([m, 1]) * MSCALE
            outputp = rtp[pv:sz].reshape([m, 3])
            output = np.concatenate([outputm, outputp], axis=1).reshape([m * 4])
            yield year, input, output
            lasth = ht


@data(swap=[0, 2, 3, 4, 1])
@sequential(['ds.x'], ['ds.y'], layout_in=[SIZE, INPUT, 5], layout_out=[SIZE, OUTPUT, 4])
@divid(lengths=[SIZE], names=['ds'])
@segment(segment_size=SIZE)
@attributes('yr', 'x', 'y')
def dataset(n):
    return generator(INPUT, OUTPUT, SIZE * n)


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
            nn.Conv2d(256, 5, kernel_size=3, padding=1),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Tanh(),
        )

        self.lstm = ConvLSTM(4, 4, 3, padding=1, width=WINDOW, height=(INPUT + OUTPUT), bsize=self.batch)

    def batch_size_changed(self, new_val, orig_val):
        self.batch = new_val
        self.lstm.batch_size_changed(new_val, orig_val)
        self.lstm.reset()

    def forward(self, x):
        x = th.squeeze(x, dim=2)
        m = x[:, 0:1, :, :]
        dh = x[:, 1:2, :, :]
        p = x[:, 2:5, :, :]

        result = Variable(th.zeros(self.batch, 3, SIZE, OUTPUT))
        self.merror = Variable(th.zeros(self.batch, 1, 1, 1))
        self.divrg = Variable(th.zeros(self.batch, 1, 1, 1))

        init_m = m[:, :, 0:WINDOW, :]
        init_p = p[:, :, 0:WINDOW, :]
        init_dh = dh[:, :, 0:WINDOW, :]
        init = th.cat((init_p, init_dh), dim=1)

        guess = self.guess(init)
        guess = guess.view(self.batch, 5, WINDOW, OUTPUT)
        gmass = guess[:, 0:1, :, :]
        gposn = guess[:, 1:4, :, :]
        gdelh = guess[:, 4:5, :, :]

        self.tmass = m
        self.gmass = th.cat((init_m, gmass), dim=3)
        self.posn = th.cat((init_p, gposn), dim=3)
        self.delh = th.cat((init_dh, gdelh), dim=3)
        self.state = th.cat((self.posn, self.delh), dim=1)

        self.lstm.reset()

        for i in range(SIZE):
            if i < WINDOW:
                result[:, :, i, :] = self.posn[:, :, i, INPUT:(INPUT + OUTPUT)]
            else:
                self.state = self.lstm(self.state)

                if i < SIZE - WINDOW:
                    curr_p = p[:, :, i:i+WINDOW, :]
                    curr_dh = dh[:, :, i:i+WINDOW, :]
                    curr = th.cat((curr_p, curr_dh), dim=1)
                    gcurr = self.guess(curr)
                    gcurr = gcurr.view(self.batch, 5, WINDOW, OUTPUT)
                else:
                    curr_p = p[:, :, i:SIZE, :]
                    curr_dh = dh[:, :, i:SIZE, :]
                    curr = th.cat((curr_p, curr_dh), dim=1)
                    gcurr = self.guess(curr)
                    gcurr = gcurr.view(self.batch, 5, SIZE - i, OUTPUT)

                left = self.state[:, 0:3, -1, INPUT:(INPUT + OUTPUT)]
                right = gcurr[:, 2:5, -1, :]
                result[:, :, i, :] = (left + right) / 2.0

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

    ms = ys[:, 0:1, 0, :, :]
    ps = ys[:, 1:4, 0, :, :]

    lss = mse(result, ps)
    merror = mse(model.gmass, ms)
    print('-----------------------------')
    print('loss:', th.max(lss.data))
    print('merror:', th.max(merror.data))
    print('-----------------------------')
    sys.stdout.flush()

    if counter % 1 == 0:
        input = xs.data.numpy().reshape([model.batch, 5, SIZE, INPUT])[0, 2:5, :, :]
        truth = ps.data.numpy().reshape([model.batch, 3, SIZE, OUTPUT])[0, :, :, :]
        guess = result.data.numpy().reshape([model.batch, 3, SIZE, OUTPUT])[0, :, :, :]
        gmass = model.gmass[0, 0, 0, :].data.numpy()
        tmass = model.tmass[0, 0, 0, :].data.numpy()
        mass = ms[0, 0, 0, :].data.numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(input[0, :, 0], input[1, :, 0], input[2, :, 0], 'o', markersize=tmass[0] * 6)
        ax.plot(input[0, :, 1], input[1, :, 1], input[2, :, 1], 'o', markersize=tmass[1] * 6)
        ax.plot(input[0, :, 2], input[1, :, 2], input[2, :, 2], 'o', markersize=tmass[2] * 6)
        ax.plot(input[0, :, 3], input[1, :, 3], input[2, :, 3], 'o', markersize=tmass[3] * 6)
        plt.savefig('data/obsv.png')
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(truth[0, :, 0], truth[1, :, 0], truth[2, :, 0], 'ro', markersize=mass[0] * 6)
        ax.plot(truth[0, :, 1], truth[1, :, 1], truth[2, :, 1], 'bo', markersize=mass[1] * 6)
        ax.plot(guess[0, :, 0], guess[1, :, 0], guess[2, :, 0], 'r+', markersize=gmass[0] * 6)
        ax.plot(guess[0, :, 1], guess[1, :, 1], guess[2, :, 1], 'b+', markersize=gmass[1] * 6)
        plt.savefig('data/pred.png')
        plt.close()

    return th.sum(lss + model.divrg + merror)


learner = StandardLearner(model, predict, loss, optimizer, batch=BATCH)

if __name__ == '__main__':
    for epoch in range(10000):
        print('.')
        learner.learn(dataset(10), dataset(1))

    print('--------------------------------')
    errsum = 0.0
    for epoch in range(1000):
        err = learner.test(dataset(10))
        print(err)
        errsum += err

    print('--------------------------------')
    print(errsum / 1000)
