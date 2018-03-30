# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')


import sys
import time
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
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from flare.learner import StandardLearner, cast
from flare.dataset.decorators import attributes, segment, divid, sequential, shuffle, data, rebatch


epsilon = 0.00001

SCALE = 120.0
MSCALE = 500.0

BATCH = 3
REPEAT = 600
SIZE = 36
WINDOW = 12
INPUT = 4
OUTPUT = 2

lr = 1e-5

mass = None
sun = None


lasttime = time.time()


def shufflefn(xs, ys):
    # permute on different input
    perm = np.arange(xs.shape[-1])
    np.random.shuffle(perm)
    xs = xs[:, :, :, :, perm]

    # permute on different out
    perm = np.arange(ys.shape[-1])
    np.random.shuffle(perm)
    ys = ys[:, :, :, :, perm]

    # permute on different space dims
    seg = np.arange(2, 5, 1)
    np.random.shuffle(seg)

    perm = np.concatenate((np.array([0, 1]), seg))
    xs = xs[:, perm, :, :, :]

    seg = seg - 1
    perm = np.concatenate((np.array([0]), seg))
    ys = ys[:, perm, :, :, :]

    return xs, ys


def divergence(xs, ys):
    xs = xs - sun
    ys = ys - sun
    rx = np.linalg.norm(xs)
    ry = np.linalg.norm(ys)
    ux = xs / rx
    uy = ys / ry
    da = np.empty(ux.shape[0])
    for i in range(ux.shape[0]):
        da[i] = 1 - np.dot(ux[i], uy[i].T)
    dr = (rx - ry) * (rx - ry)
    return np.sum(da + dr)


def divergence_th(xs, ys):
    sz = xs.size()
    b = sz[0]
    v = sz[2] * sz[3]
    s = Variable(cast(sun), requires_grad = False)
    s = th.cat([s for _ in range(b // s.size()[0])], dim=0)

    xs = xs.permute(0, 2, 3, 1).contiguous().view(b, v, 3)
    ys = ys.permute(0, 2, 3, 1).contiguous().view(b, v, 3)
    xs = xs - s
    ys = ys - s
    rx = th.norm(xs, p=2, dim=-1, keepdim=True)
    ry = th.norm(ys, p=2, dim=-1, keepdim=True)
    ux = xs / rx
    uy = ys / ry
    da = 1 - th.bmm(ux.view(b * v, 1, 3), uy.view(b * v, 3, 1)).view(b, v, 1)
    dr = ((rx - ry) * (rx - ry)).view(b, v, 1)
    return th.sum(da + dr, dim=2)


def generator(n, m, yrs, btch):
    global lasttime
    lasttime = time.time()

    global mass, sun

    n = int(n)
    m = int(m)
    pv = int(n + 1)
    sz = n + m + 1
    mass = xp.array(xp.random.rand(btch, sz) / MSCALE, dtype=np.float)
    mass[:, 0] = np.ones([btch])

    x = SCALE / 4.0 * (2 * xp.random.rand(btch, sz, 3) - 1)
    x[:, 0, :] = xp.zeros([btch, 3], dtype=xp.float64)

    r = xp.sqrt(x[:, :, 0] * x[:, :, 0] + x[:, :, 1] * x[:, :, 1] + x[:, :, 2] * x[:, :, 2] + epsilon).reshape([btch, INPUT + OUTPUT + 1, 1])
    v = xp.sqrt(au.G / r) * (2 * xp.random.rand(btch, sz, 3) - 1)
    v[:, 0, :] = - np.sum((mass[:, 1:, np.newaxis] * v[:, 1:, :]) / mass[:, 0:1, np.newaxis], axis=1)

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

            sun = rtp[:, 0:1, :].reshape([btch, 1, 3])

            inputm = mass[:, 1:INPUT+1].reshape([btch, n, 1]) / MSCALE
            inputp = rtp[:, 1:pv].reshape([btch, n, 3])
            inputdh = dh[:, 1:pv].reshape([btch, n, 1])
            input = np.concatenate([inputm, inputdh, inputp], axis=2).reshape([btch, n * 5])

            outputm = mass[:, INPUT+1:].reshape([btch, m, 1]) / MSCALE
            outputp = rtp[:, pv:sz].reshape([btch, m, 3])
            output = np.concatenate([outputm, outputp], axis=2).reshape([btch, m * 4])
            yield year, input, output
            lasth = ht

    print('gen:', time.time() - lasttime)
    sys.stdout.flush()
    lasttime = time.time()


@rebatch(repeat=REPEAT)
@shuffle(shufflefn, repeat=REPEAT)
@data(swap=[0, 2, 3, 4, 1])
@sequential(['ds.x'], ['ds.y'], layout_in=[BATCH, SIZE, INPUT, 5], layout_out=[BATCH, SIZE, OUTPUT, 4])
@divid(lengths=[SIZE], names=['ds'])
@segment(segment_size=SIZE)
@attributes('yr', 'x', 'y')
def dataset():
    return generator(INPUT, OUTPUT, SIZE, BATCH)


class Guess(nn.Module):
    def __init__(self, num_classes=120):
        super(Guess, self).__init__()

        self.normal = nn.BatchNorm1d(192)
        self.layer1 = self._make_layer(192, 256)
        self.layer2 = self._make_layer(256, 512)
        self.layer3 = self._make_layer(512, 1024)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, num_in, num_out):
        return nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU(),
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.normal(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = out.view(out.size(0), 5, 12, 2)
        out = F.tanh(out)
        return out


class Encoder(nn.Module):
    def __init__(self, num_classes=576):
        super(Encoder, self).__init__()

        self.normal = nn.BatchNorm1d(360)
        self.layer1 = self._make_layer(360, 512)
        self.layer2 = self._make_layer(512, 1024)
        self.layer3 = self._make_layer(1024, 2048)
        self.linear = nn.Linear(2048, num_classes)

    def _make_layer(self, num_in, num_out):
        return nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU(),
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.normal(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = out.view(out.size(0), 8, 12, 6)
        out = F.tanh(out)
        return out


class Decoder(nn.Module):
    def __init__(self, num_classes=6):
        super(Decoder, self).__init__()

        self.normal = nn.BatchNorm1d(648)
        self.layer1 = self._make_layer(648, 1024)
        self.layer2 = self._make_layer(1024, 2048)
        self.layer3 = self._make_layer(2048, 2048)
        self.linear = nn.Linear(2048, num_classes)

    def _make_layer(self, num_in, num_out):
        return nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU(),
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.normal(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = out.view(out.size(0), 3, 1, 2)
        out = F.tanh(out)
        return out


class Evolve(nn.Module):
    def __init__(self, num_classes=576):
        super(Evolve, self).__init__()

        self.normal = nn.BatchNorm1d(648)
        self.layer1 = self._make_layer(648, 1024)
        self.layer2 = self._make_layer(1024, 2048)
        self.layer3 = self._make_layer(2048, 2048)
        self.linear = nn.Linear(2048, num_classes)

    def _make_layer(self, num_in, num_out):
        return nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU(),
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.normal(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = out.view(out.size(0), 8, 12, 6)
        out = F.tanh(out)
        return out


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

        self.normal = nn.BatchNorm1d(648)
        self.layer1 = self._make_layer(648, 1024)
        self.layer2 = self._make_layer(1024, 2048)
        self.layer3 = self._make_layer(2048, 2048)
        self.linear = nn.Linear(2048, 6)

    def _make_layer(self, num_in, num_out):
        return nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU(),
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.normal(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = out.view(out.size(0), 3, 1, 2)
        out = F.sigmoid(out)
        return out


class Model(nn.Module):
    def __init__(self, bsize=1, omass=None):
        super(Model, self).__init__()
        self.batch = bsize
        self.omass = omass

        self.guess = Guess()
        self.encode = Encoder()
        self.decode = Decoder()
        self.evolve = Evolve()
        self.gate = Gate()

    def batch_size_changed(self, new_val, orig_val):
        self.batch = new_val

    def forward(self, x):
        x = th.squeeze(x, dim=2).permute(0, 2, 1, 3, 4).contiguous()
        sr, sb, sc, ss, si = tuple(x.size())
        x = x.view(sr * sb, sc, ss, si)

        m = x[:, 0:1, :, :]
        dh = x[:, 1:2, :, :]
        p = x[:, 2:5, :, :]

        result = Variable(cast(np.zeros([sr * sb, 3, SIZE, OUTPUT])))
        self.merror = Variable(cast(np.zeros([sr * sb, 1, 1, 1])))

        init_m = m[:, :, 0:WINDOW, :]
        init_p = p[:, :, 0:WINDOW, :]
        init_dh = dh[:, :, 0:WINDOW, :]
        init = th.cat((init_p, init_dh), dim=1)

        guess = self.guess(init)
        guess = guess.view(sr * sb, 5, WINDOW, OUTPUT)
        gmass = guess[:, 0:1, :, :]
        gposn = guess[:, 1:4, :, :]
        gdelh = guess[:, 4:5, :, :]

        self.tmass = m
        self.gmass = th.cat((init_m, gmass), dim=3)
        self.posn = th.cat((init_p, gposn), dim=3)
        self.delh = th.cat((init_dh, gdelh), dim=3)

        self.state = self.encode(th.cat((self.gmass, self.posn, self.delh), dim=1))
        for i in range(SIZE):
            envr = th.cat((self.gmass, self.state), dim=1)
            self.state = self.evolve(envr)
            gate = self.gate(envr)
            target = gate * self.decode(envr)
            result[:, :, i::SIZE, :] = target
            #print('currt:', th.max(target.data), th.min(target.data), th.mean(target.data))
            #sys.stdout.flush()

        gmass = th.cat([gmass for _ in range(int(SIZE / WINDOW))], dim=2)
        result = th.cat([gmass, result], dim=1)

        return result


mse = nn.MSELoss()

model = Model(bsize=BATCH)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)


def predict(xs):
    global lasttime
    print('cns:', time.time() - lasttime)
    sys.stdout.flush()
    lasttime = time.time()

    result = model(xs)
    return result


counter = 0


def loss(xs, ys, result):
    global counter, lasttime
    counter = counter + 1

    xs = xs.permute(0, 2, 1, 3, 4).contiguous()
    sr, sb, sc, ss, si = tuple(xs.size())
    xs = xs.view(sr * sb, sc, ss, si)

    ys = ys.permute(0, 2, 1, 3, 4).contiguous()
    sr, sb, sc, ss, si = tuple(ys.size())
    ys = ys.view(sr * sb, sc, ss, si)

    ms = ys[:, 0:1, :, :]
    ps = ys[:, 1:4, :, :]

    gm = result[:, 0:1, :, :]
    gp = result[:, 1:4, :, :]

    div = divergence_th(gp, ps)
    sizes = tuple(div.size())
    zeros = Variable(cast(np.zeros(sizes)), requires_grad=False)
    lss = mse(div, zeros)
    merror = mse(gm, ms)

    print('-----------------------------')
    print('dur:', time.time() - lasttime)
    print('lss:', th.mean(th.sqrt(lss).data))
    print('mer:', th.mean(th.sqrt(merror).data))
    print('ttl:', th.mean(th.sqrt(lss + merror / 50).data))
    print('-----------------------------')
    sys.stdout.flush()
    lasttime = time.time()

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * (0.1 ** (counter // 100))

    if counter % 1 == 0:
        if th.cuda.is_available():
            input = xs.data.cpu().numpy()[0, 2:5, :, :]
            truth = ps.data.cpu().numpy()[0, :, :, :]
            guess = result.data.cpu().numpy()[0, :, :, :]
            mass = ms[0, 0, 0, :].data.cpu().numpy()
            gmass = model.gmass[0, 0, 0, :].data.cpu().numpy()
            tmass = model.tmass[0, 0, 0, :].data.cpu().numpy()
        else:
            input = xs.data.numpy()[0, 2:5, :, :]
            truth = ps.data.numpy()[0, :, :, :]
            guess = result.data.numpy()[0, :, :, :]
            mass = ms[0, 0, 0, :].data.numpy()
            gmass = model.gmass[0, 0, 0, :].data.numpy()
            tmass = model.tmass[0, 0, 0, :].data.numpy()

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

    return th.mean(lss + merror / 50)


learner = StandardLearner(model, predict, loss, optimizer, batch=BATCH * REPEAT)

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
