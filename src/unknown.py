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
from flare.nn.lstm import ConvLSTM, StackedConvLSTM
from flare.nn.nri import MLPEncoder, MLPDecoder, get_tril_offdiag_indices, get_triu_offdiag_indices
from flare.nn.nri import gumbel_softmax, my_softmax, encode_onehot, nll_gaussian, kl_categorical_uniform
from flare.dataset.decorators import attributes, segment, divid, sequential, shuffle, data, rebatch


epsilon = 0.00000001

SCALE = 120.0
MSCALE = 500.0
VSCALE = 10000.0

BATCH = 3
REPEAT = 5
SIZE = 32
WINDOW = 8
INPUT = 5
OUTPUT = 2

lr = 1e-5

mass = None
sun = None


lasttime = time.time()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mnorm(x):
    return 2 * sigmoid(6 / (1 - np.log(x))) - 1


def msize(x):
    return int(1 + x * 10.0)


def shufflefn(xs, ys):
    # permute on different input
    seg = np.arange(1, xs.shape[-2], 1)
    np.random.shuffle(seg)
    perm = np.concatenate((np.array([0]), seg))
    xs = xs[:, :, :, perm, :]

    # permute on different out
    perm = np.arange(ys.shape[-2])
    np.random.shuffle(perm)
    ys = ys[:, :, :, perm, :]

    # permute on different space dims
    seg = np.arange(2, 5, 1)
    np.random.shuffle(seg)
    perm = np.concatenate((np.array([0, 1]), seg, seg + 3))

    xs = xs[:, :, :, :, perm]
    ys = ys[:, :, :, :, perm]

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

    sz = n + m
    mass = xp.array(xp.random.rand(btch, sz) / MSCALE, dtype=np.float)
    mass[:, 0] = np.ones([btch])

    x = SCALE / 2.0 * (2 * xp.random.rand(btch, sz, 3) - 1)
    x[:, 0, :] = xp.zeros([btch, 3], dtype=xp.float64)

    r = xp.sqrt(x[:, :, 0] * x[:, :, 0] + x[:, :, 1] * x[:, :, 1] + x[:, :, 2] * x[:, :, 2] + epsilon).reshape([btch, sz, 1])
    u = 2 * xp.random.rand(btch, sz, 3) - 1
    u = u / xp.sqrt(u[:, :, 0] * u[:, :, 0] + u[:, :, 1] * u[:, :, 1] + u[:, :, 2] * u[:, :, 2] + epsilon).reshape([btch, sz, 1])
    v = xp.sqrt(au.G / r) * u
    v[:, 0, :] = - np.sum((mass[:, 1:, np.newaxis] * v[:, 1:, :]) / mass[:, 0:1, np.newaxis], axis=1)

    center = (np.sum(mass.reshape([btch, sz, 1]) * x, axis=1) / np.sum(mass, axis=1).reshape([btch, 1])).reshape([btch, 1, 3])
    x = x - center

    solver = ode.verlet(nbody.acceleration_of(au, mass))
    h = hamilton.hamiltonian(au, mass)
    lastha = h(x, v, limit=sz)
    lasthl = h(x, v, limit=n)

    t = 0
    lastyear = 0
    for epoch in range(366 * (yrs + 1)):
        t, x, v = solver(t, x, v, 1)

        center = (np.sum(mass.reshape([btch, sz, 1]) * x, axis=1) / np.sum(mass, axis=1).reshape([btch, 1])).reshape([btch, 1, 3])
        x = x - center

        year = int(t / 365.256363004)
        if year != lastyear:
            lastyear = year
            rtp = x / SCALE
            rtv = v / SCALE
            ha = h(x, v, limit=sz)
            hl = h(x, v, limit=n)
            dha = ha - lastha
            dhl = hl - lasthl

            sun = rtp[:, 0:1, :].reshape([btch, 1, 3])

            inputm = mnorm(mass[:, 0:n].reshape([btch, n, 1]))
            inputp = rtp[:, 0:n].reshape([btch, n, 3])
            inputv = rtv[:, 0:n].reshape([btch, n, 3]) * VSCALE
            inputdh = dhl[:, 0:n].reshape([btch, n, 1]) / au.G * SCALE
            input = np.concatenate([inputm, inputdh, inputp, inputv], axis=2).reshape([btch, n * 8])

            outputm = mnorm(mass[:, n:].reshape([btch, m, 1]))
            outputp = rtp[:, n:].reshape([btch, m, 3])
            outputv = rtv[:, n:].reshape([btch, m, 3]) * VSCALE
            outputdh = dha[:, n:].reshape([btch, m, 1]) / au.G * SCALE
            output = np.concatenate([outputm, outputdh, outputp, outputv], axis=2).reshape([btch, m * 8])
            yield year, input, output
            lastha = ha
            lasthl = hl

            #print('-----------------------------')
            #print('im:', np.max(inputm), np.min(inputm))
            #print('ip:', np.max(inputp), np.min(inputp))
            #print('iv:', np.max(inputv), np.min(inputv))
            #print('ih:', np.max(inputdh), np.min(inputdh))
            #print('om:', np.max(outputm), np.min(outputm))
            #print('op:', np.max(outputp), np.min(outputp))
            #print('ov:', np.max(outputv), np.min(outputv))
            #print('oh:', np.max(outputdh), np.min(outputdh))
            #print('-----------------------------')
            #sys.stdout.flush()

    print('gen:', time.time() - lasttime)
    sys.stdout.flush()
    lasttime = time.time()


@rebatch(repeat=REPEAT)
@shuffle(shufflefn, repeat=REPEAT)
@data()
@sequential(['ds.x'], ['ds.y'], layout_in=[SIZE, BATCH, INPUT, 8], layout_out=[SIZE, BATCH, OUTPUT, 8])
@divid(lengths=[SIZE], names=['ds'])
@segment(segment_size=SIZE)
@attributes('yr', 'x', 'y')
def dataset():
    return generator(INPUT, OUTPUT, SIZE, BATCH)


class Guess(nn.Module):
    def __init__(self, num_classes=8 * WINDOW * OUTPUT):
        super(Guess, self).__init__()

        self.lstm = StackedConvLSTM(3, 8 * WINDOW * INPUT, 2048, 2048, 1, padding=0, bsize=REPEAT*BATCH, width=1, height=1)
        self.linear = nn.Linear(2048, num_classes)

    def batch_size_changed(self, new_val, orig_val):
        new_val = new_val * REPEAT
        self.batch = new_val
        self.lstm.batch_size_changed(orig_val, orig_val, force=True)
        self.lstm.reset()

    def forward(self, x):
        out = x.view(x.size(0), -1, 1, 1)
        out = self.lstm(out)
        out = out.view(out.size(0), -1)
        out = th.tanh(self.linear(out))
        out = out.view(out.size(0), 8, WINDOW, OUTPUT)

        mn = th.sigmoid(out[:, 0:1, :, :])
        hn = out[:, 1:2, :, :]
        pn = out[:, 2:5, :, :]
        vn = out[:, 5:8, :, :]
        out = th.cat([mn, hn, pn, vn], dim=1)

        print('guessm:', th.max(mn.data), th.min(mn.data))
        print('guessh:', th.max(hn.data), th.min(hn.data))
        print('guessx:', th.max(pn.data), th.min(pn.data))
        print('guessv:', th.max(vn.data), th.min(vn.data))
        sys.stdout.flush()

        return out


class Evolve(nn.Module):
    def __init__(self):
        super(Evolve, self).__init__()
        n = INPUT + OUTPUT
        w = WINDOW
        c = 8
        d = c * w

        off_diag = np.ones([n, n]) - np.eye(n)
        self.rel_rec = Variable(cast(np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)))
        self.rel_send = Variable(cast(np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)))

        self.encoder = MLPEncoder(d, 2048, 1)
        self.decoder = MLPDecoder(c, 1, 2048, 2048, 2048)

    def forward(self, x, w=WINDOW):
        mo = x[:, 0:1, :, :]
        out = x.permute(0, 3, 2, 1).contiguous()

        logits = self.encoder(out, self.rel_rec, self.rel_send)
        edges = gumbel_softmax(logits)
        self.prob = my_softmax(logits, -1)
        out = self.decoder(out, edges, self.rel_rec, self.rel_send, w)
        out = out.permute(0, 3, 2, 1).contiguous()

        hn = out[:, 1:2, :, :]
        pn = out[:, 2:5, :, :]
        vn = out[:, 5:8, :, :]
        out = th.cat([mo, hn, pn, vn], dim=1)

        print('evolvm:', th.max(mo.data), th.min(mo.data))
        print('evolvh:', th.max(hn.data), th.min(hn.data))
        print('evolvx:', th.max(pn.data), th.min(pn.data))
        print('evolvv:', th.max(vn.data), th.min(vn.data))
        sys.stdout.flush()

        return out


class Ratio(nn.Module):
    def __init__(self):
        super(Ratio, self).__init__()

        self.lstm = ConvLSTM(16 * WINDOW * (INPUT + OUTPUT), 2048, 1, padding=0, bsize=REPEAT*BATCH, width=1, height=1)
        self.linear = nn.Linear(2048, 8 * WINDOW * (INPUT + OUTPUT))

    def batch_size_changed(self, new_val, orig_val):
        new_val = new_val * REPEAT
        self.batch = new_val
        self.lstm.batch_size_changed(orig_val, orig_val, force=True)
        self.lstm.reset()

    def forward(self, x):
        out = x.view(x.size(0), -1, 1, 1)
        out = self.lstm(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = out.view(out.size(0), 8, WINDOW, INPUT + OUTPUT)
        out = F.sigmoid(out)

        print('ratio:', th.max(out.data), th.min(out.data))
        sys.stdout.flush()

        return out


class Model(nn.Module):
    def __init__(self, bsize=1):
        super(Model, self).__init__()
        self.batch = bsize
        self.basedim = 8 * WINDOW * (INPUT + OUTPUT)

        self.guess = Guess()
        self.evolve = Evolve()
        self.ratio = Ratio()

    def batch_size_changed(self, new_val, orig_val):
        self.batch = new_val
        self.guess.batch_size_changed(new_val, orig_val)
        self.ratio.batch_size_changed(new_val, orig_val)

    def forward(self, x):
        x = x.permute(0, 2, 4, 1, 3).contiguous()
        sr, sb, sc, ss, si = tuple(x.size())
        x = x.view(sr * sb, sc, ss, si)

        result = Variable(cast(np.zeros([sr * sb, 8, SIZE, INPUT + OUTPUT])))

        init = x[:, :, 0:WINDOW, :]
        guess = self.guess(init.contiguous())
        state = th.cat((init, guess), dim=3)
        result[:, :, 0::SIZE, :] = state[:, :, 0::SIZE, :]

        for i in range(1, SIZE, 1):
            print('-----------------------------')
            print('idx:', i)
            sys.stdout.flush()

            statenx = self.evolve(state)
            input = statenx[:, :, :, :INPUT]
            if i < SIZE - WINDOW:
                init = x[:, :, i:WINDOW+i, :]
                guess = self.guess(init.contiguous())
                stategs = th.cat((init, guess), dim=3)
            else:
                guess = self.guess(input.contiguous())
                stategs = th.cat((input, guess), dim=3)

            ratio = self.ratio(th.cat([statenx, stategs], dim=1))
            state = ratio * statenx + (1 - ratio) * stategs

            result[:, :, i::SIZE, :] = state[:, :, 0::SIZE, :]

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

triu_indices = get_triu_offdiag_indices(8 * WINDOW * (INPUT + OUTPUT))
tril_indices = get_tril_offdiag_indices(8 * WINDOW * (INPUT + OUTPUT))
if th.cuda.is_available():
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()


def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def loss(xs, ys, result):

    global counter, lasttime
    counter = counter + 1

    xs = xs.permute(0, 2, 4, 1, 3).contiguous()
    sr, sb, sc, ss, si = tuple(xs.size())
    xs = xs.view(sr * sb, sc, ss, si)

    ys = ys.permute(0, 2, 4, 1, 3).contiguous()
    sr, sb, sc, ss, si = tuple(ys.size())
    ys = ys.view(sr * sb, sc, ss, si)

    im = xs[:, 0:1, :, :]
    ip = xs[:, 2:5, :, :]
    iv = xs[:, 5:8, :, :]

    ms = ys[:, 0:1, :, :]
    ps = ys[:, 2:5, :, :]
    vs = ys[:, 5:8, :, :]

    tm = th.cat([im, ms], dim=3)
    tp = th.cat([ip, ps], dim=3)
    tv = th.cat([iv, vs], dim=3)
    t = th.cat([tm, tp, tv], dim=1)

    gm = result[:, 0:1, :, :]
    gp = result[:, 2:5, :, :]
    gv = result[:, 5:8, :, :]
    g = th.cat([gm, gp, gv], dim=1)

    loss_nll = nll_gaussian(t, g, 5e-5)
    loss_kl = kl_categorical_uniform(model.evolve.prob, INPUT + OUTPUT, 1)

    print('-----------------------------')
    print('dur:', time.time() - lasttime)
    print('per:', th.mean(th.sqrt((tp - gp) * (tp - gp)).data))
    print('ver:', th.mean(th.sqrt((tv - gv) * (tv - gv)).data))
    print('mer:', th.mean(th.sqrt((tm - gm) * (tm - gm)).data))
    print('lss:', th.mean(loss_nll.data))
    print('lkl:', th.mean(loss_kl.data))
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
            imass = im[0, 0, 0, :].data.cpu().numpy()
            gmass = gm[0, 0, 0, :].data.cpu().numpy()
            tmass = ms[0, 0, 0, :].data.cpu().numpy()
        else:
            input = xs.data.numpy()[0, 2:5, :, :]
            truth = ps.data.numpy()[0, :, :, :]
            guess = result.data.numpy()[0, :, :, :]
            imass = im[0, 0, 0, :].data.numpy()
            gmass = gm[0, 0, 0, :].data.numpy()
            tmass = ms[0, 0, 0, :].data.numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        ax.plot(input[0, :, 1], input[1, :, 1], input[2, :, 1], 'o', markersize=msize(imass[1]))
        ax.plot(input[0, :, 2], input[1, :, 2], input[2, :, 2], 'o', markersize=msize(imass[2]))
        ax.plot(input[0, :, 3], input[1, :, 3], input[2, :, 3], 'o', markersize=msize(imass[3]))
        ax.plot(input[0, :, 4], input[1, :, 4], input[2, :, 4], 'o', markersize=msize(imass[4]))
        set_aspect_equal_3d(ax)
        plt.savefig('data/obsv.png')
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        ax.plot(truth[0, :, 0], truth[1, :, 0], truth[2, :, 0], 'ro', markersize=msize(tmass[0]))
        ax.plot(truth[0, :, 1], truth[1, :, 1], truth[2, :, 1], 'bo', markersize=msize(tmass[1]))
        ax.plot(guess[0, :, 0], guess[1, :, 0], guess[2, :, 0], 'r+', markersize=msize(gmass[0]))
        ax.plot(guess[0, :, 1], guess[1, :, 1], guess[2, :, 1], 'b+', markersize=msize(gmass[1]))
        set_aspect_equal_3d(ax)
        plt.savefig('data/pred.png')
        plt.close()

    #return th.sum(pe + ve + me / 50 + he)
    return loss_nll + loss_kl


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
