# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')


import sys
import time
import numpy as np

from physics import ode, hamilton, nbody
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

from flare.learner import StandardLearner, cast
from flare.nn.nri import MLPEncoder, MLPDecoder, get_tril_offdiag_indices, get_triu_offdiag_indices
from flare.nn.nri import gumbel_softmax, my_softmax, encode_onehot, nll_gaussian, kl_categorical_uniform
from flare.dataset.decorators import attributes, segment, divid, sequential, shuffle, data, rebatch


epsilon = 0.00000001

MSCALE = 10
VSCALE = 100.0
SCALE = 10.0

BATCH = 5
REPEAT = 12
SIZE = 8
BODYCOUNT = 3

lr = 1e-5

mass = None
sun = None


lasttime = time.time()


def msize(x):
    return int(1 + MSCALE * x / 10)


def shufflefn(xs, ys):
    # permute on different input
    perm = np.arange(xs.shape[-2])
    np.random.shuffle(perm)
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


def generator(sz, yrs, btch):
    global lasttime
    lasttime = time.time()

    global mass

    mass = xp.random.rand(btch, sz) * MSCALE
    x = xp.random.rand(btch, sz, 3) * SCALE
    v = xp.zeros([btch, sz, 3])

    center = (np.sum(mass.reshape([btch, sz, 1]) * x, axis=1) / np.sum(mass, axis=1).reshape([btch, 1])).reshape([btch, 1, 3])
    x = x - center

    solver = ode.verlet(nbody.acceleration_of(au, mass))
    h = hamilton.hamiltonian(au, mass)
    lastha = h(x, v, limit=sz)

    t = 0
    lastyear = 0
    for epoch in range(yrs * 144):
        t, x, v = solver(t, x, v, 0.1)
        center = (np.sum(mass.reshape([btch, sz, 1]) * x, axis=1) / np.sum(mass, axis=1).reshape([btch, 1])).reshape([btch, 1, 3])
        x = x - center

        year = int(t)
        if 10 * int(year / 10) == lastyear + 10:
            lastyear = year
            rtp = x / SCALE
            rtv = v
            ha = h(x, v, limit=sz)
            dha = ha - lastha

            inputm = mass[:, :].reshape([btch, sz, 1]) / MSCALE
            inputp = xp.tanh(rtp.reshape([btch, sz, 3]))
            inputv = xp.tanh(rtv.reshape([btch, sz, 3]) * VSCALE)
            inputdh = xp.tanh(dha.reshape([btch, sz, 1]) / au.G * SCALE)
            input = np.concatenate([inputm, inputdh, inputp, inputv], axis=2).reshape([btch, sz * 8])
            yield year, input
            lastha = ha

            #print('-----------------------------')
            #print('m:', np.max(inputm), np.min(inputm))
            #print('p:', np.max(inputp), np.min(inputp))
            #print('v:', np.max(inputv), np.min(inputv))
            #print('h:', np.max(inputdh), np.min(inputdh))
            #print('-----------------------------')
            #sys.stdout.flush()

    print('gen:', time.time() - lasttime)
    sys.stdout.flush()
    lasttime = time.time()


@rebatch(repeat=REPEAT)
@shuffle(shufflefn, repeat=REPEAT)
@data()
@sequential(['xs.d'], ['ys.d'], layout_in=[SIZE, BATCH, BODYCOUNT, 8], layout_out=[2 * SIZE, BATCH, BODYCOUNT, 8])
@divid(lengths=[SIZE, 2 * SIZE], names=['xs', 'ys'])
@segment(segment_size = 3 * SIZE)
@attributes('yr', 'd')
def dataset():
    return generator(BODYCOUNT, 3 * SIZE, BATCH)


class Evolve(nn.Module):
    def __init__(self):
        super(Evolve, self).__init__()
        w = SIZE
        c = 8
        d = c * w

        off_diag = np.ones([BODYCOUNT, BODYCOUNT]) - np.eye(BODYCOUNT)
        self.rel_rec = Variable(cast(np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)))
        self.rel_send = Variable(cast(np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)))

        self.encoder = MLPEncoder(d, 2048, 1)
        self.decoder = MLPDecoder(c, 1, 2048, 2048, 2048)

    def forward(self, x, w=SIZE):
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


class Model(nn.Module):
    def __init__(self, bsize=1):
        super(Model, self).__init__()
        self.batch = bsize
        self.evolve = Evolve()

    def forward(self, x):
        x = x.permute(0, 2, 4, 1, 3).contiguous()
        sr, sb, sc, ss, si = tuple(x.size())
        state = x.view(sr * sb, sc, ss, si)
        result = Variable(cast(np.zeros([sr * sb, 8, 2 * SIZE, BODYCOUNT])))
        for i in range(3):
            print('-----------------------------')
            print('idx:', i)
            sys.stdout.flush()

            state = self.evolve(state, w=SIZE)
            result[:, :, i*SIZE:(i+1)*SIZE, :] = state[:, :, :, :]

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

triu_indices = get_triu_offdiag_indices(8 * SIZE * BODYCOUNT)
tril_indices = get_tril_offdiag_indices(8 * SIZE * BODYCOUNT)
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

    ms = ys[:, 0:1, :, :]
    ps = ys[:, 2:5, :, :]
    vs = ys[:, 5:8, :, :]

    gm = result[:, 0:1, :, :]
    gp = result[:, 2:5, :, :]
    gv = result[:, 5:8, :, :]

    loss_nll = nll_gaussian(ys, result, 5e-5)
    loss_kl = kl_categorical_uniform(model.evolve.prob, BODYCOUNT, 1)

    print('-----------------------------')
    print('dur:', time.time() - lasttime)
    print('per:', th.mean(th.sqrt((ps - gp) * (ps - gp)).data))
    print('ver:', th.mean(th.sqrt((vs - gv) * (vs - gv)).data))
    print('mer:', th.mean(th.sqrt((ms - gm) * (ms - gm)).data))
    print('lss:', th.mean(loss_nll.data))
    print('lkl:', th.mean(loss_kl.data))
    print('-----------------------------')
    sys.stdout.flush()
    lasttime = time.time()

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * (0.1 ** (counter // 100))

    if counter % 1 == 0:
        if th.cuda.is_available():
            truth = ps.data.cpu().numpy()[0, :, :, :]
            guess = result.data.cpu().numpy()[0, :, :, :]
            gmass = gm[0, 0, 0, :].data.cpu().numpy()
            tmass = ms[0, 0, 0, :].data.cpu().numpy()
        else:
            truth = ps.data.numpy()[0, :, :, :]
            guess = result.data.numpy()[0, :, :, :]
            gmass = gm[0, 0, 0, :].data.numpy()
            tmass = ms[0, 0, 0, :].data.numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        ax.plot(truth[0, :, 0], truth[1, :, 0], truth[2, :, 0], 'ro', markersize=msize(tmass[0]))
        ax.plot(truth[0, :, 1], truth[1, :, 1], truth[2, :, 1], 'go', markersize=msize(tmass[1]))
        ax.plot(truth[0, :, 2], truth[1, :, 2], truth[2, :, 2], 'bo', markersize=msize(tmass[2]))
        ax.plot(guess[0, :, 0], guess[1, :, 0], guess[2, :, 0], 'r+', markersize=msize(gmass[0]))
        ax.plot(guess[0, :, 1], guess[1, :, 1], guess[2, :, 1], 'g+', markersize=msize(gmass[1]))
        ax.plot(guess[0, :, 2], guess[1, :, 2], guess[2, :, 2], 'b+', markersize=msize(gmass[2]))
        set_aspect_equal_3d(ax)
        plt.savefig('data/3body.png')
        plt.close()

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
