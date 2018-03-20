# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, d1, d2, d3, bsize=1):
        super(VAE, self).__init__()
        self.batch = bsize

        self.dim = d1
        self.fc1 = nn.Linear(d1, d2)
        self.fc21 = nn.Linear(d2, d3)
        self.fc22 = nn.Linear(d2, d3)
        self.fc3 = nn.Linear(d3, d2)
        self.fc4 = nn.Linear(d2, d1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def batch_size_changed(self, new_val, orig_val):
        self.batch = new_val

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


bce = nn.BCELoss()
mse = nn.MSELoss()


def vae_loss(batch, dim, recon_x, x, mu, logvar):
    kld = - 0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse(recon_x, x) + kld
