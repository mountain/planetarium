# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_inner, kernel_size=3, padding=1):
        super(VAE, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_inner = dim_inner

        self.fc1 = nn.Conv2d(dim_input, dim_hidden, kernel_size=kernel_size, padding=padding)
        self.fc21 = nn.Conv2d(dim_hidden, dim_inner, kernel_size=kernel_size, padding=padding)
        self.fc22 = nn.Conv2d(dim_hidden, dim_inner, kernel_size=kernel_size, padding=padding)
        self.fc3 = nn.Conv2d(dim_inner, dim_hidden, kernel_size=kernel_size, padding=padding)
        self.fc4 = nn.Conv2d(dim_hidden, dim_input, kernel_size=kernel_size, padding=padding)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

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
        return self.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


mse = nn.MSELoss()
bce = nn.BCELoss()
kld = nn.KLDivLoss()


def vae_loss(recon_x, x, mu, logvar):
    var = logvar.exp()
    dist = th.distributions.Normal(mu, var).sample()
    unit = th.distributions.Normal(mu * 0.0, var.pow(0)).sample()
    return bce(th.sigmoid(recon_x), th.sigmoid(x)) + (kld(dist, unit) + kld(unit, dist)) / 2.0
