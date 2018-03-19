# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from torch.autograd import Variable
from torch.nn import functional as F


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


# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss(batch, dim, recon_x, x, mu, logvar):
    lss = 0
    for i in range(batch):
        a, b = recon_x[i], x[i].view(-1, dim)
        BCE = F.binary_cross_entropy(a, b, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())
        lss += (BCE + KLD)

    return lss
