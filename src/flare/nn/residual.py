# -*- coding: utf-8 -*-

import torch.nn as nn

from flare.nn.attn import Attention1d, Attention


class ResidualBlock1D(nn.Module):

    def __init__(self, dim):
        super(ResidualBlock1D, self).__init__()

        self.norm1 = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)

        self.relu = nn.ReLU()

        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)

        self.attention = Attention1d(dim, int(dim / 3))

    def forward(self, x):
        b, c, _ = x.size()

        y = self.norm1(x).view(b, c)
        y = self.relu(y)
        y = self.l1(y)
        y = self.norm2(y)
        y = self.relu(y)
        y = self.l2(y).view(b, c, 1)

        a = self.attention(y)

        return x + y * a


class ResidualBlock2D(nn.Module):

    def __init__(self, dim, ksize=3, padding=1):
        super(ResidualBlock2D, self).__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.relu = nn.ReLU()

        self.l1 = nn.Conv2d(dim, dim, ksize, padding=padding)
        self.l2 = nn.Conv2d(dim, dim, ksize, padding=padding)

        self.attention = Attention(dim, int(dim / 3))

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.norm1(x)
        y = self.relu(y)
        y = self.l1(y)
        y = self.norm2(y)
        y = self.relu(y)
        y = self.l2(y)

        a = self.attention(y)

        return x + y * a

