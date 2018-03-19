# -*- coding: utf-8 -*-

import torch.nn as nn


class Attention1d(nn.Module):

    def __init__(self, dim, mdim):
        super(Attention1d, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
                nn.Linear(dim, mdim),
                nn.ReLU(inplace=True),
                nn.Linear(mdim, dim),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()

        a = self.avg_pool(x).view(b, c)
        a = self.fc(a).view(b, c, 1)

        return a


class Attention(nn.Module):

    def __init__(self, dim, mdim):
        super(Attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
                nn.Linear(dim, mdim),
                nn.ReLU(inplace=True),
                nn.Linear(mdim, dim),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        a = self.avg_pool(x).view(b, c)
        a = self.fc(a).view(b, c, 1, 1)

        return a

