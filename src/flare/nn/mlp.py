# -*- coding: utf-8 -*-

import torch.nn as nn


class MLP(nn.Sequential):
    def __init__(self, dims, bsize=1):
        self.batch = bsize
        layers = []
        last = len(dims) - 1
        for i in range(last):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i != last:
                layers.append(nn.ReLU())

        super(MLP, self).__init__(*layers)

    def batch_size_changed(self, new_val, orig_val):
        self.batch = new_val

