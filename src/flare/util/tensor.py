# -*- coding: utf-8 -*-

import torch as th


def squeeze(tensor, n=2, m=-1):
    dims = len(tensor.size())
    if m == -1:
        m = dims
    if dims > n:
        for i in range(n, m):
            tensor = th.squeeze(tensor, dim=n)

    return tensor