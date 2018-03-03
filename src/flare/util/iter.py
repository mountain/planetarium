# -*- coding: utf-8 -*-

import numpy as np
import torch as th


def is_indexable(data):
    if isinstance(data, tuple):
        return True
    elif isinstance(data, list):
        return True
    elif isinstance(data, np.ndarray):
        return True
    elif isinstance(data, th._TensorBase):
        return True
    else:
        return False


def deep_size(indexable, depth=0):
    if is_indexable(indexable) and depth < 10:
        return [len(indexable)] + deep_size(indexable[0], depth=depth+1)
    else:
        return []
