# -*- coding: utf-8 -*-

import random

from flare.dataset.decorators import attributes, data


def normalize(val):
    return val > 0.5


def xor(a, b):
    if a < 0.5 <= b:
        return 1.0
    elif a >= 0.5 > b:
        return 1.0
    else:
        return 0.0


@attributes('x1', 'x2', 'x3')
def generator(gate, n):
    for _ in range(n):
        a = random.random()
        b = random.random()
        c = gate(a, b)

        yield a, b, c