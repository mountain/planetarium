# -*- coding: utf-8 -*-

import numpy as np

from functools import reduce

from flare import pipe as fp


class Sequential(list):
    def __init__(self, seq=None, **kwargs):
        super(Sequential, self).__init__(seq, **kwargs)

    def assertDuplication(self):
        result = True
        for elm in self:
            result = result and (np.all(elm == self[0]))
            if not result:
                break
        return result


class HierarchicalDict(dict):
    def __init__(self, seq=None, **kwargs):
        super(HierarchicalDict, self).__init__(seq, **kwargs)

    def __getitem__(self, subkeys):
        if isinstance(subkeys, str):
            return dict.__getitem__(self, subkeys)
        elif isinstance(subkeys, list) or isinstance(subkeys, tuple):
            val = self
            for subkey in subkeys:
                if isinstance(val, Sequential):
                    val = Sequential([elm[subkey] for elm in val])
                else:
                    val = val[subkey]
            return val
        else:
            raise(Exception('type error'))

    def find_keys(self, key):
        subkeys = key.split('.')
        val = self
        results = []
        try:
            result = []
            for subkey in subkeys:
                if isinstance(val, Sequential):
                    val = val[0][subkey]
                else:
                    val = val[subkey]
                result.append(subkey)
            results.append(result)

            if isinstance(val, HierarchicalDict):
                for k in val.keys():
                    if k[0] != '_':
                        subresults = val.find_keys(k)
                        for subresult in subresults:
                            results.append(result + subresult)
            elif isinstance(val, Sequential):
                for k in val[0].keys():
                    if k[0] != '_':
                        subresults = val[0].find_keys(k)
                        for subresult in subresults:
                            results.append(result + subresult)

        except KeyError:
            results = []

        return results


class ObservableDict(HierarchicalDict):
    def __init__(self, inputs, outputs, layout=[], layout_in=None, layout_out=None, layouts_in=None):
        super(ObservableDict, self).__init__([])
        self.list_input = inputs
        self.list_output = outputs
        self.fields_input = set(inputs)
        self.fields_output = set(outputs)
        self.fields_processed = set()
        self.layout = layout

        if layout_in is None:
            layout_in = layout
        if layout_out is None:
            layout_out = layout

        self.layout_in = layout_in
        self.layout_out = layout_out

        self.layout_input = [1, len(self.fields_input)] + layout_in
        self.layout_output = [1, len(self.fields_output)] + layout_out
        self.data_input = np.zeros(self.layout_input)
        self.data_output = np.zeros(self.layout_output)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)

        subkeys = self.find_keys(key)
        for subkey in subkeys:
            keystr = '.'.join(subkey)
            if keystr in self.list_input:
                pos = self.list_input.index(keystr)
                if len(self.layout_input) == 5:
                    val = np.array(self[subkey])
                    if np.isscalar(val):
                        value = val * np.ones(self.layout)
                    else:
                        value = val.reshape(self.layout_in)
                    self.data_input[0, pos, :, :, :] = np.array(value).reshape(self.layout_in)
                elif len(self.layout_input) == 4:
                    self.data_input[0, pos, :, :] = np.array(self[subkey]).reshape(self.layout_in)
                elif len(self.layout_input) == 3:
                    self.data_input[0, pos, :] = np.array(self[subkey]).reshape(self.layout_in)
                elif len(self.layout_input) == 2:
                    self.data_input[0, pos] = np.array(self[subkey]).reshape(self.layout_in)
                else:
                    raise(Exception('layout error'))

            if keystr in self.list_output:
                pos = self.list_output.index(keystr)
                if len(self.layout_output) == 5:
                    val = np.array(self[subkey])

                    if np.isscalar(val):
                        value = val * np.ones(self.layout)
                    else:
                        value = val.reshape(self.layout_out)

                        if len(val.shape) == len(self.layout_out) and val.shape != self.layout_out:
                            value = val * np.ones(self.layout_out)

                    self.data_output[0, pos, :, :, :] = np.array(value).reshape(self.layout_out)
                elif len(self.layout_output) == 4:
                    self.data_output[0, pos, :, :] = np.array(self[subkey]).reshape(self.layout_out)
                elif len(self.layout_output) == 3:
                    self.data_output[0, pos, :] = np.array(self[subkey]).reshape(self.layout_out)
                elif len(self.layout_output) == 2:
                    self.data_output[0, pos] = np.array(self[subkey]).reshape(self.layout_out)
                else:
                    raise(Exception('layout error'))

    def update(self, another):
        for k in another.keys():
            self[k] = another[k]


class memo(object):
    def __init__(self, f):
        self.fn = f
        self._curr_ = None
        self._last_ = None

    def __call__(self, *args, **kwargs):
        for data in self.fn(*args, **kwargs):
            self._curr_ = data
            yield data

    def _swap_(self, *args, **kwargs):
        self._last_ = self._curr_


def last(f):
    def wrapped(*args, **kwargs):
        for data in f(*args, **kwargs):
            if '_last_' in dir(f):
                data.update({'_last_': f._last_})
                f._swap_()
            yield data
    return wrapped


def attributes(*names):
    def wrapper(f):
        def wrapped(*args, **kwargs):
            for data in f(*args, **kwargs):
                kvs = zip(names, data)
                result = HierarchicalDict({k: v for k, v in kvs})
                if '_last_' in dir(f):
                    result.update({'_last_': f._last_})
                    f._swap()
                yield result
        return wrapped

    return wrapper


def discrete(inputs, outputs, layout=[]):
    def wrapper(f):
        def wrapped(*args, **kwargs):
            for data in f(*args, **kwargs):
                result = ObservableDict(inputs, outputs, layout)
                result.update(data)
                yield result
        return wrapped

    return wrapper


def sequential(inputs, outputs, layout=[], layout_in=None, layout_out=None):
    def wrapper(f):
        def wrapped(*args, **kwargs):
            for data in f(*args, **kwargs):
                result = ObservableDict(inputs, outputs, layout, layout_in, layout_out)
                result.update(data)
                yield result
        return wrapped

    return wrapper


def feature(g, inputs, outputs):
    def wrapper(f):
        def wrapped(*args, **kwargs):
            for result in f(*args, **kwargs):
                kvin = {k: result[k] for k in inputs}
                result.update({k: v for k, v in zip(outputs, g(**kvin))})
                yield result
        return wrapped

    return wrapper


def filter(g, inputs):
    def wrapper(f):
        def wrapped(*args, **kwargs):
            for result in f(*args, **kwargs):
                if isinstance(result, HierarchicalDict):
                    kvin = {k: result[k] for k in inputs}
                    if g(**kvin):
                        yield result
                else:
                    flags = []
                    filtered = []
                    for elm in result:
                        kvin = {k: elm[k] for k in inputs}
                        filtered.append(elm)
                        flags.append(g(**kvin))
                    if reduce(bool.__and__, flags, True):
                        yield filtered
        return wrapped

    return wrapper


def window(window_size=1):
    def wrapper(f):
        def wrapped(*args, **kwargs):
            return fp.roll(f(*args, **kwargs), window_size=window_size)

        return wrapped

    return wrapper


def segment(segment_size=1):
    def wrapper(f):
        def wrapped(*args, **kwargs):
            for seg in fp.batches(f(*args, **kwargs), batch_size=segment_size):
                yield Sequential(seg)

        return wrapped

    return wrapper


def divid(lengths=[1], names=['x']):
    begins = {}
    ends = {}
    pos = 0
    for k, l in zip(names, lengths):
        begins[k] = pos
        ends[k] = pos + l
        pos = ends[k]

    def wrapper(f):
        def wrapped(*args, **kwargs):
            for data in f(*args, **kwargs):
                yield HierarchicalDict({k:Sequential(data[begins[k]:ends[k]]) for k in names})
        return wrapped

    return wrapper


def assertDup(keys):
    keys = keys.split('.')

    def wrapper(f):
        def wrapped(*args, **kwargs):
            for value in f(*args, **kwargs):
                seq = value()[keys]
                if isinstance(seq, Sequential):
                    assert(seq.assertDuplication())
            return f(*args, **kwargs)
        return wrapped

    return wrapper


def assertNoDup(keys):
    keys = keys.split('.')

    def wrapper(f):
        def wrapped(*args, **kwargs):
            for value in f(*args, **kwargs):
                seq = value[keys]
                if isinstance(seq, Sequential):
                    assert(not seq.assertDuplication())
            return f(*args, **kwargs)
        return wrapped

    return wrapper


mapping = feature


def debug():
    def wrapper(f):
        def wrapped(*args, **kwargs):
            for result in f(*args, **kwargs):
                print(result)
                yield result
        return wrapped

    return wrapper


def data(swap=None):
    def wrapper(f):
        def wrapped(*args, **kwargs):
            for result in f(*args, **kwargs):
                if swap is None:
                    yield [(result.data_input, result.data_output)]
                else:
                    idx = range(len(swap))
                    yield [(np.moveaxis(result.data_input, idx, swap), np.moveaxis(result.data_output, idx, swap))]
        return wrapped

    return wrapper


def shuffle(fn, repeat=1):
    def wrapper(f):
        def wrapped(*args, **kwargs):
            results = [result for result in f(*args, **kwargs)]
            for _ in range(repeat):
                for result in results:
                    for pair in result:
                        xs, ys = pair
                        yield [fn(xs, ys)]
        return wrapped

    return wrapper


def batch(repeat=1):
    def wrapper(f):
        def wrapped(*args, **kwargs):
            batch_count = 0
            results = [result for result in f(*args, **kwargs)]
            for result in results:
                if batch_count % repeat == 0:
                    result_batch_xs, result_batch_ys = [], []
                batch_count += 1
                for pair in result:
                    xs, ys = pair
                    shape = xs.shape
                    shape[0] = xs.shape[0] * repeat
                    result_batch_xs.append(xs), result_batch_xs.append(ys)
                    if batch_count % repeat == 0:
                        array_xs = np.array(result_batch_xs, dtype=np.float32).reshape(shape)
                        array_ys = np.array(result_batch_ys, dtype=np.float32).reshape(shape)
                        yield [(array_xs, array_ys)]

        return wrapped

    return wrapper

