# -*- coding: utf-8 -*-


def filter_true(**pwargs):
    return True


def validate_true(*args):
    return True


def identity(x):
    return {'value': x}


def pick(attrs):
    pass


def pick_all(record):
    return [record[key] for key in record.iteritems()]


def pick_value(record):
    return record['value']


def batches(input_handler, retriever=identity, filter=filter_true, validator=validate_true, pick=pick_value, batch_size=1):
    it = iter(input_handler)
    result = []
    for elem in it:
        record = retriever(elem)
        if filter(**record):
            if len(result) < batch_size:
                result.append(pick(record))
            else:
                result = [pick(record)]

            if validator(*result):
                if len(result) == batch_size:
                    yield result


def roll(input_handler, retriever=identity, filter=filter_true, validator=validate_true, pick=pick_value, window_size=1):
    it = iter(input_handler)
    result = []
    for elem in it:
        record = retriever(elem)
        if filter(**record):
            if len(result) < window_size:
                result.append(pick(record))
            else:
                result = result[1:]
                result.append(pick(record))

            if validator(*result):
                if len(result) == window_size:
                    yield result
