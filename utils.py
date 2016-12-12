from __future__ import division

import numpy

import theano.tensor as T


def glorot_uniform(shape):
    if len(shape) == 1:
        fan_in = fan_out = shape[0] ** 0.5
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        raise NotImplementedError('glorot_uniform, len(shape) > 2')

    s = (6 / (fan_in + fan_out)) ** 0.5
    return numpy.random.uniform(-s, s, shape).astype('float32')


def positive_clip(x, eps=1e-7):
    return T.maximum(x, eps)


def iter_batches(data, batch_size, shuffle=True):
    samples_count = data.shape[0]

    assert samples_count % batch_size == 0

    if shuffle:
        indices = numpy.arange(samples_count)
        numpy.random.shuffle(indices)
        data = data[indices]

    for j in xrange(0, samples_count, batch_size):
        yield data[j: j + batch_size]
