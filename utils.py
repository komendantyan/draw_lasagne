import numpy
import scipy.stats

import theano.tensor as T


def truncnorm(shape):
    rvs = scipy.stats.truncnorm(-3.0, 3.0).rvs
    return rvs(shape).astype('float32')


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
