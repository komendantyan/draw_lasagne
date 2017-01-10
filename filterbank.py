from __future__ import division

import theano.tensor as T


def filterbank(center, width, logsigma2, shape):
    assert len(shape) == 3

    batch_size, window_size, image_size = shape
    w = T.patternbroadcast(
        T.arange(window_size, dtype='float32').reshape((1, window_size, 1)),
        [True, False, True],
    )
    i = T.patternbroadcast(
        T.arange(image_size, dtype='float32').reshape((1, 1, image_size)),
        [True, True, False],
    )

    center = T.patternbroadcast(
        center.reshape((batch_size, 1, 1)),
        [False, True, True])
    width = T.patternbroadcast(
        width.reshape((batch_size, 1, 1)),
        [False, True, True])
    logsigma2 = T.patternbroadcast(
        logsigma2.reshape((batch_size, 1, 1)),
        [False, True, True])

    mu = (image_size - 1) * \
        ((1 + center) / 2 + width * (w / (window_size - 1) - 0.5))

    F = T.exp(-(mu - i)**2 / (2 * T.exp(logsigma2 / 2)))
    F = F / T.maximum(T.sum(F, 2, keepdims=True), 1e-7)

    return F
