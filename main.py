#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
import abc
import cPickle
import datetime

import numpy
import scipy.stats
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


BATCH_SIZE = 100
Z_EPSILON = 1e-3
LEARNING_RATE = 1e-1


def truncnorm(shape):
    rvs = scipy.stats.truncnorm(-3.0, 3.0).rvs
    return rvs(shape).astype('float32')


class BaseCell(object):
    __metaclass__ = abc.ABCMeta

    childs = []
    weights = []

    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self):
        pass

    @property
    def descendants(self):
        cells = [self]
        for cell in self.childs:
            cells.extend(cell.descendants)
        return cells


class DenseCell(BaseCell):
    def __init__(self, input_size, output_size, activation):
        self._W = theano.shared(truncnorm(
            (BATCH_SIZE, input_size, output_size)) / (input_size + output_size))
        self._b = theano.shared(truncnorm(
            (BATCH_SIZE, output_size)) / output_size)
        self.weights = [self._W, self._b]

        self._activation = activation

        self.childs = []

    def __call__(self, x):
        y = T.batched_dot(x, self._W) + self._b
        if self._activation is not None:
            return self._activation(y)
        else:
            return y


class AttentionCell(BaseCell):
    def _filterbank(self, image_size, window_size, g, delta, logsigma2):
        w = T.patternbroadcast(
            T.arange(window_size, dtype='float32').reshape((1, window_size, 1)),
            [True, False, True],
        )
        i = T.patternbroadcast(
            T.arange(image_size, dtype='float32').reshape((1, 1, image_size)),
            [True, True, False],
        )

        g = T.patternbroadcast(g.reshape((BATCH_SIZE, 1, 1)),
                               [False, True, True])
        delta = T.patternbroadcast(delta.reshape((BATCH_SIZE, 1, 1)),
                                   [False, True, True])
        logsigma2 = T.patternbroadcast(logsigma2.reshape((BATCH_SIZE, 1, 1)),
                                       [False, True, True])

        mu = (image_size - 1) * \
            ((1 + g) / 2 + delta * (w / (window_size - 1) - 0.5))

        return T.maximum(T.exp(-(mu - i)**2 / (2 * T.exp(logsigma2 / 2))), 1e-7)

    @abc.abstractmethod
    def _apply_filter(self, image, Fx, Fy, gamma):
        pass

    def __init__(self, input_size, image_shape, window_shape):
        self._params_cell = DenseCell(input_size, 5, None)
        self.childs = [self._params_cell]
        self.weights = []

        self._image_shape = image_shape
        self._window_shape = window_shape
        assert image_shape[0] == window_shape[0]

    def __call__(self, h, image):
        params = self._params_cell(h)
        gX, gY, delta, logsigma2, gamma = [params[:, j] for j in xrange(5)]

        Fx = self._filterbank(
            self._image_shape[2], self._window_shape[2], gX, delta, logsigma2)
        Fy = self._filterbank(
            self._image_shape[1], self._window_shape[1], gY, delta, logsigma2)

        window = self._apply_filter(
            image, Fx, Fy,
            T.patternbroadcast(
                gamma.reshape((BATCH_SIZE, 1, 1, 1)),
                [False, True, True, True]
            )
        )

        return window


class ReadCell(AttentionCell):
    def _apply_filter(self, image, Fx, Fy, gamma):
        tmp = T.batched_tensordot(image, Fy, [2, 2])
        window = T.batched_tensordot(tmp, Fx, [2, 2])
        return gamma * window


class WriteCell(AttentionCell):
    def _apply_filter(self, image, Fx, Fy, gamma):
        tmp = T.batched_tensordot(image, Fy, [2, 1])
        window = T.batched_tensordot(tmp, Fx, [2, 1])
        return gamma * window


class LSTMCell(BaseCell):
    def __init__(self, input_size, output_size):
        self._fg_cell = DenseCell(input_size + output_size, output_size, T.nnet.sigmoid)
        self._ig_cell = DenseCell(input_size + output_size, output_size, T.nnet.sigmoid)
        self._cv_cell = DenseCell(input_size + output_size, output_size, T.tanh)
        self._og_cell = DenseCell(input_size + output_size, output_size, T.nnet.sigmoid)

        self.childs = [self._fg_cell, self._ig_cell,
                       self._cv_cell, self._og_cell]
        self.weights = []

    def __call__(self, i, h, c):
        input = T.concatenate([i, h], 1)
        fg = self._fg_cell(input)
        ig = self._ig_cell(input)
        cv = self._cv_cell(input)
        og = self._og_cell(input)

        c_next = c * fg + cv * ig
        h_next = T.tanh(c_next) * og

        return h_next, c_next


class SamplingCell(BaseCell):
    def __init__(self, input_size, output_size, z_epsilon):
        self._epsilon = RandomStreams().normal(
            (BATCH_SIZE, output_size),
            std=z_epsilon
        )

        self._z_mean_cell = DenseCell(input_size, output_size, None)
        self._z_logvar2_cell = DenseCell(input_size, output_size, None)

        self.weights = []
        self.childs = [self._z_mean_cell, self._z_logvar2_cell]

    def __call__(self, h):
        mean = self._z_mean_cell(h)
        logvar2 = self._z_logvar2_cell(h)

        return mean + T.exp(logvar2 / 2) * self._epsilon


class Model(BaseCell):
    def __init__(self, image_shape, window_shape, internal_size, z_ndim,
                 time_rounds):
        self._image_shape = image_shape
        self._window_shape = window_shape
        self._internal_size = internal_size
        self._z_ndim = z_ndim
        self._time_rounds = time_rounds

        self._h_enc = [theano.shared(truncnorm((BATCH_SIZE, internal_size)))]
        self._h_dec = [theano.shared(truncnorm((BATCH_SIZE, internal_size)))]
        self._c_enc = [theano.shared(truncnorm((BATCH_SIZE, internal_size)))]
        self._c_dec = [theano.shared(truncnorm((BATCH_SIZE, internal_size)))]
        self._canvas = [
            theano.shared(numpy.zeros((BATCH_SIZE,) + image_shape, 'float32'))
        ]

        self._read_cell = ReadCell(internal_size, image_shape, window_shape)

        self._lstm_enc_cell = LSTMCell(
            2 * numpy.prod(window_shape) + internal_size,
            internal_size
        )

        self._sampling_cell = SamplingCell(internal_size, z_ndim, Z_EPSILON)

        self._lstm_dec_cell = LSTMCell(z_ndim, internal_size)

        self._post_dec_cell = DenseCell(internal_size, numpy.prod(window_shape), None)
        self._write_cell = WriteCell(internal_size, image_shape, window_shape)

        self.childs = [self._read_cell, self._lstm_enc_cell, self._sampling_cell,
                       self._lstm_dec_cell, self._post_dec_cell, self._write_cell]
        self.weights = []

    def __call__(self):
        self._x = T.tensor4('x')

        for t in xrange(self._time_rounds):
            self._x_err = self._x - T.nnet.sigmoid(self._canvas[-1])
            r = self._read_cell(self._h_dec[-1], T.concatenate([self._x, self._x_err], 1))
            h_enc_next, c_enc_next = self._lstm_enc_cell(
                T.concatenate([self._h_dec[-1], T.flatten(r, 2)], 1),
                self._h_enc[-1],
                self._c_enc[-1],
            )
            self._h_enc.append(h_enc_next)
            self._c_enc.append(c_enc_next)

            z = self._sampling_cell(self._h_enc[-1])

            h_dec_next, c_dec_next = self._lstm_dec_cell(
                z, self._h_dec[-1], self._c_dec[-1])
            self._h_dec.append(h_dec_next)
            self._c_dec.append(c_dec_next)

            post_dec = self._post_dec_cell(self._h_dec[-1]).reshape(
                (BATCH_SIZE,) + self._window_shape)
            canvas_next = self._canvas[-1] + self._write_cell(self._h_dec[-1], post_dec)
            self._canvas.append(canvas_next)

        self._output = T.nnet.sigmoid(self._canvas[-1])

        self.loss = self._make_loss()
        self.fit = self._make_fit_function()
        self.predict = self._make_predict_function()

    def _make_loss(self):
        crop = lambda x: T.maximum(x, 1e-7)

        x = self._x
        x_pred = self._output

        return - T.mean(self._x * T.log(crop(self._output)) + \
                        (1 - self._x) * T.log(crop(1 - self._output)))

    def _make_predict_function(self):
        return theano.function([self._x], self._output)

    def _make_fit_function(self):
        wrt = reduce(list.__add__, [c.weights for c in self.descendants])
        grad = theano.grad(self.loss, wrt)

        updates = [(w, w - LEARNING_RATE * g) for (w, g) in zip(wrt, grad)]

        return theano.function([self._x], updates=updates)

    def save_weights(self, path):
        weights = reduce(list.__add__, [c.weights for c in self.descendants])
        weights_values = [w.get_value() for w in weights]
        with open(path, 'w') as file_:
            cPickle.dump(weights_values, file_)

    def load_weights(self, path):
        weights = reduce(list.__add__, [c.weights for c in self.descendants])
        with open(path) as file_:
            weights_values = cPickle.load(file_)
        for w, v in zip(weights, weights_values):
            w.set_value(v)


def log(str, *args):
    print ("[%s] " + str) % ((datetime.datetime.now().isoformat(' '),) + args)


if __name__ == '__main__':
    model = Model((1, 28, 28), (1, 7, 7), 100, 3, 5)
    model()

    from keras.datasets import mnist
    mnist = mnist.load_data()
    train_data = (mnist[0][0] / 255.0).astype('float32').reshape((-1, 1, 28, 28))
    test_data = (mnist[1][0] / 255.0).astype('float32').reshape((-1, 1, 28, 28))

    loss = theano.function([model._x], model.loss)

    for epoch in range(100):
        log("epoch %d", epoch)

        sample = numpy.arange(60000)
        numpy.random.shuffle(sample)
        data = train_data[sample]
        for j in range(0, 60000, BATCH_SIZE):
            batch_data = data[j: j+BATCH_SIZE]
            model.fit(batch_data)
            if j % 10000 == 0:
                log("j %d", j)
                log("loss100 %f", loss(test_data[:BATCH_SIZE]))


    model.save_weights("weights.pkl")
