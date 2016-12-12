# pylint: disable=invalid-name

from __future__ import division

import abc
from collections import OrderedDict
import logging

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import truncnorm, positive_clip


logger = logging.getLogger(__name__)


class BaseCell(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kargs):
        self.config = kargs

        self.subcells = OrderedDict()
        self.weights = OrderedDict()
        self.internal = OrderedDict()

        logger.debug('%r.__init__', self.name)

    @property
    def name(self):
        if 'name' in self.internal:
            return self.internal['name']
        else:
            return str(type(self))

    @abc.abstractmethod
    def __call__(self):
        pass

    @property
    def descendants(self):
        cells = [self]
        for cell in self.subcells.itervalues():
            cells.extend(cell.descendants)
        return cells

    @property
    def descendants_weighs(self):
        return reduce(
            list.__add__,
            [c.weights.values() for c in self.descendants],
            []
        )


class DenseCell(BaseCell):
    def __init__(self, input_size, output_size, activation):
        config = vars()
        config.pop('self')
        super(DenseCell, self).__init__(**config)

        self.weights['W'] = theano.shared(
            truncnorm((input_size, output_size)) / (input_size + output_size))
        self.weights['b'] = theano.shared(
            truncnorm((output_size,)) / output_size)

    def __call__(self, x):
        y = T.dot(x, self.weights['W']) + self.weights['b']
        if self.config['activation'] is not None:
            return self.config['activation'](y)
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

        g = T.patternbroadcast(
            g.reshape((self.config['batch_size'], 1, 1)),
            [False, True, True])
        delta = T.patternbroadcast(
            delta.reshape((self.config['batch_size'], 1, 1)),
            [False, True, True])
        logsigma2 = T.patternbroadcast(
            logsigma2.reshape((self.config['batch_size'], 1, 1)),
            [False, True, True])

        mu = (image_size - 1) * \
            ((1 + g) / 2 + delta * (w / (window_size - 1) - 0.5))

        F = T.exp(-(mu - i)**2 / (2 * T.exp(logsigma2 / 2)))
        F = F / positive_clip(T.sum(F, 2, keepdims=True))

        return F

    @abc.abstractmethod
    def _apply_filter(self, image, Fx, Fy, gamma):
        pass

    def __init__(self, batch_size, input_size, image_shape, window_shape):
        config = vars()
        config.pop('self')
        super(AttentionCell, self).__init__(**config)

        assert image_shape[0] == window_shape[0]

        self.subcells['params'] = DenseCell(input_size, 5, None)

    def __call__(self, h, image):
        params = self.subcells['params'](h)
        gX, gY, delta, logsigma2, gamma = [params[:, j] for j in xrange(5)]

        Fx = self._filterbank(
            self.config['image_shape'][2], self.config['window_shape'][2],
            gX, delta, logsigma2)
        Fy = self._filterbank(
            self.config['image_shape'][1], self.config['window_shape'][1],
            gY, delta, logsigma2)

        window = self._apply_filter(
            image, Fx, Fy,
            T.patternbroadcast(
                gamma.reshape((self.config['batch_size'], 1, 1, 1)),
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
        config = vars()
        config.pop('self')
        super(LSTMCell, self).__init__(**config)

        self.subcells['fg'] = DenseCell(input_size + output_size, output_size, T.nnet.sigmoid)
        self.subcells['ig'] = DenseCell(input_size + output_size, output_size, T.nnet.sigmoid)
        self.subcells['cv'] = DenseCell(input_size + output_size, output_size, T.tanh)
        self.subcells['og'] = DenseCell(input_size + output_size, output_size, T.nnet.sigmoid)

    def __call__(self, i, h, c):
        input_ = T.concatenate([i, h], 1)
        fg = self.subcells['fg'](input_)
        ig = self.subcells['ig'](input_)
        cv = self.subcells['cv'](input_)
        og = self.subcells['og'](input_)

        c_next = c * fg + cv * ig
        h_next = T.tanh(c_next) * og

        return h_next, c_next


class SamplingCell(BaseCell):
    def __init__(self, batch_size, input_size, output_size, z_epsilon):
        config = vars()
        config.pop('self')
        super(SamplingCell, self).__init__(**config)

        self._epsilon = RandomStreams().normal(
            (batch_size, output_size),
            std=z_epsilon
        )

        self.subcells['z_mean'] = DenseCell(input_size, output_size, None)
        self.subcells['z_logvar2'] = DenseCell(input_size, output_size, None)

    def __call__(self, h):
        mean = self.subcells['z_mean'](h)
        logvar2 = self.subcells['z_logvar2'](h)

        return mean + T.exp(logvar2 / 2) * self._epsilon
