from __future__ import division

import theano
import theano.tensor as T

import lasagne.layers as ll


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


class AttentionLayer(ll.MergeLayer):
    def __init__(self, incomings, window_shape, **kwargs):
        if len(window_shape) != 2 or None in window_shape:
            raise ValueError(
                "An AttentionLayer window_shape should be 2-dimensional tuple "
                "of int: (window_height, window_width). Got %r." % window_shape)

        self.window_shape = window_shape
        super(AttentionLayer, self).__init__(incomings, **kwargs)

        if len(self.input_shapes) != 2:
            raise ValueError(
                "An AttentionLayer requires two input layers, got %r." %
                len(self.input_shapes))
        if len(self.input_shapes[0]) != 2 or self.input_shapes[0][1] != 6:
            raise ValueError(
                "An AttentionLayer requires 2-dimensional first input with "
                "size 6 of last one: (center_x, center_y, width_x, width_y, "
                "logsigma2, gamma), Got shape %r." % self.input_shapes[0])
        if len(self.input_shapes[1]) != 4 or None in self.input_shapes[2:]:
            raise ValueError(
                "An AttentionLayer requires 4-dimensional second input with "
                "fixed size of last two dimensions. Got shape %r." %
                self.input_shapes[1])
        if self.input_shapes[0][0] is None or \
                self.input_shapes[1][0] is None or \
                self.input_shapes[0][0] != self.input_shapes[1][0]:
            raise ValueError(
                "An AttentionLayer requires equals, not None batch_size of both "
                "inputs (first diminsion). Got input_shapes %r." %
                self.input_shapes)

    def get_output_shape_for(self, input_shapes):
        batch_size, channel_size, _, _ = self.input_shapes[1]
        window_height, window_width = self.window_shape

        return (batch_size, channel_size, window_height, window_width)

    def get_output_for(self, inputs, **kwargs):
        batch_size, _, image_height, image_width = self.input_shapes[1]
        window_height, window_width = self.window_shape

        center_x, center_y, width_x, width_y, logsigma2, gamma = [
            inputs[0][:, j] for j in xrange(6)
        ]

        Fx = filterbank(
            center_x, width_x, logsigma2,
            shape=(batch_size, window_width, image_width)
        )
        Fy = filterbank(
            center_y, width_y, logsigma2,
            shape=(batch_size, window_height, image_height)
        )

        gamma = T.patternbroadcast(
            T.reshape(gamma, (batch_size, 1, 1, 1)),
            [False, True, True, True]
        )

        _tmp = T.batched_tensordot(inputs[1], Fy, [2, 2])
        _tmp = T.batched_tensordot(_tmp, Fx, [2, 2])
        return _tmp * gamma


class SamplingLayer(ll.MergeLayer):
    def __init__(self, incomings, variation_coeff=1.0,
                 random_stream=theano.sandbox.rng_mrg.MRG_RandomStreams(),
                 **kwargs):
        super(SamplingLayer, self).__init__(incomings, **kwargs)
        self.variation_coeff = variation_coeff
        self.random_stream = random_stream

        self.random_variable = self.add_param(
            self.random_stream.normal(self.input_shapes[0],
                                      std=self.variation_coeff),
            self.input_shapes[0],
            name='random_variable',
            trainable=False
        )

        if len(self.input_shapes) != 2 or \
                self.input_shapes[0] != self.input_shapes[1]:
            raise ValueError(
                "An SamplingLayer requires two equal-shape inputs. Got %r." %
                self.input_shapes)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        mean, logvar2 = inputs
        return mean + T.exp(logvar2 / 2.0) * self.random_variable
