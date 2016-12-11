#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=invalid-name

from __future__ import division

import logging

import numpy
import theano
import theano.tensor as T

import cells
import models
from utils import (
    truncnorm,
    iter_batches,
)
import optimizers


BATCH_SIZE = 100
Z_NDIM = 3
Z_EPSILON = 1e-2
TIME_ROUNDS = 5
NB_EPOCH = 0


class Model(models.BaseModel):
    def __init__(self, batch_size, image_shape, window_shape, internal_size,
                 z_ndim, z_epsilon, time_rounds):
        config = vars()
        config.pop('self')
        super(Model, self).__init__(**config)

        self.internal['h_enc'] = [theano.shared(truncnorm((batch_size, internal_size)))]
        self.internal['h_dec'] = [theano.shared(truncnorm((batch_size, internal_size)))]
        self.internal['c_enc'] = [theano.shared(truncnorm((batch_size, internal_size)))]
        self.internal['c_dec'] = [theano.shared(truncnorm((batch_size, internal_size)))]
        self.internal['canvas'] = [
            theano.shared(numpy.zeros((batch_size,) + image_shape, 'float32'))
        ]

        self.subcells['read'] = cells.ReadCell(batch_size, internal_size, image_shape, window_shape)
        self.subcells['lstm_enc'] = cells.LSTMCell(
            2 * numpy.prod(window_shape) + internal_size,
            internal_size
        )
        self.subcells['sampling'] = cells.SamplingCell(batch_size, internal_size, z_ndim, z_epsilon)
        self.subcells['lstm_dec'] = cells.LSTMCell(z_ndim, internal_size)
        self.subcells['post_dec'] = cells.DenseCell(internal_size, numpy.prod(window_shape), None)
        self.subcells['write'] = cells.WriteCell(batch_size, internal_size, image_shape, window_shape)

    def __call__(self):
        x = T.tensor4('x')

        for t in xrange(self.config['time_rounds']):
            x_err = x - T.nnet.sigmoid(self.internal['canvas'][-1])
            r = self.subcells['read'](self.internal['h_dec'][-1], T.concatenate([x, x_err], 1))
            h_enc_next, c_enc_next = self.subcells['lstm_enc'](
                T.concatenate([self.internal['h_dec'][-1], T.flatten(r, 2)], 1),
                self.internal['h_enc'][-1],
                self.internal['c_enc'][-1],
            )
            self.internal['h_enc'].append(h_enc_next)
            self.internal['c_enc'].append(c_enc_next)

            z = self.subcells['sampling'](self.internal['h_enc'][-1])

            h_dec_next, c_dec_next = self.subcells['lstm_dec'](
                z,
                self.internal['h_dec'][-1],
                self.internal['c_dec'][-1]
            )
            self.internal['h_dec'].append(h_dec_next)
            self.internal['c_dec'].append(c_dec_next)

            post_dec = self.subcells['post_dec'](self.internal['h_dec'][-1]).reshape(
                (self.config['batch_size'],) + self.config['window_shape'])
            canvas_next = self.internal['canvas'][-1] + \
                self.subcells['write'](self.internal['h_dec'][-1], post_dec)
            self.internal['canvas'].append(canvas_next)

        output = T.nnet.sigmoid(self.internal['canvas'][-1])
        loss = T.mean(T.nnet.binary_crossentropy(output, x)) * self.config['batch_size']

        self.internal['input'] = x
        self.internal['output'] = output
        self.internal['loss'] = loss

        self.predict = theano.function([x], output)
        self.fit = theano.function(
            [x],
            loss,
            updates=optimizers.adadelta(
                loss,
                self.descendants_weighs,
                0.1, 0.1, 1.0
            )
        )
        self.loss = theano.function([x], loss)


if __name__ == '__main__':
    logging.basicConfig(
        format=(
            "[%(asctime)s %(relativeCreated)8d] "
            "[%(levelname).1s]\t%(message)s"
        ),
        datefmt="%F %T",
        level=logging.INFO
    )
    logging.info('logging initialized')

    model = Model(BATCH_SIZE, (1, 28, 28), (1, 7, 7),
                  100, Z_NDIM, Z_EPSILON, TIME_ROUNDS)
    logging.info('model created')
    model()
    logging.info('model built')

    from keras.datasets import mnist
    mnist = mnist.load_data()
    train_data = (mnist[0][0] / 255.0).astype('float32').reshape((-1, 1, 28, 28))
    test_data = (mnist[1][0] / 255.0).astype('float32').reshape((-1, 1, 28, 28))
    logging.info('dataset loaded')

    # model.load_weights('weights.pkl')
    # logging.info('weights loaded')

    logging.info('Starting training')
    try:
        for epoch in range(1, NB_EPOCH + 1):
            logging.info('epoch %d/%d started', epoch, NB_EPOCH)

            for j, batch_data in enumerate(iter_batches(train_data, BATCH_SIZE)):
                train_loss = model.fit(batch_data)
                if j % 10 == 0:
                    logging.debug('epoch %4d/%d, objects %5d/60000, train_loss=%f',
                                 epoch, NB_EPOCH, j * BATCH_SIZE, train_loss / BATCH_SIZE)

            val_loss = 0.0
            for batch_data in iter_batches(test_data, BATCH_SIZE, shuffle=False):
                val_loss += model.loss(batch_data)
            logging.info('epoch %3d/%d, val_loss=%f',
                         epoch, NB_EPOCH, val_loss / test_data.shape[0])

    except KeyboardInterrupt:
        logging.error('training was interrupted from keyboard')

    logging.info('saving weights')
    model.save_weights("weights.pkl")
    logging.info('weights saved')

    val_loss = 0.0
    for batch_data in iter_batches(test_data, BATCH_SIZE, shuffle=False):
        val_loss += model.loss(batch_data)
    logging.info('finally val_loss=%f',
                 val_loss / test_data.shape[0])

    logging.info('completed')
