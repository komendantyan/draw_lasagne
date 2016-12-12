#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=invalid-name

import logging
import functools

import numpy

import theano
import theano.tensor as T

import cells
import models
import optimizers


BATCH_SIZE = 100
NB_EPOCH = 500
EARLY_STOPPING_STEPS = 3
EARLY_STOPPING_ACCURACY = 1e-4

logger = logging.getLogger(__name__)


class Model(models.BaseModel):
    def __init__(self, batch_size, optimizer):
        config = vars()
        config.pop('self')
        super(Model, self).__init__(**config)

        self.subcells['bn_1'] = cells.BatchNormalizationCell((self.config['batch_size'], 28**2))
        self.subcells['dense_1'] = cells.DenseCell(28**2, 10, T.nnet.relu)
        self.subcells['bn_2'] = cells.BatchNormalizationCell((self.config['batch_size'], 10))
        self.subcells['dense_2'] = cells.DenseCell(10, 28**2, T.nnet.sigmoid)

    def __call__(self):
        x = T.matrix('x')
        x_norm = self.subcells['bn_1'](x)
        encoded = self.subcells['dense_1'](x_norm)
        encoded_norm = self.subcells['bn_2'](encoded)
        decoded = self.subcells['dense_2'](encoded_norm)

        loss = T.mean(T.nnet.binary_crossentropy(decoded, x))

        self.internal['input'] = x
        self.internal['encoded'] = encoded
        self.internal['output'] = decoded
        self.internal['loss'] = loss


if __name__ == '__main__':
    logging.basicConfig(
        format=(
            "[%(asctime)s %(relativeCreated)8d] "
            "[%(levelname).1s]\t%(message)s"
        ),
        datefmt="%F %T",
        level=logging.INFO
    )

    from keras.datasets import mnist
    mnist = mnist.load_data()
    train_data = (mnist[0][0] / 255.0).astype('float32').reshape((-1, 28**2))
    test_data = (mnist[1][0] / 255.0).astype('float32').reshape((-1, 28**2))

    model = Model(
        BATCH_SIZE,
        optimizer=functools.partial(
            optimizers.adadelta,
            gamma_g=0.95, gamma_dtheta=0.95, learning_rate=1.0
        )
    )
    model()

    val_loss = [model.loss(test_data)]
    logger.info('initial val_loss=%f', val_loss[-1])

    try:
        stagnation_steps = 0
        for epoch in range(1, NB_EPOCH + 1):
            model.fit(train_data)
            val_loss.append(model.loss(test_data))
            delta_val_loss = val_loss[-1] - val_loss[-2]
            if - delta_val_loss <= EARLY_STOPPING_ACCURACY:
                stagnation_steps += 1
            else:
                stagnation_steps = 0

            logger.info('epoch %3d/%d, val_loss=%f, delta_val_loss=%.2e, stagnation_steps=%d/%d',
                        epoch, NB_EPOCH, val_loss[-1], delta_val_loss,
                        stagnation_steps, EARLY_STOPPING_STEPS)
            if stagnation_steps == EARLY_STOPPING_STEPS:
                logger.info('early stopping')
                break

    except KeyboardInterrupt:
        logger.error('training was interrupted from keyboard')

    model.save_weights("weights.pkl")

    logger.info('finally val_loss=%f', model.loss(test_data))

    predict_data = model.predict(test_data)
    numpy.save('./predict_data.npy', predict_data)
    numpy.save('./test_data.npy', test_data)
