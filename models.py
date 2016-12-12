# pylint: disable=invalid-name

from __future__ import division

import cPickle
import logging

import numpy

import theano

import cells
from utils import iter_batches


logger = logging.getLogger(__name__)


class BaseModel(cells.BaseCell):
    _batch_fit = None
    _batch_predict = None
    _batch_loss = None

    def predict(self, data):
        logger.debug('call %r.predict', self.name)
        if self._batch_predict is None:
            logger.debug('compiling %r._batch_predict', self.name)
            self._batch_predict = theano.function(
                [self.internal['input']],
                self.internal['output'],
            )

        if data.shape[0] == self.config['batch_size']:
            return self._batch_predict(data)
        else:
            return numpy.concatenate([
                self._batch_predict(batch_data)
                for batch_data in iter_batches(data, self.config['batch_size'],
                                               shuffle=False)
            ])


    def fit(self, data):
        logger.debug('call %r.fit', self.name)
        if self._batch_fit is None:
            logger.debug('compiling %r._batch_fit', self.name)
            self._batch_fit = theano.function(
                [self.internal['input']],
                self.internal['loss'],
                updates=self.config['optimizer'](
                    self.internal['loss'],
                    self.descendants_weighs,
                )
            )

        return numpy.mean([
            self._batch_fit(batch_data)
            for batch_data in iter_batches(data, self.config['batch_size'])
        ])

    def loss(self, data):
        logger.debug('call %r.loss', self.name)
        if self._batch_loss is None:
            logger.debug('compiling %r._batch_loss', self.name)
            self._batch_loss = theano.function(
                [self.internal['input']],
                self.internal['loss'],
            )

        return numpy.mean([
            self._batch_loss(batch_data)
            for batch_data in iter_batches(data, self.config['batch_size'])
        ])

    def save_weights(self, path):
        logger.info('call %r.save_weights(%r)', self.name, path)
        weights_values = [w.get_value() for w in self.descendants_weighs]
        with open(path, 'w') as file_:
            cPickle.dump(weights_values, file_)

    def load_weights(self, path):
        logger.info('call %r.load_weights(%r)', self.name, path)
        with open(path) as file_:
            weights_values = cPickle.load(file_)
        for w, v in zip(self.descendants_weighs, weights_values):
            w.set_value(v)
