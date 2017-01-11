#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=invalid-name

from __future__ import division

import logging
import cPickle
import os

import numpy
import theano
import theano.tensor as T

import lasagne
from lasagne import (
    layers as ll,
    nonlinearities as ln,
    init as li,
)

import advanced_layers
import utils


BS = 100
CH = 1
IH = 28
IW = 28
WH = 7
WW = 7

HS = 100

ENC_NDIM = 10
ENC_VAR = 1.0

TIME_ROUNDS = 1

NB_EPOCHS = 1000
EARLY_STOPPING_STEPS = 5
EARLY_STOPPING_ACCURACY = 0.0


logging.basicConfig(
    format=(
        "[%(asctime)s %(relativeCreated)8d] "
        "[%(levelname).1s]\t%(message)s"
    ),
    datefmt="%F %T",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def make_model():
    image = ll.InputLayer(
        (BS, CH, IH, IW),
        name='step1.image'
    )

    h_read_init = ll.InputLayer(
        (HS,),
        lasagne.utils.create_param(li.Uniform(), (HS,),
                                   name='step1.tensor.h_read_init'),
        name='step1.h_read_init'
    )
    h_read_init.add_param(h_read_init.input_var, (HS,))

    h_write_init = ll.InputLayer(
        (HS,),
        lasagne.utils.create_param(li.Uniform(), (HS,),
                                   name='step1.tensor.h_write_init'),
        name='step1.h_write_init'
    )
    h_write_init.add_param(h_write_init.input_var, (HS,))

    h_read = ll.ExpressionLayer(
        h_read_init,
        lambda t: T.tile(T.reshape(t, (1, HS)), (BS, 1)),
        (BS, HS),
        name='step1.h_read'
    )

    h_write = ll.ExpressionLayer(
        h_write_init,
        lambda t: T.tile(T.reshape(t, (1, HS)), (BS, 1)),
        (BS, HS),
        name='step1.h_write'
    )

    canvas = ll.InputLayer(
        (BS, CH, IH, IW),
        lasagne.utils.create_param(li.Constant(0.0), (BS, CH, IH, IW),
                                   name='step1.tensor.canvas'),
        name='step1.canvas'
    )

    image_prev = ll.NonlinearityLayer(canvas, ln.sigmoid,
                                      name='step1.image_prev')

    image_error = ll.ElemwiseSumLayer([image, image_prev], coeffs=[1, -1],
                                      name='step1.image_error')
    image_stack = ll.ConcatLayer([image, image_error],
                                 name='step1.image_stack')

    read_params = ll.DenseLayer(h_write, 6, nonlinearity=None,
                                name='step1.read_params')
    read_window = advanced_layers.AttentionLayer(
        [read_params, image_stack], (WH, WW), name='step1.read_window')

    read_flat = ll.FlattenLayer(read_window, name='step1.read_flat')
    read_code = ll.ConcatLayer([read_flat, h_write], name='step1.read_code')

    read_code_sequence = ll.ReshapeLayer(
        read_code, (BS, 1, read_code.output_shape[-1]),
        name='step1.read_code_sequence'
    )

    read_rnn = ll.GRULayer(
        read_code_sequence, HS, only_return_final=True,
        hid_init=h_read,
        name='step1.read_rnn',
    )

    sample_mean = ll.DenseLayer(read_rnn, ENC_NDIM, nonlinearity=None,
                                name='step1.sample_mean')
    sample_logvar2 = ll.DenseLayer(read_rnn, ENC_NDIM, nonlinearity=None,
                                   name='step1.sample_logvar2')
    sample = advanced_layers.SamplingLayer(
        [sample_mean, sample_logvar2], ENC_VAR,
        name='step1.sample'
    )

    write_code = ll.DenseLayer(sample, HS, name='step1.write_code')
    write_code_sequence = ll.ReshapeLayer(
        write_code, (BS, 1, write_code.output_shape[-1]),
        name='step1.write_code_sequence'
    )
    write_rnn = ll.GRULayer(
        write_code_sequence, HS, only_return_final=True,
        hid_init=h_write,
        name='step1.write_rnn',
    )
    write_window_flat = ll.DenseLayer(write_rnn, CH * WH * WW,
                                      name='step1.write_window_flat')
    write_window = ll.ReshapeLayer(write_window_flat, (BS, CH, WH, WW),
                                   name='step1.write_window')

    write_params = ll.DenseLayer(h_write, 6, nonlinearity=None,
                                 name='step1.write_params')
    write_image = advanced_layers.AttentionLayer(
        [write_params, write_window], (IH, IW),
        name='step1.write_image'
    )
    canvas_next = ll.ElemwiseSumLayer([canvas, write_image],
                                      name='step1.canvas_next')

    def rename(name):
        if name is None:
            return None
        step, real_name = name.split('.', 1)
        step = int(step[4:])
        return 'step%d.%s' % (step + 1, real_name)

    for step in xrange(1, TIME_ROUNDS):
        sample_random_variable_next = sample.random_stream.normal(
            sample.input_shapes[0],
            std=sample.variation_coeff,
        )
        sample_random_variable_next.name = 'step%d.sample.random_variable' % \
            (step + 1)

        canvas, canvas_next = (
            canvas_next,
            utils.modified_copy(
                canvas_next,
                modify={
                    h_read: read_rnn,
                    h_write: write_rnn,
                    canvas: canvas_next,
                    sample.random_stream: sample.random_stream,
                    sample.random_variable: sample_random_variable_next,
                },
                rename=rename,
            )
        )

        h_read = read_rnn
        h_write = write_rnn
        read_rnn = utils.layer_by_name(canvas_next, 'step%d.read_rnn' % (step + 1))
        write_rnn = utils.layer_by_name(canvas_next, 'step%d.write_rnn' % (step + 1))
        sample = utils.layer_by_name(canvas_next, 'step%d.sample' % (step + 1))

    output = ll.NonlinearityLayer(canvas_next, ln.sigmoid, name='output')

    return output


if __name__ == '__main__':
    mnist = utils.load_mnist(process=lambda x: (x > 0.8).astype('float32'))
    model = make_model()

    logger.info('visualize model to model.svg')
    utils.visualize_model(model, 'model.svg')

    image = utils.layer_by_name(model, 'step1.image').input_var

    output_layers = [
        utils.layer_by_name(model, name)
        for name in (
            ['output'] +
            ['step%d.sample_mean' % j for j in range(1, TIME_ROUNDS + 1)] +
            ['step%d.sample_logvar2' % j for j in range(1, TIME_ROUNDS + 1)]
        )
    ]

    output_tensors = ll.get_output(output_layers)
    output = output_tensors[0]
    mean = output_tensors[1: 1 + TIME_ROUNDS]
    logvar2 = output_tensors[1 + TIME_ROUNDS: 1 + 2 * TIME_ROUNDS]

    bc_loss = T.mean(lasagne.objectives.binary_crossentropy(output, image))
    kl_loss = T.mean([
        0.5 * T.mean(mean[step] ** 2 + T.exp(logvar2[step]) - logvar2[step] - 1)
        for step in range(TIME_ROUNDS)
    ])
    loss = bc_loss + 0.01 * kl_loss

    params = lasagne.layers.get_all_params(model, trainable=True)
    updates = lasagne.updates.adadelta(loss, params)

    f_train_batch = theano.function(
        [image], loss, updates=updates,
        # mode=theano.compile.MonitorMode(post_func=inspect_node)
    )
    f_test_batch = theano.function(
        [image], loss,
        # mode=theano.compile.MonitorMode(post_func=inspect_node)
    )
    f_predict_batch = theano.function(
        [image], output,
        # mode=theano.compile.MonitorMode(post_func=inspect_node)
    )
    f_raw_loss_batch = theano.function(
        [image], bc_loss
    )

    f_train = utils.batch_wrapper(f_train_batch, BS, aggregate=numpy.mean)
    f_test = utils.batch_wrapper(f_test_batch, BS, aggregate=numpy.mean)
    f_predict = utils.batch_wrapper(f_predict_batch, BS, cut=True, shuffle=False)
    f_raw_loss = utils.batch_wrapper(f_raw_loss_batch, BS, aggregate=numpy.mean)

    logger.info('initial val_loss=%f, raw_loss=%f',
                f_test(mnist[1][0]), f_raw_loss(mnist[1][0]))

    if os.path.exists('init_weights.pkl'):
        with open('init_weights.pkl') as file_:
            weights = cPickle.load(file_)
            ll.set_all_param_values(model, weights, trainable=True)

    try:
        val_loss = []
        for epoch in range(1, NB_EPOCHS + 1):
            train_loss = f_train(mnist[0][0], _verbose=True)
            val_loss.append(f_test(mnist[1][0]))
            raw_loss = f_raw_loss(mnist[1][0])

            logger.info(
                'epoch %3d/%d, train_loss=%f, val_loss=%f, raw_loss=%f',
                epoch, NB_EPOCHS, train_loss, val_loss[-1], raw_loss
            )

            if epoch > EARLY_STOPPING_STEPS:
                recent_best = min(val_loss[-EARLY_STOPPING_STEPS:])
                best = val_loss[-(EARLY_STOPPING_STEPS+1)]
                if recent_best > best + EARLY_STOPPING_ACCURACY:
                    logger.info('early stopping')
                    break

    except KeyboardInterrupt:
        logger.error('interrupted from keyboard')

    with open('weights.pkl', 'w') as file_:
        cPickle.dump(
            ll.get_all_param_values(model, trainable=True),
            file_
        )

    logger.info('finally val_loss=%f, raw_loss=%f',
                f_test(mnist[1][0]), f_raw_loss(mnist[1][0]))

    all_outputs = [
        ll.NonlinearityLayer(
            utils.layer_by_name(model, 'step%d.canvas_next' % j),
            nonlinearity=ln.sigmoid
        )
        for j in xrange(1, TIME_ROUNDS + 1)
    ]

    f_predict_steps_batch = theano.function([image], ll.get_output(all_outputs))
    f_predict_steps = utils.batch_wrapper(f_predict_steps_batch, BS, cut=True,
                                          shuffle=False)

    steps_value = f_predict_steps(mnist[1][0])
    for step, value in enumerate(steps_value, 1):
        numpy.save('./predictions/last/step%.3d.npy' % step, value)
