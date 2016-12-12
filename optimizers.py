# pylint: disable=invalid-name

import numpy

import theano


def adadelta(loss, wrt, gamma_dtheta, gamma_g, learning_rate, eps=1e-7):
    grad = theano.grad(loss, wrt)

    updates = []

    for theta, g in zip(wrt, grad):
        shape = theta.get_value().shape
        E_dtheta = theano.shared(numpy.zeros(shape, 'float32'))
        E_g = theano.shared(numpy.zeros(shape, 'float32'))

        E_g_next = gamma_g * E_g + (1 - gamma_g) * g ** 2
        dtheta = ((E_dtheta + eps) ** 0.5) * (g / (E_g_next + eps) ** 0.5)
        E_dtheta_next = gamma_dtheta * E_dtheta + (1 - gamma_dtheta) * dtheta ** 2

        updates.extend([
            (E_dtheta, E_dtheta_next),
            (E_g, E_g_next),
            (theta, theta - learning_rate * dtheta),
        ])

    return updates


def gradient_descent(loss, wrt, learning_rate):
    grad = theano.grad(loss, wrt)

    updates = []

    for theta, g in zip(wrt, grad):
        updates.extend([
            (theta, theta - learning_rate * g),
        ])

    return updates
