from __future__ import division


__all__ = ['load_mnist', 'load_cifar10']


def load_mnist(shape=None, flat=False):
    if shape is None:
        shape = (28**2,) if flat else (1, 28, 28)
    batch_shape = (-1,) + shape

    from keras.datasets import mnist
    mnist = mnist.load_data()
    train_data = (mnist[0][0] / 255.0).astype('float32').reshape(batch_shape)
    test_data = (mnist[1][0] / 255.0).astype('float32').reshape(batch_shape)

    return ((train_data, mnist[0][1]), (test_data, mnist[1][1]))


def load_cifar10(shape=None, flat=False):
    if shape is None:
        shape = (3 * 32**2,) if flat else (3, 32, 32)
    batch_shape = (-1,) + shape

    from keras.datasets import cifar10
    cifar10 = cifar10.load_data()
    train_data = (cifar10[0][0] / 255.0).astype('float32').reshape(batch_shape)
    test_data = (cifar10[1][0] / 255.0).astype('float32').reshape(batch_shape)

    return ((train_data, cifar10[0][1]), (test_data, cifar10[1][1]))
