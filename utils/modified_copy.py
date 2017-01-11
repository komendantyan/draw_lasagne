#!/usr/bin/env python
# -*- coding: utf-8 -*-

# thanks for inspiration to
# https://github.com/yandexdataschool/AgentNet/blob/master/agentnet/utils/clone.py

import copy
import collections

import lasagne.layers as ll


__all__ = ['modified_copy']


def _check_assertations(locals_):
    layers = locals_['layers']
    modify = locals_['memo']
    rename = locals_['rename']

    def is_list_of_layers(var):
        return (
            isinstance(var, list) and
            all(isinstance(l, ll.Layer) for l in var)
        )

    if not (isinstance(layers, ll.Layer) or
            is_list_of_layers(layers) and layers):
        raise ValueError('layers should be isinstance of Layer class or '
                         'non-empty list of Layers isinstances. Got %r' %
                         layers)

    if not (modify is None or isinstance(modify, dict)):
        raise ValueError('modify should be None or dict. Got %r' % modify)

    if not (rename is None or isinstance(rename, collections.Callable)):
        raise ValueError('rename should be None or Callable. Got %r' % rename)


def modified_copy(layers, modify=None, rename=None):
    # _check_assertations(locals())

    layers_of_original = ll.get_all_layers(layers)
    input_layers = [l for l in layers_of_original
                    if isinstance(l, ll.InputLayer)]

    params_of_original = ll.get_all_params(layers)

    memo = {id(x): x for x in input_layers + params_of_original}
    memo.update({id(k): v for k, v in modify.iteritems()})

    modified_layers = copy.deepcopy(layers, memo)

    if rename is not None:
        layers_of_modified = ll.get_all_layers(modified_layers)
        for layer in layers_of_modified:
            if layer not in layers_of_original:
                layer.name = rename(layer.name)

        params_of_modified = ll.get_all_params(modified_layers)
        for param in params_of_modified:
            if param not in params_of_original:
                param.name = rename(param.name)

    return modified_layers
