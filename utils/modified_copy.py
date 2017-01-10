#!/usr/bin/env python
# -*- coding: utf-8 -*-

# thanks to https://github.com/yandexdataschool/AgentNet/blob/master/agentnet/utils/clone.copyright

import copy
import collections  # to check isinstance(rename_rule, Callable)

import lasagne.layers as ll


__all__ = ['modified_copy']


def _check_assertations(locals_):
    stop_layers = locals_['stop_layers']
    start_layers = locals_['start_layers']
    modify = locals_['modify']
    rename_rule = locals_['rename_rule']

    def is_list_of_layers(var):
        return isinstance(var, list) and \
            all(isinstance(l, ll.Layer) for l in var)

    if not (isinstance(stop_layers, ll.Layer) or
            is_list_of_layers(stop_layers) and stop_layers):
        raise ValueError('stop_layers should be isinstance of Layer class or '
                         'non-empty list of such isinstances. Got %r' %
                         stop_layers)

    if not (start_layers is None or is_list_of_layers(start_layers)):
        raise ValueError('start_layers should be None or list of Layer '
                         'instances. Got %r' % start_layers)

    if not (modify is None or isinstance(modify, dict)):
        raise ValueError('modify should be None or dict. Got %r' % modify)

    if not (rename_rule is None or
            isinstance(rename_rule, collections.Callable)):
        raise ValueError('rename_rule should be None or Callable. Got %r' %
                         rename_rule)


def modified_copy(stop_layers, start_layers=None, modify=None, rename_rule=None,
                  unwrap_shared=True):
    _check_assertations(locals())

    start_layers = start_layers or []
    modify = modify or {}

    layers_of_original = ll.get_all_layers(stop_layers)
    input_layers = [l for l in layers_of_original
                    if isinstance(l, ll.InputLayer)]
    start_layers.extend(input_layers)  # TODO check intersection and warn

    params_of_original = ll.get_all_params(stop_layers,
                                           unwrap_shared=unwrap_shared)

    memo = {id(x): x for x in start_layers + params_of_original}  # x -> x
    memo.update({id(k): v for k, v in modify.iteritems()})  # k -> v

    modified_layers = copy.deepcopy(stop_layers, memo)

    if rename_rule is not None:
        layers_of_modified = set(ll.get_all_layers(modified_layers))
        for layer in layers_of_modified:
            if layer not in layers_of_original:
                layer.name = rename_rule(layer.name)

        params_of_modified = ll.get_all_params(modified_layers,
                                               unwrap_shared=unwrap_shared)
        for param in params_of_modified:
            if param not in params_of_original:
                param.name = rename_rule(param.name)

    return modified_layers
