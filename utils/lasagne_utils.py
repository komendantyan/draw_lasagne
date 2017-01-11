import re
import lasagne.layers as ll


__all__ = ['layer_by_name', 'layers_by_regexp']


def layer_by_name(layer, name):
    ans = [l for l in ll.get_all_layers(layer)
           if l.name == name]
    if len(ans) > 1:
        raise ValueError('%d layers have name %s' % (len(ans), name))
    elif not ans:
        raise KeyError('no layer with name %s' % name)
    else:
        return ans[0]


def layers_by_regexp(layer, pattern):
    regexp = re.compile(pattern)

    return [
        l
        for l in ll.get_all_layers(layer)
        if l.name is not None and regexp.search(l.name) is not None
    ]
