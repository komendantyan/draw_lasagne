import lasagne.layers as ll


__all__ = ['layer_by_name']


def layer_by_name(layer, name):
    ans = [l for l in ll.get_all_layers(layer)
           if l.name == name]
    if len(ans) > 1:
        raise ValueError('%d layers have name %s' % (len(ans), name))
    elif not ans:
        raise KeyError('no layer with name %s' % name)
    else:
        return ans[0]
