from __future__ import division

from collections import OrderedDict
import re

from theano.printing import debugprint
from lasagne import layers as ll
import pydot as pydot


__all__ = ['save_debugprint', 'visualize_model']


def get_color(layer):
    if layer.__module__ == 'advanced_layers':
        return 'coral'
    elif isinstance(layer, ll.DenseLayer):
        return 'palegreen'
    if isinstance(layer, ll.InputLayer):
        return 'lightcoral'
    elif isinstance(layer, ll.MergeLayer):
        return 'lightblue'
    elif isinstance(layer, (ll.FlattenLayer, ll.ReshapeLayer,
                            ll.NonlinearityLayer)):
        return None
    else:
        return 'red'


def get_cluster_name(layer):
    if isinstance(layer, ll.InputLayer):
        return 'input'
    else:
        return layer.name.split('.', 1)[0]


def layer_to_node(layer):
    node = pydot.Node(name=layer.name, shape='record', style='filled')

    params = []

    for param in layer.get_params(unwrap_shared=False):
        if hasattr(param, 'name') and param.name is not None:
            name_or_repr = param.name
        else:
            name_or_repr = re.sub(r'(?=[{}<>])', r'\\', repr(param))
            name_or_repr += ' (%d)' % id(param)

        params.append(name_or_repr)

    node.set_label(  # pylint: disable=no-member
        '{%s}' % '|'.join([
            layer.name,
            layer.__class__.__name__ + (' (%d)' % id(layer)),
            r'\l'.join(params) + r'\l',
            repr(layer.output_shape)
        ])
    )

    color = get_color(layer)
    if color is not None:
        node.set_fillcolor(color)  # pylint: disable=no-member

    return node


def layers_to_edge(layer_src, layer_dst):
    edge = pydot.Edge(layer_src.name, layer_dst.name)
    return edge


def get_input_layers(layer):
    if hasattr(layer, 'input_layer'):
        return [layer.input_layer]
    elif hasattr(layer, 'input_layers'):
        return layer.input_layers
    else:
        return []


def make_model_graph(layers):
    graph = pydot.Dot('model', splines='line', outputorder='edgesfirst',
                      ranksep=2, nodesep=2)
    clusters = OrderedDict()

    for layer in ll.get_all_layers(layers):
        assert layer.name is not None

        cluster_name = get_cluster_name(layer)
        if cluster_name is not None:
            try:
                cluster = clusters[cluster_name]
            except KeyError:
                clusters[cluster_name] = pydot.Cluster(cluster_name,
                                                       label=cluster_name,
                                                       style='filled',
                                                       color='lightgrey')
                cluster = clusters[cluster_name]

            cluster.add_node(layer_to_node(layer))
        else:
            graph.add_node(layer_to_node(layer))

        for input_layer in get_input_layers(layer):
            input_cluster_name = get_cluster_name(input_layer)
            if cluster_name is not None and input_cluster_name is not None and \
                    cluster_name == input_cluster_name:
                cluster.add_edge(layers_to_edge(input_layer, layer))
            else:
                edge = layers_to_edge(input_layer, layer)
                edge.set_constraint(False)  # pylint: disable=no-member
                edge.set_style('dashed')  # pylint: disable=no-member
                edge.set_color('dimgrey')  # pylint: disable=no-member
                edge.set_headlabel(input_layer.name)  # pylint: disable=no-member
                edge.set_fontcolor('dimgrey')  # pylint: disable=no-member
                cluster.add_edge(edge)

    for cluster in clusters.itervalues():
        graph.add_subgraph(cluster)

    return graph


def visualize_model(layers, file='model.png'):
    with open(file, 'w') as file_:
        graph = make_model_graph(layers)
        image = graph.create(format=file.rsplit('.', 1)[-1])
        file_.write(image)


def save_debugprint(obj, **kargs):
    file = kargs.pop('file', 'debugprint.log')
    with open(file, 'w') as file_:
        debugprint(obj, file=file_, **kargs)
