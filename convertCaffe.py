from __future__ import print_function

import argparse

import onnx
from onnx import shape_inference
from onnx import optimizer

import caffe
from caffe.proto import caffe_pb2
caffe.set_mode_cpu()

from onnx2caffe._transformers import ConvAddFuser, ConstantsToInitializers
from onnx2caffe._graph import Graph
import onnx2caffe._operators as cvt
import onnx2caffe._weightloader as wlr
from onnx2caffe._error_utils import ErrorHandling

transformers = [
    ConstantsToInitializers(),
    ConvAddFuser(),
]


def convertToCaffe(graph,
                   prototxt_save_path,
                   caffemodel_save_path,
                   convert_leaky_relu,
                   max_inputs=-1):

    if convert_leaky_relu:
        cvt._ONNX_NODE_REGISTRY['LeakyRelu'] = cvt._ONNX_NODE_REGISTRY['Relu']
        wlr._ONNX_NODE_REGISTRY['LeakyRelu'] = wlr._ONNX_NODE_REGISTRY['Relu']

    exist_edges = []
    layers = []
    exist_nodes = []
    err = ErrorHandling()
    if max_inputs > 0:
        graph.inputs = graph.inputs[:max_inputs]

    for i in graph.inputs:
        edge_name = i[0]
        input_layer = cvt.make_input(i)
        layers.append(input_layer)
        exist_edges.append(i[0])
        dims_input = graph.shape_dict[edge_name]
        for dim in dims_input:
            assert dim > 0, 'Please export the ONNX graph without dynamic shapes.'
        graph.channel_dims[edge_name] = dims_input[1]

    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type
        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False

        for inp in inputs:
            if inp not in exist_edges and inp not in inputs_tensor:
                input_non_exist_flag = True
                break
        if input_non_exist_flag:
            continue
        if op_type not in cvt._ONNX_NODE_REGISTRY:
            err.unsupported_op(node)
            continue
        converter_fn = cvt._ONNX_NODE_REGISTRY[op_type]
        layer = converter_fn(node, graph, err)
        if type(layer) == tuple:
            for l in layer:
                layers.append(l)
        else:
            layers.append(layer)
        outs = node.outputs
        for out in outs:
            exist_edges.append(out)

    net = caffe_pb2.NetParameter()
    for id, layer in enumerate(layers):
        layers[id] = layer._to_proto()
    net.layer.extend(layers)

    with open(prototxt_save_path, 'w') as f:
        print(net, file=f)

    caffe.set_mode_cpu()
    deploy = prototxt_save_path
    net = caffe.Net(deploy, caffe.TEST)

    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type
        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False
        if op_type not in wlr._ONNX_NODE_REGISTRY:
            err.unsupported_op(node)
            continue
        converter_fn = wlr._ONNX_NODE_REGISTRY[op_type]
        converter_fn(net, node, graph, err)

    net.save(caffemodel_save_path)
    return net


def getGraph(onnx_path, with_opt=False):
    model = onnx.load(onnx_path)
    if with_opt:
        opt_passes = ['eliminate_nop_pad', 'eliminate_identity']
        model = optimizer.optimize(model, opt_passes)
    model = shape_inference.infer_shapes(model)
    model_graph = model.graph
    graph = Graph.from_onnx(model_graph)
    graph = graph.transformed(transformers)
    graph.channel_dims = {}
    return graph


def main(args):
    onnx_path = args.onnx
    prototxt_path = args.prototxt
    caffemodel_path = args.caffemodel
    convert_leaky_relu = args.noleaky
    max_inputs = args.max_inputs
    with_opt = not args.disable_onnx_opts
    graph = getGraph(onnx_path, with_opt)
    convertToCaffe(graph, prototxt_path, caffemodel_path, convert_leaky_relu,
                   max_inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert ONNX to Caffe .prototxt and .caffemodel')
    parser.add_argument('-i',
                        '--onnx',
                        type=str,
                        required=True,
                        help='Path to ONNX model (input) file')
    parser.add_argument('-p',
                        '--prototxt',
                        type=str,
                        required=True,
                        help='Path to Caffe .prototxt (output) file')
    parser.add_argument('-c',
                        '--caffemodel',
                        type=str,
                        help='Path to .caffemodel (output) file')
    parser.add_argument(
        '--noleaky',
        help='Flag whether to use ReLU instead of Leaky ReLU.'
        'Just for run-time measurements: Will not produce the same results as with Leaky ReLU.',
        action='store_true')
    parser.add_argument(
        '--max-inputs',
        help=
        'Only use the first N input tensors of the ONNX graph. Default: -1 (do not restrict).',
        type=int,
        default=-1)
    parser.add_argument(
        '--disable-onnx-opts',
        '-d',
        help='Disable all ONNX optimization passes (fails for some models).',
        action='store_true')

    args = parser.parse_args()
    main(args)
