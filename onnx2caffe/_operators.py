from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe import params as P
import math
import numpy as np
from ._graph import Node, Graph
from MyCaffe import Function as myf


def _compare(a, b, encoding="utf8"):  #type: (Text, Text, Text) -> bool
    if isinstance(a, bytes):
        a = a.decode(encoding)
    if isinstance(b, bytes):
        b = b.decode(encoding)
    return a == b


def make_input(input):
    name = input[0]
    if name[0].isnumeric():
        name = 'node_{}'.format(name)
    output = input[0]
    output = [output]
    shape = input[2]
    shape = list(shape)
    input_layer = myf("Input",
                      name, [],
                      output,
                      input_param=dict(shape=dict(dim=shape)))
    return input_layer


def _convert_conv(node, graph, err):
    weight_name = node.inputs[1]
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(
            node, "Weight tensor: {} not found in the graph initializer".format(
                weight_name,))
    is_deconv = False
    if node.op_type.endswith("Transpose"):
        is_deconv = True
    bias_flag = False
    bias = None
    if len(node.inputs) > 2:
        bias = node.input_tensors[node.inputs[2]]
        bias_flag = True
    dilations = node.attrs.get("dilations", [1, 1])
    groups = node.attrs.get("group", 1)
    kernel_shape = node.attrs["kernel_shape"]
    pads = node.attrs.get("pads", [0, 0, 0, 0])
    if node.attrs.get("auto_pad", None) == b"SAME_LOWER":
        pads = [kernel_shape[0] // 2, kernel_shape[1] // 2]
    strides = node.attrs["strides"]
    num_output = W.shape[0]
    assert len(
        W.shape
    ) == 4, 'Only 2d Convs with 4d weigths supported, got {}d weights.'.format(
        len(W.shape))
    layer = myf("Convolution",
                node_name, [input_name], [output_name],
                kernel_h=kernel_shape[0],
                kernel_w=kernel_shape[1],
                stride_h=strides[0],
                stride_w=strides[1],
                group=groups,
                pad_h=pads[0],
                pad_w=pads[1],
                num_output=num_output,
                dilation=max(dilations),
                bias_term=bias_flag)

    graph.channel_dims[output_name] = num_output
    return layer


def _convert_leakyrelu(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    name = str(node.name)

    if input_name == output_name:
        inplace = True
    else:
        inplace = False
    negative_slope = node.attrs.get("alpha", 0.1)
    layer = myf("ReLU",
                name, [input_name], [output_name],
                in_place=inplace,
                negative_slope=negative_slope)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return layer


def _convert_relu(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    name = str(node.name)

    if input_name == output_name:
        inplace = True
    else:
        inplace = False

    layer = myf("ReLU", name, [input_name], [output_name], in_place=inplace)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return layer


def _convert_sigmoid(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    name = str(node.name)

    if input_name == output_name:
        inplace = True
    else:
        inplace = False

    layer = myf("Sigmoid", name, [input_name], [output_name], in_place=inplace)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return layer


def _convert_BatchNorm(node, graph, err):
    epsilon = node.attrs.get("epsilon", 1e-5)
    scale = node.input_tensors[node.inputs[1]]
    bias = node.input_tensors[node.inputs[2]]
    mean = node.input_tensors[node.inputs[3]]
    var = node.input_tensors[node.inputs[4]]
    node_name = node.name

    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])

    bn_layer_name = node_name + "_bn"
    scale_layer_name = node_name + "_scale"
    bn_layer = myf("BatchNorm",
                   bn_layer_name, [input_name], [bn_layer_name],
                   eps=epsilon,
                   use_global_stats=True,
                   in_place=False)
    scale_layer = myf("Scale",
                      scale_layer_name, [bn_layer_name], [output_name],
                      in_place=False,
                      bias_term=True)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return bn_layer, scale_layer


def _convert_Add(node, graph, err):
    input_name_list = [str(i) for i in node.inputs]
    output_name = str(node.outputs[0])
    node_name = node.name

    max_dim = 0
    for name in input_name_list:
        if graph.channel_dims[name] > max_dim:
            max_dim = graph.channel_dims[name]

    if 'broadcast' in node.attrs:
        if node.attrs['broadcast'] == 1:
            input_node_number = len(input_name_list)
            if input_node_number != 2:
                return err.unsupported_op_configuration(
                    node, "Broadcast Add must has 2 input, not {}".format(
                        input_node_number))
            axis = node.attrs['axis']
            flat_layer = myf("Flatten", node_name + '_flat',
                             [input_name_list[1]], [output_name + '_flat'])
            layer = myf("Bias",
                        node_name, [input_name_list[0], output_name + '_flat'],
                        [output_name],
                        axis=axis)
            # layer = myf("Bias", node_name, input_name_list, [output_name], bias_term = False, axis = axis)
            graph.channel_dims[output_name] = graph.channel_dims[
                input_name_list[0]]
            return flat_layer, layer

    layer = myf("Eltwise",
                node_name,
                input_name_list, [output_name],
                operation=P.Eltwise.SUM)
    graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]
    return layer


def _convert_Mul(node, graph, err):
    input_name_list = [str(i) for i in node.inputs]
    output_name = str(node.outputs[0])
    node_name = node.name

    if 'broadcast' in node.attrs:
        if node.attrs['broadcast'] == 1:
            input_node_number = len(input_name_list)
            if input_node_number != 2:
                return err.unsupported_op_configuration(
                    node, "Broadcast Mul must has 2 input, not {}".format(
                        input_node_number))
            axis = node.attrs['axis']
            flat_layer = myf("Flatten", node_name + '_flat',
                             [input_name_list[1]], [output_name + '_flat'])
            layer = myf("Scale",
                        node_name, [input_name_list[0], output_name + '_flat'],
                        [output_name],
                        bias_term=False,
                        axis=axis)
            graph.channel_dims[output_name] = graph.channel_dims[
                input_name_list[0]]
            return flat_layer, layer

    layer = myf("Eltwise",
                node_name,
                input_name_list, [output_name],
                operation=P.Eltwise.PROD)
    graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]
    return layer


def _convert_Reshape(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    if len(node.inputs) == 1:
        shape = tuple(node.attrs.get('shape', ()))
    else:
        shape = tuple(node.input_tensors[node.inputs[1]])
    # if shape == ():

    if input_name == output_name:
        inplace = True
    else:
        inplace = False
    if len(shape) == 2:
        layer = myf("Flatten",
                    node_name, [input_name], [output_name],
                    in_place=inplace)
        graph.channel_dims[output_name] = shape[1]
        return layer
    elif len(shape) == 4:
        graph.channel_dims[output_name] = shape[1]
        layer = myf("Reshape",
                    node_name, [input_name], [output_name],
                    reshape_param=dict(shape=dict(dim=list(shape))))
        return layer
    else:
        return err.unsupported_op_configuration(
            node, "Reshape dimention number shall be 2 or 4")


def _convert_Flatten(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    # shape = tuple(node.attrs.get('shape', ()))
    if input_name == output_name:
        inplace = True
    else:
        inplace = False
    layer = myf("Flatten",
                node_name, [input_name], [output_name],
                in_place=inplace)
    # graph.channel_dims[output_name] = shape[1]
    return layer


def _convert_pool(node, graph, err):

    def _modify_to_caffe_pad(kernel_shape, strides, pads_in, index):
        pad_out = pads_in[index]
        modify_pad = True
        modify_pad &= (kernel_shape[index] % 2 == 1)
        modify_pad &= (4 > strides[index] > 1)
        modify_pad &= (2 * pads_in[index] + 1 == kernel_shape[index])
        modify_pad &= (pads_in[index] > 0)
        if modify_pad:
            pad_out -= 1
        return pad_out

    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    if node.op_type.endswith("MaxPool"):
        pool_type = P.Pooling.MAX
    elif node.op_type.endswith("AveragePool"):
        pool_type = P.Pooling.AVE
    else:
        return err.unsupported_op_configuration(node, "Unsupported pool type")

    kernel_shape = node.attrs["kernel_shape"]
    strides = node.attrs.get('strides', [1, 1])
    pads = node.attrs.get('pads', [0, 0, 0, 0])
    pads[0] = _modify_to_caffe_pad(kernel_shape, strides, pads, index=0)
    # if kernel_shape[1] % 2 and 4 > strides[1] > 1 and pads[1] > 0:
    pads[1] = _modify_to_caffe_pad(kernel_shape, strides, pads, index=1)

    layer = myf("Pooling",
                node_name, [input_name], [output_name],
                pooling_param=dict(pool=pool_type,
                                   kernel_h=kernel_shape[0],
                                   kernel_w=kernel_shape[1],
                                   stride_h=strides[0],
                                   stride_w=strides[1],
                                   pad_h=pads[0],
                                   pad_w=pads[1]))
    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer


def _convert_dropout(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    ratio = node.attrs.get('ratio', 0.5)
    layer = myf("Dropout",
                node_name, [input_name], [output_name],
                dropout_ratio=ratio)
    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer


def _convert_gemm(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    weight_name = node.inputs[1]
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(
            node, "Weight tensor: {} not found in the graph initializer".format(
                weight_name,))
        return

    if node.attrs.get("broadcast", 1) != 1 or node.attrs.get("transB", 1) != 1:
        return err.unsupported_op_configuration(
            node, "Gemm is supported only for inner_product layer")

    b = None
    bias_flag = False
    if len(node.inputs) > 2:
        b = node.input_tensors[node.inputs[2]]

    if len(W.shape) != 2 or (b is not None and len(b.shape) != 1):
        return err.unsupported_op_configuration(
            node, "Gemm is supported only for inner_product layer")
    if b is not None:
        bias_flag = True
        if W.shape[0] != b.shape[0]:
            return err.unsupported_op_configuration(
                node, "Gemm is supported only for inner_product layer")

    layer = myf("InnerProduct",
                node_name, [input_name], [output_name],
                num_output=W.shape[0],
                bias_term=bias_flag)
    graph.channel_dims[output_name] = W.shape[0]

    return layer


def _convert_upsample(node, graph, err):
    if 'height_scale' in node.attrs.keys():
        factor = int(node.attrs['height_scale'])
        assert factor == int(node.attrs['width_scale'])
    elif 'scales' in node.attrs.keys():
        scales = node.attrs['scales']
        assert len(scales) == 4
        assert scales[2] == scales[3]
        factor = int(scales[2])
    else:
        raise ValueError('Could not find scale values in Upsample node: %s' %
                         str(node.attrs))
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    channels = graph.channel_dims[input_name]
    pad = int(math.ceil((factor - 1) / 2.))
    mode = node.attrs["mode"]
    # print(mode)
    # exit(0)
    #https://github.com/pytorch/pytorch/issues/6900
    if mode == b"bilinear":
        layer = myf("Deconvolution",
                    node_name, [input_name], [output_name],
                    convolution_param=dict(num_output=channels,
                                           kernel_size=2 * factor - factor % 2,
                                           stride=factor,
                                           pad=pad,
                                           group=channels,
                                           bias_term=False,
                                           weight_filler=dict(type="bilinear")))
    else:
        layer = myf("Deconvolution",
                    node_name, [input_name], [output_name],
                    convolution_param=dict(
                        num_output=channels,
                        kernel_size=factor,
                        stride=factor,
                        group=channels,
                        bias_term=False,
                    ))

    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer


def _convert_concat(node, graph, err):
    node_name = node.name
    input_name_list = [str(i) for i in node.inputs]
    output_name = str(node.outputs[0])
    axis = node.attrs.get("axis", 1)

    layer = myf('Concat', node_name, input_name_list, [output_name], axis=axis)
    if axis == 1:
        dim = 0
        for name in input_name_list:
            dim += graph.channel_dims[name]
        graph.channel_dims[output_name] = dim
    else:
        graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]

    return layer


def _convert_conv_transpose(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    weight_name = node.inputs[1]
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(
            node, "Weight tensor: {} not found in the graph initializer".format(
                weight_name,))
    bias_flag = False
    bias = None
    if len(node.inputs) > 2:
        bias = node.input_tensors[node.inputs[2]]
        bias_flag = True
    dilations = node.attrs.get("dilations", [1, 1])
    groups = node.attrs.get("group", 1)
    kernel_shape = node.attrs["kernel_shape"]
    pads = node.attrs.get("pads", [0, 0, 0, 0])
    strides = node.attrs["strides"]
    num_output = groups * W.shape[1]
    print(num_output, node.op_type)

    layer = myf('Deconvolution',
                node_name, [input_name], [output_name],
                convolution_param=dict(
                    num_output=num_output,
                    kernel_h=kernel_shape[0],
                    kernel_w=kernel_shape[1],
                    stride_h=strides[0],
                    stride_w=strides[1],
                    group=groups,
                    pad_h=pads[0],
                    pad_w=pads[1],
                    bias_term=bias_flag,
                ))

    graph.channel_dims[output_name] = W.shape[1]
    return layer


_ONNX_NODE_REGISTRY = {
    "Conv": _convert_conv,
    "Relu": _convert_relu,
    "Clip": _convert_relu,
    "LeakyRelu": _convert_leakyrelu,
    "BatchNormalization": _convert_BatchNorm,
    "Add": _convert_Add,
    "Mul": _convert_Mul,
    "Reshape": _convert_Reshape,
    "MaxPool": _convert_pool,
    "AveragePool": _convert_pool,
    "Dropout": _convert_dropout,
    "Gemm": _convert_gemm,
    "Upsample": _convert_upsample,
    "Concat": _convert_concat,
    "ConvTranspose": _convert_conv_transpose,
    "Sigmoid": _convert_sigmoid,
    "Flatten": _convert_Flatten,
}
