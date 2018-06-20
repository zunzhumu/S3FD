# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import numpy as np

def deformable_conv_unit(data, num_filter, workspace, name, dropout=False):
    if dropout:
        data = mx.sym.Dropout(data)

    conv1_offset = mx.sym.Convolution(data=data, num_filter=18, kernel=(3, 3), stride=(1, 1), pad=(1, 1)
                                      , workspace=workspace, name=name + '_offset')

    conv1 = mx.contrib.sym.DeformableConvolution(data=data, offset=conv1_offset, num_filter=int(num_filter), kernel=(3, 3), stride=(1, 1), pad=(2, 2),
                               num_deformable_group=1, dilate=(2, 2), no_bias=True, workspace=workspace, name=name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

    return act1

def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", use_batchnorm=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="{}_conv".format(name))
    if use_batchnorm:
        conv = mx.symbol.BatchNorm(data=conv, name="{}_bn".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type, \
        name="{}_{}".format(name, act_type))
    return relu

def legacy_conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", use_batchnorm=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    assert not use_batchnorm, "batchnorm not yet supported"
    bias = mx.symbol.Variable(name="conv{}_bias".format(name),
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
    conv = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="conv{}".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type, \
        name="{}{}".format(act_type, name))
    if use_batchnorm:
        relu = mx.symbol.BatchNorm(data=relu, name="bn{}".format(name))
    return conv, relu

def multi_layer_feature(body, from_layers, num_filters, strides, pads, min_filter=128):
    """Wrapper function to extract features from base network, attaching extra
    layers and SSD specific layers

    Parameters
    ----------
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    min_filter : int
        minimum number of filters used in 1x1 convolution

    Returns
    -------
    list of mx.Symbols

    """
    # arguments check
    assert len(from_layers) > 0
    assert isinstance(from_layers[0], str) and len(from_layers[0].strip()) > 0
    assert len(from_layers) == len(num_filters) == len(strides) == len(pads)
    # viz = mx.viz.plot_network(body, shape={"data":(1,3,640,640)})
    # viz.view()
    internals = body.get_internals()
    layers = []
    for k, params in enumerate(zip(from_layers, num_filters, strides, pads)):
        from_layer, num_filter, s, p = params
        if from_layer.strip():
            # extract from base network
            layer = internals[from_layer.strip() + '_output']
            layers.append(layer)
        else:
            # attach from last feature layer
            assert len(layers) > 0
            assert num_filter > 0
            layer = layers[-1]
            num_1x1 = max(min_filter, num_filter // 2)
            conv_1x1 = conv_act_layer(layer, 'multi_feat_%d_conv_1x1' % (k),
                num_1x1, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
            conv_3x3 = conv_act_layer(conv_1x1, 'multi_feat_%d_conv_3x3' % (k),
                num_filter, kernel=(3, 3), pad=(p, p), stride=(s, s), act_type='relu')
            layers.append(conv_3x3)
    # for i in range(len(layers)):
    #     arg_shape, output_shape, aux_shape = layers[i].infer_shape(data=(1, 3, 640, 640))
    #     print  'layers'+str(i)+',output_shape, ', output_shape
    #(256, 160, 160)
    # (512, 80, 80)
    # (1024, 40, 40)
    # (2048, 20, 20)
    # (256, 10, 10)
    # (128, 5, 5)
    # layers0, output_shape, [(1L, 256L, 160L, 160L)]
    # layers1, output_shape, [(1L, 512L, 80L, 80L)]
    # layers2, output_shape, [(1L, 512L, 40L, 40L)]
    # layers3, output_shape, [(1L, 256L, 20L, 20L)]
    # layers4, output_shape, [(1L, 256L, 10L, 10L)]
    # layers5, output_shape, [(1L, 128L, 5L, 5L)]
    return layers





def multi_layer_feature_FPN(body, from_layers, num_filters, strides, pads, min_filter=128):
    """Wrapper function to extract features from base network, attaching extra
    layers and SSD specific layers

    Parameters
    ----------
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    min_filter : int
        minimum number of filters used in 1x1 convolution

    Returns
    -------
    list of mx.Symbols

    """
    # arguments check
    assert len(from_layers) > 0
    assert isinstance(from_layers[0], str) and len(from_layers[0].strip()) > 0
    assert len(from_layers) == len(num_filters) == len(strides) == len(pads)

    layers = []
    internals = body.get_internals()
    # __plus15 (b, 2048, 10, 10)
    #  __plus12 (b, 1024, 19, 19)
    #  __plus6 (b, 512, 38, 38)
    #  __plus2 (b, 256, 75, 75)
    two_layer = ['_plus2', '_plus6']
    frist_layer = internals[two_layer[0].strip() + '_output']
    second_layer = internals[two_layer[1].strip() + '_output']

    # frist__layer_downsampling (b, 512, 38, 38)
    frist_layer_downsampling = conv_act_layer(frist_layer, 'frist_layer_downsampling', 64, kernel=(3, 3), pad=(1, 1),
                                              stride=(2, 2), act_type='relu', use_batchnorm=True)
    second_layer_EWS = mx.symbol.Concat(*[frist_layer_downsampling, second_layer], name='second_layer_Concat')

    # second_layer_EWS_downsampling (b, 1024, 19, 19)
    second_layer_EWS_downsampling = conv_act_layer(second_layer_EWS, 'second_layer_Concat_downsampling', 64, kernel=(3, 3), pad=(1, 1),
                                                   stride=(2, 2), act_type='relu', use_batchnorm=True)
    # arg_shape, output_shape, aux_shape = second_layer_EWS_downsampling.infer_shape(data=(1, 3, 300, 300))
    # print  'output_shape', output_shape
    for k, params in enumerate(zip(from_layers, num_filters, strides, pads)):
        from_layer, num_filter, s, p = params
        if from_layer.strip():
            # extract from base network
            layer = internals[from_layer.strip() + '_output']
            if k == 0:
                # layer (b, 1024, 19, 19)
                layer = mx.symbol.Concat(*[layer, second_layer_EWS_downsampling], name='Concat_plus12')
                layer = deformable_conv_unit(data=layer, num_filter=64, workspace=2048, name = 'plus12_deformable')
            elif k == 1:
                plus12_layer_downsampling = conv_act_layer(layers[-1], 'plus12_layer_downsampling', 64, kernel=(3, 3), pad=(1, 1),
                                              stride=(2, 2), act_type='relu', use_batchnorm=True)
                layer = mx.symbol.Concat(*[layer, plus12_layer_downsampling], name='Concat_plus15')
                layer = deformable_conv_unit(data=layer, num_filter=32, workspace=2048, name='plus15_deformable')
            layers.append(layer)
        else:
            # attach from last feature layer
            assert len(layers) > 0
            assert num_filter > 0
            layer = layers[-1]
            num_1x1 = max(min_filter, num_filter // 2)
            conv_1x1 = conv_act_layer(layer, 'multi_feat_%d_conv_1x1' % (k),
                                      num_1x1, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
            conv_3x3 = conv_act_layer(conv_1x1, 'multi_feat_%d_conv_3x3' % (k),
                                      num_filter, kernel=(3, 3), pad=(p, p), stride=(s, s), act_type='relu')

            layers.append(conv_3x3)

    new_layer=[]
    layer1 = layers
    layers = layers[::-1]
    for k, layer in enumerate(layers):
        if k == 0:
            layer = conv_act_layer(layer, 'multi_feat_FPN%d_conv_1x1_1' % (k),
                                      128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
            new_layer.append(layer)
        else:
            num_3x3 = conv_act_layer(layer, 'multi_feat_FPN%d_conv_1x1_1' % (k),
                                      64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu')
            num_3x3 = mx.symbol.Convolution(data=num_3x3, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1,1), name='multi_feat_FPN%d_conv_1x1_2' % (k))
            deconv = mx.symbol.Deconvolution(data=new_layer[-1], num_filter=256, kernel=(4,4), stride=(2,2), pad=(1,1), name='multi_feat_FPN%d_deconv' % (k))
            deconv_crop = mx.symbol.Crop(*[deconv, num_3x3], name='FPN_crop%d' % (k))
            elew_sum = mx.symbol.Concat(*[num_3x3, deconv_crop], name='FPN_Concat%d' % (k))
            bn = mx.sym.BatchNorm(data=elew_sum, fix_gamma=False, eps=2e-5, momentum=0.9, name='FPN_bn%d'% (k))
            act = mx.sym.Activation(data=bn, act_type='relu', name='FPN_relu%d'% (k))
            layer = conv_act_layer(act, 'multi_feat_FPN%d_conv_1x1_3' % (k), 64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu')
            new_layer.append(layer)

    layer2 = new_layer[::-1]

    # for i in range(len(layer1)):
    #     arg_shape, output_shape, aux_shape = layer1[i].infer_shape(data=(1, 3, 300, 300))
    #     print  'layer1'+str(i)+',output_shape, ', output_shape
    # for i in range(len(layer2)):
    #     arg_shape, output_shape, aux_shape = layer2[i].infer_shape(data=(1, 3, 300, 300))
    #     print  'layer2'+str(i)+',output_shape, ', output_shape
    return layer1, layer2



def multi_layer_feature_FPN_new(body, from_layers, num_filters, strides, pads, min_filter=128):
    """Wrapper function to extract features from base network, attaching extra
    layers and SSD specific layers

    Parameters
    ----------
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    min_filter : int
        minimum number of filters used in 1x1 convolution

    Returns
    -------
    list of mx.Symbols

    """
    # arguments check
    assert len(from_layers) > 0
    assert isinstance(from_layers[0], str) and len(from_layers[0].strip()) > 0
    assert len(from_layers) == len(num_filters) == len(strides) == len(pads)
    # __plus15 (b, 2048, 10, 10)
    #  __plus12 (b, 1024, 19, 19)
    #  __plus6 (b, 512, 38, 38)
    #  __plus2 (b, 256, 75, 75)
    internals = body.get_internals()
    layers = []
    for k, params in enumerate(zip(from_layers, num_filters, strides, pads)):
        from_layer, num_filter, s, p = params
        if from_layer.strip():
            # extract from base network
            layer = internals[from_layer.strip() + '_output']
            layers.append(layer)
        else:
            # attach from last feature layer
            assert len(layers) > 0
            assert num_filter > 0
            layer = layers[-1]
            num_1x1 = max(min_filter, num_filter // 2)
            conv_1x1 = conv_act_layer(layer, 'multi_feat_%d_conv_1x1' % (k),
                                      num_1x1, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
            conv_3x3 = conv_act_layer(conv_1x1, 'multi_feat_%d_conv_3x3' % (k),
                                      num_filter, kernel=(3, 3), pad=(p, p), stride=(s, s), act_type='relu')
            layers.append(conv_3x3)
    # for i in range(len(layers)):
    #     arg_shape, output_shape, aux_shape = layers[i].infer_shape(data=(1, 3, 300, 300))
    #     print  'layers'+str(i)+',output_shape, ', output_shape

    layer1 = layers
    new_layer = []
    layers = layers[::-1]
    for k, layer in enumerate(layers):
        if k == 0:
            layer = conv_act_layer(layer, 'multi_feat_FPN%d_conv_1x1_1' % (k),
                                      128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
            new_layer.append(layer)
        else:
            num_3x3 = conv_act_layer(layer, 'multi_feat_FPN%d_conv_1x1_1' % (k),
                                      64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu')
            num_3x3 = mx.symbol.Convolution(data=num_3x3, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1,1), name='multi_feat_FPN%d_conv_1x1_2' % (k))
            deconv = mx.symbol.Deconvolution(data=new_layer[-1], num_filter=256, kernel=(4,4), stride=(2,2), pad=(1,1), name='multi_feat_FPN%d_deconv' % (k))
            deconv_crop = mx.symbol.Crop(*[deconv, num_3x3], name='FPN_crop%d' % (k))
            elew_sum = mx.symbol.Concat(*[num_3x3, deconv_crop], name='FPN_Concat%d' % (k))
            bn = mx.sym.BatchNorm(data=elew_sum, fix_gamma=False, eps=2e-5, momentum=0.9, name='FPN_bn%d'% (k))
            act = mx.sym.Activation(data=bn, act_type='relu', name='FPN_relu%d'% (k))
            layer = conv_act_layer(act, 'multi_feat_FPN%d_conv_1x1_3' % (k), 64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu')
            new_layer.append(layer)

    layer2 = new_layer[::-1]

    # for i in range(len(layer1)):
    #     arg_shape, output_shape, aux_shape = layer1[i].infer_shape(data=(1, 3, 300, 300))
    #     print  'layer1'+str(i)+',output_shape, ', output_shape
    # for i in range(len(layer2)):
    #     arg_shape, output_shape, aux_shape = layer2[i].infer_shape(data=(1, 3, 300, 300))
    #     print  'layer2'+str(i)+',output_shape, ', output_shape
    return layer1, layer2

def multibox_layer(from_layers, num_classes, sizes=[.2, .95],
                    ratios=[1], normalization=-1, num_channels=[],
                    clip=False, interm_layer=0, steps=[]):
    """
    the basic aggregation module for SSD detection. Takes in multiple layers,
    generate multiple object detection targets by customized layers

    Parameters:
    ----------
    from_layers : list of mx.symbol
        generate multibox detection from layers
    num_classes : int
        number of classes excluding background, will automatically handle
        background in this function
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    num_channels : list of int
        number of input layer channels, used when normalization is enabled, the
        length of list should equals to number of normalization layers
    clip : bool
        whether to clip out-of-image boxes
    interm_layer : int
        if > 0, will add a intermediate Convolution layer
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions

    Returns:
    ----------
    list of outputs, as [loc_preds, cls_preds, anchor_boxes]
    loc_preds : localization regression prediction
    cls_preds : classification prediction
    anchor_boxes : generated anchor boxes
    """
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 0, \
        "num_classes {} must be larger than 0".format(num_classes)

    assert len(ratios) > 0, "aspect ratios must not be empty list"
    if not isinstance(ratios[0], list):
        # provided only one ratio list, broadcast to all from_layers
        ratios = [ratios] * len(from_layers)
    assert len(ratios) == len(from_layers), \
        "ratios and from_layers must have same length"

    assert len(sizes) > 0, "sizes must not be empty list"
    if len(sizes) == 2 and not isinstance(sizes[0], list):
         # provided size range, we need to compute the sizes for each layer
         start_offset = 1
         assert sizes[0] > 0 and sizes[0] < 1
         assert sizes[1] > 0 and sizes[1] < 1 and sizes[1] > sizes[0]
         tmp = np.linspace(sizes[0], sizes[1], num=(len(from_layers)-1))
         min_sizes = [start_offset] + tmp.tolist()
         max_sizes = tmp.tolist() + [tmp[-1]+start_offset]
         sizes = zip(min_sizes, max_sizes)
    assert len(sizes) == len(from_layers), \
        "sizes and from_layers must have same length"

    if not isinstance(normalization, list):
        normalization = [normalization] * len(from_layers)
    assert len(normalization) == len(from_layers)

    assert sum(x > 0 for x in normalization) <= len(num_channels), \
        "must provide number of channels for each normalized layer"

    if steps:
        assert len(steps) == len(from_layers), "provide steps for all layers or leave empty"

    loc_pred_layers = []
    cls_pred_layers = []
    anchor_layers = []
    num_classes += 1 # always use background as label 0

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        # normalize
        if normalization[k] > 0:
            from_layer = mx.symbol.L2Normalization(data=from_layer, \
                mode="channel", name="{}_norm".format(from_name))
            scale = mx.symbol.Variable(name="{}_scale".format(from_name),
                shape=(1, num_channels.pop(0), 1, 1),
                init=mx.init.Constant(normalization[k]),
                attr={'__wd_mult__': '0.1'})
            from_layer = mx.symbol.broadcast_mul(lhs=scale, rhs=from_layer)
        if interm_layer > 0:
            from_layer = mx.symbol.Convolution(data=from_layer, kernel=(3,3), \
                stride=(1,1), pad=(1,1), num_filter=interm_layer, \
                name="{}_inter_conv".format(from_name))
            from_layer = mx.symbol.Activation(data=from_layer, act_type="relu", \
                name="{}_inter_relu".format(from_name))

        # estimate number of anchors per location
        # here I follow the original version in caffe
        # TODO: better way to shape the anchors??
        size = sizes[k]
        assert len(size) > 0, "must provide at least one size"
        size_str = "(" + ",".join([str(x) for x in size]) + ")"
        ratio = ratios[k]
        assert len(ratio) > 0, "must provide at least one ratio"
        ratio_str = "(" + ",".join([str(x) for x in ratio]) + ")"
        num_anchors = len(size) -1 + len(ratio)

        # create location prediction layer
        num_loc_pred = num_anchors * 4
        bias = mx.symbol.Variable(name="{}_loc_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        loc_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
            stride=(1,1), pad=(1,1), num_filter=num_loc_pred, \
            name="{}_loc_pred_conv".format(from_name))
        loc_pred = mx.symbol.transpose(loc_pred, axes=(0,2,3,1))
        loc_pred = mx.symbol.Flatten(data=loc_pred)
        loc_pred_layers.append(loc_pred)

        # create class prediction layer

        num_cls_pred = num_anchors * num_classes

        if from_name == 'relu3_3':
            ## maxout
            num_cls_pred += 2 ##0,1,2 background 3 object
            bias = mx.symbol.Variable(name="{}_maxout_cls_pred_conv_bias".format(from_name),
                                      init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
            cls_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
                stride=(1,1), pad=(1,1), num_filter=num_cls_pred, \
                name="{}_maxout_cls_pred_conv".format(from_name))
            object = mx.symbol.slice_axis(cls_pred, axis=1, begin=3, end=4)
            background = mx.symbol.slice_axis(cls_pred, axis=1, begin=0, end=3)
            background = mx.symbol.max_axis(background, axis=1)
            background = mx.symbol.Reshape(data=background, shape=(-4, -1, 1,-2))
            cls_pred = mx.symbol.Concat(*[object, background], dim=1)
            # arg_shape, output_shape, aux_shape = cls_pred.infer_shape(data=(2, 3, 608, 608))
            # print 'output_shape', output_shape
        else:
            bias = mx.symbol.Variable(name="{}_cls_pred_conv_bias".format(from_name),
                                      init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
            cls_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3, 3), \
                                             stride=(1, 1), pad=(1, 1), num_filter=num_cls_pred, \
                                             name="{}_cls_pred_conv".format(from_name))

        cls_pred = mx.symbol.transpose(cls_pred, axes=(0, 2, 3, 1))
        cls_pred = mx.symbol.Flatten(data=cls_pred)
        cls_pred_layers.append(cls_pred)

        # create anchor generation layer
        if steps:
            step = (steps[k], steps[k])
        else:
            step = '(-1.0, -1.0)'
        anchors = mx.symbol.contrib.MultiBoxPrior(from_layer, sizes=size_str, ratios=ratio_str,
                                                  clip=clip, name="{}_anchors".format(from_name),
                                                  steps=step)
        anchors = mx.symbol.Flatten(data=anchors)
        anchor_layers.append(anchors)

    loc_preds = mx.symbol.Concat(*loc_pred_layers, num_args=len(loc_pred_layers), \
        dim=1, name="multibox_loc_pred")
    cls_preds = mx.symbol.Concat(*cls_pred_layers, num_args=len(cls_pred_layers), \
        dim=1)
    cls_preds = mx.symbol.Reshape(data=cls_preds, shape=(0, -1, num_classes))
    cls_preds = mx.symbol.transpose(cls_preds, axes=(0, 2, 1), name="multibox_cls_pred")

    anchor_boxes = mx.symbol.Concat(*anchor_layers, \
        num_args=len(anchor_layers), dim=1)
    anchor_boxes = mx.symbol.Reshape(data=anchor_boxes, shape=(0, -1, 4), name="multibox_anchors")
    # arg_shape, output_shape, aux_shape = anchor_boxes.infer_shape(data=(2, 3, 512, 512))
    # print  'output_shape, ', output_shape
    return [loc_preds, cls_preds, anchor_boxes]
