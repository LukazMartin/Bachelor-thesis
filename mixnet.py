import numpy as np
import tensorflow as tf
from layers import layers
from losses.yololoss import logistic_activate_tensor

slim = tf.contrib.slim


def build(inputs, nclasses, grid_size, boxes_per_cell, image_size, is_training, priors=None, type='yolo'):
    if type == 'yolo':
        output_size = (grid_size * grid_size) * (nclasses + boxes_per_cell * 5)
        net_output = YOLONet(inputs, output_size, is_training=is_training)
        preds = unroll_predictions(net_output, nclasses, grid_size, boxes_per_cell, image_size)
    elif type == 'yolo2':
        net_output = YOLONet2(inputs, nclasses, boxes_per_cell, training=is_training)
        preds = unroll_predictions2(net_output, nclasses, grid_size, boxes_per_cell, image_size, priors)
    elif type == 'myyolo':
        net_output = MyYOLONet(inputs, nclasses, boxes_per_cell, training=is_training)
        preds = my_unroll_predictions4(net_output[1], nclasses, grid_size, boxes_per_cell, image_size, priors)
    elif type == 'myyolomixnetm':
        net_output = MyYOLOMixNetM(inputs, nclasses, boxes_per_cell, training=is_training)
        preds = my_unroll_predictions4(net_output[1], nclasses, grid_size, boxes_per_cell, image_size, priors)
    # model_variables = [n.name for n in tf.global_variables()]
    
    # return net_output, preds, model_variables

    return net_output, preds


def MyYOLOMixNetS(images, num_classes, num_anchors, alpha=0.1, training=False, center=True, scope='yolo2_darknet'):
    def batch_norm(net):
        net = slim.batch_norm(net, center=center, scale=True, epsilon=1e-5, is_training=training)
        if not center:
            net = tf.nn.bias_add(net,
                                 slim.variable('biases', shape=[tf.shape(net)[-1]], initializer=tf.zeros_initializer()))
        return net

    def _split_channels(total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    net = tf.pad(images, np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), name=scope + '/pad_1')

    with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3], normalizer_fn=batch_norm,
                        activation_fn=leaky_relu(alpha)), slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2, 2],
                                                                         padding='SAME'):

        index = 0
        channels = 16

        # 608x608xC -> 304x304x16 (3x3 kernel)
        net = slim.layers.conv2d(net, channels, kernel_size=[3, 3], scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1

        # 304x304x16->304x304x16 (3x3 kernel)
        net = slim.layers.conv2d(net, channels, kernel_size=[3, 3], scope='%s/conv%d' % (scope, index))
        index += 1

        # 304x304x16->152x152x24 (3x3 kernel)
        channels += channels / 2
        net = slim.layers.conv2d(net, channels, kernel_size=[3, 3], scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1

        # 152x152x24-> 152x152x24 (3x3 kernel)
        net = slim.layers.conv2d(net, channels, kernel_size=[3, 3], scope='%s/conv%d' % (scope, index))
        index += 1

        split_input = _split_channels(24, 3)
        split_output = _split_channels(40, 3)
        # channels = 40
        # 152x152x24-> 76x76x40 (3x3, 5x5, 7x7 kernel)
        net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                  scope='%s/conv%d' % (scope, index))
        index += 1
        net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                  kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
        index += 1
        net3 = slim.layers.conv2d(net[:, :, :, split_input[0] + split_input[1]:], split_output[2],
                                  kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
        net = tf.concat([net1, net2, net3], 3)
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1

        split_input = _split_channels(40, 2)
        split_output = _split_channels(40, 2)
        # 76x76x40-> 76x76x40 (3x3, 5x5 kernel) x3
        for i in range(3):
            net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                      scope='%s/conv%d' % (scope, index))
            index += 1
            net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:], split_output[1], kernel_size=[5, 5],
                                      scope='%s/conv%d' % (scope, index))
            net = tf.concat([net1, net2], 3)
            index += 1

        # channels *= 2
        # 76x76x40-> 38x38x80 (3x3, 5x5, 7x7 kernel)
        split_input = _split_channels(40, 3)
        split_output = _split_channels(80, 3)
        net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                  scope='%s/conv%d' % (scope, index))
        index += 1
        net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                  kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
        index += 1
        net3 = slim.layers.conv2d(net[:, :, :, split_input[0] + split_input[1]:], split_output[2],
                                  kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
        net = tf.concat([net1, net2, net3], 3)
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1

        split_input = _split_channels(80, 2)
        split_output = _split_channels(80, 2)
        # 38x38x80-> 38x38x80 (3x3, 5x5 kernel) x2
        for _ in range(2):
            net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                      scope='%s/conv%d' % (scope, index))
            index += 1
            net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:], split_output[1], kernel_size=[5, 5],
                                      scope='%s/conv%d' % (scope, index))
            net = tf.concat([net1, net2], 3)
            index += 1

        # channels += channels / 2
        # 38x38x80-> 38x38x120 (3x3, 5x5, 7x7 kernel)
        split_input = _split_channels(80, 3)
        split_output = _split_channels(120, 3)
        net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                  scope='%s/conv%d' % (scope, index))
        index += 1
        net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                  kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
        index += 1
        net3 = slim.layers.conv2d(net[:, :, :, split_input[0] + split_input[1]:], split_output[2],
                                  kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
        net = tf.concat([net1, net2, net3], 3)
        index += 1

        split_input = _split_channels(120, 3)
        split_output = _split_channels(120, 3)
        # 38x38x120 -> 38x38x120(3x3, 5x5, 7x7 kernel) x2
        for x in range(2):
            net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                      scope='%s/conv%d' % (scope, index))
            index += 1
            net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                      kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
            index += 1
            net3 = slim.layers.conv2d(net[:, :, :,
                                      split_input[0] + split_input[1]:],
                                      split_output[2], kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
            net = tf.concat([net1, net2, net3], 3)
            index += 1

        split_input = _split_channels(120, 3)
        split_output = _split_channels(200, 3)
        # channels = 200
        # 38x38x120-> 19x19x200 (3x3, 5x5, 7x7 kernel)
        net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                  scope='%s/conv%d' % (scope, index))
        index += 1
        net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                  kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
        index += 1
        net3 = slim.layers.conv2d(net[:, :, :,
                                  split_input[0] + split_input[1]:],
                                  split_output[2], kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
        net = tf.concat([net1, net2, net3], 3)
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1

        split_input = _split_channels(200, 3)
        split_output = _split_channels(200, 3)
        # 19x19x200->19x19x200 (3x3, 5x5, 7x7 kernel) x2
        for j in range(2):
            net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                      scope='%s/conv%d' % (scope, index))
            index += 1
            net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                      kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
            index += 1
            net3 = slim.layers.conv2d(net[:, :, :,
                                      split_input[0] + split_input[1]:],
                                      split_output[2], kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
            net = tf.concat([net1, net2, net3], 3)
            index += 1

            # net = slim.dropout(net, keep_prob=0.2,is_training=training, scope='dropout')
        net_feat = []
        for o in range(num_anchors):
            net1 = slim.layers.conv2d(net, channels, scope='%s/conv_feat%d' % (scope, o))
            net_feat.append(net1)

    net_out = []
    for o in range(num_anchors):
        net = slim.layers.conv2d(net_feat[o], 4 + num_classes, kernel_size=[1, 1], activation_fn=None,
                                 scope='%s/conv_fc%d' % (scope, o))
        net = tf.expand_dims(net, 3)
        net_out.append(net)
    net = tf.concat(net_out, 3)
    net = tf.identity(net, name='%s/output' % scope)
    return [net_feat, net]


def MyYOLOMixNetM(images, num_classes, num_anchors, alpha=0.1, training=False, center=True, scope='yolo2_darknet'):
    def batch_norm(net):
        net = slim.batch_norm(net, center=center, scale=True, epsilon=1e-5, is_training=training)
        if not center:
            net = tf.nn.bias_add(net,
                                 slim.variable('biases', shape=[tf.shape(net)[-1]], initializer=tf.zeros_initializer()))
        return net

    def _split_channels(total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    net = tf.pad(images, np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), name=scope + '/pad_1')

    with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3], normalizer_fn=batch_norm,
                        activation_fn=leaky_relu(alpha)), slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2, 2],
                                                                         padding='SAME'):

        index = 0
        channels = 24

        # 608x608xC -> 304x304x24 (3x3 kernel)
        net = slim.layers.conv2d(net, channels, kernel_size=[3, 3], scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1

        # 304x304x24->304x304x24 (3x3 kernel)
        net = slim.layers.conv2d(net, channels, kernel_size=[3, 3], scope='%s/conv%d' % (scope, index))
        index += 1

        #304x304x24->152x152x32 (3x3, 5x5, 7x7 kernel)
        split_input = _split_channels(24, 3)
        split_output = _split_channels(32, 3)
        net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                  scope='%s/conv%d' % (scope, index))
        index += 1
        net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                  kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
        index += 1
        net3 = slim.layers.conv2d(net[:, :, :, split_input[0] + split_input[1]:], split_output[2],
                                  kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
        net = tf.concat([net1, net2, net3], 3)
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1

        # 152x152x32-> 152x152x32 (3x3 kernel)
        channels = 32
        net = slim.layers.conv2d(net, channels, kernel_size=[3, 3], scope='%s/conv%d' % (scope, index))
        index += 1

        split_input = _split_channels(32, 4)
        split_output = _split_channels(40, 4)
        # 152x152x32-> 76x76x40 (3x3, 5x5, 7x7, 9x9 kernel)
        net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                  scope='%s/conv%d' % (scope, index))
        index += 1
        net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                  kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
        index += 1
        net3 = slim.layers.conv2d(net[:, :, :,
                                  split_input[0] + split_input[1]:split_input[0] + split_input[1] +
                                                                  split_input[2]],
                                  split_output[2], kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
        index += 1
        net4 = slim.layers.conv2d(net[:, :, :, split_input[0] + split_input[1] + split_input[2]:],
                                  split_output[3], kernel_size=[9, 9], scope='%s/conv%d' % (scope, index))
        net = tf.concat([net1, net2, net3, net4], 3)
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1

        split_input = _split_channels(40, 2)
        split_output = _split_channels(40, 2)
        # 76x76x40-> 76x76x40 (3x3, 5x5 kernel) x3
        for i in range(3):
            net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                      scope='%s/conv%d' % (scope, index))
            index += 1
            net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:], split_output[1], kernel_size=[5, 5],
                                      scope='%s/conv%d' % (scope, index))
            net = tf.concat([net1, net2], 3)
            index += 1

        # 76x76x40-> 38x38x80 (3x3, 5x5, 7x7 kernel)
        split_input = _split_channels(40, 3)
        split_output = _split_channels(80, 3)
        net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                  scope='%s/conv%d' % (scope, index))
        index += 1
        net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                  kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
        index += 1
        net3 = slim.layers.conv2d(net[:, :, :, split_input[0] + split_input[1]:], split_output[2],
                                  kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
        net = tf.concat([net1, net2, net3], 3)
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1

        split_input = _split_channels(80, 4)
        split_output = _split_channels(80, 4)
        # 38x38x80-> 38x38x80 (3x3, 5x5,7x7, 9x9 kernel) x3
        for _ in range(3):
            net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                      scope='%s/conv%d' % (scope, index))
            index += 1
            net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                      kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
            index += 1
            net3 = slim.layers.conv2d(net[:, :, :,
                                      split_input[0] + split_input[1]:split_input[0] + split_input[1] +
                                                                      split_input[2]],
                                      split_output[2], kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
            index += 1
            net4 = slim.layers.conv2d(net[:, :, :, split_input[0] + split_input[1] + split_input[2]:],
                                      split_output[3], kernel_size=[9, 9], scope='%s/conv%d' % (scope, index))
            net = tf.concat([net1, net2, net3, net4], 3)
            index += 1

        # 38x38x80-> 38x38x120 (3x3 kernel)
        channels = 120
        net = slim.layers.conv2d(net, channels, kernel_size=[3, 3], scope='%s/conv%d' % (scope, index))
        index += 1

        split_input = _split_channels(120, 4)
        split_output = _split_channels(120, 4)
        # 38x38x120 -> 38x38x120(3x3, 5x5, 7x7, 9x9 kernel) x3
        for x in range(3):
            net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                      scope='%s/conv%d' % (scope, index))
            index += 1
            net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                      kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
            index += 1
            net3 = slim.layers.conv2d(net[:, :, :,
                                      split_input[0] + split_input[1] :split_input[0] + split_input[1] +
                                                                                       split_input[2]],
                                      split_output[2], kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
            index += 1
            net4 = slim.layers.conv2d(net[:, :, :, split_input[0] + split_input[1] + split_input[2]:],
                                      split_output[3], kernel_size=[9, 9], scope='%s/conv%d' % (scope, index))
            net = tf.concat([net1, net2, net3, net4], 3)
            index += 1

        split_input = _split_channels(120, 4)
        split_output = _split_channels(200, 4)
        # 38x38x120-> 19x19x200 (3x3, 5x5, 7x7, 9x9 kernel)
        net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                  scope='%s/conv%d' % (scope, index))
        index += 1
        net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                  kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
        index += 1
        net3 = slim.layers.conv2d(net[:, :, :,
                                  split_input[0] + split_input[1]:split_input[0] + split_input[1] +
                                                                  split_input[2]],
                                  split_output[2], kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
        index += 1
        net4 = slim.layers.conv2d(net[:, :, :, split_input[0] + split_input[1] + split_input[2]:],
                                  split_output[3], kernel_size=[9, 9], scope='%s/conv%d' % (scope, index))
        net = tf.concat([net1, net2, net3, net4], 3)
        index += 1
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1

        split_input = _split_channels(200, 4)
        split_output = _split_channels(200, 4)
        # 19x19x200->19x19x200 (3x3, 5x5, 7x7, 9x9 kernel) x3
        for j in range(3):
            net1 = slim.layers.conv2d(net[:, :, :, :split_input[0]], split_output[0], kernel_size=[3, 3],
                                      scope='%s/conv%d' % (scope, index))
            index += 1
            net2 = slim.layers.conv2d(net[:, :, :, split_input[0]:split_input[0] + split_input[1]], split_output[1],
                                      kernel_size=[5, 5], scope='%s/conv%d' % (scope, index))
            index += 1
            net3 = slim.layers.conv2d(net[:, :, :,
                                      split_input[0] + split_input[1]:split_input[0] + split_input[1] + split_input[2]],
                                      split_output[2], kernel_size=[7, 7], scope='%s/conv%d' % (scope, index))
            index += 1
            net4 = slim.layers.conv2d(net[:, :, :, split_input[0] + split_input[1] + split_input[2]:],
                                      split_output[3], kernel_size=[9, 9], scope='%s/conv%d' % (scope, index))
            net = tf.concat([net1, net2, net3, net4], 3)
            index += 1

        net_feat = []
        for o in range(num_anchors):
            net1 = slim.layers.conv2d(net, channels, scope='%s/conv_feat%d' % (scope, o))
            net_feat.append(net1)

    net_out = []
    for o in range(num_anchors):
        net = slim.layers.conv2d(net_feat[o], 4 + num_classes, kernel_size=[1, 1], activation_fn=None,
                                 scope='%s/conv_fc%d' % (scope, o))
        net = tf.expand_dims(net, 3)
        net_out.append(net)
    net = tf.concat(net_out, 3)
    net = tf.identity(net, name='%s/output' % scope)

    return [net_feat, net]
