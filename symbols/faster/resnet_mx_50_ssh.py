# --------------------------------------------------------
# SSH: Single Stage Headless Face Detector
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Mahyar Najibi
# --------------------------------------------------------

import mxnet as mx
from lib.symbol import Symbol
from operator_py.box_annotator_ohem import *
from operator_py.debug_data import *
import numpy as np

def checkpoint_callback(bbox_param_names, prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        weight = arg[bbox_param_names[0]]
        bias = arg[bbox_param_names[1]]
        arg[bbox_param_names[0] + '_test'] = (weight.T * mx.nd.array(stds)).T
        arg[bbox_param_names[1] + '_test'] = bias * mx.nd.array(stds) + mx.nd.array(means)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop(bbox_param_names[0] + '_test')
        arg.pop(bbox_param_names[1] + '_test')

    return _callback


class resnet_v1_50_ssh(Symbol):
    def __init__(self, n_proposals=900):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [256, 512, 1024, 2048]
        self.n_proposals = n_proposals

    def get_bbox_param_names(self):
        return ['bbox_pred_weight', 'bbox_pred_bias']

    def get_resnet_v1_conv4(self, data):
        # pred1 = mx.sym.Deconvolution(data=data, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=3, name='up')
        # pred1 = mx.symbol.UpSampling(data=data, scale=2, sample_type='bilinear', num_filter=3, num_args = 2, name='up', workspace = 8192)
        # conv1 = mx.symbol.Convolution(name='conv1', data=pred1, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False,
                                       eps=self.eps)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                                  stride=(2, 2), pool_type='max')
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0),
                                              kernel=(1, 1),
                                              stride=(1, 1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1),
                                               stride=(1, 1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a, act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b, act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        res3b_branch2a = mx.symbol.Convolution(name='res3b_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b_branch2a = mx.symbol.BatchNorm(name='bn3b_branch2a', data=res3b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3b_branch2a = bn3b_branch2a
        res3b_branch2a_relu = mx.symbol.Activation(name='res3b_branch2a_relu', data=scale3b_branch2a, act_type='relu')
        res3b_branch2b = mx.symbol.Convolution(name='res3b_branch2b', data=res3b_branch2a_relu, num_filter=128,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b_branch2b = mx.symbol.BatchNorm(name='bn3b_branch2b', data=res3b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3b_branch2b = bn3b_branch2b
        res3b_branch2b_relu = mx.symbol.Activation(name='res3b_branch2b_relu', data=scale3b_branch2b, act_type='relu')
        res3b_branch2c = mx.symbol.Convolution(name='res3b_branch2c', data=res3b_branch2b_relu, num_filter=512,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b_branch2c = mx.symbol.BatchNorm(name='bn3b_branch2c', data=res3b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3b_branch2c = bn3b_branch2c
        res3b = mx.symbol.broadcast_add(name='res3b', *[res3a_relu, scale3b_branch2c])
        res3b_relu = mx.symbol.Activation(name='res3b_relu', data=res3b, act_type='relu')
        res3c_branch2a = mx.symbol.Convolution(name='res3c_branch2a', data=res3b_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3c_branch2a = mx.symbol.BatchNorm(name='bn3c_branch2a', data=res3c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3c_branch2a = bn3c_branch2a
        res3c_branch2a_relu = mx.symbol.Activation(name='res3c_branch2a_relu', data=scale3c_branch2a, act_type='relu')
        res3c_branch2b = mx.symbol.Convolution(name='res3c_branch2b', data=res3c_branch2a_relu, num_filter=128,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3c_branch2b = mx.symbol.BatchNorm(name='bn3c_branch2b', data=res3c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3c_branch2b = bn3c_branch2b
        res3c_branch2b_relu = mx.symbol.Activation(name='res3c_branch2b_relu', data=scale3c_branch2b, act_type='relu')
        res3c_branch2c = mx.symbol.Convolution(name='res3c_branch2c', data=res3c_branch2b_relu, num_filter=512,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3c_branch2c = mx.symbol.BatchNorm(name='bn3c_branch2c', data=res3c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3c_branch2c = bn3c_branch2c
        res3c = mx.symbol.broadcast_add(name='res3c', *[res3b_relu, scale3c_branch2c])
        res3c_relu = mx.symbol.Activation(name='res3c_relu', data=res3c, act_type='relu')
        res3d_branch2a = mx.symbol.Convolution(name='res3d_branch2a', data=res3c_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3d_branch2a = mx.symbol.BatchNorm(name='bn3d_branch2a', data=res3d_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3d_branch2a = bn3d_branch2a
        res3d_branch2a_relu = mx.symbol.Activation(name='res3d_branch2a_relu', data=scale3d_branch2a, act_type='relu')
        res3d_branch2b = mx.symbol.Convolution(name='res3d_branch2b', data=res3d_branch2a_relu, num_filter=128,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3d_branch2b = mx.symbol.BatchNorm(name='bn3d_branch2b', data=res3d_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3d_branch2b = bn3d_branch2b
        res3d_branch2b_relu = mx.symbol.Activation(name='res3d_branch2b_relu', data=scale3d_branch2b, act_type='relu')
        res3d_branch2c = mx.symbol.Convolution(name='res3d_branch2c', data=res3d_branch2b_relu, num_filter=512,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3d_branch2c = mx.symbol.BatchNorm(name='bn3d_branch2c', data=res3d_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3d_branch2c = bn3d_branch2c
        res3d = mx.symbol.broadcast_add(name='res3d', *[res3c_relu, scale3d_branch2c])
        res3d_relu = mx.symbol.Activation(name='res3d_relu', data=res3d, act_type='relu')
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3d_relu, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3d_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        res4b_branch2a = mx.symbol.Convolution(name='res4b_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b_branch2a = mx.symbol.BatchNorm(name='bn4b_branch2a', data=res4b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4b_branch2a = bn4b_branch2a
        res4b_branch2a_relu = mx.symbol.Activation(name='res4b_branch2a_relu', data=scale4b_branch2a, act_type='relu')
        res4b_branch2b = mx.symbol.Convolution(name='res4b_branch2b', data=res4b_branch2a_relu, num_filter=256,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b_branch2b = mx.symbol.BatchNorm(name='bn4b_branch2b', data=res4b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4b_branch2b = bn4b_branch2b
        res4b_branch2b_relu = mx.symbol.Activation(name='res4b_branch2b_relu', data=scale4b_branch2b, act_type='relu')
        res4b_branch2c = mx.symbol.Convolution(name='res4b_branch2c', data=res4b_branch2b_relu, num_filter=1024,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b_branch2c = mx.symbol.BatchNorm(name='bn4b_branch2c', data=res4b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4b_branch2c = bn4b_branch2c
        res4b = mx.symbol.broadcast_add(name='res4b', *[res4a_relu, scale4b_branch2c])
        res4b_relu = mx.symbol.Activation(name='res4b_relu', data=res4b, act_type='relu')
        res4c_branch2a = mx.symbol.Convolution(name='res4c_branch2a', data=res4b_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4c_branch2a = mx.symbol.BatchNorm(name='bn4c_branch2a', data=res4c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4c_branch2a = bn4c_branch2a
        res4c_branch2a_relu = mx.symbol.Activation(name='res4c_branch2a_relu', data=scale4c_branch2a, act_type='relu')
        res4c_branch2b = mx.symbol.Convolution(name='res4c_branch2b', data=res4c_branch2a_relu, num_filter=256,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4c_branch2b = mx.symbol.BatchNorm(name='bn4c_branch2b', data=res4c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4c_branch2b = bn4c_branch2b
        res4c_branch2b_relu = mx.symbol.Activation(name='res4c_branch2b_relu', data=scale4c_branch2b, act_type='relu')
        res4c_branch2c = mx.symbol.Convolution(name='res4c_branch2c', data=res4c_branch2b_relu, num_filter=1024,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4c_branch2c = mx.symbol.BatchNorm(name='bn4c_branch2c', data=res4c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4c_branch2c = bn4c_branch2c
        res4c = mx.symbol.broadcast_add(name='res4c', *[res4b_relu, scale4c_branch2c])
        res4c_relu = mx.symbol.Activation(name='res4c_relu', data=res4c, act_type='relu')
        res4d_branch2a = mx.symbol.Convolution(name='res4d_branch2a', data=res4c_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4d_branch2a = mx.symbol.BatchNorm(name='bn4d_branch2a', data=res4d_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4d_branch2a = bn4d_branch2a
        res4d_branch2a_relu = mx.symbol.Activation(name='res4d_branch2a_relu', data=scale4d_branch2a, act_type='relu')
        res4d_branch2b = mx.symbol.Convolution(name='res4d_branch2b', data=res4d_branch2a_relu, num_filter=256,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4d_branch2b = mx.symbol.BatchNorm(name='bn4d_branch2b', data=res4d_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4d_branch2b = bn4d_branch2b
        res4d_branch2b_relu = mx.symbol.Activation(name='res4d_branch2b_relu', data=scale4d_branch2b, act_type='relu')
        res4d_branch2c = mx.symbol.Convolution(name='res4d_branch2c', data=res4d_branch2b_relu, num_filter=1024,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4d_branch2c = mx.symbol.BatchNorm(name='bn4d_branch2c', data=res4d_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4d_branch2c = bn4d_branch2c
        res4d = mx.symbol.broadcast_add(name='res4d', *[res4c_relu, scale4d_branch2c])
        res4d_relu = mx.symbol.Activation(name='res4d_relu', data=res4d, act_type='relu')
        res4e_branch2a = mx.symbol.Convolution(name='res4e_branch2a', data=res4d_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4e_branch2a = mx.symbol.BatchNorm(name='bn4e_branch2a', data=res4e_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4e_branch2a = bn4e_branch2a
        res4e_branch2a_relu = mx.symbol.Activation(name='res4e_branch2a_relu', data=scale4e_branch2a, act_type='relu')
        res4e_branch2b = mx.symbol.Convolution(name='res4e_branch2b', data=res4e_branch2a_relu, num_filter=256,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4e_branch2b = mx.symbol.BatchNorm(name='bn4e_branch2b', data=res4e_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4e_branch2b = bn4e_branch2b
        res4e_branch2b_relu = mx.symbol.Activation(name='res4e_branch2b_relu', data=scale4e_branch2b, act_type='relu')
        res4e_branch2c = mx.symbol.Convolution(name='res4e_branch2c', data=res4e_branch2b_relu, num_filter=1024,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4e_branch2c = mx.symbol.BatchNorm(name='bn4e_branch2c', data=res4e_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4e_branch2c = bn4e_branch2c
        res4e = mx.symbol.broadcast_add(name='res4e', *[res4d_relu, scale4e_branch2c])
        res4e_relu = mx.symbol.Activation(name='res4e_relu', data=res4e, act_type='relu')
        res4f_branch2a = mx.symbol.Convolution(name='res4f_branch2a', data=res4e_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4f_branch2a = mx.symbol.BatchNorm(name='bn4f_branch2a', data=res4f_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4f_branch2a = bn4f_branch2a
        res4f_branch2a_relu = mx.symbol.Activation(name='res4f_branch2a_relu', data=scale4f_branch2a, act_type='relu')
        res4f_branch2b = mx.symbol.Convolution(name='res4f_branch2b', data=res4f_branch2a_relu, num_filter=256,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4f_branch2b = mx.symbol.BatchNorm(name='bn4f_branch2b', data=res4f_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4f_branch2b = bn4f_branch2b
        res4f_branch2b_relu = mx.symbol.Activation(name='res4f_branch2b_relu', data=scale4f_branch2b, act_type='relu')
        res4f_branch2c = mx.symbol.Convolution(name='res4f_branch2c', data=res4f_branch2b_relu, num_filter=1024,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4f_branch2c = mx.symbol.BatchNorm(name='bn4f_branch2c', data=res4f_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4f_branch2c = bn4f_branch2c
        res4f = mx.symbol.broadcast_add(name='res4f', *[res4e_relu, scale4f_branch2c])
        res4f_relu = mx.symbol.Activation(name='res4f_relu', data=res4f, act_type='relu')
        return res4f_relu

    def get_resnet_v1_conv5(self, conv_feat):
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=conv_feat, num_filter=2048, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=conv_feat, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
        res5a_branch2b_offset = mx.symbol.Convolution(name='res5a_branch2b_offset', data=res5a_branch2a_relu,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1),
                                                      dilate=(2, 2), cudnn_off=True)
        res5a_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5a_branch2b', data=res5a_branch2a_relu,
                                                                 offset=res5a_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3),
                                                                 num_deformable_group=4,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')
        res5b_branch2b_offset = mx.symbol.Convolution(name='res5b_branch2b_offset', data=res5b_branch2a_relu,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1),
                                                      dilate=(2, 2), cudnn_off=True)
        res5b_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5b_branch2b', data=res5b_branch2a_relu,
                                                                 offset=res5b_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3),
                                                                 num_deformable_group=4,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')
        res5c_branch2b_offset = mx.symbol.Convolution(name='res5c_branch2b_offset', data=res5c_branch2a_relu,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1),
                                                      dilate=(2, 2), cudnn_off=True)
        res5c_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5c_branch2b', data=res5c_branch2a_relu,
                                                                 offset=res5c_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3),
                                                                 num_deformable_group=4,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')
        return res5c_relu

    def get_symbol_rcnn(self, cfg, is_train=True):
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
            gt_boxes = mx.sym.Variable(name='gt_boxes')
            valid_ranges = mx.sym.Variable(name='valid_ranges')
            im_info = mx.sym.Variable(name='im_info')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name='im_info')
            im_ids = mx.sym.Variable(name='im_ids')

        res4 = self.resnetc4(data, fp16=cfg.TRAIN.fp16)
        res5 = self.resnetc5(res4, deform=True)
        if cfg.TRAIN.fp16:
            res4 = mx.sym.Cast(data=res4, dtype=np.float32)
            res5 = mx.sym.Cast(data=res5, dtype=np.float32)

        ############## SSH Res4 Backwards for M1 ###########
        # Reduce res5 dimension
        res5_reduced = mx.sym.Convolution(data=res5, kernel=(1, 1), num_filter=256, name="conv_res5_reduced")
        res5_reduced_relu = mx.sym.Activation(data=res5_reduced, act_type='relu', name='conv_res5_reduced_relu')
        # Upsample res5
        res5_upsampled = mx.sym.UpSampling(data=res5_reduced_relu, scale=2.0, num_filter=256, sample_type='bilinear',
                                           name='up_res5')
        # Reduce res4 dimension
        res4_reduced = mx.sym.Convolution(data=res4, kernel=(1, 1), num_filter=256, name="conv_res4_reduced")
        res4_reduced_relu = mx.sym.Activation(data=res4_reduced, act_type='relu', name='conv_res4_reduced_relu')

        #Eltwise summation
        fuse_sum = mx.sym.ElementWiseSum(*[res5_upsampled, res4_reduced_relu], name='fuse_sum')

        # Final 3x3 Conv
        fuse = mx.sym.Convolution(data=fuse_sum, kernel=(3, 3), num_filter=256, name="conv_fusion")
        final_fuse = mx.sym.Activation(data=fuse, act_type='relu', name='conv_fusion_relu')

        ############## M3@SSH #############
        # Stride 2 convolution
        m3_3x3 = mx.symbol.Convolution(name='m3_3x3', data=res5, num_filter=256, pad=(1, 1),
                                               kernel=(3, 3), stride=(2, 2), no_bias=True)
        m3_3x3_relu = mx.sym.Activation(data=m3_3x3, act_type='relu', name='m3_3x3_relu')
        # Dim reduction
        res6_reduced = mx.symbol.Convolution(name='res6_reduced', data=m3_3x3_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1))
        res6_reduced_relu = mx.sym.Activation(data=res6_reduced, act_type='relu', name='conv_res6_reduced_relu')

        # 5x5
        m3_5x5 = mx.symbol.Convolution(name='m3_5x5', data=res6_reduced_relu, num_filter=128, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1))
        m3_5x5_relu = mx.sym.Activation(data=m3_5x5, act_type='relu', name='m3_5x5_relu')

        # 7x7
        m3_7x7_1 = mx.symbol.Convolution(name='m3_7x7_1', data=res6_reduced_relu, num_filter=128, pad=(1, 1),
                                       kernel=(3, 3), stride=(1, 1))
        m3_7x7_relu_1 = mx.sym.Activation(data=m3_7x7_1, act_type='relu', name='m3_7x7_relu_1')

        m3_7x7 = mx.symbol.Convolution(name='m3_7x7', data=m3_7x7_relu_1, num_filter=128, pad=(1, 1),
                                         kernel=(3, 3), stride=(1, 1))
        m3_7x7_relu = mx.sym.Activation(data=m3_7x7, act_type='relu', name='m3_7x7_relu')

        # Concat branches
        m3 = mx.symbol.Concat(*[m3_3x3_relu, m3_5x5_relu, m3_7x7_relu], name='cat_m3')

        # Put SSH heads
        m3_score = mx.sym.Convolution(
            data=m3, kernel=(1, 1), pad=(0, 0), num_filter=4, name="m3_score")
        m3_bbox_pred = mx.sym.Convolution(
            data=m3, kernel=(1, 1), pad=(0, 0), num_filter=8, name="m3_bbox_pred")
        m3_score_reshape = mx.sym.Reshape(data=m3_score, shape=(0, 2, -1, 0),
                                               name="m3_score_reshape")

        ############## M2@SSH #############
        # Stride 2 convolution
        m2_3x3 = mx.symbol.Convolution(name='m2_3x3', data=res5, num_filter=256, pad=(1, 1),
                                       kernel=(3, 3), stride=(1, 1), no_bias=True)
        m2_3x3_relu = mx.sym.Activation(data=m2_3x3, act_type='relu', name='m2_3x3_relu')
        # Dim reduction
        res5_reduced = mx.symbol.Convolution(name='res5_reduced', data=res5, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1))
        res5_reduced_relu = mx.sym.Activation(data=res5_reduced, act_type='relu', name='conv_res5_reduced_relu')

        # 5x5
        m2_5x5 = mx.symbol.Convolution(name='m2_5x5', data=res5_reduced_relu, num_filter=128, pad=(1, 1),
                                       kernel=(3, 3), stride=(1, 1))
        m2_5x5_relu = mx.sym.Activation(data=m2_5x5, act_type='relu', name='m2_5x5_relu')

        # 7x7
        m2_7x7_1 = mx.symbol.Convolution(name='m2_7x7_1', data=res5_reduced_relu, num_filter=128, pad=(1, 1),
                                         kernel=(3, 3), stride=(1, 1))
        m2_7x7_relu_1 = mx.sym.Activation(data=m2_7x7_1, act_type='relu', name='m2_7x7_relu_1')

        m2_7x7 = mx.symbol.Convolution(name='m2_7x7', data=m2_7x7_relu_1, num_filter=128, pad=(1, 1),
                                       kernel=(3, 3), stride=(1, 1))
        m2_7x7_relu = mx.sym.Activation(data=m2_7x7, act_type='relu', name='m2_7x7_relu')

        # Concat branches
        m2 = mx.symbol.Concat(*[m2_3x3_relu, m2_5x5_relu, m2_7x7_relu], name='cat_m2')

        # Put SSH heads
        m2_score = mx.sym.Convolution(
            data=m2, kernel=(1, 1), pad=(0, 0), num_filter=4, name="m2_score")
        m2_bbox_pred = mx.sym.Convolution(
            data=m2, kernel=(1, 1), pad=(0, 0), num_filter=8, name="m2_bbox_pred")
        m2_score_reshape = mx.sym.Reshape(data=m2_score, shape=(0, 2, -1, 0),
                                          name="m2_score_reshape")

        ############## M1@SSH #############
        # Stride 2 convolution
        m1_3x3 = mx.symbol.Convolution(name='m1_3x3', data=final_fuse, num_filter=128, pad=(1, 1),
                                       kernel=(3, 3), stride=(1, 1), no_bias=True)
        m1_3x3_relu = mx.sym.Activation(data=m1_3x3, act_type='relu', name='m1_3x3_relu')
        # Dim reduction
        res4_reduced = mx.symbol.Convolution(name='res4_reduced', data=final_fuse, num_filter=64, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1))
        res4_reduced_relu = mx.sym.Activation(data=res4_reduced, act_type='relu', name='conv_res4_reduced_relu')

        # 5x5
        m1_5x5 = mx.symbol.Convolution(name='m1_5x5', data=res4_reduced_relu, num_filter=64, pad=(1, 1),
                                       kernel=(3, 3), stride=(1, 1))
        m1_5x5_relu = mx.sym.Activation(data=m1_5x5, act_type='relu', name='m1_5x5_relu')

        # 7x7
        m1_7x7_1 = mx.symbol.Convolution(name='m1_7x7_1', data=res4_reduced_relu, num_filter=64, pad=(1, 1),
                                         kernel=(3, 3), stride=(1, 1))
        m1_7x7_relu_1 = mx.sym.Activation(data=m1_7x7_1, act_type='relu', name='m1_7x7_relu_1')

        m1_7x7 = mx.symbol.Convolution(name='m1_7x7', data=m1_7x7_relu_1, num_filter=64, pad=(1, 1),
                                       kernel=(3, 3), stride=(1, 1))
        m1_7x7_relu = mx.sym.Activation(data=m1_7x7, act_type='relu', name='m1_7x7_relu')

        # Concat branches
        m1 = mx.symbol.Concat(*[m1_3x3_relu, m1_5x5_relu, m1_7x7_relu], name='cat_m1')

        # Put SSH heads
        m1_score = mx.sym.Convolution(
            data=m1, kernel=(1, 1), pad=(0, 0), num_filter=4, name="m1_score")
        m1_bbox_pred = mx.sym.Convolution(
            data=m1, kernel=(1, 1), pad=(0, 0), num_filter=8, name="m1_bbox_pred")
        m1_score_reshape = mx.sym.Reshape(data=m1_score, shape=(0, 2, -1, 0),
                                          name="m1_score_reshape")

        if is_train:
            m3_rois, m3_label, m3_bbox_target, m3_bbox_weight = mx.sym.MultiProposalTarget(cls_prob=m3_score_reshape,
                                                                               bbox_pred=m3_bbox_pred, im_info=im_info,
                                                                               gt_boxes=gt_boxes,
                                                                               valid_ranges=valid_ranges,
                                                                               batch_size=cfg.TRAIN.BATCH_IMAGES,
                                                                               scales=(16.0, 32.0),
                                                                               ratios=(1.0,),
                                                                               feature_stride=32.0,
                                                                               name='m3_target')

            m2_rois, m2_label, m2_bbox_target, m2_bbox_weight = mx.sym.MultiProposalTarget(cls_prob=m2_score_reshape,
                                                                                           bbox_pred=m2_bbox_pred,
                                                                                           im_info=im_info,
                                                                                           gt_boxes=gt_boxes,
                                                                                           valid_ranges=valid_ranges,
                                                                                           batch_size=cfg.TRAIN.BATCH_IMAGES,
                                                                                           scales=(4.0, 8.0),
                                                                                           ratios=(1.0,),
                                                                                           feature_stride=16.0,
                                                                                           name='m2_target')

            m1_rois, m1_label, m1_bbox_target, m1_bbox_weight = mx.sym.MultiProposalTarget(cls_prob=m1_score_reshape,
                                                                                           bbox_pred=m1_bbox_pred,
                                                                                           im_info=im_info,
                                                                                           gt_boxes=gt_boxes,
                                                                                           valid_ranges=valid_ranges,
                                                                                           batch_size=cfg.TRAIN.BATCH_IMAGES,
                                                                                           scales=(1.0, 2.0),
                                                                                           ratios=(1.0,),
                                                                                           feature_stride=8.0,
                                                                                           name='m1_target')

            labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                           num_reg_classes=num_reg_classes,
                                                           roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                           cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                           bbox_targets=bbox_target, bbox_weights=bbox_weight)

            if cfg.TRAIN.fp16 == True:
                grad_scale = float(cfg.TRAIN.scale)
            else:
                grad_scale = 1.0

            m3_label = mx.symbol.Reshape(data=m3_label, shape=(-1,), name='m3_label_reshape')
            m2_label = mx.symbol.Reshape(data=m2_label, shape=(-1,), name='m2_label_reshape')
            m1_label = mx.symbol.Reshape(data=m1_label, shape=(-1,), name='m1_label_reshape')
            m3_bbox_loss_ = m3_bbox_weight * mx.sym.smooth_l1(name='m3_bbox_loss_', scalar=1.0,
                                                                data=(m3_bbox_pred - m3_bbox_target))
            m3_bbox_loss = mx.sym.MakeLoss(name='m3_bbox_loss', data=m3_bbox_loss_,
                                            grad_scale=3 * grad_scale / float(
                                                cfg.TRAIN.BATCH_IMAGES * cfg.TRAIN.RPN_BATCH_SIZE))
            m2_bbox_loss_ = m2_bbox_weight * mx.sym.smooth_l1(name='m2_bbox_loss_', scalar=1.0,
                                                              data=(m2_bbox_pred - m2_bbox_target))
            m2_bbox_loss = mx.sym.MakeLoss(name='m2_bbox_loss', data=m2_bbox_loss_,
                                           grad_scale=3 * grad_scale / float(
                                               cfg.TRAIN.BATCH_IMAGES * cfg.TRAIN.RPN_BATCH_SIZE))
            m1_bbox_loss_ = m1_bbox_weight * mx.sym.smooth_l1(name='m1_bbox_loss_', scalar=1.0,
                                                              data=(m1_bbox_pred - m1_bbox_target))
            m1_bbox_loss = mx.sym.MakeLoss(name='m1_bbox_loss', data=m1_bbox_loss_,
                                           grad_scale=3 * grad_scale / float(
                                               cfg.TRAIN.BATCH_IMAGES * cfg.TRAIN.RPN_BATCH_SIZE))



            # m3_offset_t = mx.contrib.sym.DeformablePSROIPooling(name='m3_offset_t', data=m3, rois=m3_rois,
            #                                                  group_size=1, pooled_size=7,
            #                                                  sample_per_part=4, no_trans=True, part_size=7,
            #                                                  output_dim=512, spatial_scale=0.03125)
            # m3_offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
            #
            #
            # offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")
            #
            # deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool',
            #                                                             data=conv_new_1_relu, rois=rois,
            #                                                             trans=offset_reshape, group_size=1,
            #                                                             pooled_size=7, sample_per_part=4,
            #                                                             no_trans=False, part_size=7, output_dim=256,
            #                                                             spatial_scale=0.0625, trans_std=0.1)
            # # 2 fc
            # fc_new_1 = mx.sym.FullyConnected(name='fc_new_1', data=deformable_roi_pool, num_hidden=1024)
            # fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')
            #
            # fc_new_2 = mx.sym.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
            # fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')
            # num_classes = 81
            # num_reg_classes = 1
            # cls_score = mx.sym.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
            # bbox_pred = mx.sym.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)

            if cfg.TRAIN.fp16 == True:
                grad_scale = float(cfg.TRAIN.scale)
            else:
                grad_scale = 1.0

            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid',
                                            use_ignore=True, ignore_label=-1,
                                            grad_scale=grad_scale)
            bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                        data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=grad_scale / (188.0 * 16.0))
            rcnn_label = label

            # reshape output
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')

            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=1.0,
                                                                data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                            grad_scale=3 * grad_scale / float(
                                                cfg.TRAIN.BATCH_IMAGES * cfg.TRAIN.RPN_BATCH_SIZE))
            if not cfg.TRAIN.visualize:
                group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
            else:
                group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label),
                                      mx.sym.BlockGrad(bbox_pred), mx.sym.BlockGrad(rois),
                                      mx.sym.BlockGrad(rcnn_label)])
        else:
            # ROI Proposal
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')

            rois, _ = mx.sym.MultiProposal(cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info,
                                           name='rois', batch_size=self.test_nbatch,
                                           rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N,
                                           rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                                           rpn_min_size=cfg.TEST.RPN_MIN_SIZE,
                                           threshold=cfg.TEST.RPN_NMS_THRESH,
                                           feature_stride=cfg.network.RPN_FEAT_STRIDE,
                                           ratios=tuple(cfg.network.ANCHOR_RATIOS),
                                           scales=tuple(cfg.network.ANCHOR_SCALES))

            offset_t = mx.contrib.sym.DeformablePSROIPooling(name='offset_t', data=conv_new_1_relu, rois=rois,
                                                             group_size=1, pooled_size=7,
                                                             sample_per_part=4, no_trans=True, part_size=7,
                                                             output_dim=256, spatial_scale=0.0625)
            offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
            offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")

            deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool',
                                                                        data=conv_new_1_relu, rois=rois,
                                                                        trans=offset_reshape, group_size=1,
                                                                        pooled_size=7, sample_per_part=4,
                                                                        no_trans=False, part_size=7, output_dim=256,
                                                                        spatial_scale=0.0625, trans_std=0.1)
            # 2 fc
            fc_new_1 = mx.sym.FullyConnected(name='fc_new_1', data=deformable_roi_pool, num_hidden=1024)
            fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

            fc_new_2 = mx.sym.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
            fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')
            num_classes = 81
            num_reg_classes = 1
            cls_score = mx.sym.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
            bbox_pred = mx.sym.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(self.test_nbatch, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(self.test_nbatch, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')

            group = mx.sym.Group([rois, cls_prob, bbox_pred, im_ids])

        self.sym = group
        return group

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        arg_params['res5a_branch2b_offset_weight'] = mx.nd.zeros(
            shape=self.arg_shape_dict['res5a_branch2b_offset_weight'])
        arg_params['res5a_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5a_branch2b_offset_bias'])
        arg_params['res5b_branch2b_offset_weight'] = mx.nd.zeros(
            shape=self.arg_shape_dict['res5b_branch2b_offset_weight'])
        arg_params['res5b_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5b_branch2b_offset_bias'])
        arg_params['res5c_branch2b_offset_weight'] = mx.nd.zeros(
            shape=self.arg_shape_dict['res5c_branch2b_offset_weight'])
        arg_params['res5c_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5c_branch2b_offset_bias'])
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_weight'])
        arg_params['offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_bias'])
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

