# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Guodong Zhang
# --------------------------------------------------------

import cPickle
import mxnet as mx
from symbol import Symbol
from operator_py.box_annotator_ohem import *

def checkpoint_callback(bbox_param_names, prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        weight = arg[bbox_param_names[0]]
        bias = arg[bbox_param_names[1]]
        arg[bbox_param_names[0]+'_test'] = (weight.T * mx.nd.array(stds)).T
        arg[bbox_param_names[1]+'_test'] =bias * mx.nd.array(stds) + mx.nd.array(means)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop(bbox_param_names[0]+'_test')
        arg.pop(bbox_param_names[1]+'_test')
    return _callback


class resnet_v1_50_msfast(Symbol):
    def __init__(self,n_proposals):
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

    def get_resnet_v1_conv4(self, data1, data2):
        
        conv1_weight = mx.sym.Variable("conv1_weight")
        conv1 = mx.symbol.Convolution(name='conv1', data=data1, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2), weight=conv1_weight,
                                      no_bias=True)
        f_conv1 = mx.symbol.Convolution(name='f_conv1', data=data2, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2), weight=conv1_weight,
                                        no_bias=True)
        
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn_conv1 = mx.symbol.BatchNorm(name='f_bn_conv1', data=f_conv1, use_global_stats=True, fix_gamma=False, eps=self.eps)

        scale_conv1 = bn_conv1
        f_scale_conv1 = f_bn_conv1

        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        f_conv1_relu = mx.symbol.Activation(name='f_conv1_relu', data=f_scale_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                                  stride=(2, 2), pool_type='max')
        f_pool1 = mx.symbol.Pooling(name='f_pool1', data=f_conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                                  stride=(2, 2), pool_type='max')
        res2a_branch1_weight = mx.sym.Variable('res2a_branch1_weight')
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0), kernel=(1, 1), weight=res2a_branch1_weight,
                                              stride=(1, 1), no_bias=True)
        f_res2a_branch1 = mx.symbol.Convolution(name='f_res2a_branch1', data=f_pool1, num_filter=256, pad=(0, 0), kernel=(1, 1), weight=res2a_branch1_weight,
                                              stride=(1, 1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn2a_branch1 = mx.symbol.BatchNorm(name='f_bn2a_branch1', data=f_res2a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale2a_branch1 = bn2a_branch1
        f_scale2a_branch1 = f_bn2a_branch1
        res2a_branch2a_weight = mx.sym.Variable('res2a_branch2a_weight')
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0), kernel=(1, 1), weight=res2a_branch2a_weight,
                                               stride=(1, 1), no_bias=True)
        f_res2a_branch2a = mx.symbol.Convolution(name='f_res2a_branch2a', data=f_pool1, num_filter=64, pad=(0, 0), kernel=(1, 1), weight=res2a_branch2a_weight,
                                               stride=(1, 1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn2a_branch2a = mx.symbol.BatchNorm(name='f_bn2a_branch2a', data=f_res2a_branch2a, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale2a_branch2a = bn2a_branch2a
        f_scale2a_branch2a = f_bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        f_res2a_branch2a_relu = mx.symbol.Activation(name='f_res2a_branch2a_relu', data=f_scale2a_branch2a, act_type='relu')
        res2a_branch2b_weight = mx.sym.Variable('res2a_branch2b_weight')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64, pad=(1, 1), weight=res2a_branch2b_weight,
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res2a_branch2b = mx.symbol.Convolution(name='f_res2a_branch2b', data=f_res2a_branch2a_relu, num_filter=64, pad=(1, 1), weight=res2a_branch2b_weight,
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn2a_branch2b = mx.symbol.BatchNorm(name='f_bn2a_branch2b', data=f_res2a_branch2b, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale2a_branch2b = bn2a_branch2b
        f_scale2a_branch2b = f_bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        f_res2a_branch2b_relu = mx.symbol.Activation(name='f_res2a_branch2b_relu', data=f_scale2a_branch2b, act_type='relu')
        res2a_branch2c_weight = mx.sym.Variable('res2a_branch2c_weight')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256, pad=(0, 0), weight=res2a_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res2a_branch2c = mx.symbol.Convolution(name='f_res2a_branch2c', data=f_res2a_branch2b_relu, num_filter=256, pad=(0, 0), weight=res2a_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn2a_branch2c = mx.symbol.BatchNorm(name='f_bn2a_branch2c', data=f_res2a_branch2c, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale2a_branch2c = bn2a_branch2c
        f_scale2a_branch2c = f_bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
        f_res2a = mx.symbol.broadcast_add(name='f_res2a', *[f_scale2a_branch1, f_scale2a_branch2c])
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a, act_type='relu')
        f_res2a_relu = mx.symbol.Activation(name='f_res2a_relu', data=f_res2a, act_type='relu')
        res2b_branch2a_weight = mx.sym.Variable('res2b_branch2a_weight')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0), weight=res2b_branch2a_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res2b_branch2a = mx.symbol.Convolution(name='f_res2b_branch2a', data=f_res2a_relu, num_filter=64, pad=(0, 0), weight=res2b_branch2a_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn2b_branch2a = mx.symbol.BatchNorm(name='f_bn2b_branch2a', data=f_res2b_branch2a, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale2b_branch2a = bn2b_branch2a
        f_scale2b_branch2a = f_bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
        f_res2b_branch2a_relu = mx.symbol.Activation(name='f_res2b_branch2a_relu', data=f_scale2b_branch2a, act_type='relu')
        res2b_branch2b_weight = mx.sym.Variable('res2b_branch2b_weight')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64, pad=(1, 1), weight=res2b_branch2b_weight,
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res2b_branch2b = mx.symbol.Convolution(name='f_res2b_branch2b', data=f_res2b_branch2a_relu, num_filter=64, pad=(1, 1), weight=res2b_branch2b_weight,
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn2b_branch2b = mx.symbol.BatchNorm(name='f_bn2b_branch2b', data=f_res2b_branch2b, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale2b_branch2b = bn2b_branch2b
        f_scale2b_branch2b = f_bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        f_res2b_branch2b_relu = mx.symbol.Activation(name='f_res2b_branch2b_relu', data=f_scale2b_branch2b, act_type='relu')
        res2b_branch2c_weight = mx.sym.Variable('res2b_branch2c_weight')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256, pad=(0, 0), weight=res2b_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res2b_branch2c = mx.symbol.Convolution(name='f_res2b_branch2c', data=f_res2b_branch2b_relu, num_filter=256, pad=(0, 0), weight=res2b_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn2b_branch2c = mx.symbol.BatchNorm(name='f_bn2b_branch2c', data=f_res2b_branch2c, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale2b_branch2c = bn2b_branch2c
        f_scale2b_branch2c = f_bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
        f_res2b = mx.symbol.broadcast_add(name='f_res2b', *[f_res2a_relu, f_scale2b_branch2c])
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b, act_type='relu')
        f_res2b_relu = mx.symbol.Activation(name='f_res2b_relu', data=f_res2b, act_type='relu')
        res2c_branch2a_weight = mx.sym.Variable('res2c_branch2a_weight')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0), weight=res2c_branch2a_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res2c_branch2a = mx.symbol.Convolution(name='f_res2c_branch2a', data=f_res2b_relu, num_filter=64, pad=(0, 0), weight=res2c_branch2a_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn2c_branch2a = mx.symbol.BatchNorm(name='f_bn2c_branch2a', data=f_res2c_branch2a, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale2c_branch2a = bn2c_branch2a
        f_scale2c_branch2a = f_bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
        f_res2c_branch2a_relu = mx.symbol.Activation(name='f_res2c_branch2a_relu', data=f_scale2c_branch2a, act_type='relu')
        res2c_branch2b_weight = mx.sym.Variable('res2c_branch2b_weight')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64, pad=(1, 1), weight=res2c_branch2b_weight,
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res2c_branch2b = mx.symbol.Convolution(name='f_res2c_branch2b', data=f_res2c_branch2a_relu, num_filter=64, pad=(1, 1), weight=res2c_branch2b_weight,
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn2c_branch2b = mx.symbol.BatchNorm(name='f_bn2c_branch2b', data=f_res2c_branch2b, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale2c_branch2b = bn2c_branch2b
        f_scale2c_branch2b = f_bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        f_res2c_branch2b_relu = mx.symbol.Activation(name='f_res2c_branch2b_relu', data=f_scale2c_branch2b, act_type='relu')
        res2c_branch2c_weight = mx.sym.Variable('res2c_branch2c_weight')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256, pad=(0, 0), weight=res2c_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res2c_branch2c = mx.symbol.Convolution(name='f_res2c_branch2c', data=f_res2c_branch2b_relu, num_filter=256, pad=(0, 0), weight=res2c_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn2c_branch2c = mx.symbol.BatchNorm(name='f_bn2c_branch2c', data=f_res2c_branch2c, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale2c_branch2c = bn2c_branch2c
        f_scale2c_branch2c = f_bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        f_res2c = mx.symbol.broadcast_add(name='f_res2c', *[f_res2b_relu, f_scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')
        f_res2c_relu = mx.symbol.Activation(name='f_res2c_relu', data=f_res2c, act_type='relu')
        res3a_branch1_weight = mx.sym.Variable('res3a_branch1_weight')
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0), weight=res3a_branch1_weight,
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        f_res3a_branch1 = mx.symbol.Convolution(name='f_res3a_branch1', data=f_res2c_relu, num_filter=512, pad=(0, 0), weight=res3a_branch1_weight,
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn3a_branch1 = mx.symbol.BatchNorm(name='f_bn3a_branch1', data=f_res3a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale3a_branch1 = bn3a_branch1
        f_scale3a_branch1 = f_bn3a_branch1
        res3a_branch2a_weight = mx.sym.Variable('res3a_branch2a_weight')
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu, num_filter=128, pad=(0, 0), weight=res3a_branch2a_weight,
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        f_res3a_branch2a = mx.symbol.Convolution(name='f_res3a_branch2a', data=f_res2c_relu, num_filter=128, pad=(0, 0), weight=res3a_branch2a_weight,
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn3a_branch2a = mx.symbol.BatchNorm(name='f_bn3a_branch2a', data=f_res3a_branch2a, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale3a_branch2a = bn3a_branch2a
        f_scale3a_branch2a = f_bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        f_res3a_branch2a_relu = mx.symbol.Activation(name='f_res3a_branch2a_relu', data=f_scale3a_branch2a, act_type='relu')
        res3a_branch2b_weight = mx.sym.Variable('res3a_branch2b_weight')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128, pad=(1, 1), weight=res3a_branch2b_weight,
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res3a_branch2b = mx.symbol.Convolution(name='f_res3a_branch2b', data=f_res3a_branch2a_relu, num_filter=128, pad=(1, 1), weight=res3a_branch2b_weight,
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn3a_branch2b = mx.symbol.BatchNorm(name='f_bn3a_branch2b', data=f_res3a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2b = bn3a_branch2b
        f_scale3a_branch2b = f_bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        f_res3a_branch2b_relu = mx.symbol.Activation(name='f_res3a_branch2b_relu', data=f_scale3a_branch2b, act_type='relu')
        res3a_branch2c_weight = mx.sym.Variable('res3a_branch2c_weight')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512, pad=(0, 0), weight=res3a_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res3a_branch2c = mx.symbol.Convolution(name='f_res3a_branch2c', data=f_res3a_branch2b_relu, num_filter=512, pad=(0, 0), weight=res3a_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn3a_branch2c = mx.symbol.BatchNorm(name='f_bn3a_branch2c', data=f_res3a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2c = bn3a_branch2c
        f_scale3a_branch2c = f_bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        f_res3a = mx.symbol.broadcast_add(name='f_res3a', *[f_scale3a_branch1, f_scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        f_res3a_relu = mx.symbol.Activation(name='f_res3a_relu', data=f_res3a, act_type='relu')
        res3b_branch2a_weight = mx.sym.Variable('res3b_branch2a_weight')
        res3b_branch2a = mx.symbol.Convolution(name='res3b_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0), weight=res3b_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res3b_branch2a = mx.symbol.Convolution(name='f_res3b_branch2a', data=f_res3a_relu, num_filter=128, pad=(0, 0), weight=res3b_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b_branch2a = mx.symbol.BatchNorm(name='bn3b_branch2a', data=res3b_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn3b_branch2a = mx.symbol.BatchNorm(name='f_bn3b_branch2a', data=f_res3b_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b_branch2a = bn3b_branch2a
        f_scale3b_branch2a = f_bn3b_branch2a
        res3b_branch2a_relu = mx.symbol.Activation(name='res3b_branch2a_relu', data=scale3b_branch2a, act_type='relu')
        f_res3b_branch2a_relu = mx.symbol.Activation(name='f_res3b_branch2a_relu', data=f_scale3b_branch2a, act_type='relu')
        res3b_branch2b_weight = mx.sym.Variable('res3b_branch2b_weight')
        res3b_branch2b = mx.symbol.Convolution(name='res3b_branch2b', data=res3b_branch2a_relu, num_filter=128, weight=res3b_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res3b_branch2b = mx.symbol.Convolution(name='f_res3b_branch2b', data=f_res3b_branch2a_relu, num_filter=128, weight=res3b_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b_branch2b = mx.symbol.BatchNorm(name='bn3b_branch2b', data=res3b_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn3b_branch2b = mx.symbol.BatchNorm(name='f_bn3b_branch2b', data=f_res3b_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b_branch2b = bn3b_branch2b
        f_scale3b_branch2b = f_bn3b_branch2b
        res3b_branch2b_relu = mx.symbol.Activation(name='res3b_branch2b_relu', data=scale3b_branch2b, act_type='relu')
        f_res3b_branch2b_relu = mx.symbol.Activation(name='f_res3b_branch2b_relu', data=f_scale3b_branch2b, act_type='relu')
        res3b_branch2c_weight = mx.sym.Variable('res3b_branch2c_weight')
        res3b_branch2c = mx.symbol.Convolution(name='res3b_branch2c', data=res3b_branch2b_relu, num_filter=512, weight=res3b_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res3b_branch2c = mx.symbol.Convolution(name='f_res3b_branch2c', data=f_res3b_branch2b_relu, num_filter=512, weight=res3b_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b_branch2c = mx.symbol.BatchNorm(name='bn3b_branch2c', data=res3b_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn3b_branch2c = mx.symbol.BatchNorm(name='f_bn3b_branch2c', data=f_res3b_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b_branch2c = bn3b_branch2c
        f_scale3b_branch2c = f_bn3b_branch2c
        res3b = mx.symbol.broadcast_add(name='res3b', *[res3a_relu, scale3b_branch2c])
        f_res3b = mx.symbol.broadcast_add(name='f_res3b', *[f_res3a_relu, f_scale3b_branch2c])
        res3b_relu = mx.symbol.Activation(name='res3b_relu', data=res3b, act_type='relu')
        f_res3b_relu = mx.symbol.Activation(name='f_res3b_relu', data=f_res3b, act_type='relu')
        res3c_branch2a_weight = mx.sym.Variable('res3c_branch2a_weight')
        res3c_branch2a = mx.symbol.Convolution(name='res3c_branch2a', data=res3b_relu, num_filter=128, pad=(0, 0), weight=res3c_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res3c_branch2a = mx.symbol.Convolution(name='f_res3c_branch2a', data=f_res3b_relu, num_filter=128, pad=(0, 0), weight=res3c_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3c_branch2a = mx.symbol.BatchNorm(name='bn3c_branch2a', data=res3c_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn3c_branch2a = mx.symbol.BatchNorm(name='f_bn3c_branch2a', data=f_res3c_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3c_branch2a = bn3c_branch2a
        f_scale3c_branch2a = f_bn3c_branch2a
        res3c_branch2a_relu = mx.symbol.Activation(name='res3c_branch2a_relu', data=scale3c_branch2a, act_type='relu')
        f_res3c_branch2a_relu = mx.symbol.Activation(name='f_res3c_branch2a_relu', data=f_scale3c_branch2a, act_type='relu')
        res3c_branch2b_weight = mx.sym.Variable('res3c_branch2b_weight')
        res3c_branch2b = mx.symbol.Convolution(name='res3c_branch2b', data=res3c_branch2a_relu, num_filter=128, weight=res3c_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res3c_branch2b = mx.symbol.Convolution(name='f_res3c_branch2b', data=f_res3c_branch2a_relu, num_filter=128, weight=res3c_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3c_branch2b = mx.symbol.BatchNorm(name='bn3c_branch2b', data=res3c_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn3c_branch2b = mx.symbol.BatchNorm(name='f_bn3c_branch2b', data=f_res3c_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3c_branch2b = bn3c_branch2b
        f_scale3c_branch2b = f_bn3c_branch2b
        res3c_branch2b_relu = mx.symbol.Activation(name='res3c_branch2b_relu', data=scale3c_branch2b, act_type='relu')
        f_res3c_branch2b_relu = mx.symbol.Activation(name='f_res3c_branch2b_relu', data=f_scale3c_branch2b, act_type='relu')
        res3c_branch2c_weight = mx.sym.Variable('res3c_branch2c_weight')
        res3c_branch2c = mx.symbol.Convolution(name='res3c_branch2c', data=res3c_branch2b_relu, num_filter=512, weight=res3c_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res3c_branch2c = mx.symbol.Convolution(name='f_res3c_branch2c', data=f_res3c_branch2b_relu, num_filter=512, weight=res3c_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3c_branch2c = mx.symbol.BatchNorm(name='bn3c_branch2c', data=res3c_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn3c_branch2c = mx.symbol.BatchNorm(name='f_bn3c_branch2c', data=f_res3c_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3c_branch2c = bn3c_branch2c
        f_scale3c_branch2c = f_bn3c_branch2c
        res3c = mx.symbol.broadcast_add(name='res3c', *[res3b_relu, scale3c_branch2c])
        f_res3c = mx.symbol.broadcast_add(name='f_res3c', *[f_res3b_relu, f_scale3c_branch2c])
        res3c_relu = mx.symbol.Activation(name='res3c_relu', data=res3c, act_type='relu')
        f_res3c_relu = mx.symbol.Activation(name='f_res3c_relu', data=f_res3c, act_type='relu')
        res3d_branch2a_weight = mx.sym.Variable('res3d_branch2a_weight')
        res3d_branch2a = mx.symbol.Convolution(name='res3d_branch2a', data=res3c_relu, num_filter=128, pad=(0, 0), weight=res3d_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res3d_branch2a = mx.symbol.Convolution(name='f_res3d_branch2a', data=f_res3c_relu, num_filter=128, pad=(0, 0), weight=res3d_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3d_branch2a = mx.symbol.BatchNorm(name='bn3d_branch2a', data=res3d_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn3d_branch2a = mx.symbol.BatchNorm(name='f_bn3d_branch2a', data=f_res3d_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3d_branch2a = bn3d_branch2a
        f_scale3d_branch2a = f_bn3d_branch2a
        res3d_branch2a_relu = mx.symbol.Activation(name='res3d_branch2a_relu', data=scale3d_branch2a, act_type='relu')
        f_res3d_branch2a_relu = mx.symbol.Activation(name='f_res3d_branch2a_relu', data=f_scale3d_branch2a, act_type='relu')
        res3d_branch2b_weight = mx.sym.Variable('res3d_branch2b_weight')
        res3d_branch2b = mx.symbol.Convolution(name='res3d_branch2b', data=res3d_branch2a_relu, num_filter=128, weight=res3d_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res3d_branch2b = mx.symbol.Convolution(name='f_res3d_branch2b', data=f_res3d_branch2a_relu, num_filter=128, weight=res3d_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3d_branch2b = mx.symbol.BatchNorm(name='bn3d_branch2b', data=res3d_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn3d_branch2b = mx.symbol.BatchNorm(name='f_bn3d_branch2b', data=f_res3d_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3d_branch2b = bn3d_branch2b
        f_scale3d_branch2b = f_bn3d_branch2b
        res3d_branch2b_relu = mx.symbol.Activation(name='res3d_branch2b_relu', data=scale3d_branch2b, act_type='relu')
        f_res3d_branch2b_relu = mx.symbol.Activation(name='f_res3d_branch2b_relu', data=f_scale3d_branch2b, act_type='relu')
        res3d_branch2c_weight = mx.sym.Variable('res3d_branch2c_weight')
        res3d_branch2c = mx.symbol.Convolution(name='res3d_branch2c', data=res3d_branch2b_relu, num_filter=512, weight=res3d_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res3d_branch2c = mx.symbol.Convolution(name='f_res3d_branch2c', data=f_res3d_branch2b_relu, num_filter=512, weight=res3d_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3d_branch2c = mx.symbol.BatchNorm(name='bn3d_branch2c', data=res3d_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn3d_branch2c = mx.symbol.BatchNorm(name='f_bn3d_branch2c', data=f_res3d_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3d_branch2c = bn3d_branch2c
        f_scale3d_branch2c = f_bn3d_branch2c
        res3d = mx.symbol.broadcast_add(name='res3d', *[res3c_relu, scale3d_branch2c])
        f_res3d = mx.symbol.broadcast_add(name='f_res3d', *[f_res3c_relu, f_scale3d_branch2c])
        res3d_relu = mx.symbol.Activation(name='res3d_relu', data=res3d, act_type='relu')
        f_res3d_relu = mx.symbol.Activation(name='f_res3d_relu', data=f_res3d, act_type='relu')
        res4a_branch1_weight = mx.sym.Variable('res4a_branch1_weight')
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3d_relu, num_filter=1024, pad=(0, 0), weight=res4a_branch1_weight,
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        f_res4a_branch1 = mx.symbol.Convolution(name='f_res4a_branch1', data=f_res3d_relu, num_filter=1024, pad=(0, 0), weight=res4a_branch1_weight,
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn4a_branch1 = mx.symbol.BatchNorm(name='f_bn4a_branch1', data=f_res4a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale4a_branch1 = bn4a_branch1
        f_scale4a_branch1 = f_bn4a_branch1
        res4a_branch2a_weight = mx.sym.Variable('res4a_branch2a_weight')
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3d_relu, num_filter=256, pad=(0, 0), weight=res4a_branch2a_weight,
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        f_res4a_branch2a = mx.symbol.Convolution(name='f_res4a_branch2a', data=f_res3d_relu, num_filter=256, pad=(0, 0), weight=res4a_branch2a_weight,
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn4a_branch2a = mx.symbol.BatchNorm(name='f_bn4a_branch2a', data=f_res4a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2a = bn4a_branch2a
        f_scale4a_branch2a = f_bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        f_res4a_branch2a_relu = mx.symbol.Activation(name='f_res4a_branch2a_relu', data=f_scale4a_branch2a, act_type='relu')
        res4a_branch2b_weight = mx.sym.Variable('res4a_branch2b_weight')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256, pad=(1, 1), weight=res4a_branch2b_weight,
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res4a_branch2b = mx.symbol.Convolution(name='f_res4a_branch2b', data=f_res4a_branch2a_relu, num_filter=256, pad=(1, 1), weight=res4a_branch2b_weight,
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn4a_branch2b = mx.symbol.BatchNorm(name='f_bn4a_branch2b', data=f_res4a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2b = bn4a_branch2b
        f_scale4a_branch2b = f_bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        f_res4a_branch2b_relu = mx.symbol.Activation(name='f_res4a_branch2b_relu', data=f_scale4a_branch2b, act_type='relu')
        res4a_branch2c_weight = mx.sym.Variable('res4a_branch2c_weight')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024, pad=(0, 0), weight=res4a_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res4a_branch2c = mx.symbol.Convolution(name='f_res4a_branch2c', data=f_res4a_branch2b_relu, num_filter=1024, pad=(0, 0), weight=res4a_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn4a_branch2c = mx.symbol.BatchNorm(name='f_bn4a_branch2c', data=f_res4a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2c = bn4a_branch2c
        f_scale4a_branch2c = f_bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        f_res4a = mx.symbol.broadcast_add(name='f_res4a', *[f_scale4a_branch1, f_scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        f_res4a_relu = mx.symbol.Activation(name='f_res4a_relu', data=f_res4a, act_type='relu')
        res4b_branch2a_weight = mx.sym.Variable('res4b_branch2a_weight')
        res4b_branch2a = mx.symbol.Convolution(name='res4b_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0), weight=res4b_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res4b_branch2a = mx.symbol.Convolution(name='f_res4b_branch2a', data=f_res4a_relu, num_filter=256, pad=(0, 0), weight=res4b_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b_branch2a = mx.symbol.BatchNorm(name='bn4b_branch2a', data=res4b_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4b_branch2a = mx.symbol.BatchNorm(name='f_bn4b_branch2a', data=f_res4b_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b_branch2a = bn4b_branch2a
        f_scale4b_branch2a = f_bn4b_branch2a
        res4b_branch2a_relu = mx.symbol.Activation(name='res4b_branch2a_relu', data=scale4b_branch2a, act_type='relu')
        f_res4b_branch2a_relu = mx.symbol.Activation(name='f_res4b_branch2a_relu', data=f_scale4b_branch2a, act_type='relu')
        res4b_branch2b_weight = mx.sym.Variable('res4b_branch2b_weight')
        res4b_branch2b = mx.symbol.Convolution(name='res4b_branch2b', data=res4b_branch2a_relu, num_filter=256, weight=res4b_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res4b_branch2b = mx.symbol.Convolution(name='f_res4b_branch2b', data=f_res4b_branch2a_relu, num_filter=256, weight=res4b_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b_branch2b = mx.symbol.BatchNorm(name='bn4b_branch2b', data=res4b_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4b_branch2b = mx.symbol.BatchNorm(name='f_bn4b_branch2b', data=f_res4b_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b_branch2b = bn4b_branch2b
        f_scale4b_branch2b = f_bn4b_branch2b
        res4b_branch2b_relu = mx.symbol.Activation(name='res4b_branch2b_relu', data=scale4b_branch2b, act_type='relu')
        f_res4b_branch2b_relu = mx.symbol.Activation(name='f_res4b_branch2b_relu', data=f_scale4b_branch2b, act_type='relu')
        res4b_branch2c_weight = mx.sym.Variable('res4b_branch2c_weight')
        res4b_branch2c = mx.symbol.Convolution(name='res4b_branch2c', data=res4b_branch2b_relu, num_filter=1024, weight=res4b_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res4b_branch2c = mx.symbol.Convolution(name='f_res4b_branch2c', data=f_res4b_branch2b_relu, num_filter=1024, weight=res4b_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b_branch2c = mx.symbol.BatchNorm(name='bn4b_branch2c', data=res4b_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4b_branch2c = mx.symbol.BatchNorm(name='f_bn4b_branch2c', data=f_res4b_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b_branch2c = bn4b_branch2c
        f_scale4b_branch2c = f_bn4b_branch2c
        res4b = mx.symbol.broadcast_add(name='res4b', *[res4a_relu, scale4b_branch2c])
        f_res4b = mx.symbol.broadcast_add(name='f_res4b', *[f_res4a_relu, f_scale4b_branch2c])
        res4b_relu = mx.symbol.Activation(name='res4b_relu', data=res4b, act_type='relu')
        f_res4b_relu = mx.symbol.Activation(name='f_res4b_relu', data=f_res4b, act_type='relu')
        res4c_branch2a_weight = mx.sym.Variable('res4c_branch2a_weight')
        res4c_branch2a = mx.symbol.Convolution(name='res4c_branch2a', data=res4b_relu, num_filter=256, pad=(0, 0), weight=res4c_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res4c_branch2a = mx.symbol.Convolution(name='f_res4c_branch2a', data=f_res4b_relu, num_filter=256, pad=(0, 0), weight=res4c_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4c_branch2a = mx.symbol.BatchNorm(name='bn4c_branch2a', data=res4c_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4c_branch2a = mx.symbol.BatchNorm(name='f_bn4c_branch2a', data=f_res4c_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4c_branch2a = bn4c_branch2a
        f_scale4c_branch2a = f_bn4c_branch2a
        res4c_branch2a_relu = mx.symbol.Activation(name='res4c_branch2a_relu', data=scale4c_branch2a, act_type='relu')
        f_res4c_branch2a_relu = mx.symbol.Activation(name='f_res4c_branch2a_relu', data=f_scale4c_branch2a, act_type='relu')
        res4c_branch2b_weight = mx.sym.Variable('res4c_branch2b_weight')
        res4c_branch2b = mx.symbol.Convolution(name='res4c_branch2b', data=res4c_branch2a_relu, num_filter=256, weight=res4c_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res4c_branch2b = mx.symbol.Convolution(name='f_res4c_branch2b', data=f_res4c_branch2a_relu, num_filter=256, weight=res4c_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4c_branch2b = mx.symbol.BatchNorm(name='bn4c_branch2b', data=res4c_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4c_branch2b = mx.symbol.BatchNorm(name='f_bn4c_branch2b', data=f_res4c_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4c_branch2b = bn4c_branch2b
        f_scale4c_branch2b = f_bn4c_branch2b
        res4c_branch2b_relu = mx.symbol.Activation(name='res4c_branch2b_relu', data=scale4c_branch2b, act_type='relu')
        f_res4c_branch2b_relu = mx.symbol.Activation(name='f_res4c_branch2b_relu', data=f_scale4c_branch2b, act_type='relu')
        res4c_branch2c_weight = mx.sym.Variable('res4c_branch2c_weight')
        res4c_branch2c = mx.symbol.Convolution(name='res4c_branch2c', data=res4c_branch2b_relu, num_filter=1024, weight=res4c_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res4c_branch2c = mx.symbol.Convolution(name='f_res4c_branch2c', data=f_res4c_branch2b_relu, num_filter=1024, weight=res4c_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4c_branch2c = mx.symbol.BatchNorm(name='bn4c_branch2c', data=res4c_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4c_branch2c = mx.symbol.BatchNorm(name='f_bn4c_branch2c', data=f_res4c_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4c_branch2c = bn4c_branch2c
        f_scale4c_branch2c = f_bn4c_branch2c
        res4c = mx.symbol.broadcast_add(name='res4c', *[res4b_relu, scale4c_branch2c])
        f_res4c = mx.symbol.broadcast_add(name='f_res4c', *[f_res4b_relu, f_scale4c_branch2c])
        res4c_relu = mx.symbol.Activation(name='res4c_relu', data=res4c, act_type='relu')
        f_res4c_relu = mx.symbol.Activation(name='f_res4c_relu', data=f_res4c, act_type='relu')
        res4d_branch2a_weight = mx.sym.Variable('res4d_branch2a_weight')
        res4d_branch2a = mx.symbol.Convolution(name='res4d_branch2a', data=res4c_relu, num_filter=256, pad=(0, 0), weight=res4d_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res4d_branch2a = mx.symbol.Convolution(name='f_res4d_branch2a', data=f_res4c_relu, num_filter=256, pad=(0, 0), weight=res4d_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4d_branch2a = mx.symbol.BatchNorm(name='bn4d_branch2a', data=res4d_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4d_branch2a = mx.symbol.BatchNorm(name='f_bn4d_branch2a', data=f_res4d_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4d_branch2a = bn4d_branch2a
        f_scale4d_branch2a = f_bn4d_branch2a
        res4d_branch2a_relu = mx.symbol.Activation(name='res4d_branch2a_relu', data=scale4d_branch2a, act_type='relu')
        f_res4d_branch2a_relu = mx.symbol.Activation(name='f_res4d_branch2a_relu', data=f_scale4d_branch2a, act_type='relu')
        res4d_branch2b_weight = mx.sym.Variable('res4d_branch2b_weight')
        res4d_branch2b = mx.symbol.Convolution(name='res4d_branch2b', data=res4d_branch2a_relu, num_filter=256, weight=res4d_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res4d_branch2b = mx.symbol.Convolution(name='f_res4d_branch2b', data=f_res4d_branch2a_relu, num_filter=256, weight=res4d_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4d_branch2b = mx.symbol.BatchNorm(name='bn4d_branch2b', data=res4d_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4d_branch2b = mx.symbol.BatchNorm(name='f_bn4d_branch2b', data=f_res4d_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4d_branch2b = bn4d_branch2b
        f_scale4d_branch2b = f_bn4d_branch2b
        res4d_branch2b_relu = mx.symbol.Activation(name='res4d_branch2b_relu', data=scale4d_branch2b, act_type='relu')
        f_res4d_branch2b_relu = mx.symbol.Activation(name='f_res4d_branch2b_relu', data=f_scale4d_branch2b, act_type='relu')
        res4d_branch2c_weight = mx.sym.Variable('res4d_branch2c_weight')
        res4d_branch2c = mx.symbol.Convolution(name='res4d_branch2c', data=res4d_branch2b_relu, num_filter=1024, weight=res4d_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res4d_branch2c = mx.symbol.Convolution(name='f_res4d_branch2c', data=f_res4d_branch2b_relu, num_filter=1024, weight=res4d_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4d_branch2c = mx.symbol.BatchNorm(name='bn4d_branch2c', data=res4d_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4d_branch2c = mx.symbol.BatchNorm(name='f_bn4d_branch2c', data=f_res4d_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4d_branch2c = bn4d_branch2c
        f_scale4d_branch2c = f_bn4d_branch2c
        res4d = mx.symbol.broadcast_add(name='res4d', *[res4c_relu, scale4d_branch2c])
        f_res4d = mx.symbol.broadcast_add(name='f_res4d', *[f_res4c_relu, f_scale4d_branch2c])
        res4d_relu = mx.symbol.Activation(name='res4d_relu', data=res4d, act_type='relu')
        f_res4d_relu = mx.symbol.Activation(name='f_res4d_relu', data=f_res4d, act_type='relu')
        res4e_branch2a_weight = mx.sym.Variable('res4e_branch2a_weight')
        res4e_branch2a = mx.symbol.Convolution(name='res4e_branch2a', data=res4d_relu, num_filter=256, pad=(0, 0), weight=res4e_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res4e_branch2a = mx.symbol.Convolution(name='f_res4e_branch2a', data=f_res4d_relu, num_filter=256, pad=(0, 0), weight=res4e_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4e_branch2a = mx.symbol.BatchNorm(name='bn4e_branch2a', data=res4e_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4e_branch2a = mx.symbol.BatchNorm(name='f_bn4e_branch2a', data=f_res4e_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4e_branch2a = bn4e_branch2a
        f_scale4e_branch2a = f_bn4e_branch2a
        res4e_branch2a_relu = mx.symbol.Activation(name='res4e_branch2a_relu', data=scale4e_branch2a, act_type='relu')
        f_res4e_branch2a_relu = mx.symbol.Activation(name='f_res4e_branch2a_relu', data=f_scale4e_branch2a, act_type='relu')
        res4e_branch2b_weight = mx.sym.Variable('res4e_branch2b_weight')
        res4e_branch2b = mx.symbol.Convolution(name='res4e_branch2b', data=res4e_branch2a_relu, num_filter=256, weight=res4e_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res4e_branch2b = mx.symbol.Convolution(name='f_res4e_branch2b', data=f_res4e_branch2a_relu, num_filter=256, weight=res4e_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4e_branch2b = mx.symbol.BatchNorm(name='bn4e_branch2b', data=res4e_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4e_branch2b = mx.symbol.BatchNorm(name='f_bn4e_branch2b', data=f_res4e_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4e_branch2b = bn4e_branch2b
        f_scale4e_branch2b = f_bn4e_branch2b
        res4e_branch2b_relu = mx.symbol.Activation(name='res4e_branch2b_relu', data=scale4e_branch2b, act_type='relu')
        f_res4e_branch2b_relu = mx.symbol.Activation(name='f_res4e_branch2b_relu', data=f_scale4e_branch2b, act_type='relu')
        res4e_branch2c_weight = mx.sym.Variable('res4e_branch2c_weight')
        res4e_branch2c = mx.symbol.Convolution(name='res4e_branch2c', data=res4e_branch2b_relu, num_filter=1024, weight=res4e_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res4e_branch2c = mx.symbol.Convolution(name='f_res4e_branch2c', data=f_res4e_branch2b_relu, num_filter=1024, weight=res4e_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4e_branch2c = mx.symbol.BatchNorm(name='bn4e_branch2c', data=res4e_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4e_branch2c = mx.symbol.BatchNorm(name='f_bn4e_branch2c', data=f_res4e_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4e_branch2c = bn4e_branch2c
        f_scale4e_branch2c = f_bn4e_branch2c
        res4e = mx.symbol.broadcast_add(name='res4e', *[res4d_relu, scale4e_branch2c])
        f_res4e = mx.symbol.broadcast_add(name='f_res4e', *[f_res4d_relu, f_scale4e_branch2c])
        res4e_relu = mx.symbol.Activation(name='res4e_relu', data=res4e, act_type='relu')
        f_res4e_relu = mx.symbol.Activation(name='f_res4e_relu', data=f_res4e, act_type='relu')
        res4f_branch2a_weight = mx.sym.Variable('res4f_branch2a_weight')
        res4f_branch2a = mx.symbol.Convolution(name='res4f_branch2a', data=res4e_relu, num_filter=256, pad=(0, 0), weight=res4f_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res4f_branch2a = mx.symbol.Convolution(name='f_res4f_branch2a', data=f_res4e_relu, num_filter=256, pad=(0, 0), weight=res4f_branch2a_weight,
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4f_branch2a = mx.symbol.BatchNorm(name='bn4f_branch2a', data=res4f_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4f_branch2a = mx.symbol.BatchNorm(name='f_bn4f_branch2a', data=f_res4f_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4f_branch2a = bn4f_branch2a
        f_scale4f_branch2a = f_bn4f_branch2a
        res4f_branch2a_relu = mx.symbol.Activation(name='res4f_branch2a_relu', data=scale4f_branch2a, act_type='relu')
        f_res4f_branch2a_relu = mx.symbol.Activation(name='f_res4f_branch2a_relu', data=f_scale4f_branch2a, act_type='relu')
        res4f_branch2b_weight = mx.sym.Variable('res4f_branch2b_weight')
        res4f_branch2b = mx.symbol.Convolution(name='res4f_branch2b', data=res4f_branch2a_relu, num_filter=256, weight=res4f_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        f_res4f_branch2b = mx.symbol.Convolution(name='f_res4f_branch2b', data=f_res4f_branch2a_relu, num_filter=256, weight=res4f_branch2b_weight,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4f_branch2b = mx.symbol.BatchNorm(name='bn4f_branch2b', data=res4f_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4f_branch2b = mx.symbol.BatchNorm(name='f_bn4f_branch2b', data=f_res4f_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4f_branch2b = bn4f_branch2b
        f_scale4f_branch2b = f_bn4f_branch2b
        res4f_branch2b_relu = mx.symbol.Activation(name='res4f_branch2b_relu', data=scale4f_branch2b, act_type='relu')
        f_res4f_branch2b_relu = mx.symbol.Activation(name='f_res4f_branch2b_relu', data=f_scale4f_branch2b, act_type='relu')
        res4f_branch2c_weight = mx.sym.Variable('res4f_branch2c_weight')
        res4f_branch2c = mx.symbol.Convolution(name='res4f_branch2c', data=res4f_branch2b_relu, num_filter=1024, weight=res4f_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res4f_branch2c = mx.symbol.Convolution(name='f_res4f_branch2c', data=f_res4f_branch2b_relu, num_filter=1024, weight=res4f_branch2c_weight,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4f_branch2c = mx.symbol.BatchNorm(name='bn4f_branch2c', data=res4f_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        f_bn4f_branch2c = mx.symbol.BatchNorm(name='f_bn4f_branch2c', data=f_res4f_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4f_branch2c = bn4f_branch2c
        f_scale4f_branch2c = f_bn4f_branch2c
        res4f = mx.symbol.broadcast_add(name='res4f', *[res4e_relu, scale4f_branch2c])
        f_res4f = mx.symbol.broadcast_add(name='f_res4f', *[f_res4e_relu, f_scale4f_branch2c])
        res4f_relu = mx.symbol.Activation(name='res4f_relu', data=res4f, act_type='relu')
        f_res4f_relu = mx.symbol.Activation(name='f_res4f_relu', data=f_res4f, act_type='relu')
        return res4f_relu, f_res4f_relu

    def get_resnet_v1_conv5(self, conv_feat, f_conv_feat):
        res5a_branch1_weight = mx.sym.Variable("res5a_branch1_weight")
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=conv_feat, num_filter=2048, pad=(0, 0), weight=res5a_branch1_weight,
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res5a_branch1 = mx.symbol.Convolution(name='f_res5a_branch1', data=f_conv_feat, num_filter=2048, pad=(0, 0), weight = res5a_branch1_weight,
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)

        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        f_bn5a_branch1 = mx.symbol.BatchNorm(name='f_bn5a_branch1', data=f_res5a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale5a_branch1 = bn5a_branch1
        f_scale5a_branch1 = f_bn5a_branch1

        res5a_branch2a_weight = mx.sym.Variable("res5a_branch2a_weight")
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=conv_feat, num_filter=512, pad=(0, 0), weight=res5a_branch2a_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res5a_branch2a = mx.symbol.Convolution(name='f_res5a_branch2a', data=f_conv_feat, num_filter=512, pad=(0, 0), weight=res5a_branch2a_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)

        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn5a_branch2a = mx.symbol.BatchNorm(name='f_bn5a_branch2a', data=f_res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2a = bn5a_branch2a
        f_scale5a_branch2a = f_bn5a_branch2a

        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
        f_res5a_branch2a_relu = mx.symbol.Activation(name='f_res5a_branch2a_relu', data=f_scale5a_branch2a, act_type='relu')
        res5a_branch2b_offset_weight = mx.sym.Variable("res5a_branch2b_offset_weight")
        res5a_branch2b_offset_bias = mx.sym.Variable("res5a_branch2b_offset_bias")
        
        res5a_branch2b_offset = mx.symbol.Convolution(name='res5a_branch2b_offset', data = res5a_branch2a_relu, weight=res5a_branch2b_offset_weight, bias=res5a_branch2b_offset_bias,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        f_res5a_branch2b_offset = mx.symbol.Convolution(name='f_res5a_branch2b_offset', data = f_res5a_branch2a_relu, weight=res5a_branch2b_offset_weight, bias=res5a_branch2b_offset_bias,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)

        res5a_branch2b_weight = mx.sym.Variable("res5a_branch2b_weight")
        res5a_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5a_branch2b', data=res5a_branch2a_relu, offset=res5a_branch2b_offset, 
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4, weight=res5a_branch2b_weight,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        f_res5a_branch2b = mx.contrib.symbol.DeformableConvolution(name='f_res5a_branch2b', data=f_res5a_branch2a_relu, offset=f_res5a_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4, weight=res5a_branch2b_weight,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)

        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn5a_branch2b = mx.symbol.BatchNorm(name='f_bn5a_branch2b', data=f_res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2b = bn5a_branch2b
        f_scale5a_branch2b = f_bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        f_res5a_branch2b_relu = mx.symbol.Activation(name='f_res5a_branch2b_relu', data=f_scale5a_branch2b, act_type='relu')

        res5a_branch2c_weight = mx.sym.Variable('res5a_branch2c_weight')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048, pad=(0, 0), weight=res5a_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res5a_branch2c = mx.symbol.Convolution(name='f_res5a_branch2c', data=f_res5a_branch2b_relu, num_filter=2048, pad=(0, 0), weight=res5a_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)

        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn5a_branch2c = mx.symbol.BatchNorm(name='f_bn5a_branch2c', data=f_res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2c = bn5a_branch2c
        f_scale5a_branch2c = f_bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        f_res5a = mx.symbol.broadcast_add(name='f_res5a', *[f_scale5a_branch1, f_scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')
        f_res5a_relu = mx.symbol.Activation(name='f_res5a_relu', data=f_res5a, act_type='relu')
        res5b_branch2a_weight = mx.sym.Variable('res5b_branch2a_weight')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0), weight=res5b_branch2a_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res5b_branch2a = mx.symbol.Convolution(name='f_res5b_branch2a', data=f_res5a_relu, num_filter=512, pad=(0, 0), weight=res5b_branch2a_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)

        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn5b_branch2a = mx.symbol.BatchNorm(name='f_bn5b_branch2a', data=f_res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2a = bn5b_branch2a
        f_scale5b_branch2a = f_bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')
        f_res5b_branch2a_relu = mx.symbol.Activation(name='f_res5b_branch2a_relu', data=f_scale5b_branch2a, act_type='relu')
        res5b_branch2b_offset_weight = mx.sym.Variable('res5b_branch2b_offset_weight')
        res5b_branch2b_offset_bias = mx.sym.Variable('res5b_branch2b_offset_bias')
        res5b_branch2b_offset = mx.symbol.Convolution(name='res5b_branch2b_offset', data = res5b_branch2a_relu, weight = res5b_branch2b_offset_weight, bias = res5b_branch2b_offset_bias,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        f_res5b_branch2b_offset = mx.symbol.Convolution(name='f_res5b_branch2b_offset', data = f_res5b_branch2a_relu, weight = res5b_branch2b_offset_weight, bias = res5b_branch2b_offset_bias,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)

        res5b_branch2b_weight = mx.sym.Variable("res5b_branch2b_weight")
        res5b_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5b_branch2b', data=res5b_branch2a_relu, offset=res5b_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4, weight=res5b_branch2b_weight,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        f_res5b_branch2b = mx.contrib.symbol.DeformableConvolution(name='f_res5b_branch2b', data=f_res5b_branch2a_relu, offset=f_res5b_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4, weight=res5b_branch2b_weight,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)

        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn5b_branch2b = mx.symbol.BatchNorm(name='f_bn5b_branch2b', data=f_res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2b = bn5b_branch2b
        f_scale5b_branch2b = f_bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        f_res5b_branch2b_relu = mx.symbol.Activation(name='f_res5b_branch2b_relu', data=f_scale5b_branch2b, act_type='relu')
        
        res5b_branch2c_weight = mx.sym.Variable("res5b_branch2c_weight")
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048, pad=(0, 0), weight=res5b_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res5b_branch2c = mx.symbol.Convolution(name='f_res5b_branch2c', data=f_res5b_branch2b_relu, num_filter=2048, pad=(0, 0), weight=res5b_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)

        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn5b_branch2c = mx.symbol.BatchNorm(name='f_bn5b_branch2c', data=f_res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2c = bn5b_branch2c
        f_scale5b_branch2c = f_bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        f_res5b = mx.symbol.broadcast_add(name='f_res5b', *[f_res5a_relu, f_scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')
        f_res5b_relu = mx.symbol.Activation(name='f_res5b_relu', data=f_res5b, act_type='relu')
        res5c_branch2a_weight = mx.sym.Variable("res5c_branch2a_weight")
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0), weight=res5c_branch2a_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res5c_branch2a = mx.symbol.Convolution(name='f_res5c_branch2a', data=f_res5b_relu, num_filter=512, pad=(0, 0), weight=res5c_branch2a_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)

        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn5c_branch2a = mx.symbol.BatchNorm(name='f_bn5c_branch2a', data=f_res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2a = bn5c_branch2a
        f_scale5c_branch2a = f_bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')
        f_res5c_branch2a_relu = mx.symbol.Activation(name='f_res5c_branch2a_relu', data=f_scale5c_branch2a, act_type='relu')
        res5c_branch2b_offset_weight = mx.sym.Variable("res5c_branch2b_offset_weight")
        res5c_branch2b_offset_bias = mx.sym.Variable("res5c_branch2b_offset_bias")
        res5c_branch2b_offset = mx.symbol.Convolution(name='res5c_branch2b_offset', data = res5c_branch2a_relu, weight=res5c_branch2b_offset_weight, bias=res5c_branch2b_offset_bias, num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        f_res5c_branch2b_offset = mx.symbol.Convolution(name='f_res5c_branch2b_offset', data = f_res5c_branch2a_relu, weight=res5c_branch2b_offset_weight, bias=res5c_branch2b_offset_bias, num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)

        res5c_branch2b_weight = mx.sym.Variable("res5c_branch2b_weight")
        res5c_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5c_branch2b', data=res5c_branch2a_relu, offset=res5c_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4, weight=res5c_branch2b_weight,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        f_res5c_branch2b = mx.contrib.symbol.DeformableConvolution(name='f_res5c_branch2b', data=f_res5c_branch2a_relu, offset=f_res5c_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4, weight=res5c_branch2b_weight,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)

        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn5c_branch2b = mx.symbol.BatchNorm(name='f_bn5c_branch2b', data=f_res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2b = bn5c_branch2b
        f_scale5c_branch2b = f_bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        f_res5c_branch2b_relu = mx.symbol.Activation(name='f_res5c_branch2b_relu', data=f_scale5c_branch2b, act_type='relu')
        res5c_branch2c_weight = mx.sym.Variable("res5c_branch2c_weight")
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048, pad=(0, 0), weight=res5c_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        f_res5c_branch2c = mx.symbol.Convolution(name='f_res5c_branch2c', data=f_res5c_branch2b_relu, num_filter=2048, pad=(0, 0), weight=res5c_branch2c_weight,
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)

        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        f_bn5c_branch2c = mx.symbol.BatchNorm(name='f_bn5c_branch2c', data=f_res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2c = bn5c_branch2c
        f_scale5c_branch2c = f_bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        f_res5c = mx.symbol.broadcast_add(name='f_res5c', *[f_res5b_relu, f_scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')
        f_res5c_relu = mx.symbol.Activation(name='f_res5c_relu', data=f_res5c, act_type='relu')
        
        conv_new_1_weight = mx.sym.Variable("conv_new_1_weight")
        conv_new_1_bias = mx.sym.Variable("conv_new_1_bias")
        
        conv_new_1 = mx.sym.Convolution(data=res5c_relu, kernel=(1, 1), num_filter=256, name="conv_new_1", weight=conv_new_1_weight, bias=conv_new_1_bias)
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')

        f_conv_new_1 = mx.sym.Convolution(data=f_res5c_relu, kernel=(1, 1), num_filter=256, name="f_conv_new_1", weight=conv_new_1_weight, bias=conv_new_1_bias)
        f_conv_new_1_relu = mx.sym.Activation(data=f_conv_new_1, act_type='relu', name='f_conv_new_1_relu')
        return conv_new_1_relu, f_conv_new_1_relu


    def get_symbol_rcnn(self, cfg, is_train=True):
        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        # input init
        if is_train:
            datas = mx.symbol.Variable(name="data1")
            datal = mx.symbol.Variable(name="data2")
            rois = mx.symbol.Variable(name='rois')
            #roisl = mx.symbol.Variable(name='rois2')
            label = mx.symbol.Variable(name='label')
            bbox_target = mx.symbol.Variable(name='bbox_target')
            bbox_weight = mx.symbol.Variable(name='bbox_weight')
            # reshape input
            rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
            #roisl = mx.symbol.Reshape(data=roisl, shape=(-1, 5), name='roisl_reshape')
            #label = mx.symbol.Reshape(data=label, shape=(-1,), name='label_reshape')
            #bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_reg_classes), name='bbox_target_reshape')
            #bbox_weight = mx.symbol.Reshape(data=bbox_weight, shape=(-1, 4 * num_reg_classes), name='bbox_weight_reshape')
        else:
            datas = mx.sym.Variable(name="data1")
            datal = mx.sym.Variable(name="data2")
            roiss = mx.symbol.Variable(name='rois')
            roiss = mx.symbol.Reshape(data=roiss, shape=(-1, 5), name='roiss_reshape')
            roisl = mx.symbol.Variable(name='rois2')
            roisl = mx.symbol.Reshape(data=roisl, shape=(-1, 5), name='roisl_reshape')

        # shared convolutional layers
        conv_feats, conv_featl = self.get_resnet_v1_conv4(datas, datal)
        conv_new_1_relus, conv_new_1_relul = self.get_resnet_v1_conv5(conv_feats, conv_featl)
        #conv_new_1_relus = self.get_resnet_v1_conv5(conv_feats)
        #conv_new_1_relul = self.get_fresnet_v1_conv5(conv_featl)

        offset_ts = mx.contrib.sym.DeformablePSROIPooling(name='offset_t', data=conv_new_1_relus, rois=rois, group_size=1, pooled_size=7,
                                                         sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=0.0625)
        offset_weight = mx.sym.Variable("offset_weight", lr_mult=0.01)
        offset_bias = mx.sym.Variable("offset_bias", lr_mult=0.01)
        offsets = mx.sym.FullyConnected(name='offset', data=offset_ts, num_hidden=7 * 7 * 2, lr_mult=0.01, weight = offset_weight, bias=offset_bias)
        offset_reshapes = mx.sym.Reshape(data=offsets, shape=(-1, 2, 7, 7), name="offset_reshape")

        deformable_roi_pools = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool', data=conv_new_1_relus, rois=rois,
                                                                    trans=offset_reshapes, group_size=1, pooled_size=7, sample_per_part=4,
                                                                    no_trans=False, part_size=7, output_dim=256, spatial_scale=0.0625, trans_std=0.1)


        offset_tl = mx.contrib.sym.DeformablePSROIPooling(name='f_offset_t', data=conv_new_1_relul, rois=rois, group_size=1, pooled_size=7,
                                                          sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=0.0390625)
        offsetl = mx.sym.FullyConnected(name='f_offset', data=offset_tl, num_hidden=7 * 7 * 2, lr_mult=0.01, weight = offset_weight, bias=offset_bias)
        offset_reshapel = mx.sym.Reshape(data=offsetl, shape=(-1, 2, 7, 7), name="f_offset_reshape")

        deformable_roi_pooll = mx.contrib.sym.DeformablePSROIPooling(name='f_deformable_roi_pool', data=conv_new_1_relul, rois=rois,
                                                                    trans=offset_reshapel, group_size=1, pooled_size=7, sample_per_part=4,
                                                                    no_trans=False, part_size=7, output_dim=256, spatial_scale=0.0390625, trans_std=0.1)

        #deformable_roi_pool = mx.sym.Concat(*[deformable_roi_pools, deformable_roi_pooll], dim=1)
        #deformable_roi_pool = (deformable_roi_pools + deformable_roi_pooll) * 0.5
        deformable_roi_pool = mx.sym.maximum(deformable_roi_pools, deformable_roi_pooll)

        # 2 fc
        fc_new_1 = mx.sym.FullyConnected(name='fc_new_1', data=deformable_roi_pool, num_hidden=1024)
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

        fc_new_2 = mx.sym.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
        fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')

        # cls_score/bbox_pred
        cls_score = mx.sym.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
        bbox_pred = mx.sym.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1,self.n_proposals, num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, self.n_proposals,4 * num_reg_classes))

        if is_train:            
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes,
                                                               roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)


                cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
                labels_ohem = mx.symbol.Reshape(data=labels_ohem, shape=(-1,), name='label_reshape')
                bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))
                bbox_weights_ohem = mx.symbol.Reshape(data=bbox_weights_ohem, shape=(-1, 4 * num_reg_classes),
                                            name='bbox_weight_reshape')
                bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_reg_classes),
                                           name='bbox_target_reshape')

                #sample_weight = mx.sym.Custom(op_type='grad_scale', label=labels_ohem, assignment=assignment)
                #sample_weight = mx.sym.Reshape(data=sample_weight, shape=(-1, 1))
                #box_scale = mx.sym.tile(sample_weight, reps=(1, 8))
                #score_scale = mx.sym.tile(sample_weight, reps=(1, num_classes))

                #bbox_weights_ohem = bbox_weights_ohem * box_scale
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                                  data=(bbox_pred - bbox_target))
                    
                #cls_prob = score_scale * mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                                normalization='valid', use_ignore=True, ignore_label=-1, grad_scale=1.0)                

                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=1.0 / (128*cfg.TRAIN.BATCH_IMAGES))
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid',
                                                grad_scale=1.0)
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                            data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label), mx.sym.BlockGrad(bbox_pred)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([roiss, cls_prob, bbox_pred])

        self.sym = group
        return group

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        """arg_params['res5a_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5a_branch2b_offset_weight'])
        arg_params['res5a_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5a_branch2b_offset_bias'])
        arg_params['res5b_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5b_branch2b_offset_weight'])
        arg_params['res5b_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5b_branch2b_offset_bias'])
        arg_params['res5c_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5c_branch2b_offset_weight'])
        arg_params['res5c_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5c_branch2b_offset_bias'])
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
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
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])"""



