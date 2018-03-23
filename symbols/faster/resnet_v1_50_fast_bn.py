# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Guodong Zhang
# --------------------------------------------------------

import cPickle
import mxnet as mx
from lib.symbol import Symbol
#from operator_py.debug import *
from operator_py.box_annotator_ohem import *
from operator_py.debug_data import *



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

class resnet_v1_50_fast_bn(Symbol):
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
        #pred1 = mx.sym.Deconvolution(data=data, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=3, name='up')
        #pred1 = mx.symbol.UpSampling(data=data, scale=2, sample_type='bilinear', num_filter=3, num_args = 2, name='up', workspace = 8192)
        #conv1 = mx.symbol.Convolution(name='conv1', data=pred1, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                                  stride=(2, 2), pool_type='max')
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0), kernel=(1, 1),
                                              stride=(1, 1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0), kernel=(1, 1),
                                               stride=(1, 1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256, pad=(0, 0),
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
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256, pad=(0, 0),
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
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')

        
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1, momentum=0.9, fix_gamma=False, eps=self.eps)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        res3b_branch2a = mx.symbol.Convolution(name='res3b_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b_branch2a = mx.symbol.BatchNorm(name='bn3b_branch2a', data=res3b_branch2a, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale3b_branch2a = bn3b_branch2a
        res3b_branch2a_relu = mx.symbol.Activation(name='res3b_branch2a_relu', data=scale3b_branch2a, act_type='relu')
        res3b_branch2b = mx.symbol.Convolution(name='res3b_branch2b', data=res3b_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b_branch2b = mx.symbol.BatchNorm(name='bn3b_branch2b', data=res3b_branch2b, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale3b_branch2b = bn3b_branch2b
        res3b_branch2b_relu = mx.symbol.Activation(name='res3b_branch2b_relu', data=scale3b_branch2b, act_type='relu')
        res3b_branch2c = mx.symbol.Convolution(name='res3b_branch2c', data=res3b_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b_branch2c = mx.symbol.BatchNorm(name='bn3b_branch2c', data=res3b_branch2c, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale3b_branch2c = bn3b_branch2c
        res3b = mx.symbol.broadcast_add(name='res3b', *[res3a_relu, scale3b_branch2c])
        res3b_relu = mx.symbol.Activation(name='res3b_relu', data=res3b, act_type='relu')
        res3c_branch2a = mx.symbol.Convolution(name='res3c_branch2a', data=res3b_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3c_branch2a = mx.symbol.BatchNorm(name='bn3c_branch2a', data=res3c_branch2a, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale3c_branch2a = bn3c_branch2a
        res3c_branch2a_relu = mx.symbol.Activation(name='res3c_branch2a_relu', data=scale3c_branch2a, act_type='relu')
        res3c_branch2b = mx.symbol.Convolution(name='res3c_branch2b', data=res3c_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3c_branch2b = mx.symbol.BatchNorm(name='bn3c_branch2b', data=res3c_branch2b, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale3c_branch2b = bn3c_branch2b
        res3c_branch2b_relu = mx.symbol.Activation(name='res3c_branch2b_relu', data=scale3c_branch2b, act_type='relu')
        res3c_branch2c = mx.symbol.Convolution(name='res3c_branch2c', data=res3c_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3c_branch2c = mx.symbol.BatchNorm(name='bn3c_branch2c', data=res3c_branch2c, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale3c_branch2c = bn3c_branch2c
        res3c = mx.symbol.broadcast_add(name='res3c', *[res3b_relu, scale3c_branch2c])
        res3c_relu = mx.symbol.Activation(name='res3c_relu', data=res3c, act_type='relu')
        res3d_branch2a = mx.symbol.Convolution(name='res3d_branch2a', data=res3c_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3d_branch2a = mx.symbol.BatchNorm(name='bn3d_branch2a', data=res3d_branch2a, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale3d_branch2a = bn3d_branch2a
        res3d_branch2a_relu = mx.symbol.Activation(name='res3d_branch2a_relu', data=scale3d_branch2a, act_type='relu')
        res3d_branch2b = mx.symbol.Convolution(name='res3d_branch2b', data=res3d_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3d_branch2b = mx.symbol.BatchNorm(name='bn3d_branch2b', data=res3d_branch2b, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale3d_branch2b = bn3d_branch2b
        res3d_branch2b_relu = mx.symbol.Activation(name='res3d_branch2b_relu', data=scale3d_branch2b, act_type='relu')
        res3d_branch2c = mx.symbol.Convolution(name='res3d_branch2c', data=res3d_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3d_branch2c = mx.symbol.BatchNorm(name='bn3d_branch2c', data=res3d_branch2c, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale3d_branch2c = bn3d_branch2c
        res3d = mx.symbol.broadcast_add(name='res3d', *[res3c_relu, scale3d_branch2c])
        res3d_relu = mx.symbol.Activation(name='res3d_relu', data=res3d, act_type='relu')
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3d_relu, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1, momentum=0.9, fix_gamma=False, eps=self.eps)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3d_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        res4b_branch2a = mx.symbol.Convolution(name='res4b_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b_branch2a = mx.symbol.BatchNorm(name='bn4b_branch2a', data=res4b_branch2a, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4b_branch2a = bn4b_branch2a
        res4b_branch2a_relu = mx.symbol.Activation(name='res4b_branch2a_relu', data=scale4b_branch2a, act_type='relu')
        res4b_branch2b = mx.symbol.Convolution(name='res4b_branch2b', data=res4b_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b_branch2b = mx.symbol.BatchNorm(name='bn4b_branch2b', data=res4b_branch2b, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4b_branch2b = bn4b_branch2b
        res4b_branch2b_relu = mx.symbol.Activation(name='res4b_branch2b_relu', data=scale4b_branch2b, act_type='relu')
        res4b_branch2c = mx.symbol.Convolution(name='res4b_branch2c', data=res4b_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b_branch2c = mx.symbol.BatchNorm(name='bn4b_branch2c', data=res4b_branch2c, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4b_branch2c = bn4b_branch2c
        res4b = mx.symbol.broadcast_add(name='res4b', *[res4a_relu, scale4b_branch2c])
        res4b_relu = mx.symbol.Activation(name='res4b_relu', data=res4b, act_type='relu')
        res4c_branch2a = mx.symbol.Convolution(name='res4c_branch2a', data=res4b_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4c_branch2a = mx.symbol.BatchNorm(name='bn4c_branch2a', data=res4c_branch2a, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4c_branch2a = bn4c_branch2a
        res4c_branch2a_relu = mx.symbol.Activation(name='res4c_branch2a_relu', data=scale4c_branch2a, act_type='relu')
        res4c_branch2b = mx.symbol.Convolution(name='res4c_branch2b', data=res4c_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4c_branch2b = mx.symbol.BatchNorm(name='bn4c_branch2b', data=res4c_branch2b, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4c_branch2b = bn4c_branch2b
        res4c_branch2b_relu = mx.symbol.Activation(name='res4c_branch2b_relu', data=scale4c_branch2b, act_type='relu')
        res4c_branch2c = mx.symbol.Convolution(name='res4c_branch2c', data=res4c_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4c_branch2c = mx.symbol.BatchNorm(name='bn4c_branch2c', data=res4c_branch2c, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4c_branch2c = bn4c_branch2c
        res4c = mx.symbol.broadcast_add(name='res4c', *[res4b_relu, scale4c_branch2c])
        res4c_relu = mx.symbol.Activation(name='res4c_relu', data=res4c, act_type='relu')
        res4d_branch2a = mx.symbol.Convolution(name='res4d_branch2a', data=res4c_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4d_branch2a = mx.symbol.BatchNorm(name='bn4d_branch2a', data=res4d_branch2a, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4d_branch2a = bn4d_branch2a
        res4d_branch2a_relu = mx.symbol.Activation(name='res4d_branch2a_relu', data=scale4d_branch2a, act_type='relu')
        res4d_branch2b = mx.symbol.Convolution(name='res4d_branch2b', data=res4d_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4d_branch2b = mx.symbol.BatchNorm(name='bn4d_branch2b', data=res4d_branch2b, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4d_branch2b = bn4d_branch2b
        res4d_branch2b_relu = mx.symbol.Activation(name='res4d_branch2b_relu', data=scale4d_branch2b, act_type='relu')
        res4d_branch2c = mx.symbol.Convolution(name='res4d_branch2c', data=res4d_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4d_branch2c = mx.symbol.BatchNorm(name='bn4d_branch2c', data=res4d_branch2c, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4d_branch2c = bn4d_branch2c
        res4d = mx.symbol.broadcast_add(name='res4d', *[res4c_relu, scale4d_branch2c])
        res4d_relu = mx.symbol.Activation(name='res4d_relu', data=res4d, act_type='relu')
        res4e_branch2a = mx.symbol.Convolution(name='res4e_branch2a', data=res4d_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4e_branch2a = mx.symbol.BatchNorm(name='bn4e_branch2a', data=res4e_branch2a, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4e_branch2a = bn4e_branch2a
        res4e_branch2a_relu = mx.symbol.Activation(name='res4e_branch2a_relu', data=scale4e_branch2a, act_type='relu')
        res4e_branch2b = mx.symbol.Convolution(name='res4e_branch2b', data=res4e_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4e_branch2b = mx.symbol.BatchNorm(name='bn4e_branch2b', data=res4e_branch2b, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4e_branch2b = bn4e_branch2b
        res4e_branch2b_relu = mx.symbol.Activation(name='res4e_branch2b_relu', data=scale4e_branch2b, act_type='relu')
        res4e_branch2c = mx.symbol.Convolution(name='res4e_branch2c', data=res4e_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4e_branch2c = mx.symbol.BatchNorm(name='bn4e_branch2c', data=res4e_branch2c, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4e_branch2c = bn4e_branch2c
        res4e = mx.symbol.broadcast_add(name='res4e', *[res4d_relu, scale4e_branch2c])
        res4e_relu = mx.symbol.Activation(name='res4e_relu', data=res4e, act_type='relu')
        res4f_branch2a = mx.symbol.Convolution(name='res4f_branch2a', data=res4e_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4f_branch2a = mx.symbol.BatchNorm(name='bn4f_branch2a', data=res4f_branch2a, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4f_branch2a = bn4f_branch2a
        res4f_branch2a_relu = mx.symbol.Activation(name='res4f_branch2a_relu', data=scale4f_branch2a, act_type='relu')
        res4f_branch2b = mx.symbol.Convolution(name='res4f_branch2b', data=res4f_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4f_branch2b = mx.symbol.BatchNorm(name='bn4f_branch2b', data=res4f_branch2b, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4f_branch2b = bn4f_branch2b
        res4f_branch2b_relu = mx.symbol.Activation(name='res4f_branch2b_relu', data=scale4f_branch2b, act_type='relu')
        res4f_branch2c = mx.symbol.Convolution(name='res4f_branch2c', data=res4f_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4f_branch2c = mx.symbol.BatchNorm(name='bn4f_branch2c', data=res4f_branch2c, momentum=0.9,
                                             fix_gamma=False, eps=self.eps)
        scale4f_branch2c = bn4f_branch2c
        res4f = mx.symbol.broadcast_add(name='res4f', *[res4e_relu, scale4f_branch2c])
        res4f_relu = mx.symbol.Activation(name='res4f_relu', data=res4f, act_type='relu')
        return res4f_relu

    def get_resnet_v1_conv5(self, conv_feat):
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=conv_feat, num_filter=2048, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1, momentum=0.9, fix_gamma=False, eps=self.eps)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=conv_feat, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
        res5a_branch2b_offset = mx.symbol.Convolution(name='res5a_branch2b_offset', data = res5a_branch2a_relu,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        res5a_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5a_branch2b', data=res5a_branch2a_relu, offset=res5a_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')
        res5b_branch2b_offset = mx.symbol.Convolution(name='res5b_branch2b_offset', data = res5b_branch2a_relu,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        res5b_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5b_branch2b', data=res5b_branch2a_relu, offset=res5b_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')
        res5c_branch2b_offset = mx.symbol.Convolution(name='res5c_branch2b_offset', data = res5c_branch2a_relu,
                                                      num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        res5c_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5c_branch2b', data=res5c_branch2a_relu, offset=res5c_branch2b_offset,
                                                                 num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
                                                                 stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c, momentum=0.9,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')
        return res5c_relu


    def get_symbol_rcnn(self, cfg, is_train=True):
        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        # input init
        if is_train:
            data = mx.symbol.Variable(name="data")
            rois = mx.symbol.Variable(name='rois')
            label = mx.symbol.Variable(name='label')
            bbox_target = mx.symbol.Variable(name='bbox_target')
            bbox_weight = mx.symbol.Variable(name='bbox_weight')
            # reshape input
            rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
            label = mx.symbol.Reshape(data=label, shape=(-1,), name='label_reshape')
            bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_reg_classes), name='bbox_target_reshape')
            bbox_weight = mx.symbol.Reshape(data=bbox_weight, shape=(-1, 4 * num_reg_classes), name='bbox_weight_reshape')
        else:
            data = mx.sym.Variable(name="data")
            rois = mx.symbol.Variable(name='rois')
            # reshape input
            rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')

        # shared convolutional layers
        #data = mx.sym.Cast(data=data, dtype=np.float16)        
        conv_feat = self.get_resnet_v1_conv4(data)
        # res5
        relu1 = self.get_resnet_v1_conv5(conv_feat)

        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=256, name="conv_new_1")
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')
        #conv_new_1_relu = mx.sym.Cast(data=conv_new_1_relu, dtype=np.float32)
        offset_t = mx.contrib.sym.DeformablePSROIPooling(name='offset_t', data=conv_new_1_relu, rois=rois, group_size=1, pooled_size=7,
                                                         sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=0.0625)
        offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
        offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")

        deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool', data=conv_new_1_relu, rois=rois,
                                                                    trans=offset_reshape, group_size=1, pooled_size=7, sample_per_part=4,
                                                                    no_trans=False, part_size=7, output_dim=256, spatial_scale=0.0625, trans_std=0.1)
        #deformable_roi_pool = mx.sym.Cast(data=deformable_roi_pool, dtype=np.float16)
        # 2 fc
        fc_new_1 = mx.sym.FullyConnected(name='fc_new_1', data=deformable_roi_pool, num_hidden=1024)
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

        fc_new_2 = mx.sym.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
        fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')

        #fc_new_2_relu = mx.sym.Cast(data=fc_new_2_relu, dtype=np.float32)
        
        # cls_score/bbox_pred
        cls_score = mx.sym.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
        bbox_pred = mx.sym.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)
        
        #cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1,self.n_proposals, num_classes))
        #bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, self.n_proposals,4 * num_reg_classes))
        if is_train:
            if False:
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


                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                                normalization='valid', use_ignore=True, ignore_label=-1, grad_scale=1.0)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                                  data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=1.0 / (cfg.TRAIN.BATCH_ROIS_OHEM*cfg.TRAIN.BATCH_IMAGES))
                rcnn_label = labels_ohem
            else:
                #cls_score = mx.sym.Custom(op_type='debug_data', datai1=cls_score, datai2=label, datai3=bbox_pred, datai4=bbox_target)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid', use_ignore=True, ignore_label=-1, 
                                                grad_scale=1.0)
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                            data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / (188*16))
                rcnn_label = label

            # reshape output
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')
            group = mx.sym.Group([cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        arg_params['res5a_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5a_branch2b_offset_weight'])
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
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

