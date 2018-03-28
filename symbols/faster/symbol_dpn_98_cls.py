import mxnet as mx
# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yuwen Xiong, Xizhou Zhu
# --------------------------------------------------------

import cPickle
import mxnet as mx
from lib.symbol import Symbol
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

k_R = 160

G   = 40

k_sec  = {  2: 3, \
            3: 6, \
            4: 20, \
            5: 3   }

inc_sec= {  2: 16, \
            3: 32, \
            4: 32, \
            5: 128 }

def BK(data):
    return mx.symbol.BlockGrad(data=data)

bn_momentum = 0.9
# - - - - - - - - - - - - - - - - - - - - - - -
# Fundamental Elements
def BN(data, fix_gamma=False, momentum=0.9, name=None):
    bn     = mx.symbol.BatchNorm( data=data, fix_gamma=fix_gamma, momentum = 0.95, name=('%s__bn'%name))
    return bn

def AC(data, act_type='relu', name=None):
    act    = mx.symbol.Activation(data=data, act_type=act_type, name=('%s__%s' % (name, act_type)))
    return act

def BN_AC(data, momentum=bn_momentum, name=None):
    bn     = BN(data=data, name=name, fix_gamma=False)
    bn_ac  = AC(data=bn,   name=name)
    return bn_ac

def Conv(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, no_bias=True, w=None, b=None, attr=None, num_group=1, dconv=False):
    Convolution = mx.symbol.Convolution
    if dconv:
        offset = mx.symbol.Convolution(name=('%s__offset' % name), data=data,
                                       num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1),
                                       dilate=(2, 2), cudnn_off=True)
        conv = mx.contrib.symbol.DeformableConvolution(data=data, num_filter=num_filter, num_group=num_group, offset=offset, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
                                                       stride=(1, 1), dilate=(2, 2), no_bias=True, name=('%s__conv' % name))
    else:
        if w is None:
            conv     = Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s__conv' %name), no_bias=no_bias, attr=attr)
        else:
            if b is None:
                conv = Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s__conv' %name), no_bias=no_bias, weight=w, attr=attr)
            else:
                conv = Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s__conv' %name), no_bias=False, bias=b, weight=w, attr=attr)
    return conv


# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Common functions < CVPR >
def Conv_BN(   data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    cov    = Conv(   data=data,   num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    cov_bn = BN(     data=cov,    name=('%s__bn' % name))
    return cov_bn

def Conv_BN_AC(data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    cov_bn = Conv_BN(data=data,   num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    cov_ba = AC(     data=cov_bn, name=('%s__ac' % name))
    return cov_ba

# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Common functions < ECCV >
def BN_Conv(   data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    bn     = BN(     data=data,   name=('%s__bn' % name))
    bn_cov = Conv(   data=bn,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    return bn_cov

def AC_Conv(   data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1, dconv=False):
    ac     = AC(     data=data,   name=('%s__ac' % name))
    ac_cov = Conv(   data=ac,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr, dconv=dconv)
    return ac_cov

def BN_AC_Conv(data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1, dconv=False):
    bn     = BN(     data=data,   name=('%s__bn' % name))
    ba_cov = AC_Conv(data=bn,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr, dconv=dconv)
    return ba_cov


def DualPathFactory(data, num_1x1_a, num_3x3_b, num_1x1_c, name, inc, G, _type='normal', ndflag=True):
    kw = 3
    kh = 3
    pw = (kw - 1) / 2
    ph = (kh - 1) / 2

    # type
    if _type is 'proj':
        key_stride = 1
        if ndflag:
            key_name = 1
        else:
            key_name = 2
        has_proj = True
    if _type is 'down':
        key_stride = 2
        key_name = 2
        has_proj = True
    if _type is 'normal':
        key_stride = 1
        key_name = 1
        has_proj = False

    # PROJ
    if type(data) is list:
        data_in = mx.symbol.Concat(*[data[0], data[1]], name=('%s_cat-input' % name))
    else:
        data_in = data


    if has_proj:
        c1x1_w = BN_AC_Conv(data=data_in, num_filter=(num_1x1_c + 2 * inc), kernel=(1, 1),
                            stride=(key_stride, key_stride), name=('%s_c1x1-w(s/%d)' % (name, key_name)), pad=(0, 0))
        data_o1 = mx.symbol.slice_axis(data=c1x1_w, axis=1, begin=0, end=num_1x1_c,
                                       name=('%s_c1x1-w(s/%d)-split1' % (name, key_name)))
        data_o2 = mx.symbol.slice_axis(data=c1x1_w, axis=1, begin=num_1x1_c, end=(num_1x1_c + 2 * inc),
                                       name=('%s_c1x1-w(s/%d)-split2' % (name, key_name)))
    else:
        data_o1 = data[0]
        data_o2 = data[1]

    # MAIN

    c1x1_a = BN_AC_Conv(data=data_in, num_filter=num_1x1_a, kernel=(1, 1), pad=(0, 0), name=('%s_c1x1-a' % name))
    c3x3_b = BN_AC_Conv(data=c1x1_a, num_filter=num_3x3_b, kernel=(kw, kh), pad=(pw, ph),
                            name=('%s_c%dx%d-b' % (name, kw, kh)), stride=(key_stride, key_stride), num_group=G, dconv=not ndflag)
    c1x1_c = BN_AC_Conv(data=c3x3_b, num_filter=(num_1x1_c + inc), kernel=(1, 1), pad=(0, 0), name=('%s_c1x1-c' % name))

    c1x1_c1 = mx.symbol.slice_axis(data=c1x1_c, axis=1, begin=0, end=num_1x1_c, name=('%s_c1x1-c-split1' % name))
    c1x1_c2 = mx.symbol.slice_axis(data=c1x1_c, axis=1, begin=num_1x1_c, end=(num_1x1_c + inc),
                                   name=('%s_c1x1-c-split2' % name))

    # OUTPUTS
    summ = mx.symbol.ElementWiseSum(*[data_o1, c1x1_c1], name=('%s_sum' % name))
    dense = mx.symbol.Concat(*[data_o2, c1x1_c2], name=('%s_cat' % name))

    return [summ, dense]

class symbol_dpn_98_cls(Symbol):
    def __init__(self, n_proposals=400):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 2e-5
        self.n_proposals = n_proposals

    def get_bbox_param_names(self):
        return ['bbox_pred_weight', 'bbox_pred_bias']

    def get_conv4(self, data):

        # conv1
        conv1_x_1 = Conv(data=data, num_filter=96, kernel=(7, 7), name='conv1_x_1', pad=(3, 3), stride=(2, 2))
        conv1_x_1 = BN_AC(conv1_x_1, name='conv1_x_1__relu-sp')
        conv1_x_x = mx.symbol.Pooling(data=conv1_x_1, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                                      name="pool1")

        # conv2
        bw = 256
        inc = inc_sec[2]
        R = (k_R * bw) / 256
        conv2_x_x = DualPathFactory(conv1_x_x, R, R, bw, 'conv2_x__1', inc, G, 'proj')
        for i_ly in range(2, k_sec[2] + 1):
            conv2_x_x = DualPathFactory(conv2_x_x, R, R, bw, ('conv2_x__%d' % i_ly), inc, G, 'normal')

        # conv3
        bw = 512
        inc = inc_sec[3]
        R = (k_R * bw) / 256
        conv3_x_x = DualPathFactory(conv2_x_x, R, R, bw, 'conv3_x__1', inc, G, 'down')
        for i_ly in range(2, k_sec[3] + 1):
            conv3_x_x = DualPathFactory(conv3_x_x, R, R, bw, ('conv3_x__%d' % i_ly), inc, G, 'normal')

        # conv4
        bw = 1024
        inc = inc_sec[4]
        R = (k_R * bw) / 256
        conv4_x_x = DualPathFactory(conv3_x_x, R, R, bw, 'conv4_x__1', inc, G, 'down')
        for i_ly in range(2, k_sec[4] + 1):
            conv4_x_x = DualPathFactory(conv4_x_x, R, R, bw, ('conv4_x__%d' % i_ly), inc, G, 'normal')
        return conv4_x_x

    def get_conv5(self, conv4_x_x):
        bw = 2048
        inc = inc_sec[5]
        R = (k_R * bw) / 256
        conv5_x_x = DualPathFactory(conv4_x_x, R, R, bw, 'conv5_x__1', inc, G, 'proj', ndflag=False)
        for i_ly in range(2, k_sec[5] + 1):
            conv5_x_x = DualPathFactory(conv5_x_x, R, R, bw, ('conv5_x__%d' % i_ly), inc, G, 'normal', ndflag=False)

        # output: concat
        conv5_x_x = mx.symbol.Concat(*[conv5_x_x[0], conv5_x_x[1]], name='conv5_x_x_cat-final')
        conv5_x_x = BN_AC(conv5_x_x, name='conv5_x_x__relu-sp')
        return conv5_x_x


    def get_symbol_rcnn(self, cfg, is_train=True):

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

        if is_train:
	    data = mx.sym.Cast(data=data, dtype=np.float16)
        conv_feat = self.get_conv4(data)
        # res5
        relu1 = self.get_conv5(conv_feat)

        # conv_new_1
        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=1024, name="conv_new_1", lr_mult=3.0)
	conv_new_1_bn = mx.symbol.BatchNorm(name='conv_new_1_bn', data=conv_new_1, momentum=0.95, fix_gamma=False, eps=self.eps)
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1_bn, act_type='relu', name='relu1')

        if is_train:        
	    conv_new_1_relu = mx.sym.Cast(data=conv_new_1_relu, dtype=np.float32)
            
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
                                                grad_scale=100.0)
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                            data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=100.0 / (188*16))
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
	arg_params['conv_new_1_bn_beta'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bn_beta'])
        arg_params['conv_new_1_bn_gamma'] = mx.nd.ones(shape=self.arg_shape_dict['conv_new_1_bn_gamma'])
        aux_params['conv_new_1_bn_moving_mean'] = mx.nd.zeros(shape=self.aux_shape_dict['conv_new_1_bn_moving_mean'])
        aux_params['conv_new_1_bn_moving_var'] = mx.nd.ones(shape=self.aux_shape_dict['conv_new_1_bn_moving_var'])
        
        arg_params['conv5_x__1_c3x3-b__offset_weight'] = mx.nd.zeros(
            shape=self.arg_shape_dict['conv5_x__1_c3x3-b__offset_weight'])
        arg_params['conv5_x__1_c3x3-b__offset_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['conv5_x__1_c3x3-b__offset_bias'])
        arg_params['conv5_x__2_c3x3-b__offset_weight'] = mx.nd.zeros(
            shape=self.arg_shape_dict['conv5_x__2_c3x3-b__offset_weight'])
        arg_params['conv5_x__2_c3x3-b__offset_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['conv5_x__2_c3x3-b__offset_bias'])
        arg_params['conv5_x__3_c3x3-b__offset_weight'] = mx.nd.zeros(
            shape=self.arg_shape_dict['conv5_x__3_c3x3-b__offset_weight'])
        arg_params['conv5_x__3_c3x3-b__offset_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['conv5_x__3_c3x3-b__offset_bias'])

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

