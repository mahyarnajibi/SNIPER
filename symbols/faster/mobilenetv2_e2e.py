# coding=utf-8
## initial script written by <zhangcycat@gmail.com>,
## list of changes made by liangfu <liangfu.chen@harman.com> :
# 1. add multiplier argument
# 2. add an argument to global average pooling
# 3. make number of expansion filters depend on the input tensor shape

# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# by Mahyar Najibi and Bharat Singh
# --------------------------------------------------------------

import mxnet as mx
from symbols.symbol import Symbol
import numpy as np

def relu6(data, prefix):
    return mx.sym.clip(data, 0, 6, name='%s-relu6' % prefix)


def shortcut(data_in, data_residual, prefix):
    out = mx.sym.elemwise_add(data_in, data_residual, name='%s-shortcut' % prefix)
    return out


def mobilenet_unit(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, if_act=True, prefix=''):
    conv = mx.sym.Convolution(
        data=data,
        num_filter=num_filter,
        kernel=kernel,
        num_group=num_group,
        stride=stride,
        pad=pad,
        no_bias=True,
        name='%s-conv2d' % prefix)
    bn = mx.sym.BatchNorm(data=conv, name='%s-batchnorm' % prefix, fix_gamma=False, momentum=0.995, eps=1e-5)
    if if_act:
        act = relu6(bn, prefix)
        return act
    else:
        return bn


def inverted_residual_unit(data, num_in_filter, num_filter, ifshortcut, stride, kernel, pad, expansion_factor, prefix):
    num_expfilter = int(round(num_in_filter * expansion_factor))

    channel_expand = mobilenet_unit(
        data=data,
        num_filter=num_expfilter,
        kernel=(1, 1),
        stride=(1, 1),
        pad=(0, 0),
        num_group=1,
        if_act=True,
        prefix='%s-exp' % prefix,
    )
    bottleneck_conv = mobilenet_unit(
        data=channel_expand,
        num_filter=num_expfilter,
        stride=stride,
        kernel=kernel,
        pad=pad,
        num_group=num_expfilter,
        if_act=True,
        prefix='%s-depthwise' % prefix,
    )
    linear_out = mobilenet_unit(
        data=bottleneck_conv,
        num_filter=num_filter,
        kernel=(1, 1),
        stride=(1, 1),
        pad=(0, 0),
        num_group=1,
        if_act=False,
        prefix='%s-linear' % prefix
    )
    if ifshortcut:
        out = shortcut(
            data_in=data,
            data_residual=linear_out,
            prefix=prefix,
        )
        return out
    else:
        return linear_out


def invresi_blocks(data, in_c, t, c, n, s, prefix):
    first_block = inverted_residual_unit(
        data=data,
        num_in_filter=in_c,
        num_filter=c,
        ifshortcut=False,
        stride=(s, s),
        kernel=(3, 3),
        pad=(1, 1),
        expansion_factor=t,
        prefix='%s-block0' % prefix
    )

    last_residual_block = first_block
    last_c = c

    for i in range(1, n):
        last_residual_block = inverted_residual_unit(
            data=last_residual_block,
            num_in_filter=last_c,
            num_filter=c,
            ifshortcut=True,
            stride=(1, 1),
            kernel=(3, 3),
            pad=(1, 1),
            expansion_factor=t,
            prefix='%s-block%d' % (prefix, i)
        )
    return last_residual_block


MNETV2_CONFIGS_MAP = {
    (224, 224): {
        'firstconv_filter_num': 32,  # 3*224*224 -> 32*112*112
        # t, c, n, s
        'bottleneck_params_list': [
            (1, 16, 1, 1),  # 32x112x112 -> 16x112x112
            (6, 24, 2, 2),  # 16x112x112 -> 24x56x56
            (6, 32, 3, 2),  # 24x56x56 -> 32x28x28
            (6, 64, 4, 2),  # 32x28x28 -> 64x14x14
            (6, 96, 3, 1),  # 64x14x14 -> 96x14x14
            (6, 160, 3, 2),  # 96x14x14 -> 160x7x7
            (6, 320, 1, 1),  # 160x7x7 -> 320x7x7
        ],
        'filter_num_before_gp': 1280,  # 320x7x7 -> 1280x7x7
    }
}


def checkpoint_callback(bbox_param_names, prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        weight = arg[bbox_param_names[0]]
        bias = arg[bbox_param_names[1]]
        stds = np.array([0.1, 0.1, 0.2, 0.2])
        arg[bbox_param_names[0] + '_test'] = (weight.T * mx.nd.array(stds)).T
        arg[bbox_param_names[1] + '_test'] = bias * mx.nd.array(stds)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop(bbox_param_names[0] + '_test')
        arg.pop(bbox_param_names[1] + '_test')

    return _callback


class mobilenetv2_e2e(Symbol):
    def __init__(self, n_proposals=400, momentum=0.95, fix_bn=False, test_nbatch=1):
        self.multiplier = 1
        self.MNetConfigs = MNETV2_CONFIGS_MAP[(224, 224)]
        self.test_nbatch = test_nbatch


    def get_bbox_param_names(self):
        return ['bbox_pred_weight', 'bbox_pred_bias']

    def get_rpn(self, conv_feat, num_anchors):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=256, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
        return rpn_cls_score, rpn_bbox_pred

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
            crowd_boxes = mx.sym.Variable(name='crowd_boxes')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name='im_info')
            im_ids = mx.sym.Variable(name='im_ids')

        # shared convolutional layers
        # self.MNetConfigs.update(configs)
        # first conv2d block
        first_c = int(round(self.MNetConfigs['firstconv_filter_num'] * self.multiplier))
        first_layer = mobilenet_unit(
            data=data,
            num_filter=first_c,
            kernel=(3, 3),
            stride=(2, 2),
            pad=(1, 1),
            if_act=True,
            prefix='first-3x3-conv'
        )
        first_layer = mx.sym.Cast(data=first_layer, dtype=np.float16)
        last_bottleneck_layer = first_layer
        in_c = first_c
        # bottleneck sequences
        for i, layer_setting in enumerate(self.MNetConfigs['bottleneck_params_list']):
            t, c, n, s = layer_setting
            last_bottleneck_layer = invresi_blocks(
                data=last_bottleneck_layer,
                in_c=in_c, t=t, c=int(round(c * self.multiplier)), n=n, s=s,
                prefix='seq-%d' % i
            )
            in_c = int(round(c * self.multiplier))
        # last conv2d block before global pooling
        last_fm = mobilenet_unit(
            data=last_bottleneck_layer,
            num_filter=int(1280 * self.multiplier) if self.multiplier > 1.0 else 1280,
            kernel=(1, 1),
            stride=(1, 1),
            pad=(0, 0),
            if_act=True,
            prefix='last-1x1-conv'
        )

        last_fm = mx.sym.Cast(data=last_fm, dtype=np.float32)
        rpn_cls_score, rpn_bbox_pred = self.get_rpn(last_fm, num_anchors)

        rpn_cls_score_reshape = mx.sym.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0),
                                               name="rpn_cls_score_reshape")
        if is_train:
            # prepare rpn data
            if cfg.TRAIN.fp16 == True:
                grad_scale = float(cfg.TRAIN.scale)
            else:
                grad_scale = 1.0

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1,
                                                name="rpn_cls_prob", grad_scale=grad_scale)

            conv_new_1 = mx.sym.Convolution(data=last_fm, kernel=(1, 1), num_filter=256, name="conv_new_1")
            conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')

            rois, label, bbox_target, bbox_weight = mx.sym.MultiProposalTarget(cls_prob=rpn_cls_prob,
                                                                               bbox_pred=rpn_bbox_pred, im_info=im_info,
                                                                               gt_boxes=gt_boxes,
                                                                               valid_ranges=valid_ranges,
                                                                               crowd_boxes=crowd_boxes,
                                                                               batch_size=cfg.TRAIN.BATCH_IMAGES,
                                                                               feature_stride=cfg.network.RPN_FEAT_STRIDE,
                                                                               scales=cfg.network.ANCHOR_SCALES,
                                                                               name='multi_proposal_target')
            label = mx.symbol.Reshape(data=label, shape=(-1,), name='label_reshape')
            offset_t = mx.contrib.sym.DeformablePSROIPooling(name='offset_t', data=conv_new_1_relu, rois=rois,
                                                             group_size=1, pooled_size=7,
                                                             sample_per_part=4, no_trans=True, part_size=7,
                                                             output_dim=256, spatial_scale=0.03125)
            offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
            offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")

            deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool',
                                                                        data=conv_new_1_relu, rois=rois,
                                                                        trans=offset_reshape, group_size=1,
                                                                        pooled_size=7, sample_per_part=4,
                                                                        no_trans=False, part_size=7, output_dim=256,
                                                                        spatial_scale=0.03125, trans_std=0.1)
            # 2 fc
            fc_new_1 = mx.sym.FullyConnected(name='fc_new_1', data=deformable_roi_pool, num_hidden=512)
            fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

            fc_new_2 = mx.sym.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=512)
            fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')

            num_reg_classes = 1
            num_classes = 81
            cls_score = mx.sym.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
            bbox_pred = mx.sym.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)

            if cfg.TRAIN.fp16 == True:
                grad_scale = float(cfg.TRAIN.scale)
            else:
                grad_scale = 1.0

            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, use_ignore=True,
                                            ignore_label=-1, grad_scale=grad_scale / (300.0 * cfg.TRAIN.BATCH_IMAGES))
            bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                        data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                        grad_scale=grad_scale / (188.0 * cfg.TRAIN.BATCH_IMAGES))
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
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            # ROI Proposal

            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')

            conv_new_1 = mx.sym.Convolution(data=last_fm, kernel=(1, 1), num_filter=256, name="conv_new_1")
            conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')

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
                                                             group_size=1,
                                                             pooled_size=7,
                                                             sample_per_part=4, no_trans=True, part_size=7,
                                                             output_dim=256,
                                                             spatial_scale=0.03125)

            offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
            offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")

            deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool',
                                                                        data=conv_new_1_relu,
                                                                        rois=rois,
                                                                        trans=offset_reshape, group_size=1,
                                                                        pooled_size=7,
                                                                        sample_per_part=4,
                                                                        no_trans=False, part_size=7, output_dim=256,
                                                                        spatial_scale=0.03125, trans_std=0.1)
            # 2 fc
            fc_new_1 = mx.sym.FullyConnected(name='fc_new_1', data=deformable_roi_pool, num_hidden=512)
            fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

            fc_new_2 = mx.sym.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=512)
            fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')

            num_reg_classes = 1
            num_classes = 81
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
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

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
