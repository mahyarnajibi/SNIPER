import cPickle
import mxnet as mx
from symbols.symbol import Symbol
from operator_py.box_annotator_ohem import *
from operator_py.debug_data import *
import numpy as np

def checkpoint_callback(bbox_param_names, prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        #weight = arg[bbox_param_names[0]]
        #bias = arg[bbox_param_names[1]]
        #stds = np.array([0.1, 0.1, 0.2, 0.2])
        #arg[bbox_param_names[0]+'_test'] = (weight.T * mx.nd.array(stds)).T
        #arg[bbox_param_names[1]+'_test'] =bias * mx.nd.array(stds)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        #arg.pop(bbox_param_names[0]+'_test')
        #arg.pop(bbox_param_names[1]+'_test')
    return _callback


class resnet_mx_101_e2e_3k(Symbol):
    def __init__(self, n_proposals=400, momentum=0.95, fix_bn=False, test_nbatch=1):
        """
        Use __init__ to define parameter network needs
        """
        self.momentum = momentum
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [64, 256, 512, 1024, 2048]
        self.fix_bn = fix_bn
        self.test_nbatch= test_nbatch        

    def get_bbox_param_names(self):
        return ['bbox_pred_weight', 'bbox_pred_bias']

    def residual_unit(self, data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=512, memonger=False,
                      fix_bn=False):
        if fix_bn or self.fix_bn:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn1')
        else:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        if fix_bn or self.fix_bn:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn2')
        else:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                   pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        if fix_bn or self.fix_bn:
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn3')
        else:
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut

    def residual_unit_dilate(self, data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=512,
                             memonger=False):
        if self.fix_bn:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn1')
        else:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        if self.fix_bn:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn2')
        else:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), dilate=(2, 2),
                                   stride=stride, pad=(2, 2),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        if self.fix_bn:
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn3')
        else:
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut

    def residual_unit_deform(self, data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=512,
                             memonger=False):
        if self.fix_bn:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn1')
        else:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        if self.fix_bn:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn2')
        else:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        offset = mx.symbol.Convolution(name=name + '_offset', data=act2,
                                       num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1),
                                       dilate=(2, 2), cudnn_off=True)
        conv2 = mx.contrib.symbol.DeformableConvolution(name=name + '_conv2', data=act2,
                                                        offset=offset,
                                                        num_filter=512, pad=(2, 2), kernel=(3, 3),
                                                        num_deformable_group=4,
                                                        stride=(1, 1), dilate=(2, 2), no_bias=True)
        if self.fix_bn:
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn3')
        else:
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn3')

        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut

    def get_rpn(self, conv_feat, num_anchors):
        conv_feat = mx.sym.Cast(data=conv_feat, dtype=np.float32)                
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
        return rpn_cls_score, rpn_bbox_pred

    def get_symbol_rcnn(self, cfg, is_train=True):
        num_anchors = cfg.network.NUM_ANCHORS
        num_classes = 3131
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

        if cfg.TRAIN.fp16 == True:
            grad_scale = float(cfg.TRAIN.scale)
        else:
            grad_scale = 1.0

        # shared convolutional layers
        conv_feat = self.resnetc4(data, fp16=cfg.TRAIN.fp16)
        # res5
        relut = self.resnetc5(conv_feat, deform=True)
        relu1 = mx.symbol.Concat(*[conv_feat, relut], name='cat4')
        relu1 = mx.sym.Cast(data=relu1, dtype=np.float32)        

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors)
        rpn_cls_score_reshape = mx.sym.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0),
                                               name="rpn_cls_score_reshape")
        if is_train:
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1,
                                                name="rpn_cls_prob", grad_scale=grad_scale)
            rois, label, bbox_target, bbox_weight, posrois, _, poslabels = mx.sym.MultiProposalTargetMask(cls_prob=rpn_cls_prob, bbox_pred=rpn_bbox_pred, im_info=im_info, gt_boxes=gt_boxes, valid_ranges=valid_ranges, batch_size=cfg.TRAIN.BATCH_IMAGES, scales=cfg.network.ANCHOR_SCALES, rfcn_3k=True, name='multi_proposal_target_mask')
            label = mx.symbol.Reshape(data=label, shape=(-1,), name='label_reshape')
            sublabel = mx.symbol.Reshape(data=poslabels, shape=(-1,), name='sublabel_reshape')            
        else:
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


        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=1024, name="conv_new_1")
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')

        conv_new_2 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=1024, name="conv_new_2")
        conv_new_2_relu = mx.sym.Activation(data=conv_new_2, act_type='relu', name='conv_new_2_relu')
        num_sub_classes = num_classes - 1
        conv_new_3 = mx.sym.Convolution(data=conv_new_2_relu, kernel=(1, 1), num_filter=num_sub_classes, name="conv_new_3")
        if is_train:
            roipooled_subcls_rois = mx.sym.ROIPooling(name='roipooled_subcls_rois', data=conv_new_3,
                                                      rois=posrois, pooled_size=(7, 7), spatial_scale=0.0625)
        else:
            roipooled_subcls_rois = mx.sym.ROIPooling(name='roipooled_subcls_rois', data=conv_new_3,
                                                      rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
            
        subcls_score = mx.sym.Pooling(name='ave_subcls_scors_rois', data=roipooled_subcls_rois, pool_type='avg',
                                          global_pool=True, kernel=(7, 7))
        subcls_score = mx.sym.Reshape(name='subcls_score_reshape', data=subcls_score, shape=(-1, num_sub_classes))

        # rfcn_cls/rfcn_bbox
        rfcn_cls = mx.sym.Convolution(data=conv_new_1, kernel=(1, 1), num_filter=7*7*2, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=conv_new_1, kernel=(1, 1), num_filter=7*7*4, name="rfcn_bbox")

        psroipooled_cls_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois,
                                                                     group_size=7, pooled_size=7, sample_per_part=4, no_trans=True, trans_std=0.1,
                                                                     output_dim=2, spatial_scale=0.0625, part_size=7)
        psroipooled_loc_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois,
                                                                     group_size=7, pooled_size=7, sample_per_part=4, no_trans=True, trans_std=0.1,
                                                                     output_dim=4, spatial_scale=0.0625, part_size=7)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, 2))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4))

        # prepare rpn data
        if cfg.TRAIN.fp16 == True:
            grad_scale = float(cfg.TRAIN.scale)
        else:
            grad_scale = 1.0

        if is_train:
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, use_ignore=True, ignore_label=-1, grad_scale=grad_scale / (300.0*16.0))
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 2),
                                  name='cls_prob_reshape')

            subcls_prob = mx.sym.SoftmaxOutput(name='subcls_prob', data=subcls_score, label=sublabel, use_ignore=True, ignore_label=-1, grad_scale=grad_scale / (200.0*16.0))
            bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
        
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=grad_scale / (188.0*16.0))
            
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=1.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                            grad_scale=3 * grad_scale / float(cfg.TRAIN.BATCH_IMAGES * cfg.TRAIN.RPN_BATCH_SIZE))
            rcnn_label = label
            rcnn_sublabel = sublabel
            #bbox_loss = mx.sym.Custom(datai1=bbox_loss, datai2=rcnn_label, datai3=rcnn_sublabel, datai4=bbox_weight, op_type='debug_data')
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, subcls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label), mx.sym.BlockGrad(rcnn_sublabel)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(self.test_nbatch, -1, 2),
                                      name='cls_prob_reshape')
            subcls_prob = mx.sym.SoftmaxActivation(name='subcls_prob', data=subcls_score)            
            subcls_prob = mx.sym.Reshape(data=subcls_prob, shape=(self.test_nbatch, -1, num_sub_classes),
                                      name='subcls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(self.test_nbatch, -1, 4),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred, im_ids, subcls_prob])
        self.sym = group
        return group

    def resnetc4(self, data, fp16=False):
        units = self.units
        filter_list = self.filter_list
        bn_mom = self.momentum
        workspace = self.workspace
        num_stage = len(units)
        memonger = False

        data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, use_global_stats=True, name='bn_data')
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        if fp16:
            body = mx.sym.Cast(data=body, dtype=np.float16)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, use_global_stats=True, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

        for i in range(num_stage - 1):
            body = self.residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                                      name='stage%d_unit%d' % (i + 1, 1), workspace=workspace,
                                      memonger=memonger, fix_bn=(i == 0))
            for j in range(units[i] - 1):
                body = self.residual_unit(body, filter_list[i + 1], (1, 1), True,
                                          name='stage%d_unit%d' % (i + 1, j + 2),
                                          workspace=workspace, memonger=memonger, fix_bn=(i == 0))

        return body

    def resnetc5(self, body, deform):
        units = self.units
        filter_list = self.filter_list
        workspace = self.workspace
        num_stage = len(units)
        memonger = False

        i = num_stage - 1
        if deform:
            body = self.residual_unit_deform(body, filter_list[i + 1], (1, 1), False,
                                             name='stage%d_unit%d' % (i + 1, 1), workspace=workspace,
                                             memonger=memonger)
        else:
            body = self.residual_unit_dilate(body, filter_list[i + 1], (1, 1), False,
                                             name='stage%d_unit%d' % (i + 1, 1), workspace=workspace,
                                             memonger=memonger)
        for j in range(units[i] - 1):
            if deform:
                body = self.residual_unit_deform(body, filter_list[i + 1], (1, 1), True,
                                                 name='stage%d_unit%d' % (i + 1, j + 2),
                                                 workspace=workspace, memonger=memonger)
            else:
                body = self.residual_unit_dilate(body, filter_list[i + 1], (1, 1), True,
                                                 name='stage%d_unit%d' % (i + 1, j + 2),
                                                 workspace=workspace, memonger=memonger)

        return body

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        """arg_params['stage4_unit1_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit1_offset_weight'])
        arg_params['stage4_unit1_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit1_offset_bias'])
        arg_params['stage4_unit2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_offset_weight'])
        arg_params['stage4_unit2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_offset_bias'])
        arg_params['stage4_unit3_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_offset_weight'])
        arg_params['stage4_unit3_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_offset_bias'])

        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])"""
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])
        
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['conv_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_2_weight'])
        arg_params['conv_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_2_bias'])
        arg_params['conv_new_3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_3_weight'])
        arg_params['conv_new_3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_3_bias'])
        arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_cls_weight'])
        arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_bias'])
        arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_bbox_weight'])
        arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_bias'])

    def init_weight_rpn(self, cfg, arg_params, aux_params):        
        arg_params['stage4_unit1_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit1_offset_weight'])
        arg_params['stage4_unit1_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit1_offset_bias'])
        arg_params['stage4_unit2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_offset_weight'])
        arg_params['stage4_unit2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_offset_bias'])
        arg_params['stage4_unit3_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_offset_weight'])
        arg_params['stage4_unit3_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_offset_bias'])

        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

        

    def init_weight(self, cfg, arg_params, aux_params):        
        self.init_weight_rcnn(cfg, arg_params, aux_params)
