# ---------------------------------------------------------------
# SNIPER: Efficient Multi-scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified from https://github.com/msracver/Deformable-ConvNets
# Modified by Mahyar Najibi
# ---------------------------------------------------------------

import mxnet as mx
import numpy as np
import os
import pickle

def get_rpn_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label

def get_mask_names():
    pred = ['mask_prob', 'mask_targets']
    label = []
    return pred, label

def get_rcnn_names(cfg):
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    if cfg.TRAIN.ENABLE_OHEM or cfg.TRAIN.END2END:
        pred.append('rcnn_label')
    if cfg.TRAIN.END2END:
        rpn_pred, rpn_label = get_rpn_names()
        pred = rpn_pred + pred
        label = rpn_label
        if cfg.TRAIN.WITH_MASK:
            mask_pred, mask_label = get_mask_names()
            pred += mask_pred
            label += mask_label

    return pred, label


def get_rcnn_names_4vis(cfg):
    pred,label = get_rcnn_names(cfg)
    pred += ['rcnn_bbox_pred', 'rois', 'rcnn_label']
    return pred, label



class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNAccFgMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNAccFgMetric, self).__init__('RCNNFgAcc')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[3]
        label = preds[4]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class MaskLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, config):
        super(MaskLogLossMetric, self).__init__('MaskLogLoss')
        self.pred, self.label = get_rcnn_names(config)

    def update(self, labels, preds):
        pred = preds[self.pred.index('mask_prob')]
        label = preds[self.pred.index('mask_targets')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

class RCNNFgLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNFgLogLossMetric, self).__init__('RCNNFgLogLoss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[3]
        label = preds[4]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label')].asnumpy()
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        if self.ohem:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            if self.e2e:
                label = preds[self.pred.index('rcnn_label')].asnumpy()
            else:
                label = labels[self.label.index('rcnn_label')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        num_inst = np.sum(label != -1)
        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

class RCNNL1LossCRCNNMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNL1LossCRCNNMetric, self).__init__('RCNNL1LossCRCNN')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)
        self.cfg = cfg

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        if self.ohem:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            if self.e2e:
                label = preds[self.pred.index('rcnn_label')].asnumpy()
            else:
                label = labels[self.label.index('rcnn_label')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        num_inst = np.sum(label != -1)
        #num_inst = np.sum(label > 0)
        self.sum_metric += np.sum(bbox_loss)
        #self.num_inst += num_inst
        self.num_inst += num_inst #(self.cfg.TRAIN.BATCH_ROIS_OHEM*self.cfg.TRAIN.BATCH_IMAGES)

class VisMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(VisMetric, self).__init__('Vis')
        self.freq = cfg.TRAIN.visualization_freq
        self.root_path = cfg.TRAIN.visualization_path
        self.pred, self.label = get_rcnn_names_4vis(cfg)
        self.nGPU = len(cfg.gpus.split(','))
        self.file_name = os.path.basename(os.path.normpath(cfg.output_path))

    def update(self, labels, preds):
        if self.num_inst % self.freq == 0 and self.num_inst%self.nGPU==0:
            pred = preds[self.pred.index('rcnn_cls_prob')].asnumpy()
            bbox_pred = preds[self.pred.index('rcnn_bbox_pred')].asnumpy()
            labels = preds[self.pred.index('rcnn_label')].asnumpy()
            rois = preds[self.pred.index('rois')].asnumpy()
            inds = np.where(rois[:, 0] == 0)[0]
            path = os.path.join(self.root_path, 'dump_preds_{}.pkl'.format(self.file_name))
            with open(path,'wb') as f:
                pickle.dump((pred[0], bbox_pred[inds], rois[inds], labels[inds]),f)
       
        self.num_inst += 1
        self.sum_metric = 0
        
