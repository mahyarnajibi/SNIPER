# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong, from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------


"""
This file has functions about generating bounding box regression targets
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from .bbox_transform import bbox_overlaps, bbox_transform


def compute_bbox_regression_targets(rois, overlaps, labels, cfg):
    """
    given rois, overlaps, gt labels, compute bounding box regression targets
    :param rois: roidb[i]['boxes'] k * 4
    :param overlaps: roidb[i]['max_overlaps'] k * 1
    :param labels: roidb[i]['max_classes'] k * 1
    :return: targets[i][class, dx, dy, dw, dh] k * 5
    """
    # Ensure ROIs are floats
    rois = rois.astype(np.float, copy=False)

    # Sanity check
    if len(rois) != len(overlaps):
        print('bbox regression: this should not happen')

    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    if len(gt_inds) == 0:
        print('something wrong : zero ground truth rois')
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_REGRESSION_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(rois[ex_inds, :], rois[gt_inds, :])

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
    return targets


def add_bbox_regression_targets(roidb, cfg):
    """
    given roidb, add ['bbox_targets'] and normalize bounding box regression targets
    :param roidb: roidb to be processed. must have gone through imdb.prepare_roidb
    :return: means, std variances of targets
    """
    print('add bounding box regression targets')
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0]

    num_images = len(roidb)
    num_classes = 2 if cfg.CLASS_AGNOSTIC else roidb[0]['gt_overlaps'].shape[1]

    for im_i in range(num_images):
        rois = roidb[im_i]['boxes']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_targets'] = compute_bbox_regression_targets(rois, max_overlaps, max_classes, cfg)

    if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
        # use fixed / precomputed means and stds instead of empirical values
        means = np.tile(np.array(cfg.TRAIN.BBOX_MEANS), (num_classes, 1))
        stds = np.tile(np.array(cfg.TRAIN.BBOX_STDS), (num_classes, 1))
    else:
        # compute mean, std values
        class_counts = np.zeros((num_classes, 1)) + 1e-14
        sums = np.zeros((num_classes, 4))
        squared_sums = np.zeros((num_classes, 4))
        for im_i in range(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in range(1, num_classes):
                cls_indexes = np.where(targets[:, 0] > 0)[0] if cfg.CLASS_AGNOSTIC else np.where(targets[:, 0] == cls)[0]
                if cls_indexes.size > 0:
                    class_counts[cls] += cls_indexes.size
                    sums[cls, :] += targets[cls_indexes, 1:].sum(axis=0)
                    squared_sums[cls, :] += (targets[cls_indexes, 1:] ** 2).sum(axis=0)

        means = sums / class_counts
        # var(x) = E(x^2) - E(x)^2
        stds = np.sqrt(squared_sums / class_counts - means ** 2)

    print('bbox target means:')
    print(means)
    print(means[1:, :].mean(axis=0))  # ignore bg class
    print('bbox target stdevs:')
    print(stds)
    print(stds[1:, :].mean(axis=0))  # ignore bg class


    # normalized targets
    for im_i in range(num_images):
        targets = roidb[im_i]['bbox_targets']
        for cls in range(1, num_classes):
            cls_indexes = np.where(targets[:, 0] > 0) if cfg.CLASS_AGNOSTIC else np.where(targets[:, 0] == cls)[0]
            roidb[im_i]['bbox_targets'][cls_indexes, 1:] -= means[cls, :]
            roidb[im_i]['bbox_targets'][cls_indexes, 1:] /= stds[cls, :]

    return means.ravel(), stds.ravel()


def expand_bbox_regression_targets(bbox_targets_data, num_classes, cfg=None):
    """
    expand from 5 to 4 * num_classes; only the right class has non-zero bbox regression targets
    :param bbox_targets_data: [k * 5]
    :param num_classes: number of classes
    :return: bbox target processed [k * 4 num_classes]
    bbox_weights ! only foreground boxes have bbox regression computation!
    """
    classes = bbox_targets_data[:, 0]
    
    num_classes = 2
    bbox_targets = np.zeros((classes.size, 4 * num_classes), dtype=np.float32)
    bbox_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    indexes = np.where(classes > 0)[0]
    for index in indexes:
        cls = classes[index]
        start = int(4 * 1 if cls > 0 else 0) 
        end = start + 4
        bbox_targets[index, start:end] = bbox_targets_data[index, 1:]
        bbox_weights[index, start:end] = np.array([1.0,1.0,1.0,1.0])
    return bbox_targets, bbox_weights

