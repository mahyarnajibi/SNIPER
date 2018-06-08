# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Inference Module
# by Mahyar Najibi and Bharat Singh
# --------------------------------------------------------------
import cv2
import mxnet as mx
import numpy as np
from nms.nms import py_nms_wrapper, soft_nms
from mask_utils import crop_polys, poly_encoder
from generate_anchor import generate_anchors
from bbox.bbox_transform import *
import numpy.random as npr
from chips import genchips
import math


class im_worker(object):
    def __init__(self, cfg, crop_size=None, target_size=None):
        self.cfg = cfg
        self.crop_size = crop_size
        if not target_size:
            self.target_size = self.cfg.TRAIN.SCALES[0]
        else:
            self.target_size = target_size

    def worker(self, data):
        imp = data[0]
        flipped = data[2]
        pixel_means = self.cfg.network.PIXEL_MEANS
        im = cv2.imread(imp, cv2.IMREAD_COLOR)
        # Flip the image
        if flipped:
            im = im[:, ::-1, :]

        # Crop if required
        if self.crop_size:
            crop = data[1]
            max_size = [self.crop_size, self.crop_size]
            im = im[int(crop[0][1]):int(crop[0][3]), int(crop[0][0]):int(crop[0][2]), :]
            scale = crop[1]
        else:
            max_size = data[1]
            # Compute scale based on config
            min_target_size = self.target_size[0]
            max_target_size = self.target_size[1]
            im_size_min = np.min(im.shape[0:2])
            im_size_max = np.max(im.shape[0:2])
            scale = float(min_target_size) / float(im_size_min)
            if np.round(scale * im_size_max) > max_target_size:
                scale = float(max_target_size) / float(im_size_max)
        # Resize the image
        try:
            im = cv2.resize(im, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        except:
            print 'Image Resize Failed!'

        rim = np.zeros((3, max_size[0], max_size[1]), dtype=np.float32)
        d1m = min(im.shape[0], max_size[0])
        d2m = min(im.shape[1], max_size[1])

        for j in range(3):
            rim[j, :d1m, :d2m] = im[:d1m, :d2m, 2 - j] - pixel_means[2 - j]

        if self.crop_size:
            return mx.nd.array(rim, dtype='float32')
        else:
            return mx.nd.array(rim, dtype='float32'), scale, (im.shape[0],im.shape[1])
            
def nms_worker(worker_args):
    return soft_nms(worker_args[0], sigma=worker_args[1], method=2)

# Compute all anchors
scales = np.array([2, 4, 7, 10, 13, 16, 24], dtype=np.float32)
ratios = (0.5, 1, 2)
feat_stride = 16
base_anchors = generate_anchors(base_size=feat_stride, ratios=list(ratios), scales=list(scales))
num_anchors = base_anchors.shape[0]
feat_width = 32
feat_height = 32
shift_x = np.arange(0, feat_width) * feat_stride
shift_y = np.arange(0, feat_height) * feat_stride
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
A = num_anchors
K = shifts.shape[0]
all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
all_anchors = all_anchors.reshape((K * A, 4))


def roidb_anchor_worker(data):
    im_info = data[0]
    cur_crop = data[1]
    im_scale = data[2]
    nids = data[3]
    gtids = data[4]
    gt_boxes = data[5]
    boxes = data[6]
    classes = data[7]
    max_n_gts = 100
    max_poly_len = 500

    has_mask = True if len(data) > 8 else False
    
    anchors = all_anchors.copy()
    inds_inside = np.where((anchors[:, 0] >= -32) &
                           (anchors[:, 1] >= -32) &
                           (anchors[:, 2] < im_info[0]+32) &
                           (anchors[:, 3] < im_info[1]+32))[0]

    anchors = anchors[inds_inside, :]
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)
    total_anchors = int(K * A)

    gt_boxes[:, 0] = gt_boxes[:, 0] - cur_crop[0]
    gt_boxes[:, 2] = gt_boxes[:, 2] - cur_crop[0]
    gt_boxes[:, 1] = gt_boxes[:, 1] - cur_crop[1]
    gt_boxes[:, 3] = gt_boxes[:, 3] - cur_crop[1]

    vgt_boxes = boxes[np.intersect1d(gtids, nids)]

    vgt_boxes[:, 0] = vgt_boxes[:, 0] - cur_crop[0]
    vgt_boxes[:, 2] = vgt_boxes[:, 2] - cur_crop[0]
    vgt_boxes[:, 1] = vgt_boxes[:, 1] - cur_crop[1]
    vgt_boxes[:, 3] = vgt_boxes[:, 3] - cur_crop[1]

    gt_boxes = clip_boxes(np.round(gt_boxes * im_scale), im_info[:2])
    vgt_boxes = clip_boxes(np.round(vgt_boxes * im_scale), im_info[:2])

    ids = filter_boxes(gt_boxes, 10)
    if len(ids) == 0:
        gt_boxes = np.zeros((0, 4))
        classes = np.zeros((0, 1))

    if has_mask:
        mask_polys = data[8]
        # Shift and crop the mask polygons
        mask_polys = crop_polys(mask_polys, cur_crop, im_scale)
        # Create the padded encoded array
        if len(ids) > 0:
            polylen = len(mask_polys)
            tmask_polys = []
            tgt_boxes = []
            tclasses = []
            for i in ids:
                if i < polylen:
                    tmask_polys.append(mask_polys[i])
                    tgt_boxes.append(gt_boxes[i])
                    tclasses.append(classes[i])
            if len(gt_boxes) > 0:
                gt_boxes = np.array(tgt_boxes)
                classes = np.array(tclasses).reshape(len(tclasses), 1)
                mask_polys = tmask_polys
            else:
                gt_boxes = np.zeros((0, 4))
                classes = np.zeros((0, 1))

            encoded_polys = poly_encoder(mask_polys, classes[:,0]-1,
                    max_poly_len=max_poly_len, max_n_gts=max_n_gts)
        else:
            encoded_polys = -np.ones((max_n_gts, max_poly_len), dtype=np.float32)
    else:
        if len(ids) > 0:
            gt_boxes = gt_boxes[ids]
            classes = classes[ids]

    agt_boxes = gt_boxes.copy()
    ids = filter_boxes(vgt_boxes, 10)
    if len(ids) > 0:
        vgt_boxes = vgt_boxes[ids]
    else:
        vgt_boxes = np.zeros((0, 4))

    if len(vgt_boxes) > 0:
        ov = bbox_overlaps(np.ascontiguousarray(gt_boxes).astype(float), np.ascontiguousarray(vgt_boxes).astype(float))
        mov = np.max(ov, axis=1)
    else:
        mov = np.zeros((len(gt_boxes)))

    invalid_gtids = np.where(mov < 1)[0]
    valid_gtids = np.where(mov == 1)[0]
    invalid_boxes = gt_boxes[invalid_gtids, :]
    gt_boxes = gt_boxes[valid_gtids, :]

    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        if invalid_boxes is not None:
            if len(invalid_boxes) > 0:
                overlapsn = bbox_overlaps(anchors.astype(np.float), invalid_boxes.astype(np.float))
                argmax_overlapsn = overlapsn.argmax(axis=1)
                max_overlapsn = overlapsn[np.arange(len(inds_inside)), argmax_overlapsn]
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        labels[max_overlaps < 0.4] = 0
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= 0.5] = 1

        if invalid_boxes is not None:
            if len(invalid_boxes) > 0:
                labels[max_overlapsn > 0.3] = -1
    else:
        labels[:] = 0
        if len(invalid_boxes) > 0:
            overlapsn = bbox_overlaps(anchors.astype(np.float), invalid_boxes.astype(np.float))
            argmax_overlapsn = overlapsn.argmax(axis=1)
            max_overlapsn = overlapsn[np.arange(len(inds_inside)), argmax_overlapsn]
            if len(invalid_boxes) > 0:
                labels[max_overlapsn > 0.3] = -1

        # subsample positive labels if we have too many
    num_fg = 128
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = 256 - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = np.array([1.0, 1.0, 1.0, 1.0])

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)

    labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, A * feat_height * feat_width)).astype(np.float16)
    bbox_targets = bbox_targets.reshape((feat_height, feat_width, A * 4)).transpose(2, 0, 1)
    bbox_weights = bbox_weights.reshape((feat_height, feat_width, A * 4)).transpose((2, 0, 1))
    pids = np.where(bbox_weights == 1)
    bbox_targets = bbox_targets[pids]

    fgt_boxes = -np.ones((100, 5))
    if len(agt_boxes) > 0:
        fgt_boxes[:min(len(agt_boxes), 100), :] = np.hstack((agt_boxes, classes))

    rval = [mx.nd.array(labels, dtype='float16'), bbox_targets, mx.nd.array(pids), mx.nd.array(fgt_boxes)]
    if has_mask:
        rval.append(mx.nd.array(encoded_polys))
    return rval

def roidb_worker(data):
    im_i = data[0]
    im_info = data[1]        
    cur_crop = data[2]
    im_scale = data[3]
    nids = data[4]    
    gt_boxes = data[5]
    gt_labs = data[6]
    boxes = data[7]
    n_expected_roi = data[8]
    num_classes = data[9]

    gt_boxes[:, 0] = gt_boxes[:, 0] - cur_crop[0]
    gt_boxes[:, 2] = gt_boxes[:, 2] - cur_crop[0]
    gt_boxes[:, 1] = gt_boxes[:, 1] - cur_crop[1]
    gt_boxes[:, 3] = gt_boxes[:, 3] - cur_crop[1]

    gt_boxes = clip_boxes(np.round(gt_boxes * im_scale), im_info[:2])
    ids = filter_boxes(gt_boxes, 5)

    if len(ids)>0:
        gt_boxes = gt_boxes[ids]
        gt_labs = gt_labs[ids]                
        gt_boxes = np.hstack((gt_boxes, gt_labs.reshape(len(gt_labs), 1)))
        if has_mask:
            mask_polys = [mask_polys[i] for i in ids]
    else:
        gt_boxes = np.zeros((0, 5))

    crois = boxes.copy()

    crois[:, 0] = crois[:, 0] - cur_crop[0]
    crois[:, 2] = crois[:, 2] - cur_crop[0]
    crois[:, 1] = crois[:, 1] - cur_crop[1]
    crois[:, 3] = crois[:, 3] - cur_crop[1]

    rois = clip_boxes(np.round(crois * im_scale), im_info[:2])

    ids = filter_boxes(rois, 10)
    tids = np.intersect1d(ids, nids)
    if len(nids) > 0:
        ids = tids
    else:
        ids = nids

    if len(ids) > 0:            
        rois = rois[ids, :]

    fg_rois_per_image = len(rois)
    rois_per_image = fg_rois_per_image
    rois, labels, bbox_targets, bbox_weights = sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes, gt_boxes=gt_boxes)

    pad_roi = np.array([[0,0,100,100]])
    pad_label = np.array(-1)
    pad_weights = np.zeros((1, 8))
    pad_targets = np.zeros((1, 8))

    if rois.shape[0] > n_expected_roi:
        rois = rois[0:n_expected_roi, :]
        bbox_weights = bbox_weights[0:n_expected_roi, :]
        bbox_targets = bbox_targets[0:n_expected_roi, :]
        labels = labels[0:n_expected_roi]
    elif rois.shape[0] < n_expected_roi:
        n_pad = n_expected_roi - rois.shape[0]
        rois = np.vstack((rois, np.repeat(pad_roi, n_pad, axis=0)))
        labels = np.hstack((labels, np.repeat(pad_label, n_pad, axis=0)))
        bbox_weights = np.vstack((bbox_weights, np.repeat(pad_weights, n_pad, axis=0)))
        bbox_targets = np.vstack((bbox_targets, np.repeat(pad_targets, n_pad, axis=0)))

    batch_index = im_i * np.ones((rois.shape[0], 1))
    rois_array_this_image = np.hstack((batch_index, rois))
    if rois_array_this_image.shape[0]==0:
        print 'Something Wrong2'
    rval = [mx.nd.array(rois_array_this_image), mx.nd.array(labels), mx.nd.array(bbox_targets), mx.nd.array(bbox_weights)]
    if has_mask:
        rval.append(encoded_polys)
    return rval

def chip_worker(r):
    width = r['width']
    height = r['height']
    im_size_max = max(width, height)
    im_scale_1 = 3
    im_scale_2 = 1.667
    im_scale_3 = 512.0 / float(im_size_max)

    gt_boxes = r['boxes'][np.where(r['max_overlaps'] == 1)[0], :]

    ws = (gt_boxes[:, 2] - gt_boxes[:, 0]).astype(np.int32)
    hs = (gt_boxes[:, 3] - gt_boxes[:, 1]).astype(np.int32)
    area = np.sqrt(ws * hs)
    ms = np.maximum(ws, hs)

    ids1 = np.where((area < 80) & (ms < 450.0 / im_scale_1) & (ws >= 2) & (hs >= 2))[0]
    ids2 = np.where((area >= 32) & (area < 150) & (ms < 450.0 / im_scale_2))[0]
    ids3 = np.where((area >= 120))[0]

    chips1 = genchips(int(r['width'] * im_scale_1), int(r['height'] * im_scale_1), gt_boxes[ids1, :] * im_scale_1, 512)
    chips2 = genchips(int(r['width'] * im_scale_2), int(r['height'] * im_scale_2), gt_boxes[ids2, :] * im_scale_2, 512)
    chips3 = genchips(int(r['width'] * im_scale_3), int(r['height'] * im_scale_3), gt_boxes[ids3, :] * im_scale_3, 512)
    chips1 = np.array(chips1) / im_scale_1
    chips2 = np.array(chips2) / im_scale_2
    chips3 = np.array(chips3) / im_scale_3

    chip_ar = []
    for chip in chips1:
        chip_ar.append([chip, im_scale_1, 512, 512])
    for chip in chips2:
        chip_ar.append([chip, im_scale_2, 512, 512])
    for chip in chips3:
        chip_ar.append([chip, im_scale_3, int(r['height'] * im_scale_3), int(r['width'] * im_scale_3)])

    return chip_ar

def props_in_chip_worker(r):
    props_in_chips = [[] for _ in range(len(r['crops']))]
    widths = (r['boxes'][:, 2] - r['boxes'][:, 0]).astype(np.int32)
    heights = (r['boxes'][:, 3] - r['boxes'][:, 1]).astype(np.int32)
    max_sizes = np.maximum(widths, heights)

    width = r['width']
    height = r['height']
    im_size_max = max(width, height)
    im_size_min = min(width, height)

    im_scale_1 = 3
    im_scale_2 = 1.667
    im_scale_3 = 512.0 / float(im_size_max)

    area = np.sqrt(widths * heights)

    sids = np.where((area < 80) & (max_sizes < 450.0 / im_scale_1) & (widths >= 2) & (heights >= 2))[0]
    mids = np.where((widths >= 2) & (heights >= 2) & (area < 150) & (max_sizes < 450.0 / im_scale_2))[0]
    bids = np.where((area >= 120))[0]

    chips1, chips2, chips3 = [], [], []
    chip_ids1, chip_ids2, chip_ids3 = [], [], []
    for ci, crop in enumerate(r['crops']):
        if crop[1] == im_scale_1:
            chips1.append(crop[0])
            chip_ids1.append(ci)
        elif crop[1] == im_scale_2:
            chips2.append(crop[0])
            chip_ids2.append(ci)
        else:
            chips3.append(crop[0])
            chip_ids3.append(ci)

    chips1 = np.array(chips1, dtype=np.float)
    chips2 = np.array(chips2, dtype=np.float)
    chips3 = np.array(chips3, dtype=np.float)
    chip_ids1 = np.array(chip_ids1)
    chip_ids2 = np.array(chip_ids2)
    chip_ids3 = np.array(chip_ids3)

    small_boxes = r['boxes'][sids].astype(np.float)
    med_boxes = r['boxes'][mids].astype(np.float)
    big_boxes = r['boxes'][bids].astype(np.float)

    small_covered = np.zeros(small_boxes.shape[0], dtype=bool)
    med_covered = np.zeros(med_boxes.shape[0], dtype=bool)
    big_covered = np.zeros(big_boxes.shape[0], dtype=bool)

    if chips1.shape[0] > 0:
        overlaps = ignore_overlaps(chips1, small_boxes)
        max_ids = overlaps.argmax(axis=0)
        for pi, cid in enumerate(max_ids):
            cur_chip = chips1[cid]
            cur_box = small_boxes[pi]
            x1 = max(cur_chip[0], cur_box[0])
            x2 = min(cur_chip[2], cur_box[2])
            y1 = max(cur_chip[1], cur_box[1])
            y2 = min(cur_chip[3], cur_box[3])
            area = math.sqrt(abs((x2 - x1) * (y2 - y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area < 80):
                props_in_chips[chip_ids1[cid]].append(sids[pi])
                small_covered[pi] = True

    if chips2.shape[0] > 0:
        overlaps = ignore_overlaps(chips2, med_boxes)
        max_ids = overlaps.argmax(axis=0)
        for pi, cid in enumerate(max_ids):
            cur_chip = chips2[cid]
            cur_box = med_boxes[pi]
            x1 = max(cur_chip[0], cur_box[0])
            x2 = min(cur_chip[2], cur_box[2])
            y1 = max(cur_chip[1], cur_box[1])
            y2 = min(cur_chip[3], cur_box[3])
            area = math.sqrt(abs((x2 - x1) * (y2 - y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 0 and area <= 150):
                props_in_chips[chip_ids2[cid]].append(mids[pi])
                med_covered[pi] = True


    if chips3.shape[0] > 0:
        overlaps = ignore_overlaps(chips3, big_boxes)
        max_ids = overlaps.argmax(axis=0)
        for pi, cid in enumerate(max_ids):
            cur_chip = chips3[cid]
            cur_box = big_boxes[pi]
            x1 = max(cur_chip[0], cur_box[0])
            x2 = min(cur_chip[2], cur_box[2])
            y1 = max(cur_chip[1], cur_box[1])
            y2 = min(cur_chip[3], cur_box[3])
            area = math.sqrt(abs((x2 - x1) * (y2 - y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 120):
                props_in_chips[chip_ids3[cid]].append(bids[pi])
                big_covered[pi] = True

    rem_small_boxes = small_boxes[np.where(small_covered == False)[0]]
    neg_sids = sids[np.where(small_covered == False)[0]]
    rem_med_boxes = med_boxes[np.where(med_covered == False)[0]]
    neg_mids = mids[np.where(med_covered == False)[0]]
    rem_big_boxes = big_boxes[np.where(big_covered == False)[0]]
    neg_bids = bids[np.where(big_covered == False)[0]]

    neg_chips1 = genchips(int(r['width'] * im_scale_1), int(r['height'] * im_scale_1), rem_small_boxes * im_scale_1,
                          512)
    neg_chips1 = np.array(neg_chips1, dtype=np.float) / im_scale_1
    chip_ids1 = np.arange(0, len(neg_chips1))
    neg_chips2 = genchips(int(r['width'] * im_scale_2), int(r['height'] * im_scale_2), rem_med_boxes * im_scale_2, 512)
    neg_chips2 = np.array(neg_chips2, dtype=np.float) / im_scale_2
    chip_ids2 = np.arange(len(neg_chips1), len(neg_chips2) + len(neg_chips1))
    neg_chips3 = genchips(int(r['width'] * im_scale_3), int(r['height'] * im_scale_3), rem_big_boxes * im_scale_3, 512)
    neg_chips3 = np.array(neg_chips3, dtype=np.float) / im_scale_3
    chip_ids3 = np.arange(len(neg_chips2) + len(neg_chips1), len(neg_chips1) + len(neg_chips2) + len(neg_chips3))

    neg_props_in_chips = [[] for _ in range(len(neg_chips1) + len(neg_chips2) + len(neg_chips3))]

    if neg_chips1.shape[0] > 0:
        overlaps = ignore_overlaps(neg_chips1, rem_small_boxes)
        max_ids = overlaps.argmax(axis=0)
        for pi, cid in enumerate(max_ids):
            cur_chip = neg_chips1[cid]
            cur_box = rem_small_boxes[pi]
            x1 = max(cur_chip[0], cur_box[0])
            x2 = min(cur_chip[2], cur_box[2])
            y1 = max(cur_chip[1], cur_box[1])
            y2 = min(cur_chip[3], cur_box[3])
            area = math.sqrt(abs((x2 - x1) * (y2 - y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area < 80):
                neg_props_in_chips[chip_ids1[cid]].append(neg_sids[pi])

    if neg_chips2.shape[0] > 0:
        overlaps = ignore_overlaps(neg_chips2, rem_med_boxes)
        max_ids = overlaps.argmax(axis=0)
        for pi, cid in enumerate(max_ids):
            cur_chip = neg_chips2[cid]
            cur_box = rem_med_boxes[pi]
            x1 = max(cur_chip[0], cur_box[0])
            x2 = min(cur_chip[2], cur_box[2])
            y1 = max(cur_chip[1], cur_box[1])
            y2 = min(cur_chip[3], cur_box[3])
            area = math.sqrt(abs((x2 - x1) * (y2 - y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 0 and area < 150):
                neg_props_in_chips[chip_ids2[cid]].append(neg_mids[pi])

    if neg_chips3.shape[0] > 0:
        overlaps = ignore_overlaps(neg_chips3, rem_big_boxes)
        max_ids = overlaps.argmax(axis=0)
        for pi, cid in enumerate(max_ids):
            cur_chip = neg_chips3[cid]
            cur_box = rem_big_boxes[pi]
            x1 = max(cur_chip[0], cur_box[0])
            x2 = min(cur_chip[2], cur_box[2])
            y1 = max(cur_chip[1], cur_box[1])
            y2 = min(cur_chip[3], cur_box[3])
            area = math.sqrt(abs((x2 - x1) * (y2 - y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 120):
                neg_props_in_chips[chip_ids3[cid]].append(neg_bids[pi])

    neg_chips = []
    final_neg_props_in_chips = []
    chip_counter = 0
    for chips, cscale in zip([neg_chips1, neg_chips2, neg_chips3], [im_scale_1, im_scale_2, im_scale_3]):
        for chip in chips:
            if len(neg_props_in_chips[chip_counter]) > 25 or (len(neg_props_in_chips[chip_counter]) > 10 and cscale != im_scale_1):
                final_neg_props_in_chips.append(np.array(neg_props_in_chips[chip_counter], dtype=int))
                if cscale != im_scale_3:
                    neg_chips.append([chip, cscale, 512, 512])
                else:
                    neg_chips.append([chip, cscale, int(r['height'] * im_scale_3), int(r['width'] * im_scale_3)])
                chip_counter += 1

    r['neg_chips'] = neg_chips
    r['neg_props_in_chips'] = final_neg_props_in_chips

    for j in range(len(props_in_chips)):
        props_in_chips[j] = np.array(props_in_chips[j], dtype=np.int32)

    return props_in_chips, neg_chips, final_neg_props_in_chips