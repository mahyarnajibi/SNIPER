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
from chips.chip_generator import chip_generator
import math
import copy_reg
import types


# Pickle dumping recipe for using classes with multi-processing map
def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


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


class anchor_worker(object):
    def __init__(self, cfg, chip_size, max_n_gts=100, max_poly_len=500):
        self.scales = np.array(cfg.network.ANCHOR_SCALES, dtype=np.float32)
        self.ratios = cfg.network.ANCHOR_RATIOS
        feat_stride = cfg.network.RPN_FEAT_STRIDE
        self.max_n_gts = max_n_gts
        self.max_poly_len = max_poly_len

        # Initializing anchors
        base_anchors = generate_anchors(base_size=feat_stride, ratios=list(self.ratios),
                                             scales=list(self.scales))
        self.num_anchors = base_anchors.shape[0]
        self.feat_width = chip_size / cfg.network.RPN_FEAT_STRIDE
        self.feat_height = chip_size / cfg.network.RPN_FEAT_STRIDE
        shift_x = np.arange(0, self.feat_width) * feat_stride
        shift_y = np.arange(0, self.feat_height) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        self.K = shifts.shape[0]
        all_anchors = base_anchors.reshape((1, self.num_anchors, 4)) + \
                      shifts.reshape((1, self.K, 4)).transpose((1, 0, 2))
        self.all_anchors = all_anchors.reshape((self.K * self.num_anchors, 4))
        self.batch_size = cfg.TRAIN.RPN_BATCH_SIZE
        self.pos_thresh = cfg.TRAIN.RPN_POSITIVE_OVERLAP
        self.neg_thresh = cfg.TRAIN.RPN_NEGATIVE_OVERLAP
        self.num_fg = int(self.batch_size * cfg.TRAIN.RPN_FG_FRACTION)

    def worker(self, data):
        im_info, cur_crop, im_scale, nids, gtids, gt_boxes, boxes, classes = data
        import pdb;pdb.set_trace()
        has_mask = True if len(data) > 8 else False

        anchors = self.all_anchors.copy()
        inds_inside = np.where((anchors[:, 0] >= -32) &
                               (anchors[:, 1] >= -32) &
                               (anchors[:, 2] < im_info[0] + 32) &
                               (anchors[:, 3] < im_info[1] + 32))[0]

        anchors = anchors[inds_inside, :]
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)
        total_anchors = int(self.K * self.num_anchors)

        gt_boxes[:, 0] -= cur_crop[0]
        gt_boxes[:, 2] -= cur_crop[0]
        gt_boxes[:, 1] -= cur_crop[1]
        gt_boxes[:, 3] -= cur_crop[1]

        vgt_boxes = boxes[np.intersect1d(gtids, nids)]

        vgt_boxes[:, 0] -= cur_crop[0]
        vgt_boxes[:, 2] -= cur_crop[0]
        vgt_boxes[:, 1] -= cur_crop[1]
        vgt_boxes[:, 3] -= cur_crop[1]

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

                encoded_polys = poly_encoder(mask_polys, classes[:, 0] - 1,
                                             max_poly_len=self.max_poly_len, max_n_gts=self.max_n_gts)
            else:
                encoded_polys = -np.ones((self.max_n_gts, self.max_poly_len), dtype=np.float32)
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
            ov = bbox_overlaps(np.ascontiguousarray(gt_boxes).astype(float),
                               np.ascontiguousarray(vgt_boxes).astype(float))
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

            labels[max_overlaps < self.neg_thresh] = 0
            labels[gt_argmax_overlaps] = 1

            # fg label: above threshold IoU
            labels[max_overlaps >= self.pos_thresh] = 1

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
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > self.num_fg:
            disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - self.num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.batch_size - np.sum(labels == 1)
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

        labels = labels.reshape((1, self.feat_height, self.feat_width, self.num_anchors)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, self.num_anchors * self.feat_height * self.feat_width)).astype(np.float16)
        bbox_targets = bbox_targets.reshape((self.feat_height, self.feat_width, self.num_anchors * 4)).transpose(2, 0, 1)
        bbox_weights = bbox_weights.reshape((self.feat_height, self.feat_width, self.num_anchors * 4)).transpose((2, 0, 1))
        pids = np.where(bbox_weights == 1)
        bbox_targets = bbox_targets[pids]

        fgt_boxes = -np.ones((100, 5))
        if len(agt_boxes) > 0:
            fgt_boxes[:min(len(agt_boxes), 100), :] = np.hstack((agt_boxes, classes))

        rval = [mx.nd.array(labels, dtype='float16'), bbox_targets, mx.nd.array(pids), mx.nd.array(fgt_boxes)]
        import pdb;pdb.set_trace()
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


class chip_worker(object):
    def __init__(self, cfg, chip_size):
        self.valid_ranges = cfg.TRAIN.VALID_RANGES
        self.scales = cfg.TRAIN.SCALES
        self.chip_size = chip_size
        self.use_cpp = cfg.TRAIN.CPP_CHIPS
        self.chip_stride = 32
        self.chip_generator = chip_generator(chip_stride=self.chip_stride, use_cpp=self.use_cpp)
        self.use_neg_chips = cfg.TRAIN.USE_NEG_CHIPS

    def chip_extractor(self, r):
        width = r['width']
        height = r['height']
        im_size_max = max(width, height)

        gt_boxes = r['boxes'][np.where(r['max_overlaps'] == 1)[0], :]

        ws = (gt_boxes[:, 2] - gt_boxes[:, 0]).astype(np.int32)
        hs = (gt_boxes[:, 3] - gt_boxes[:, 1]).astype(np.int32)
        area = np.sqrt(ws * hs)
        ms = np.maximum(ws, hs)

        chip_ar = []
        for i, im_scale in enumerate(self.scales):
            if i == len(self.scales)-1:
                # The coarsest (or possibly the only scale)
                im_scale /= float(im_size_max)
                ids = np.where((area >= self.valid_ranges[i][0]))[0]
            elif i == 0:
                # The finest scale (but not the only scale)
                ids = np.where((area < self.valid_ranges[i][1]) &
                               (ms < (self.chip_size - self.chip_stride - 1) / im_scale) & (ws >= 2) & (hs >= 2))[0]
            else:
                # An intermediate scale
                ids = np.where((area >= self.valid_ranges[i][0]) & (area < self.valid_ranges[i][1])
                       & (ms < (self.chip_size - self.chip_stride - 1) / im_scale))[0]

            cur_chips = self.chip_generator.generate(gt_boxes[ids, :] * im_scale, int(r['width'] * im_scale), int(r['height'] * im_scale),
                                 self.chip_size)
            cur_chips = np.array(cur_chips) / im_scale
            if i != len(self.scales) - 1:
                for chip in cur_chips:
                    chip_ar.append([chip, im_scale, self.chip_size, self.chip_size])
            else:
                for chip in cur_chips:
                    chip_ar.append([chip, im_scale, int(r['height'] * im_scale), int(r['width'] * im_scale)])

        return chip_ar

    def box_assigner(self, r):
        props_in_chips = [[] for _ in range(len(r['crops']))]
        widths = (r['boxes'][:, 2] - r['boxes'][:, 0]).astype(np.int32)
        heights = (r['boxes'][:, 3] - r['boxes'][:, 1]).astype(np.int32)
        max_sizes = np.maximum(widths, heights)
        width = r['width']
        height = r['height']
        area = np.sqrt(widths * heights)
        im_size_max = max(width, height)

        # ** Distribute chips based on the scales
        all_chips = [[] for _ in self.scales]
        all_chip_ids = [[] for _ in self.scales]
        for ci, crop in enumerate(r['crops']):
            for scale_i, s in enumerate(self.scales):
                if (scale_i == len(self.scales) - 1) or s == crop[1]:
                    all_chips[scale_i].append(crop[0])
                    all_chip_ids[scale_i].append(ci)
                    break

        # All chips in each of the scales:
        all_chips = [np.array(chips) for chips in all_chips]
        # The ids of chips in each of the scales:
        all_chip_ids = [np.array(chip_ids) for chip_ids in all_chip_ids]

        # ** Find valid boxes in each scale
        valid_ids = []
        for scale_i, im_scale in enumerate(self.scales):
            if scale_i == len(self.scales) - 1:
                # The coarsest scale (or the only scale)
                ids = np.where((area >= self.valid_ranges[scale_i][0]))[0]
            else:
                # Other scales
                ids = np.where((area < self.valid_ranges[scale_i][1]) &
                               (max_sizes < (self.chip_size - self.chip_stride - 1) / im_scale) &
                               (widths >= 2) & (heights >= 2))[0]
            valid_ids.append(ids)
        valid_boxes = [r['boxes'][ids].astype(np.float) for ids in valid_ids]

        # ** Assign boxes to the chips in each scale based on the maximum overlap
        # ** and keep track of the assigned boxes for neg sampling
        covered_boxes = [np.zeros(boxes.shape[0], dtype=bool) for boxes in valid_boxes]
        for scale_i, chips in enumerate(all_chips):
            if chips.shape[0]>0:
                overlaps = ignore_overlaps(chips, valid_boxes[scale_i])
                max_ids = overlaps.argmax(axis=0)
                for pi, cid in enumerate(max_ids):
                    cur_chip = chips[cid]
                    cur_box = valid_boxes[scale_i][pi]
                    x1, x2, y1, y2 = max(cur_chip[0], cur_box[0]), min(cur_chip[2], cur_box[2]), \
                                     max(cur_chip[1], cur_box[1]), min(cur_chip[3], cur_box[3])
                    area = math.sqrt(abs((x2 - x1) * (y2 - y1)))
                    if scale_i == len(self.scales) - 1:
                        # The coarsest scale (or the only scale)
                        if x2 - x1 >= 1 and y2 - y1 >= 1 and area >= self.valid_ranges[scale_i][0]:
                            props_in_chips[all_chip_ids[scale_i][cid]].append(valid_ids[scale_i][pi])
                            covered_boxes[scale_i][pi] = True
                    else:
                        if x2 - x1 >= 1 and y2 - y1 >= 1 and area <= self.valid_ranges[scale_i][1]:
                            props_in_chips[all_chip_ids[scale_i][cid]].append(valid_ids[scale_i][pi])
                            covered_boxes[scale_i][pi] = True
        if self.use_neg_chips:
            # ** Generate negative chips based on remaining boxes
            rem_valid_boxes = [valid_boxes[i][np.where(covered_boxes[i] == False)[0]] for i in range(len(self.scales))]
            neg_chips = []
            neg_props_in_chips = []
            first_available_chip_id = 0
            neg_chip_ids = []
            for scale_i, im_scale in enumerate(self.scales):
                if scale_i == len(self.scales) - 1:
                    im_scale /= float(im_size_max)
                chips = self.chip_generator.generate(rem_valid_boxes[scale_i] * im_scale, int(r['width'] * im_scale),
                                            int(r['height'] * im_scale), self.chip_size)
                neg_chips.append(np.array(chips, dtype=np.float) / im_scale)
                neg_props_in_chips += [[] for _ in chips]
                neg_chip_ids.append(np.arange(first_available_chip_id,first_available_chip_id+len(chips)))
                first_available_chip_id += len(chips)

            # ** Assign remaining boxes to negative chips based on max overlap
            neg_ids = [valid_ids[i][np.where(covered_boxes[i] == False)[0]] for i in range(len(self.scales))]
            for scale_i in range(len(self.scales)):
                if neg_chips[scale_i].shape[0]>0:
                    overlaps = ignore_overlaps(neg_chips[scale_i], rem_valid_boxes[scale_i])
                    max_ids = overlaps.argmax(axis=0)
                    for pi, cid in enumerate(max_ids):
                        cur_chip = neg_chips[scale_i][cid]
                        cur_box = rem_valid_boxes[scale_i][pi]
                        x1, x2, y1, y2 = max(cur_chip[0], cur_box[0]), min(cur_chip[2], cur_box[2]), \
                                         max(cur_chip[1], cur_box[1]), min(cur_chip[3], cur_box[3])
                        area = math.sqrt(abs((x2 - x1) * (y2 - y1)))
                        if scale_i == len(self.scales) - 1:
                            if x2 - x1 >= 1 and y2 - y1 >= 1 and area >= self.valid_ranges[scale_i][0]:
                                neg_props_in_chips[neg_chip_ids[scale_i][cid]].append(neg_ids[scale_i][pi])
                        else:
                            if x2 - x1 >= 1 and y2 - y1 >= 1 and area < self.valid_ranges[scale_i][1]:
                                neg_props_in_chips[neg_chip_ids[scale_i][cid]].append(neg_ids[scale_i][pi])
            # Final negative chips extracted:
            final_neg_chips = []
            # IDs of proposals which are valid inside each of the negative chips:
            final_neg_props_in_chips = []
            chip_counter = 0
            for scale_i, chips in enumerate(neg_chips):
                im_scale = self.scales[scale_i] / float(im_size_max) if scale_i == len(self.scales) - 1 else \
                           self.scales[scale_i]
                for chip in chips:
                    if len(neg_props_in_chips[chip_counter]) > 25 or (
                                    len(neg_props_in_chips[chip_counter]) > 10 and scale_i != 0):
                        final_neg_props_in_chips.append(np.array(neg_props_in_chips[chip_counter], dtype=int))
                        if scale_i != len(self.scales) - 1:
                            final_neg_chips.append([chip, im_scale, self.chip_size, self.chip_size])
                        else:
                            final_neg_chips.append(
                                [chip, im_scale, int(r['height'] * im_scale), int(r['width'] * im_scale)])
                        chip_counter += 1

            r['neg_chips'] = final_neg_chips
            r['neg_props_in_chips'] = final_neg_props_in_chips
        for j in range(len(props_in_chips)):
            props_in_chips[j] = np.array(props_in_chips[j], dtype=np.int32)
        if self.use_neg_chips:
            return props_in_chips, final_neg_chips, final_neg_props_in_chips
        else:
            return props_in_chips
