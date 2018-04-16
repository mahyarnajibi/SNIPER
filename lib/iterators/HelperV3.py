import cv2
import mxnet as mx
import numpy as np
import numpy.random as npr
from bbox.bbox_transform import *
from bbox.bbox_regression import expand_bbox_regression_targets
from generate_anchor import generate_anchors

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


class im_worker(object):
    def __init__(self,cfg,crop_size):
        self.cfg = cfg
        self.crop_size = crop_size

    def worker(self,data):
        imp = data[0]
        crop = data[1]
        flipped = data[2]
        crop_size = self.crop_size
        pixel_means = self.cfg.network.PIXEL_MEANS

        im = cv2.imread(imp, cv2.IMREAD_COLOR)
        
        # Crop the image
        crop_scale = crop[1]
        if flipped:
            im = im[:, ::-1, :]
        
        origim = im[int(crop[0][1]):int(crop[0][3]),int(crop[0][0]):int(crop[0][2]),:]
        
        # Scale the image
        crop_scale = crop[1]
        
        # Resize the crop
        if int(origim.shape[0]*0.625)==0 or int(origim.shape[1]*0.625)==0:
            print 'Something wrong3'
        try:
            im = cv2.resize(origim, None, None, fx=crop_scale, fy=crop_scale, interpolation=cv2.INTER_LINEAR)
        except:
            print 'Something wrong4'
        
        rim = np.zeros((3, crop_size, crop_size), dtype=np.float32)
        d1m = min(im.shape[0], crop_size)
        d2m = min(im.shape[1], crop_size)
        if not self.cfg.IS_DPN:
            for j in range(3):
                rim[j, :d1m, :d2m] = im[:d1m, :d2m, 2-j] - pixel_means[2-j]
        else:
            for j in range(3):
                rim[j, :d1m, :d2m] = (im[:d1m, :d2m, 2-j] - pixel_means[2-j]) * 0.0167


        return mx.nd.array(rim, dtype='float32')


def sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes,
        labels=None, overlaps=None, bbox_targets=None, gt_boxes=None, scale_sd=1):
    if labels is None:
        #overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float)
		if len(gt_boxes) > 0:
		    overlaps = bbox_overlaps(rois.astype(np.float), gt_boxes[:, :4].astype(np.float))
		    gt_assignment = overlaps.argmax(axis=1)
		    overlaps = overlaps.max(axis=1)
		    labels = gt_boxes[gt_assignment, 4]
		else:
			return rois, np.zeros((len(rois))), np.zeros((len(rois), 8)), np.zeros((len(rois), 8))
        

    thresh = 0.5

    if scale_sd == 0.5:
        thresh = 0.6

    if scale_sd == 0.25:
        thresh = 0.7    

    fg_indexes = np.where(overlaps >= thresh)[0]

    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < 0.5))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        #targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
        targets = bbox_transform(rois, gt_boxes[gt_assignment[keep_indexes], :4])
        targets = ((targets - np.array([0, 0, 0, 0]))
                       / (np.array([0.1, 0.1, 0.2, 0.2]) * scale_sd ))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
    expand_bbox_regression_targets(bbox_target_data, num_classes)

    return rois, labels, bbox_targets, bbox_weights


def roidb_anchor_worker(data):
    im_info = data[0]
    cur_crop = data[1]
    im_scale = data[2]
    nids = data[3]
    gtids = data[4]
    gt_boxes = data[5]
    boxes = data[6]

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
    if len(ids)>0:
        gt_boxes = gt_boxes[ids]
    else:
        gt_boxes = np.zeros((0, 4))

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

    rval = [mx.nd.array(labels, dtype='float16'), bbox_targets, mx.nd.array(pids)]
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
    else:
        gt_boxes = np.zeros((0, 5))


    #crois = new_rec['boxes'].copy()
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

    #overlaps = ignore_overlaps(rois.astype(np.float), gt_boxes.astype(np.float))
    #mov = np.max(overlaps)

    #if mov < 1:
    #    print 'Something Wrong 1'
    #    import pdb;pdb.set_trace()

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
    """rois = np.zeros((400, 4))
    bbox_targets = np.zeros((400, 8))
    bbox_weights = np.zeros((400, 8))
    gt_boxes = np.zeros((5, 5))
    labels = np.zeros((400, 1))"""

    batch_index = im_i * np.ones((rois.shape[0], 1))
    rois_array_this_image = np.hstack((batch_index, rois))
    if rois_array_this_image.shape[0]==0:
        print 'Something Wrong2'
    rval = [mx.nd.array(rois_array_this_image), mx.nd.array(labels), mx.nd.array(bbox_targets), mx.nd.array(bbox_weights)]
    return rval
