import cv2
import mxnet as mx
import numpy as np
from bbox.bbox_transform import *
from bbox.bbox_regression import expand_bbox_regression_targets

def im_worker(data):
    imp = data[0]
    crop = data[1]
    flipped = data[2]
    crop_size = data[3]
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
    rim[0, :d1m, :d2m] = im[:d1m, :d2m, 2] - 123.15
    rim[1, :d1m, :d2m] = im[:d1m, :d2m, 1] - 115.90
    rim[2, :d1m, :d2m] = im[:d1m, :d2m, 0] - 103.06
    return mx.nd.array(rim, dtype='float32')


def sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes,
        labels=None, overlaps=None, bbox_targets=None, gt_boxes=None, scale_sd=1):
    if labels is None:
        #overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float)
        overlaps = bbox_overlaps(rois.astype(np.float), gt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]

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
    ids = filter_boxes(gt_boxes, 10)
    if len(ids)>0:
        gt_boxes = gt_boxes[ids]
        gt_labs = gt_labs[ids]                
        gt_boxes = np.hstack((gt_boxes, gt_labs.reshape(len(gt_labs), 1)))


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

    overlaps = ignore_overlaps(rois.astype(np.float), gt_boxes.astype(np.float))
    mov = np.max(overlaps)

    if mov < 1:
        print 'Something Wrong 1'
        import pdb;pdb.set_trace()

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
