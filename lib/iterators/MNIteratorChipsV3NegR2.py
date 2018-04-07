import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mxnet as mx
import cv2
import numpy as np
import math
from bbox.bbox_regression import expand_bbox_regression_targets
from MNIteratorBase import MNIteratorBase
from bbox.bbox_transform import bbox_overlaps, bbox_pred, bbox_transform, clip_boxes, filter_boxes, ignore_overlaps
from bbox.bbox_regression import compute_bbox_regression_targets
from chips import genchips, genchipsones, genscorechips
from multiprocessing import Pool
import time
from HelperV3 import im_worker, roidb_worker


def clip_boxes_with_chip(boxes, chip):
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], chip[2] - 1), chip[0])
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], chip[3] - 1), chip[1])
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], chip[2] - 1), chip[0])
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], chip[3] - 1), chip[1])
    return boxes


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

    ids1 = np.where((area < 80) & (ms < 450.0/im_scale_1) & (ws >= 2) & (hs >= 2))[0]
    ids2 = np.where((area >= 32) & (area < 150) & (ms < 450.0/im_scale_2))[0]
    ids3 = np.where((area >= 120))[0]

    chips1 = genchips(int(r['width'] * im_scale_1), int(r['height'] * im_scale_1), gt_boxes[ids1, :] * im_scale_1, 512)
    chips2 = genchips(int(r['width'] * im_scale_2), int(r['height'] * im_scale_2), gt_boxes[ids2, :] * im_scale_2, 512)
    chips3 = genchips(int(r['width'] * im_scale_3), int(r['height'] * im_scale_3), gt_boxes[ids3, :] * im_scale_3, 512)
    chips1 = np.array(chips1) / im_scale_1
    chips2 = np.array(chips2) / im_scale_2
    chips3 = np.array(chips3) / im_scale_3

    chip_ar = []
    for chip in chips1:
        chip_ar.append([chip, im_scale_1])
    for chip in chips2:
        chip_ar.append([chip, im_scale_2])
    for chip in chips3:
        chip_ar.append([chip, im_scale_3])

    return chip_ar

    # return (np.array(chips1),np.array(chips2).np.array(chips3),im_scale_3)

def chip_worker_two_scales(r):
    width = r['width']
    height = r['height']
    im_size_max = max(width, height)
    im_size_min = min(width, height)
    im_scale_2 = 800.0 / im_size_min
    im_scale_3 = 512.0 / float(im_size_max)

    gt_boxes = r['boxes'][np.where(r['max_overlaps'] == 1)[0], :]

    ws = (gt_boxes[:, 2] - gt_boxes[:, 0]).astype(np.int32)
    hs = (gt_boxes[:, 3] - gt_boxes[:, 1]).astype(np.int32)
    area = np.sqrt(ws * hs)
    ms = np.maximum(ws, hs)

    ids2 = np.where((area < 150) & (ms < 450.0/im_scale_2))[0]
    ids3 = np.where((area >= 120))[0]

    chips2 = genchips(int(r['width'] * im_scale_2), int(r['height'] * im_scale_2), gt_boxes[ids2, :] * im_scale_2, 512)
    chips3 = genchips(int(r['width'] * im_scale_3), int(r['height'] * im_scale_3), gt_boxes[ids3, :] * im_scale_3, 512)
    chips2 = np.array(chips2) / im_scale_2
    chips3 = np.array(chips3) / im_scale_3

    chip_ar = []
    for chip in chips2:
        chip_ar.append([chip, im_scale_2])
    for chip in chips3:
        chip_ar.append([chip, im_scale_3])

    return chip_ar


def chip_worker_one_scale(r):
    width = r['width']
    height = r['height']
    im_size_max = max(width, height)
    im_size_min = min(width, height)
    im_scale_2 = 800.0 / im_size_min

    gt_boxes = r['boxes'][np.where(r['max_overlaps'] == 1)[0], :]

    ws = (gt_boxes[:, 2] - gt_boxes[:, 0]).astype(np.int32)
    hs = (gt_boxes[:, 3] - gt_boxes[:, 1]).astype(np.int32)

    chips2 = genchipsones(int(r['width'] * im_scale_2), int(r['height'] * im_scale_2), gt_boxes * im_scale_2, 512)
    chips2 = np.array(chips2) / im_scale_2

    chip_ar = []
    for chip in chips2:
        chip_ar.append([chip, im_scale_2])

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

    sids = np.where((area < 80) & (max_sizes < 450.0/im_scale_1) & (widths >= 2) & (heights >= 2))[0]
    mids = np.where((area >= 32) & (area < 150) & (max_sizes < 450.0/im_scale_2))[0]
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
    score_small_boxes = r['proposal_scores'][sids].astype(np.float)
    med_boxes = r['boxes'][mids].astype(np.float)
    score_med_boxes = r['proposal_scores'][mids].astype(np.float)    
    big_boxes = r['boxes'][bids].astype(np.float)
    score_big_boxes = r['proposal_scores'][bids].astype(np.float)        

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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area < 80):
                props_in_chips[chip_ids1[cid]].append(sids[pi])
                small_covered[pi] = True
            #else:
            #    print ('quack')

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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 32 and area <= 150):
                props_in_chips[chip_ids2[cid]].append(mids[pi])
                med_covered[pi] = True
            #else:
            #    print ('quack 2')

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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 120):
                props_in_chips[chip_ids3[cid]].append(bids[pi])
                big_covered[pi] = True
            #else:
            #    print ('quack 3')

    rem_small_boxes = small_boxes[np.where(small_covered == False)[0]]
    neg_sids = sids[np.where(small_covered == False)[0]]
    score_sids = score_small_boxes[np.where(small_covered == False)[0]]
    rem_med_boxes = med_boxes[np.where(med_covered == False)[0]]
    neg_mids = mids[np.where(med_covered == False)[0]]
    score_mids = score_med_boxes[np.where(med_covered == False)[0]]    
    rem_big_boxes = big_boxes[np.where(big_covered == False)[0]]
    neg_bids = bids[np.where(big_covered == False)[0]]
    score_bids = score_big_boxes[np.where(big_covered == False)[0]]    

    neg_chips1 = genscorechips(int(r['width'] * im_scale_1), int(r['height'] * im_scale_1), rem_small_boxes * im_scale_1, 512, score_sids)
    neg_chips1 = np.array(neg_chips1, dtype=np.float) / im_scale_1
    chip_ids1 = np.arange(0, len(neg_chips1))
    neg_chips2 = genscorechips(int(r['width'] * im_scale_2), int(r['height'] * im_scale_2), rem_med_boxes * im_scale_2, 512, score_mids)
    neg_chips2 = np.array(neg_chips2, dtype=np.float) / im_scale_2
    chip_ids2 = np.arange(len(neg_chips1), len(neg_chips2) + len(neg_chips1))
    neg_chips3 = genscorechips(int(r['width'] * im_scale_3), int(r['height'] * im_scale_3), rem_big_boxes * im_scale_3, 512, score_bids)
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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))            
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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 32 and area < 150):
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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 120):
                neg_props_in_chips[chip_ids3[cid]].append(neg_bids[pi])

    neg_chips = []
    final_neg_props_in_chips = []
    chip_counter = 0
    for chips, cscale in zip([neg_chips1, neg_chips2, neg_chips3], [im_scale_1, im_scale_2, im_scale_3]):
        for chip in chips:
            if len(neg_props_in_chips[chip_counter]) > 40:
                final_neg_props_in_chips.append(np.array(neg_props_in_chips[chip_counter], dtype=int))
                neg_chips.append([chip, cscale])
            chip_counter += 1

    # import pdb;pdb.set_trace()
    r['neg_chips'] = neg_chips
    r['neg_props_in_chips'] = final_neg_props_in_chips

    for j in range(len(props_in_chips)):
        props_in_chips[j] = np.array(props_in_chips[j], dtype=np.int32)

    return props_in_chips,neg_chips,final_neg_props_in_chips


def props_in_chip_worker_late(r):
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

    sids = np.where((area < 80) & (max_sizes < 450.0/im_scale_1) & (widths >= 2) & (heights >= 2))[0]
    mids = np.where((area >= 32) & (area < 150) & (max_sizes < 450.0/im_scale_2))[0]
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
    score_small_boxes = r['proposal_scores'][sids].astype(np.float)
    med_boxes = r['boxes'][mids].astype(np.float)
    score_med_boxes = r['proposal_scores'][mids].astype(np.float)    
    big_boxes = r['boxes'][bids].astype(np.float)
    score_big_boxes = r['proposal_scores'][bids].astype(np.float)        

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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area < 80):
                props_in_chips[chip_ids1[cid]].append(sids[pi])
                small_covered[pi] = True
            #else:
            #    print ('quack')

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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 32 and area <= 150):
                props_in_chips[chip_ids2[cid]].append(mids[pi])
                med_covered[pi] = True
            #else:
            #    print ('quack 2')

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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 120):
                props_in_chips[chip_ids3[cid]].append(bids[pi])
                big_covered[pi] = True
            #else:
            #    print ('quack 3')

    rem_small_boxes = small_boxes[np.where(small_covered == False)[0]]
    neg_sids = sids[np.where(small_covered == False)[0]]
    score_sids = score_small_boxes[np.where(small_covered == False)[0]]
    rem_med_boxes = med_boxes[np.where(med_covered == False)[0]]
    neg_mids = mids[np.where(med_covered == False)[0]]
    score_mids = score_med_boxes[np.where(med_covered == False)[0]]    
    rem_big_boxes = big_boxes[np.where(big_covered == False)[0]]
    neg_bids = bids[np.where(big_covered == False)[0]]
    score_bids = score_big_boxes[np.where(big_covered == False)[0]]    

    neg_chips1 = genchips(int(r['width'] * im_scale_1), int(r['height'] * im_scale_1), rem_small_boxes * im_scale_1, 512)
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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))            
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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 32 and area < 150):
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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 120):
                neg_props_in_chips[chip_ids3[cid]].append(neg_bids[pi])

    neg_chips = []
    final_neg_props_in_chips = []
    chip_counter = 0
    for chips, cscale in zip([neg_chips1, neg_chips2, neg_chips3], [im_scale_1, im_scale_2, im_scale_3]):
        for chip in chips:
            if len(neg_props_in_chips[chip_counter]) > 40:
                final_neg_props_in_chips.append(np.array(neg_props_in_chips[chip_counter], dtype=int))
                neg_chips.append([chip, cscale])
            chip_counter += 1

    # import pdb;pdb.set_trace()
    r['neg_chips'] = neg_chips
    r['neg_props_in_chips'] = final_neg_props_in_chips

    for j in range(len(props_in_chips)):
        props_in_chips[j] = np.array(props_in_chips[j], dtype=np.int32)

    return props_in_chips,neg_chips,final_neg_props_in_chips



def props_in_chip_worker_two_scales(r):
    props_in_chips = [[] for _ in range(len(r['crops']))]
    widths = (r['boxes'][:, 2] - r['boxes'][:, 0]).astype(np.int32)
    heights = (r['boxes'][:, 3] - r['boxes'][:, 1]).astype(np.int32)
    max_sizes = np.maximum(widths, heights)

    width = r['width']
    height = r['height']
    im_size_max = max(width, height)
    im_size_min = min(width, height)

    im_scale_2 = 800.0 / im_size_min
    im_scale_3 = 512.0 / float(im_size_max)

    area = np.sqrt(widths * heights)

    mids = np.where((area < 150) & (max_sizes < 450.0/im_scale_2))[0]
    bids = np.where((area >= 120))[0]

    chips2, chips3 = [], []
    chip_ids2, chip_ids3 = [], []
    for ci, crop in enumerate(r['crops']):
        if crop[1] == im_scale_2:
            chips2.append(crop[0])
            chip_ids2.append(ci)
        else:
            chips3.append(crop[0])
            chip_ids3.append(ci)

    chips2 = np.array(chips2, dtype=np.float)
    chips3 = np.array(chips3, dtype=np.float)
    chip_ids2 = np.array(chip_ids2)
    chip_ids3 = np.array(chip_ids3)

    med_boxes = r['boxes'][mids].astype(np.float)
    big_boxes = r['boxes'][bids].astype(np.float)

    med_covered = np.zeros(med_boxes.shape[0], dtype=bool)
    big_covered = np.zeros(big_boxes.shape[0], dtype=bool)

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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area <= 150):
                props_in_chips[chip_ids2[cid]].append(mids[pi])
                med_covered[pi] = True
            #else:
            #    print ('quack 2')

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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 120):
                props_in_chips[chip_ids3[cid]].append(bids[pi])
                big_covered[pi] = True
            #else:
            #    print ('quack 3')

    rem_med_boxes = med_boxes[np.where(med_covered == False)[0]]
    neg_mids = mids[np.where(med_covered == False)[0]]
    rem_big_boxes = big_boxes[np.where(big_covered == False)[0]]
    neg_bids = bids[np.where(big_covered == False)[0]]

    neg_chips2 = genchips(int(r['width'] * im_scale_2), int(r['height'] * im_scale_2), rem_med_boxes * im_scale_2, 512)
    neg_chips2 = np.array(neg_chips2, dtype=np.float) / im_scale_2
    chip_ids2 = np.arange(len(neg_chips2))
    neg_chips3 = genchips(int(r['width'] * im_scale_3), int(r['height'] * im_scale_3), rem_big_boxes * im_scale_3, 512)
    neg_chips3 = np.array(neg_chips3, dtype=np.float) / im_scale_3
    chip_ids3 = np.arange(len(neg_chips2), len(neg_chips2) + len(neg_chips3))

    neg_props_in_chips = [[] for _ in range(len(neg_chips2) + len(neg_chips3))]

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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area < 150):
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
            area = math.sqrt(abs((x2-x1)*(y2-y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 120):
                neg_props_in_chips[chip_ids3[cid]].append(neg_bids[pi])

    neg_chips = []
    final_neg_props_in_chips = []
    chip_counter = 0
    for chips, cscale in zip([neg_chips2, neg_chips3], [im_scale_2, im_scale_3]):
        for chip in chips:
            if len(neg_props_in_chips[chip_counter]) > 50:
                final_neg_props_in_chips.append(np.array(neg_props_in_chips[chip_counter], dtype=int))
                neg_chips.append([chip, cscale])
            chip_counter += 1

    # import pdb;pdb.set_trace()
    r['neg_chips'] = neg_chips
    r['neg_props_in_chips'] = final_neg_props_in_chips

    for j in range(len(props_in_chips)):
        props_in_chips[j] = np.array(props_in_chips[j], dtype=np.int32)

    return props_in_chips,neg_chips,final_neg_props_in_chips

def props_in_chip_worker_one_scale(r):
    props_in_chips = [[] for _ in range(len(r['crops']))]
    widths = (r['boxes'][:, 2] - r['boxes'][:, 0]).astype(np.int32)
    heights = (r['boxes'][:, 3] - r['boxes'][:, 1]).astype(np.int32)
    max_sizes = np.maximum(widths, heights)

    width = r['width']
    height = r['height']

    im_size_min = min(width, height)
    mids = np.where(widths > -1)[0]
    im_scale_2 = 800.0 / im_size_min

    chips2 = []
    chip_ids2 = []
    for ci, crop in enumerate(r['crops']):
        chips2.append(crop[0])
        chip_ids2.append(ci)

    chips2 = np.array(chips2, dtype=np.float)
    chip_ids2 = np.array(chip_ids2)

    med_boxes = r['boxes'].astype(np.float)

    med_covered = np.zeros(med_boxes.shape[0], dtype=bool)

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
            if (x2 - x1 >= 1 and y2 - y1 >= 1):
                props_in_chips[chip_ids2[cid]].append(mids[pi])
                med_covered[pi] = True
            #else:
            #    print ('quack 2')


    rem_med_boxes = med_boxes[np.where(med_covered == False)[0]]
    neg_mids = mids[np.where(med_covered == False)[0]]

    neg_chips2 = genchipsones(int(r['width'] * im_scale_2), int(r['height'] * im_scale_2), rem_med_boxes * im_scale_2, 512)
    neg_chips2 = np.array(neg_chips2, dtype=np.float) / im_scale_2
    chip_ids2 = np.arange(len(neg_chips2))

    neg_props_in_chips = [[] for _ in range(len(neg_chips2))]

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
            if (x2 - x1 >= 1 and y2 - y1 >= 1):
                neg_props_in_chips[chip_ids2[cid]].append(neg_mids[pi])

    neg_chips = []
    final_neg_props_in_chips = []
    chip_counter = 0
    for chips, cscale in zip([neg_chips2], [im_scale_2]):
        for chip in chips:
            if len(neg_props_in_chips[chip_counter]) > 40:
                final_neg_props_in_chips.append(np.array(neg_props_in_chips[chip_counter], dtype=int))
                neg_chips.append([chip, cscale])
            chip_counter += 1

    # import pdb;pdb.set_trace()
    r['neg_chips'] = neg_chips
    r['neg_props_in_chips'] = final_neg_props_in_chips

    for j in range(len(props_in_chips)):
        props_in_chips[j] = np.array(props_in_chips[j], dtype=np.int32)

    return props_in_chips,neg_chips,final_neg_props_in_chips

class MNIteratorChips(MNIteratorBase):
    def __init__(self, roidb, config, batch_size=4, threads=8, nGPUs=1, pad_rois_to=400, crop_size=(512, 512)):
        self.crop_size = crop_size
        self.num_classes = roidb[0]['gt_overlaps'].shape[1]
        self.bbox_means = np.tile(np.array(config.TRAIN.BBOX_MEANS), (self.num_classes, 1))
        self.bbox_stds = np.tile(np.array(config.TRAIN.BBOX_STDS), (self.num_classes, 1))
        self.data_name = ['data', 'rois']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']
        self.pool = Pool(64)
        self.context_size = 320
        self.epiter = 0
        super(MNIteratorChips, self).__init__(roidb, config, batch_size, threads, nGPUs, pad_rois_to, False)

    def reset(self):
        self.cur_i = 0
        self.n_neg_per_im = 2
        self.crop_idx = [0] * len(self.roidb)
        print self.epiter

        if self.epiter > -1:
            chips = self.pool.map(chip_worker, self.roidb)
        else:
            chips = self.pool.map(chip_worker_one_scale, self.roidb)
        #chipindex = []
        chip_count = 0
        for i, r in enumerate(self.roidb):
            cs = chips[i]
            chip_count += len(cs)
            r['crops'] = cs
            #for j in range(len(cs)):
            #    chipindex.append(i)

        if self.epiter < 3:
            all_props_in_chips = self.pool.map(props_in_chip_worker, self.roidb)
        else:
            all_props_in_chips = self.pool.map(props_in_chip_worker_late, self.roidb)

        for (props_in_chips, neg_chips, neg_props_in_chips), cur_roidb in zip(all_props_in_chips, self.roidb):
            cur_roidb['props_in_chips'] = props_in_chips
            cur_roidb['neg_crops'] = neg_chips
            cur_roidb['neg_props_in_chips'] = neg_props_in_chips

        # Append negative chips
        chipindex = []
        for i, r in enumerate(self.roidb):
            cs = r['neg_crops']
            if len(cs) > 0:
                sel_inds = np.arange(len(cs))
                if len(cs) > self.n_neg_per_im:
                    sel_inds = np.random.permutation(sel_inds)[0:self.n_neg_per_im]
                for ind in sel_inds:
                    chip_count = chip_count + 1
                    r['crops'].append(r['neg_crops'][ind])
                    r['props_in_chips'].append(r['neg_props_in_chips'][ind].astype(np.int32))
            all_crops = r['crops']
            for j in range(len(all_crops)):
                chipindex.append(i)


        print('quack N chips: {}'.format(chip_count))

        blocksize = self.batch_size
        chipindex = np.array(chipindex)
        if chipindex.shape[0] % blocksize > 0:
            extra = blocksize - (chipindex.shape[0] % blocksize)
            chipindex = np.hstack((chipindex, chipindex[0:extra]))
        allinds = np.random.permutation(chipindex)
        self.inds = np.array(allinds, dtype=int)
        for r in self.roidb:
            r['chip_order'] = np.random.permutation(np.arange(len(r['crops'])))

        self.epiter = self.epiter + 1
        self.size = len(self.inds)
        print 'Done!'

    def get_batch(self):
        if self.cur_i >= self.size:
            return False

        # cur_roidbs = [self.roidb[self.inds[i%self.size]] for i in range(self.cur_i, self.cur_i+self.batch_size)]

        # Process cur roidb
        self.batch = self._get_batch()

        self.cur_i += self.batch_size
        return True

    def _get_batch(self):
        """
        return a dict of multiple images
        :param roidb: a list of dict, whose length controls batch size
        ['images', 'flipped'] + ['gt_boxes', 'boxes', 'gt_overlap'] => ['bbox_targets']
        :return: data, label
        """

        cur_from = self.cur_i
        cur_to = self.cur_i + self.batch_size
        roidb = [self.roidb[self.inds[i]] for i in range(cur_from, cur_to)]
        # num_images = len(roidb)
        cropids = [self.roidb[self.inds[i]]['chip_order'][self.crop_idx[self.inds[i]]%len(self.roidb[self.inds[i]]['chip_order'])] for i in range(cur_from, cur_to)]
        n_batch = len(roidb)
        ims = []
        for i in range(n_batch):
            ims.append([roidb[i]['image'], roidb[i]['crops'][cropids[i]], roidb[i]['flipped'], self.crop_size[0]])

        for i in range(cur_from, cur_to):
            self.crop_idx[self.inds[i]] = self.crop_idx[self.inds[i]] + 1

        processed_list = self.thread_pool.map(im_worker, ims)
        # im_tensor, roidb = self.im_process(roidb,cropids)
        processed_roidb = []
        t1 = time.time()
        for i in range(len(roidb)):
            tmp = roidb[i].copy()
            scale = roidb[i]['crops'][cropids[i]][1]
            tmp['im_info'] = [self.crop_size[0], self.crop_size[1], scale]
            processed_roidb.append(tmp)
        im_tensor = mx.nd.zeros((n_batch, 3, self.crop_size[0], self.crop_size[1]), dtype=np.float32)
        for i in range(len(processed_list)):
            im_tensor[i] = processed_list[i]

        worker_data = []
        for i in range(len(processed_roidb)):
            cropid = cropids[i]
            tmp = processed_roidb[i]['crops'][cropid][0]
            nids = processed_roidb[i]['props_in_chips'][cropid]
            gtids = np.where(processed_roidb[i]['max_overlaps'] == 1)[0]
            gt_boxes = processed_roidb[i]['boxes'][gtids, :]
            gt_classes = processed_roidb[i]['max_classes'][gtids]
            boxes = processed_roidb[i]['boxes'].copy()
            cur_crop = processed_roidb[i]['crops'][cropid][0]
            im_scale = processed_roidb[i]['crops'][cropid][1]

            argw = [i % self.n_per_gpu, processed_roidb[i]['im_info'], cur_crop, im_scale, nids, gt_boxes, gt_classes,
                    boxes, self.n_expected_roi, self.num_classes]
            worker_data.append(argw)

        # all_labels = self.thread_pool.map(self.roidb_worker,worker_data)
        all_labels = self.pool.map(roidb_worker, worker_data)

        rois = mx.nd.zeros((n_batch, self.n_expected_roi, 5), mx.cpu(0))
        labels = mx.nd.zeros((n_batch, self.n_expected_roi), mx.cpu(0))
        bbox_targets = mx.nd.zeros((n_batch, self.n_expected_roi, 8), mx.cpu(0))
        bbox_weights = mx.nd.zeros((n_batch, self.n_expected_roi, 8), mx.cpu(0))
        for i in range(len(all_labels)):
            rois[i] = all_labels[i][0]
            labels[i] = all_labels[i][1]
            bbox_targets[i] = all_labels[i][2]
            bbox_weights[i] = all_labels[i][3]
        #self.visualize(im_tensor, rois, labels)
        self.data = [im_tensor, rois]
        self.label = [labels, bbox_targets, bbox_weights]
        return mx.io.DataBatch(data=self.data, label=self.label, pad=self.getpad(), index=self.getindex(),
                               provide_data=self.provide_data, provide_label=self.provide_label)


    def visualize(self, im_tensor, boxes, labels): # , bbox_targets, bbox_weights):
        # import pdb;pdb.set_trace()
        im_tensor = im_tensor.asnumpy()
        boxes = boxes.asnumpy()

        for imi in range(im_tensor.shape[0]):
            im = np.zeros((im_tensor.shape[2], im_tensor.shape[3], 3), dtype=np.uint8)
            for i in range(3):
                im[:, :, i] = im_tensor[imi, i, :, :] + self.pixel_mean[2 - i]
            # Visualize positives
            plt.imshow(im)
            pos_ids = np.where(labels[imi].asnumpy() > 0)[0]
            cboxes = boxes[imi][pos_ids, 1:5]
            # cboxes = boxes[imi][:, 0:4]
            for box in cboxes:
                rect = plt.Rectangle((box[0], box[1]),
                                     box[2] - box[0],
                                     box[3] - box[1], fill=False,
                                     edgecolor='green', linewidth=3.5)
                plt.gca().add_patch(rect)
            num = np.random.randint(100000)
            #plt.show()
            plt.savefig('debug/visualization/test_{}_pos.png'.format(num))
            plt.cla()
            plt.clf()
            plt.close()

