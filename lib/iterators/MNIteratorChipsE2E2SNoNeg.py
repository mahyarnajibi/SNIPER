import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mxnet as mx
import cv2
import numpy as np
import math
from bbox.bbox_regression import expand_bbox_regression_targets
from MNIteratorBase import MNIteratorBase
from bbox.bbox_transform import bbox_overlaps, bbox_pred, bbox_transform, clip_boxes, filter_boxes, ignore_overlaps
from bbox.bbox_regression import compute_bbox_regression_targets
from chips import genchips
from multiprocessing import Pool
import time
from HelperV3 import im_worker, roidb_worker, roidb_anchor_worker


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
    im_scale_2 = 1
    im_scale_3 = 512.0 / float(im_size_max)

    gt_boxes = r['boxes'][np.where(r['max_overlaps'] == 1)[0], :]

    ws = (gt_boxes[:, 2] - gt_boxes[:, 0]).astype(np.int32)
    hs = (gt_boxes[:, 3] - gt_boxes[:, 1]).astype(np.int32)
    area = np.sqrt(ws * hs)
    ms = np.maximum(ws, hs)

    ids2 = np.where((area < 300) & (ms < 450.0 / im_scale_2))[0]
    ids3 = np.where((area >= 300) | (ms >= 450.0 / im_scale_2))[0]

    chips2 = genchips(int(r['width'] * im_scale_2), int(r['height'] * im_scale_2), gt_boxes[ids2, :] * im_scale_2, 512)
    chips3 = genchips(int(r['width'] * im_scale_3), int(r['height'] * im_scale_3), gt_boxes[ids3, :] * im_scale_3, 512)
    chips2 = np.array(chips2) / im_scale_2
    chips3 = np.array(chips3) / im_scale_3

    chip_ar = []
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

    im_scale_2 = 1
    im_scale_3 = 512.0 / float(im_size_max)

    area = np.sqrt(widths * heights)

    mids = np.where((widths >= 2) & (heights >= 2) & (area < 300) & (max_sizes < 450.0 / im_scale_2))[0]
    bids = np.where((area >= 300) | (max_sizes >= 450.0 / im_scale_2))[0] 

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
            area = math.sqrt(abs((x2 - x1) * (y2 - y1)))
            if (x2 - x1 >= 1 and y2 - y1 >= 1 and area >= 0 and area <= 300):
                props_in_chips[chip_ids2[cid]].append(mids[pi])

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
            w = x2 - x1
            h = y2 - y1
            ms = max(w, h)
            area = math.sqrt(abs((x2 - x1) * (y2 - y1)))

            if x2 - x1 >= 1 and y2 - y1 >= 1 and (area > 300.0 or ms > 450.0):
                props_in_chips[chip_ids3[cid]].append(bids[pi])

    for j in range(len(props_in_chips)):
        props_in_chips[j] = np.array(props_in_chips[j], dtype=np.int32)

    return props_in_chips



class MNIteratorChips(MNIteratorBase):
    def __init__(self, roidb, config, batch_size=4, threads=8, nGPUs=1, pad_rois_to=400, crop_size=(512, 512)):
        self.crop_size = crop_size
        self.num_classes = 602 #roidb[0]['gt_overlaps'].shape[1]
        self.bbox_means = np.tile(np.array(config.TRAIN.BBOX_MEANS), (self.num_classes, 1))
        self.bbox_stds = np.tile(np.array(config.TRAIN.BBOX_STDS), (self.num_classes, 1))
        self.data_name = ['data', 'valid_ranges', 'im_info']
        self.label_name = ['label', 'bbox_target', 'bbox_weight', 'gt_boxes', 'crowd_boxes']
        self.pool = Pool(64)
        self.context_size = 320
        self.epiter = 0
        self.im_worker = im_worker(crop_size=self.crop_size[0], cfg=config)
        super(MNIteratorChips, self).__init__(roidb, config, batch_size, threads, nGPUs, pad_rois_to, False)

    def reset(self):
        self.cur_i = 0
        self.crop_idx = [0] * len(self.roidb)
        """if self.epiter != 0:
            allinds = np.random.permutation(self.chipindexsave)
            self.inds = np.array(allinds, dtype=int)
            for r in self.roidb:
                r['chip_order'] = np.random.permutation(np.arange(len(r['crops'])))
            self.epiter = self.epiter + 1                
            return"""
            
        chips = self.pool.map(chip_worker, self.roidb)
        # chipindex = []
        chip_count = 0
        for i, r in enumerate(self.roidb):
            cs = chips[i]
            chip_count += len(cs)
            r['crops'] = cs

        #all_props_in_chips = []
        #for r in self.roidb:
        #    all_props_in_chips.append(props_in_chip_worker(r))
        all_props_in_chips = self.pool.map(props_in_chip_worker, self.roidb)

        for props_in_chips, cur_roidb in zip(all_props_in_chips, self.roidb):
            cur_roidb['props_in_chips'] = props_in_chips


        chipindex = []
        for i, r in enumerate(self.roidb):
            all_crops = r['crops']
            for j in range(len(all_crops)):
                chipindex.append(i)

        print('quack N chips: {}'.format(chip_count))

        blocksize = self.batch_size
        chipindex = np.array(chipindex)
        if chipindex.shape[0] % blocksize > 0:
            extra = blocksize - (chipindex.shape[0] % blocksize)
            chipindex = np.hstack((chipindex, chipindex[0:extra]))
            
        self.chipindexsave = chipindex
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
        import time
        t1 = time.time()

        cur_from = self.cur_i
        cur_to = self.cur_i + self.batch_size
        roidb = [self.roidb[self.inds[i]] for i in range(cur_from, cur_to)]
        # num_images = len(roidb)
        cropids = [self.roidb[self.inds[i]]['chip_order'][
                       self.crop_idx[self.inds[i]] % len(self.roidb[self.inds[i]]['chip_order'])] for i in
                   range(cur_from, cur_to)]
        n_batch = len(roidb)
        ims = []
        for i in range(n_batch):
            ims.append([roidb[i]['image'], roidb[i]['crops'][cropids[i]], roidb[i]['flipped']])

        for i in range(cur_from, cur_to):
            self.crop_idx[self.inds[i]] = self.crop_idx[self.inds[i]] + 1

        # im_tensor, roidb = self.im_process(roidb,cropids)
        processed_roidb = []
        for i in range(len(roidb)):
            tmp = roidb[i].copy()
            scale = roidb[i]['crops'][cropids[i]][1]
            height = roidb[i]['crops'][cropids[i]][2]
            width = roidb[i]['crops'][cropids[i]][3]                                      
            tmp['im_info'] = [self.crop_size[0], self.crop_size[1], scale]
            processed_roidb.append(tmp)

        processed_list = self.thread_pool.map_async(self.im_worker.worker, ims)

        worker_data = []
        srange = np.zeros((len(processed_roidb), 2))
        chipinfo = np.zeros((len(processed_roidb), 3))

        scales = self.cfg.network.ANCHOR_SCALES
        ratios = self.cfg.network.ANCHOR_RATIOS
        feat_stride = self.cfg.network.RPN_FEAT_STRIDE
        feat_width = self.crop_size[1]/feat_stride
        feat_height = self.crop_size[0]/feat_stride        
        
        for i in range(len(processed_roidb)):
            cropid = cropids[i]
            nids = processed_roidb[i]['props_in_chips'][cropid]
            gtids = np.where(processed_roidb[i]['max_overlaps'] == 1)[0]
            gt_boxes = processed_roidb[i]['boxes'][gtids, :]
            boxes = processed_roidb[i]['boxes'].copy()
            crowd = processed_roidb[i]['crowd'].copy()
            cur_crop = processed_roidb[i]['crops'][cropid][0]
            im_scale = processed_roidb[i]['crops'][cropid][1]
            height = processed_roidb[i]['crops'][cropid][2]
            width = processed_roidb[i]['crops'][cropid][3]
            classes = processed_roidb[i]['max_classes'][gtids]
            if im_scale == 1:
                srange[i, 0] = 0
                srange[i, 1] = 300
            else:
                srange[i, 0] = 240*im_scale
                srange[i, 1] = 512
            chipinfo[i, 0] = height
            chipinfo[i, 1] = width
            chipinfo[i, 2] = im_scale
            
            argw = [processed_roidb[i]['im_info'], cur_crop, im_scale, nids, gtids, gt_boxes, boxes, classes.reshape(len(classes), 1), scales, ratios, feat_width, feat_height, feat_stride, crowd]

            worker_data.append(argw)

        t2 = time.time()

        # print 'q1 ' + str(t2 - t1)
        #all_labels = []
        #for w in worker_data:
        #    all_labels.append(roidb_anchor_worker(w))
        all_labels = self.pool.map(roidb_anchor_worker, worker_data)
        t3 = time.time()
        # print 'q2 ' + str(t3 - t2)
        A = len(scales)*len(ratios)
        labels = mx.nd.zeros((n_batch, A * feat_height * feat_width), mx.cpu(0))
        bbox_targets = mx.nd.zeros((n_batch, A * 4, feat_height, feat_width), mx.cpu(0))
        bbox_weights = mx.nd.zeros((n_batch, A * 4, feat_height, feat_width), mx.cpu(0))
        gt_boxes = -mx.nd.ones((n_batch, 100, 5))
        crowd_boxes = -mx.nd.ones((n_batch, 10, 5))

        for i in range(len(all_labels)):
            labels[i] = all_labels[i][0][0]
            pids = all_labels[i][2]
            if len(pids[0]) > 0:
                bbox_targets[i][pids[0], pids[1], pids[2]] = all_labels[i][1]
                bbox_weights[i][pids[0], pids[1], pids[2]] = 1.0
            gt_boxes[i] = all_labels[i][3]
            crowd_boxes[i] = all_labels[i][4]
        t4 = time.time()
        # print 'q3 ' + str(t4 - t3)

        im_tensor = mx.nd.zeros((n_batch, 3, self.crop_size[0], self.crop_size[1]), dtype=np.float32)
        processed_list = processed_list.get()
        for i in range(len(processed_list)):
            im_tensor[i] = processed_list[i]
        t5 = time.time()
        # print 'q4 ' + str(t5 - t4)
        # self.visualize(im_tensor, rois, labels)
        self.data = [im_tensor, mx.nd.array(srange), mx.nd.array(chipinfo)]
        self.label = [labels, bbox_targets, bbox_weights, gt_boxes, crowd_boxes]
        t6 = time.time()
        # print 'convert ' + str(t6 - t5)
        return mx.io.DataBatch(data=self.data, label=self.label, pad=self.getpad(), index=self.getindex(),
                               provide_data=self.provide_data, provide_label=self.provide_label)

    def visualize(self, im_tensor, boxes, labels):  # , bbox_targets, bbox_weights):
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
            # plt.show()
            plt.savefig('debug/visualization/test_{}_pos.png'.format(num))
            plt.cla()
            plt.clf()
            plt.close()
