import mxnet as mx
import random
import sys
import os
import pickle
import cv2
import numpy as np
import time
from multiprocessing.pool import ThreadPool
from bbox.bbox_transform import clip_boxes
from bbox.bbox_regression import expand_bbox_regression_targets

class MNIterator(mx.io.DataIter):
    def __init__(self, roidb, pixel_mean, config, batch_size = 4,  threads = 8, nGPUs = 1, aspect_grouping=True,horizontal=True,pad_rois_to=900):
        super(MNIterator, self).__init__()
        assert batch_size % nGPUs == 0, 'batch_size should be divisible by number of GPUs'
        
        self.cur_i = 0
        self.roidb = roidb
        self.size = 1#len(roidb)
        self.batch_size = batch_size
        self.pixel_mean = pixel_mean

        self.thread_pool = ThreadPool(threads)
        self.data_name = ['data', 'rois']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']
        self.n_per_gpu = batch_size / nGPUs
        self.batch = None
        self.hor = horizontal
        self.cfg = config
        self.n_expected_roi = pad_rois_to

        self.pad_label = np.array(-1)
        self. pad_weights = np.zeros((1, 8))
        self.pad_targets = np.zeros((1, 8))
        self.pad_roi = np.array([[0,0,100,100]])


        self.reset()
        self.get_batch()
        self.reset()

    @property
    def provide_data(self):
        return [ (k, v.shape) for k, v in zip(self.data_name, self.data) ]

    @property
    def provide_label(self):
        return [ (k, v.shape) for k, v in zip(self.label_name, self.label) ]

    @property
    def provide_data_single(self):
        return [ (k, v.shape) for k, v in zip(self.data_name, self.data) ]

    @property
    def provide_label_single(self):
        return [ (k, v.shape) for k, v in zip(self.label_name, self.label) ]

    def reset(self):
        self.cur_i = 0
        widths = np.array([r['width'] for r in self.roidb])
        heights = np.array([r['height'] for r in self.roidb])
        if self.hor:
            inds = np.where(widths>=heights)[0]
        else:
            inds = np.where(widths<heights)[0]

        inds = np.random.permutation(inds)
        extra = inds.shape[0] % self.batch_size
        extra = inds.shape[0] % self.batch_size
        inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
        row_perm = np.random.permutation(np.arange(inds_.shape[0]))
        inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
        self.inds = inds

    # def getpad(self):
    #     if self.cur_i + self.batch_size > self.size:
    #         return self.cur_i + self.batch_size - self.size
    #     else:
    #         return 0

    def iter_next(self):
        return self.get_batch()

    def next(self):
        if self.iter_next():
            return self.batch
        raise StopIteration

    def get_batch(self):
        if self.cur_i >= self.size:
            return False
        # Form cur roidb
        cur_roidbs = [self.roidb[self.inds[i%self.size]] for i in range(self.cur_i, self.cur_i+self.batch_size)]
        
        # Process cur roidb
        self.batch = self._get_batch(cur_roidbs)
        self.cur_i += self.batch_size
        return True

    def im_worker(self,data):
        im = data['im']
        crop = data['crop']
        roidb = data['roidb'].copy()


        target_size = self.cfg.SCALES[0][0]
        max_size = self.cfg.SCALES[0][1]

        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        if crop != None:
            im = im[crop[1]:crop[3], crop[0]:crop[2], :]
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_info = [im.shape[0],im.shape[1],im_scale]

        
        roidb['boxes'] = clip_boxes(np.round(roidb['boxes'] * im_scale), im_info[:2])
        roidb['oboxes'] = roidb['boxes'].copy()
        roidb['im_info'] = im_info
        roidb['invalid_boxes'] = np.empty((0, 4), dtype=np.float32)   
        roidb['lrange'] = [0, 5000 * 5000]
        roidb['lrange'] = np.array(roidb['lrange'], dtype=np.float32).reshape(1, 2)
        return {'im':im,'roidb': roidb}


    def roidb_worker(self,data):
        # infer num_classes from gt_overlaps
        
        roidb = data[0]
        im_i = data[1]
        num_classes = roidb['gt_overlaps'].shape[1]
        rois = roidb['boxes']
        labels = roidb['max_classes']
        overlaps = roidb['max_overlaps']
        bbox_targets = roidb['bbox_targets']

        im_rois, labels, bbox_targets, bbox_weights = \
            self.sample_rois(rois, self.fg_rois_per_image, self.rois_per_image, num_classes,
                            labels, overlaps, bbox_targets)

        rois = im_rois
        if rois.shape[0] > self.n_expected_roi:
            rois = rois[0:self.n_expected_roi, :]
            bbox_weights = bbox_weights[0:self.n_expected_roi, :]
            bbox_targets = bbox_targets[0:self.n_expected_roi, :]
            labels = labels[0:self.n_expected_roi]
        elif rois.shape[0] < self.n_expected_roi:
            n_pad = self.n_expected_roi - rois.shape[0]
            rois = np.vstack((rois, np.repeat(self.pad_roi, n_pad, axis=0)))
            labels = np.hstack((labels, np.repeat(self.pad_label, n_pad, axis=0)))
            bbox_weights = np.vstack((bbox_weights, np.repeat(self.pad_weights, n_pad, axis=0)))
            bbox_targets = np.vstack((bbox_targets, np.repeat(self.pad_targets, n_pad, axis=0)))

        batch_index = im_i * np.ones((rois.shape[0], 1))
        rois_array_this_image = np.hstack((batch_index, rois))

        return {'rois':rois_array_this_image,'labels':labels,
        'bbox_weights':bbox_weights,'bbox_targets':bbox_targets}
    def get_index(self):
        return self.cur_i/self.batch_size
    def _get_batch(self,roidb):
        """
        return a dict of multiple images
        :param roidb: a list of dict, whose length controls batch size
        ['images', 'flipped'] + ['gt_boxes', 'boxes', 'gt_overlap'] => ['bbox_targets']
        :return: data, label
        """
        num_images = len(roidb)
        im_tensor, roidb = self.im_process(roidb)
        assert self.cfg.TRAIN.BATCH_ROIS == -1 or self.cfg.TRAIN.BATCH_ROIS % self.cfg.TRAIN.BATCH_IMAGES == 0, \
            'BATCHIMAGES {} must divide BATCH_ROIS {}'.format(self.cfg.TRAIN.BATCH_IMAGES, self.cfg.TRAIN.BATCH_ROIS)

        if self.cfg.TRAIN.BATCH_ROIS == -1:
            self.rois_per_image = np.sum([iroidb['boxes'].shape[0] for iroidb in roidb])
            self.fg_rois_per_image = self.rois_per_image
        else:
            self.rois_per_image = self.cfg.TRAIN.BATCH_ROIS / self.cfg.TRAIN.BATCH_IMAGES
            self.fg_rois_per_image = np.round(self.cfg.TRAIN.FG_FRACTION * self.rois_per_image).astype(int)

        worker_data = [(roidb[i],i%self.n_per_gpu) for i in range(len(roidb))]        
        all_labels = self.thread_pool.map(self.roidb_worker,worker_data)
        
        rois = mx.nd.zeros((num_images,self.n_expected_roi,5),mx.cpu(0))
        labels = mx.nd.zeros((num_images,self.n_expected_roi),mx.cpu(0))
        bbox_targets = mx.nd.zeros((num_images,self.n_expected_roi,8),mx.cpu(0))
        bbox_weights = mx.nd.zeros((num_images,self.n_expected_roi,8),mx.cpu(0))
        for i,clabel in enumerate(all_labels):
            rois[i] = clabel['rois']
            labels[i] = clabel['labels']
            bbox_targets[i] = clabel['bbox_targets']
            bbox_weights[i] = clabel['bbox_weights']


        self.data = [im_tensor,rois]
        self.label = [labels,bbox_targets,bbox_weights]
        return mx.io.DataBatch(data=self.data, label=self.label, pad=self.getpad(), index=self.getindex(), provide_data=self.provide_data, provide_label=self.provide_label)



    def im_process(self,roidb):    
        n_batch = len(roidb)
        width = self.cfg.SCALES[0][1] if self.hor else self.cfg.SCALES[0][0]
        height = self.cfg.SCALES[0][0] if self.hor else self.cfg.SCALES[0][1]
        im_tensor = mx.ndarray.zeros((n_batch,
            3,height,width))
        ims = []
        for i in range(n_batch):
            im = cv2.imread(roidb[i]['image'], cv2.IMREAD_COLOR)
            #print('Time to read images: {}'.format(time.time()-stime))
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            ims.append({'im':im,'crop':None,'roidb':roidb[i]})
        #stime = time.time()
        processed_list = self.thread_pool.map(self.im_worker,ims)
        processed_roidb = [p['roidb'] for p in processed_list]
        
        for i in range(len(processed_list)):
            im = processed_list[i]['im']
            for j in range(3):
                im_tensor[i, j, 0:im.shape[0], 0:im.shape[1]] = im[:, :, 2 - j] - self.pixel_mean[2 - j]
        return im_tensor,processed_roidb


    def sample_rois(self,rois, fg_rois_per_image, rois_per_image, num_classes,
                labels=None, overlaps=None, bbox_targets=None, gt_boxes=None):
        """
        generate random sample of ROIs comprising foreground and background examples
        :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
        :param fg_rois_per_image: foreground roi number
        :param rois_per_image: total roi number
        :param num_classes: number of classes
        :param labels: maybe precomputed
        :param overlaps: maybe precomputed (max_overlaps)
        :param bbox_targets: maybe precomputed
        :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
        :return: (labels, rois, bbox_targets, bbox_weights)
        """
        cfg = self.cfg
        if labels is None:
            overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
            gt_assignment = overlaps.argmax(axis=1)
            overlaps = overlaps.max(axis=1)
            labels = gt_boxes[gt_assignment, 4]

        # foreground RoI with FG_THRESH overlap
        fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
        fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
        # Sample foreground regions without replacement
        if len(fg_indexes) > fg_rois_per_this_image:
            fg_indexes = np.random.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
        # Sample foreground regions without replacement
        if len(bg_indexes) > bg_rois_per_this_image:
            bg_indexes = np.random.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

        # indexes selected
        keep_indexes = np.append(fg_indexes, bg_indexes)

        # pad more to ensure a fixed minibatch size
        while keep_indexes.shape[0] < rois_per_image:
            gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
            gap_indexes = np.random.choice(range(len(rois)), size=gap, replace=False)
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
            targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
            if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
                targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                           / np.array(cfg.TRAIN.BBOX_STDS))
            bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

        bbox_targets, bbox_weights = \
            expand_bbox_regression_targets(bbox_target_data, num_classes, cfg)

        return rois, labels, bbox_targets, bbox_weights