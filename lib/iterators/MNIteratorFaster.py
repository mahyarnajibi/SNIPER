import mxnet as mx
import cv2
import numpy as np
from bbox.bbox_transform import clip_boxes
from bbox.bbox_regression import expand_bbox_regression_targets
from MNIteratorBase import MNIteratorBase

class MNIteratorFaster(MNIteratorBase):
    def __init__(self, roidb, config, batch_size = 4,  threads = 8, nGPUs = 1, pad_rois_to=900,single_size_change=False):
        self.data_name = ['data', 'rois']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']
        super(MNIteratorFaster,self).__init__(roidb,config,batch_size,threads,nGPUs,pad_rois_to,single_size_change)


    def im_worker(self,data):
        im = data['im']
        crop = data['crop']
        roidb = data['roidb'].copy()


        target_size = self.cfg.TRAIN.SCALES[0][0]
        max_size = self.cfg.TRAIN.SCALES[0][1]

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
        # Since we have ohem pass all rois
        rois_per_image = rois.shape[0]
        fg_rois_per_image = rois_per_image
        im_rois, labels, bbox_targets, bbox_weights = \
            self.sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes,
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

        worker_data = [(roidb[i],i%self.n_per_gpu) for i in range(len(roidb))]   
        # all_labels = []
        # for cdata in worker_data:
        #     all_labels.append(self.roidb_worker(cdata))
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
        
        ims = []
        for i in range(n_batch):
            im = cv2.imread(roidb[i]['image'], cv2.IMREAD_COLOR)
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            ims.append({'im':im,'crop':None,'roidb':roidb[i]})
        processed_list = self.thread_pool.map(self.im_worker,ims)
        processed_roidb = [p['roidb'] for p in processed_list]

        max_height = np.array([p['im'].shape[0] for p in processed_list]).max()
        max_width = np.array([p['im'].shape[1] for p in processed_list]).max()
        if max_width>=max_height:
            max_width = self.cfg.TRAIN.SCALES[0][1]
            max_height = self.cfg.TRAIN.SCALES[0][0]
        else:
            max_width = self.cfg.TRAIN.SCALES[0][0]
            max_height = self.cfg.TRAIN.SCALES[0][1]

        im_tensor = mx.ndarray.zeros((n_batch,
            3,max_height,max_width))
        #print im_tensor.shape
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

    

