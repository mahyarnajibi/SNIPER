# --------------------------------------------------------------
# AutoFocus: Efficient Multi-Scale Inference
# Licensed under The Apache-2.0 License [see LICENSE for details]
# SNIPER/AutoFocus Test Iterator
# by Mahyar Najibi
# --------------------------------------------------------------
import mxnet as mx
import numpy as np
from data_utils.data_workers import im_worker
from MNIteratorBase import MNIteratorBase
import math

class MNIteratorTestAutoFocus(MNIteratorBase):
    def __init__(self, roidb, config, test_scale, batch_size=4, threads=8, nGPUs=1, pad_rois_to=400, crop_size=(512, 512),
                 num_classes=None):
        self.crop_size = crop_size

        self.num_classes = num_classes if num_classes else roidb[0]['gt_overlaps'].shape[1]
        self.data_name = ['data', 'im_info', 'im_ids', 'chip_ids']
        self.label_name = None
        self.label = []
        self.context_size = 320
        self.im_worker = im_worker(crop_size=None if not self.crop_size else self.crop_size[0], cfg=config, target_size=test_scale)
        self.test_scale = test_scale
        super(MNIteratorTestAutoFocus, self).__init__(roidb, config, batch_size, threads, nGPUs, pad_rois_to, True)
        # Create the mapping for the 
        self.reset()
        print('Iterator has {} samples'.format(self.size))

    def set_scale(self, scale):
        self.test_scale = scale
        self.im_worker = im_worker(crop_size=None if not self.crop_size else self.crop_size[0], cfg=self.cfg,
                                   target_size=scale)

    def _get_batch(self, roidb, chip_ids, im_ids):
        n_batch = len(roidb)
        hor_flag = True if roidb[0]['width']>= roidb[0]['height'] else False
        max_size = [0, 0]
        chips = []
        scales = []
        min_target_size = self.test_scale[0]
        max_target_size = self.test_scale[1]
        local_chip_ids = np.zeros(len(roidb))
        for i, r in enumerate(roidb):
            # Compute the scale factor
            im_size_min = min(r['width'], r['height'])
            im_size_max = max(r['width'], r['height'])
            # Compute the scale factor
            scale = float(min_target_size) / float(im_size_min)
            if np.round(scale * im_size_max) > max_target_size:
                scale = float(max_target_size) / float(im_size_max)
            cchip_id = r['crop_mapping'][chip_ids[i]]
            cur_chip = r['inference_crops'][cchip_id]
            local_chip_ids[i] = cchip_id

            max_size[0] = max(max_size[0], int(math.ceil((cur_chip[3] - cur_chip[1])*scale)))
            max_size[1] = max(max_size[1], int(math.ceil((cur_chip[2] - cur_chip[0])*scale)))
            chips.append(cur_chip)
            scales.append(scale)

        ims = []
        for i in range(n_batch):
            ims.append([roidb[i]['image'], max_size ,roidb[i]['flipped'], chips[i],scales[i]])
        im_info = np.zeros((n_batch, 3))

        processed_list = self.thread_pool.map(self.im_worker.worker_fast_inference, ims)


        im_tensor = mx.nd.zeros((n_batch, 3, max_size[0], max_size[1]), dtype=np.float32)
        for i,p in enumerate(processed_list):
            im_info[i] = [p[2][0], p[2][1], p[1]]
            im_tensor[i] = p[0]
        self.data = [im_tensor,  mx.nd.array(im_info), mx.nd.array(im_ids), mx.nd.array(local_chip_ids)]
        return mx.io.DataBatch(data=self.data, label=self.label, pad=self.getpad(), index=self.getindex(),
                               provide_data=self.provide_data, provide_label=self.provide_label)



    def get_batch(self):
        if self.cur_i >= self.size:
            return False

        cur_chip_ids = [self.inds[i%self.size] for i in range(self.cur_i, self.cur_i + self.batch_size)]
        cur_roidb_ids = [self.crop2im[i] for i in cur_chip_ids]
        cur_roidbs = [self.roidb[i] for i in cur_roidb_ids]

        # Process cur roidb
        self.batch = self._get_batch(cur_roidbs, cur_chip_ids, cur_roidb_ids)
        self.cur_i += self.batch_size
        return True


    def reset(self):
        self.cur_i = 0
        # Collect the inference crop widths and heights, create the crop mapping
        self.crop2im = {}
        clustering_input = []
        crop_counter = 0
        widths = []
        heights = []
        for i, r in enumerate(self.roidb):
            local_crop_mapping = {}
            local_counter = 0
            for crop in r['inference_crops']:
                width = crop[2] - crop[0]
                height = crop[3] - crop[1]
                widths.append(width)
                heights.append(height)
                clustering_input.append([width, height])
                self.crop2im[crop_counter] = i
                local_crop_mapping[crop_counter] = local_counter
                local_counter += 1
                crop_counter += 1
            r['crop_mapping'] = local_crop_mapping

        # SORT BASED ON AREA
        clustering_input = np.array(clustering_input)
        areas = clustering_input[:,0] * clustering_input[:, 1]
        self.inds = areas.argsort()

        # Now group everything based on aspect ratio
        widths = np.array(widths)[self.inds]
        heights = np.array(heights)[self.inds]
        horz_inds = np.where(widths >= heights)[0]
        vert_inds = np.where(widths<heights)[0]

        if horz_inds.shape[0]%self.batch_size>0:
            extra_horz = self.batch_size - (horz_inds.shape[0] % self.batch_size)
            horz_inds = np.hstack((horz_inds, horz_inds[-extra_horz:]))
        if vert_inds.shape[0]%self.batch_size>0:
            extra_vert = self.batch_size - (vert_inds.shape[0]%self.batch_size)
            vert_inds = np.hstack((vert_inds, vert_inds[-extra_vert:]))
        inds = np.hstack((horz_inds, vert_inds))
        
        # Make sure the indices are divisible by batch size
        if inds.shape[0]%self.batch_size>0:
            extra = self.batch_size - (inds.shape[0]%self.batch_size)
            inds = np.hstack((inds, inds[-extra:]))
        self.inds = self.inds[inds]
        assert self.inds.shape[0]%self.batch_size==0,'The number of samples here should be divisible by batch size'
        self.size = len(self.inds)
