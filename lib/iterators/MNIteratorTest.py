import mxnet as mx
import numpy as np
from utils.data_workers import im_worker
from MNIteratorBase import MNIteratorBase
from multiprocessing import Pool

class MNIteratorTest(MNIteratorBase):
    def __init__(self, roidb, config, test_scale, batch_size=4, threads=8, nGPUs=1, pad_rois_to=400, crop_size=(512, 512)):
        self.crop_size = crop_size
        self.num_classes = roidb[0]['gt_overlaps'].shape[1]
        self.data_name = ['data', 'im_info', 'im_ids']
        self.label_name = None
        self.label = []
        self.context_size = 320
        self.im_worker = im_worker(crop_size=None if not self.crop_size else self.crop_size[0], cfg=config, target_size=test_scale)
        self.test_scale = test_scale
        super(MNIteratorTest, self).__init__(roidb, config, batch_size, threads, nGPUs, pad_rois_to, True)

        self.reset()

    
    def _get_batch(self, roidb):
        n_batch = len(roidb)
        im_ids = np.array([self.inds[i % self.size] for i in range(self.cur_i, self.cur_i + self.batch_size)])

        hor_flag = True if roidb[0]['width']>= roidb[0]['height'] else False
        max_size = [self.test_scale[0], self.test_scale[1]] if hor_flag else \
            [self.test_scale[1], self.test_scale[0]]


        ims = []
        for i in range(n_batch):
            ims.append([roidb[i]['image'], max_size ,roidb[i]['flipped']])
        im_info = np.zeros((n_batch, 3))
        # processed_list = []
        # for im in ims:
        #     processed_list.append(self.im_worker.worker(im))
        processed_list = self.thread_pool.map(self.im_worker.worker, ims)
        im_tensor = mx.nd.zeros((n_batch, 3, max_size[0], max_size[1]), dtype=np.float32)
        for i,p in enumerate(processed_list):
            im_info[i] = [p[2][0], p[2][1], p[1]]
            im_tensor[i] = p[0]
        self.data = [im_tensor,  mx.nd.array(im_info), mx.nd.array(im_ids)]
        return mx.io.DataBatch(data=self.data, label=self.label, pad=self.getpad(), index=self.getindex(),
                               provide_data=self.provide_data, provide_label=self.provide_label)

    def reset(self):
        self.cur_i = 0
        widths = np.array([r['width'] for r in self.roidb])
        heights = np.array([r['height'] for r in self.roidb])
        horz_inds = np.where(widths >= heights)[0]
        vert_inds = np.where(widths<heights)[0]
        if horz_inds.shape[0]%self.batch_size>0:
            extra_horz = self.batch_size - (horz_inds.shape[0] % self.batch_size)
            horz_inds = np.hstack((horz_inds, horz_inds[0:extra_horz]))
        if vert_inds.shape[0]%self.batch_size>0:
            extra_vert = self.batch_size - (vert_inds.shape[0]%self.batch_size)
            vert_inds = np.hstack((vert_inds, vert_inds[0:extra_vert]))
        inds = np.hstack((horz_inds, vert_inds))
        extra = inds.shape[0] % self.batch_size
        assert extra==0,'The number of samples here should be divisible by batch size'
        self.inds = inds
        self.size = len(self.inds)
