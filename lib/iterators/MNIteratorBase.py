import numpy as np
import mxnet as mx
from multiprocessing.pool import ThreadPool
class MNIteratorBase(mx.io.DataIter):

    def __init__(self, roidb, config, batch_size,  threads, nGPUs, pad_rois_to, single_size_change):
        super(MNIteratorBase, self).__init__()
        assert batch_size % nGPUs == 0, 'batch_size should be divisible by number of GPUs'
        
        self.cur_i = 0
        self.roidb = roidb
        
        self.batch_size = batch_size
        self.pixel_mean = config.network.PIXEL_MEANS

        self.thread_pool = ThreadPool(threads)

        self.n_per_gpu = batch_size / nGPUs
        self.batch = None
        
        self.cfg = config
        self.n_expected_roi = pad_rois_to

        self.pad_label = np.array(-1)
        self.pad_weights = np.zeros((1, 8))
        self.pad_targets = np.zeros((1, 8))
        self.pad_roi = np.array([[0,0,100,100]])
        self.single_size_change = single_size_change
        self.reset()
        self.get_batch()
        self.reset()

    def __len__(self):
        return len(self.inds)
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
        horz_inds = np.where(widths >= heights)[0]
        vert_inds = np.where(widths<heights)[0]
        if horz_inds.shape[0]%self.batch_size>0:
            extra_horz = self.batch_size - (horz_inds.shape[0]%self.batch_size)
            horz_inds = np.hstack((horz_inds,horz_inds[0:extra_horz]))
        if vert_inds.shape[0]%self.batch_size>0:
            extra_vert = self.batch_size - (vert_inds.shape[0]%self.batch_size)
            vert_inds = np.hstack((vert_inds,vert_inds[0:extra_vert]))
        inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
        extra = inds.shape[0] % self.batch_size
        assert extra==0,'The number of samples here should be divisible by batch size'
        if not self.single_size_change:
            inds_ = np.reshape(inds,(-1,self.batch_size))
            row_perm = np.random.permutation(np.arange(inds_.shape[0]))
            inds = np.reshape(inds_[row_perm,:],(-1,))
        self.inds = inds
        self.size = len(self.inds)

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

        #print('Im {}'.format(self.cur_i))
        cur_roidbs = [self.roidb[self.inds[i%self.size]] for i in range(self.cur_i, self.cur_i+self.batch_size)]

        # Process cur roidb
        self.batch = self._get_batch(cur_roidbs)
        self.cur_i += self.batch_size
        return True

    def get_index(self):
        return self.cur_i/self.batch_size

    def _get_batch(self,roidb):
        raise NotImplementedError("This method should be implemented in the inherited classes")