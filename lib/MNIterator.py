import mxnet as mx

class MNIterator(mx.io.DataIter):
	def __init__(self,im_rec_path,im_lst_path,pixel_mean,batch_size=4,label_pad=10000,bbox_len=8, threads=8, nGPUs=1):
		super(MNIterator,self).__init__()
		assert batch_size%nGPUs==0, 'batch_size should be divisible by number of GPUs'
		self.det_iter = mx.io.ImageDetRecordIter(
	            path_imgrec     = im_rec_path,
	            path_imglist    = im_lst_path,
	            label_width     = -1,
	            label_pad_width = label_pad,
	            batch_size      = batch_size,
	            data_shape      = (3,1000,1000),
	            mean_r          = pixel_mean[2],
	            mean_g          = pixel_mean[1],
	            mean_b          = pixel_mean[0],
	            preprocess_threads = threads,
	            resize_mode     = 'force')
		self.bbox_len = bbox_len
		self.data_name = ['data','rois']
		self.label_name= ['label','bbox_target','bbox_weight']
		self.n_per_gpu = batch_size / nGPUs
		self.get_batch()
		self.reset()



	@property
	def provide_data(self):
		return [(k, v.shape) for k, v in zip(self.data_name, self.batch.data)]
	@property
	def provide_label(self):
		return [(k, v.shape) for k, v in zip(self.label_name, self.batch.label)]
	@property
	def provide_data_single(self):
		return [(k, v.shape) for k, v in zip(self.data_name, self.batch.data)]
	@property
	def provide_label_single(self):
		return [(k, v.shape) for k, v in zip(self.label_name, self.batch.label)]
	def reset(self):
		self.det_iter.reset()
	def iter_next(self):
		return self.get_batch()
	def next(self):
		if self.iter_next():
			return self.batch
		else:
			raise StopIteration
	def get_batch(self):
		self.batch = self.det_iter.next()
		if not self.batch:
			return False
		# process the labels
		rois = []
		targets = []
		labels = []
		weights = []
		clabel = self.batch.label[0]
		ind = 0
		for i in range(clabel.shape[0]):
			mxlen = int(clabel[i][3].asscalar())
			header = int(clabel[i][4].asscalar())
			objlen = int(clabel[i][5].asscalar())
			n_objs = (mxlen-header)/objlen
			info = clabel[i][4+header:4+mxlen].reshape((n_objs,objlen))
			cur_label = mx.ndarray.slice(info,(0,0),(info.shape[0],1))
			labels.append(cur_label.reshape((1,cur_label.shape[0])))
			cur_rois = mx.ndarray.slice(info,(0,1),(info.shape[0],5))
			cur_rois = mx.ndarray.concat(mx.ndarray.ones((cur_rois.shape[0],1))*ind,cur_rois,dim=1)
			cur_rois = cur_rois.reshape((1,cur_rois.shape[0],cur_rois.shape[1]))
			rois.append(cur_rois)
			cur_targets = mx.ndarray.slice(info,(0,5),(info.shape[0],5+self.bbox_len))
			targets.append(cur_targets.reshape((1,cur_targets.shape[0],cur_targets.shape[1])))
			cur_weights = mx.ndarray.slice(info,(0,5+self.bbox_len),(info.shape[0],info.shape[1]))
			weights.append(cur_weights.reshape((1,cur_weights.shape[0],cur_weights.shape[1])))
			ind = (ind+1)%self.n_per_gpu

		

		self.batch.data.append(mx.ndarray.concat(*rois,dim=0))
		self.batch.data = self.batch.data
		self.batch.label = [mx.ndarray.concat(*labels,dim=0),
					 mx.ndarray.concat(*targets,dim=0),
					 mx.ndarray.concat(*weights,dim=0)]
		# import pdb
		# pdb.set_trace()
		self.batch = mx.io.DataBatch(data=self.batch.data,label=self.batch.label,
									pad=self.getpad(),index=self.getindex(),provide_data=self.provide_data,
									provide_label=self.provide_label)
		return True

