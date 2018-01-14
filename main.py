from lib.MNIterator import MNIterator
from easydict import EasyDict
from lib.load_model import load_param
import sys
import logging
sys.path.insert(0,'lib')
from symbols.rfcn_resnet50 import rfcn_resnet50
from configs.default_configs import config,update_config,get_opt_params
import mxnet as mx
from lib import metric,callback
import numpy as np
from lib.general_utils import get_optim_params,checkpoint_callback
import os

cfg = EasyDict()
cfg.rec_path = 'train_list.rec'
cfg.list_path = 'train_list.lst'
cfg.batch_size = 4
cfg.PIXEL_MEANS = [103.06,115.90,123.15]
cfg.external_cfg =  'configs/res50_rfcn_pascal.yml'
cfg.bbox_normalization_mean = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
cfg.bbox_normalization_stds = np.array([ 0.1,  0.1,  0.2,  0.2,  0.1,  0.1,  0.2,  0.2])
cfg.nGPUs = 2
cfg.display = 20
cfg.save_prefix = 'CRCNN'


if __name__=='__main__':

	update_config(cfg.external_cfg)
	context=[mx.gpu(i) for i in range(cfg.nGPUs)]
	if not os.path.isdir(config.output_path):
		os.mkdir(config.output_path)

	# Creating the iterator
	train_iter = MNIterator(cfg.rec_path,cfg.list_path,cfg.PIXEL_MEANS,batch_size=cfg.batch_size,nGPUs=cfg.nGPUs)
	train_iter.next()

	# Creating the module
	
	sym_inst = rfcn_resnet50()
	sym = sym_inst.get_symbol_rfcn(config)
	
	# Creating the Logger
	logging.getLogger().setLevel(logging.DEBUG) 

	# Creating the module
	mod = mx.mod.Module(symbol=sym,
					context=context,
					data_names=[k[0] for k in train_iter.provide_data_single],
					label_names=[k[0] for k in train_iter.provide_label_single])
	shape_dict = dict(train_iter.provide_data_single+train_iter.provide_label_single)
	sym_inst.infer_shape(shape_dict)
	arg_params, aux_params = load_param(config.network.pretrained,config.network.pretrained_epoch,convert=True)
	sym_inst.init_weight(config,arg_params,aux_params)


	# Creating the metrics
	eval_metric = metric.RCNNAccMetric(config)
	cls_metric  = metric.RCNNLogLossMetric(config)
	bbox_metric = metric.RCNNL1LossMetric(config)
	eval_metrics = mx.metric.CompositeEvalMetric()
	eval_metrics.add(eval_metric)
	eval_metrics.add(cls_metric)
	eval_metrics.add(bbox_metric) 


	eval_metrics = mx.metric.CompositeEvalMetric()
	eval_metrics.add(eval_metric)
	eval_metrics.add(cls_metric)
	eval_metrics.add(bbox_metric)

	optimizer_params = get_optim_params(config,roidb_len=10022,batch_size=cfg.batch_size)
	print ('Optimizer params: {}'.format(optimizer_params))

	# Checkpointing
	prefix = os.path.join(config.output_path,cfg.save_prefix)
	epoch_end_callback = [mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True),
		checkpoint_callback(sym_inst.get_bbox_param_names(),prefix, cfg.bbox_normalization_mean, cfg.bbox_normalization_stds)]

	mod.fit(train_iter,optimizer='sgd',optimizer_params=optimizer_params,
		eval_metric=eval_metrics,num_epoch=config.TRAIN.end_epoch,kvstore=config.default.kvstore,
		batch_end_callback=mx.callback.Speedometer(cfg.batch_size, cfg.display),
		epoch_end_callback=epoch_end_callback, arg_params=arg_params,aux_params=aux_params)
