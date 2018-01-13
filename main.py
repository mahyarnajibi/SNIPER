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

cfg = EasyDict()
cfg.rec_path = 'train_list.rec'
cfg.list_path = 'train_list.lst'
cfg.batch_size = 2
cfg.PIXEL_MEANS = [103.06,115.90,123.15]
cfg.external_cfg =  'configs/res50_rfcn_pascal.yml'
if __name__=='__main__':


	# Creating the iterator
	train_iter = MNIterator(cfg.rec_path,cfg.list_path,cfg.PIXEL_MEANS,batch_size=cfg.batch_size)
	train_iter.next()
	# Creating the module
	update_config(cfg.external_cfg)
	sym_inst = rfcn_resnet50()
	sym = sym_inst.get_symbol_rfcn(config)
	
		
	
	# Creating the Logger
	logging.getLogger().setLevel(logging.DEBUG) 

	# Creating the module
	context=[mx.gpu(0),mx.gpu(1)]
	mod = mx.mod.Module(symbol=sym,
                     context=context,
                     data_names=[k[0] for k in train_iter.provide_data_single],
                     label_names=[k[0] for k in train_iter.provide_label_single])
    # Bind data to moudle
	#mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
	#mod.init_params(initializer=mx.init.Uniform(scale=.1))
	shape_dict = dict(train_iter.provide_data_single+train_iter.provide_label_single)
	sym_inst.infer_shape(shape_dict)
	arg_params, aux_params = load_param(config.network.pretrained,config.network.pretrained_epoch,convert=True)
	sym_inst.init_weight(config,arg_params,aux_params)
    # Create Optimizer
	#mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.0002), ))


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

	optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': 0.0001,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}

	mod.fit(train_iter,optimizer='sgd',optimizer_params=optimizer_params,
		eval_metric=eval_metrics,num_epoch=config.TRAIN.end_epoch,kvstore=config.default.kvstore,
		batch_end_callback=mx.callback.Speedometer(2, 20), arg_params=arg_params,aux_params=aux_params)
