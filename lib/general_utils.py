import mxnet as mx
from lib.lr_scheduler import WarmupMultiBatchScheduler
import os
import logging
import time

def get_optim_params_single_epoch(cfg):

	# Create scheduler
	base_lr = cfg.TRAIN.lr
	lr_factor = cfg.TRAIN.lr_factor
	lr_iters = [162855]
	lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, cfg.TRAIN.warmup, cfg.TRAIN.warmup_lr, cfg.TRAIN.warmup_step)

	optim_params = {'momentum': cfg.TRAIN.momentum,
                        'wd': cfg.TRAIN.wd,
                        'learning_rate': base_lr,
                        'rescale_grad': 1.0,
                        'clip_gradient': None,
                        'lr_scheduler': lr_scheduler}

	return optim_params

def get_optim_params(cfg,roidb_len,batch_size):

	# Create scheduler
	base_lr = cfg.TRAIN.lr
	lr_step = cfg.TRAIN.lr_step
	lr_factor = cfg.TRAIN.lr_factor
	begin_epoch = cfg.TRAIN.begin_epoch
	end_epoch = cfg.TRAIN.end_epoch
	lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
	lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
	lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
	lr_iters = [int(epoch * roidb_len / batch_size) for epoch in lr_epoch_diff]
	lr_scheduler = WarmupMultiBatchScheduler(lr_iters, lr_factor, cfg.TRAIN.warmup, cfg.TRAIN.warmup_lr, cfg.TRAIN.warmup_step)

        if cfg.TRAIN.fp16 == True:
                optim_params = {'momentum': cfg.TRAIN.momentum,
                        'wd': cfg.TRAIN.wd*cfg.TRAIN.scale,
                        'learning_rate': base_lr/cfg.TRAIN.scale,
                        'rescale_grad': 1.0,
                        'multi_precision': True,
                        'clip_gradient': None,
                        'lr_scheduler': lr_scheduler}
        else:
                optim_params = {'momentum': cfg.TRAIN.momentum,
                        'wd': cfg.TRAIN.wd,
                        'learning_rate': base_lr,
                        'rescale_grad': 1.0,
                        'clip_gradient': None,
                        'lr_scheduler': lr_scheduler}

	return optim_params
def checkpoint_callback(bbox_param_names, prefix, means, stds):
	def _callback(iter_no, sym, arg, aux):
		weight = arg[bbox_param_names[0]]
		bias = arg[bbox_param_names[1]]
		repeat = bias.shape[0] / means.shape[0]

		arg[bbox_param_names[0]+'_test'] = weight * mx.nd.repeat(mx.nd.array(stds), repeats=repeat).reshape((bias.shape[0], 1, 1, 1))
		arg[bbox_param_names[1]+'_test'] = bias * mx.nd.repeat(mx.nd.array(stds), repeats=repeat) + mx.nd.repeat(mx.nd.array(means), repeats=repeat)
		mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
		arg.pop(bbox_param_names[0]+'_test')
		arg.pop(bbox_param_names[1]+'_test')
	return _callback

def _convert_context(params, ctx):
	"""
	:param params: dict of str to NDArray
	:param ctx: the context to convert to
	:return: dict of str of NDArray with context ctx
	"""
	new_params = dict()
	for k, v in params.items():
		new_params[k] = v.as_in_context(ctx)
	return new_params


def load_param(prefix, epoch, convert=False, ctx=None, process=False):
	"""
	wrapper for load checkpoint
	:param prefix: Prefix of model name.
	:param epoch: Epoch number of model we would like to load.
	:param convert: reference model should be converted to GPU NDArray first
	:param ctx: if convert then ctx must be designated.
	:param process: model should drop any test
	:return: (arg_params, aux_params)
	"""
	arg_params, aux_params = load_checkpoint(prefix, epoch)
	if convert:
		if ctx is None:
			ctx = mx.cpu()
		arg_params = _convert_context(arg_params, ctx)
		aux_params = _convert_context(aux_params, ctx)
	if process:
		tests = [k for k in arg_params.keys() if '_test' in k]
		for test in tests:
			arg_params[test.replace('_test', '')] = arg_params.pop(test)
	return arg_params, aux_params

def get_fixed_param_names(fixed_param_prefix,sym):
	"""
	:param fixed_param_prefix: the prefix in param names to be fixed in the model
	:param fixed_param_prefix: network symbol
	:return: [fixed_param_names]
	"""
	fixed_param_names = []
	if fixed_param_prefix is None:
		return fixed_param_names

	for name in sym.list_arguments():
		for prefix in fixed_param_prefix:
			if prefix in name:
				fixed_param_names.append(name)
	return fixed_param_names


def create_logger(root_output_path, cfg, image_set):
    # set up logger
	if not os.path.exists(root_output_path):
		os.makedirs(root_output_path)
	assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)

	cfg_name = os.path.basename(cfg).split('.')[0]
	config_output_path = os.path.join(root_output_path, '{}'.format(cfg_name))
	if not os.path.exists(config_output_path):
		os.makedirs(config_output_path)

	image_sets = [iset for iset in image_set.split('+')]
	final_output_path = os.path.join(config_output_path, '{}'.format('_'.join(image_sets)))
	if not os.path.exists(final_output_path):
		os.makedirs(final_output_path)

	log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
	head = '%(asctime)-15s %(message)s'
	logging.basicConfig(filename=os.path.join(final_output_path, log_file), format=head)
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)

	return logger, final_output_path


def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
    return startTime_for_tictoc

def toc():
    if 'startTime_for_tictoc' in globals():
        endTime = time.time()
        return endTime - startTime_for_tictoc
    else:
        return None
