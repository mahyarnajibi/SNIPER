from lib.lr_scheduler import WarmupMultiFactorScheduler
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
	lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, cfg.TRAIN.warmup, cfg.TRAIN.warmup_lr, cfg.TRAIN.warmup_step)

	optim_params = {'momentum': cfg.TRAIN.momentum,
                        'wd': cfg.TRAIN.wd,
                        'learning_rate': base_lr,
                        'rescale_grad': 1.0,
                        'clip_gradient': None,
                        'lr_scheduler': lr_scheduler}

	return optim_params