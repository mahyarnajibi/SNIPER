# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Training Module
# by Mahyar Najibi and Bharat Singh
# --------------------------------------------------------------
import init
import matplotlib
matplotlib.use('Agg')
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '2'
#os.environ['MXNET_ENABLE_GPU_P2P'] = '1'
from iterators.MNIteratorE2E import MNIteratorE2E
import sys
sys.path.insert(0, 'lib')
from symbols.faster import *
from configs.faster.default_configs import config, update_config, update_config_from_list
import mxnet as mx
from train_utils import metric
from train_utils.utils import get_optim_params, get_fixed_param_names, create_logger, load_param
from iterators.PrefetchingIter import PrefetchingIter

from data_utils.load_data import load_proposal_roidb, merge_roidb, filter_roidb
from bbox.bbox_regression import add_bbox_regression_targets
import argparse

def parser():
    arg_parser = argparse.ArgumentParser('SNIPER training module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/sniper_res101_e2e.yml',type=str)
    arg_parser.add_argument('--display', dest='display', help='Number of epochs between displaying loss info',
                            default=100, type=int)
    arg_parser.add_argument('--momentum', dest='momentum', help='BN momentum', default=0.995, type=float)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--set', dest='set_cfg_list', help='Set the configuration fields from command line',
                            default=None, nargs=argparse.REMAINDER)

    return arg_parser.parse_args()


if __name__ == '__main__':

    args = parser()
    update_config(args.cfg)
    if args.set_cfg_list:
        update_config_from_list(args.set_cfg_list)

    context = [mx.gpu(int(gpu)) for gpu in config.gpus.split(',')]
    nGPUs = len(context)
    batch_size = nGPUs * config.TRAIN.BATCH_IMAGES

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # Create roidb
    image_sets = [iset for iset in config.dataset.image_set.split('+')]
    roidbs = [load_proposal_roidb(config.dataset.dataset, image_set, config.dataset.root_path,
        config.dataset.dataset_path,
        proposal=config.dataset.proposal, append_gt=True, flip=config.TRAIN.FLIP,
        result_path=config.output_path,
        proposal_path=config.proposal_path, load_mask=config.TRAIN.WITH_MASK, only_gt=not config.TRAIN.USE_NEG_CHIPS)
        for image_set in image_sets]

    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb, config)
    bbox_means, bbox_stds = add_bbox_regression_targets(roidb, config)



    print('Creating Iterator with {} Images'.format(len(roidb)))
    train_iter = MNIteratorE2E(roidb=roidb, config=config, batch_size=batch_size, nGPUs=nGPUs,
                               threads=config.TRAIN.NUM_THREAD, pad_rois_to=400)
    print('The Iterator has {} samples!'.format(len(train_iter)))

    # Creating the Logger
    logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)

    # get list of fixed parameters
    print('Initializing the model...')
    sym_inst = eval('{}.{}'.format(config.symbol, config.symbol))(n_proposals=400, momentum=args.momentum)
    sym = sym_inst.get_symbol_rpn(config) if config.TRAIN.ONLY_PROPOSAL else sym_inst.get_symbol_rcnn(config)

    fixed_param_names = get_fixed_param_names(config.network.FIXED_PARAMS, sym)

    # Creating the module
    mod = mx.mod.Module(symbol=sym,
                        context=context,
                        data_names=[k[0] for k in train_iter.provide_data_single],
                        label_names=[k[0] for k in train_iter.provide_label_single],
                        fixed_param_names=fixed_param_names)

    shape_dict = dict(train_iter.provide_data_single + train_iter.provide_label_single)
    sym_inst.infer_shape(shape_dict)
    arg_params, aux_params = load_param(config.network.pretrained, config.network.pretrained_epoch, convert=True)

    if config.TRAIN.ONLY_PROPOSAL:
        sym_inst.init_weight_rpn(config, arg_params, aux_params)
    else:
        sym_inst.init_weight_rcnn(config, arg_params, aux_params)

    # Creating the metrics
    eval_metric = metric.RPNAccMetric()
    cls_metric = metric.RPNLogLossMetric()
    bbox_metric = metric.RPNL1LossMetric()
    rceval_metric = metric.RCNNAccMetric(config)
    rccls_metric  = metric.RCNNLogLossMetric(config)
    rcbbox_metric = metric.RCNNL1LossCRCNNMetric(config)

    eval_metrics = mx.metric.CompositeEvalMetric()
    eval_metrics.add(eval_metric)
    eval_metrics.add(cls_metric)
    eval_metrics.add(bbox_metric)
    if not config.TRAIN.ONLY_PROPOSAL:
        eval_metrics.add(rceval_metric)
        eval_metrics.add(rccls_metric)
        eval_metrics.add(rcbbox_metric)

    if config.TRAIN.WITH_MASK:
        mask_metric = metric.MaskLogLossMetric(config)
        eval_metrics.add(mask_metric)

    optimizer_params = get_optim_params(config, len(train_iter), batch_size)
    print ('Optimizer params: {}'.format(optimizer_params))

    # Checkpointing
    prefix = os.path.join(output_path, args.save_prefix)
    batch_end_callback = mx.callback.Speedometer(batch_size, args.display)
    epoch_end_callback = [mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True),
                          eval('{}.checkpoint_callback'.format(config.symbol))(sym_inst.get_bbox_param_names(), prefix, bbox_means, bbox_stds)]

    train_iter = PrefetchingIter(train_iter)
    mod.fit(train_iter, optimizer='sgd', optimizer_params=optimizer_params,
            eval_metric=eval_metrics, num_epoch=config.TRAIN.end_epoch, kvstore=config.default.kvstore,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback, arg_params=arg_params, aux_params=aux_params)
