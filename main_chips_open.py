import os

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '2'
os.environ['MXNET_ENABLE_GPU_P2P'] = '1'

import init
from iterators.MNIteratorChipsE2E2SNoNeg import MNIteratorChips
from load_model import load_param
import sys

sys.path.insert(0, 'lib')
from symbols.faster.resnet_mx_101_e2e import resnet_mx_101_e2e, checkpoint_callback
from configs.faster.default_configs import config, update_config, get_opt_params
import mxnet as mx
import metric, callback
import numpy as np
from general_utils import get_optim_params, get_fixed_param_names, create_logger
from iterators.PrefetchingIter import PrefetchingIter

from load_data import load_proposal_roidb, merge_roidb, filter_roidb, add_chip_data, remove_small_boxes
from bbox.bbox_regression import add_bbox_regression_targets
from argparse import ArgumentParser
import cPickle


def parser():
    arg_parser = ArgumentParser('Faster R-CNN training module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/open_res101_mx_e2e.yml',type=str)
    arg_parser.add_argument('--display', dest='display', help='Number of epochs between displaying loss info',
                            default=100, type=int)
    arg_parser.add_argument('--momentum', dest='momentum', help='BN momentum', default=0.995, type=float)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='CRCNN', type=str)
    arg_parser.add_argument('--threadid', dest='threadid', help='Prefix used for snapshotting the network',
                            type=int)

    return arg_parser.parse_args()


if __name__ == '__main__':
    args = parser()
    update_config(args.cfg)
    context = [mx.gpu(int(gpu)) for gpu in config.gpus.split(',')]
    nGPUs = len(context)
    batch_size = nGPUs * config.TRAIN.BATCH_IMAGES

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # Create roidb
    config.debug = False
    if config.debug == False:
        image_sets = [iset for iset in config.dataset.image_set.split('+')]
        roidbs = [load_proposal_roidb(config.dataset.dataset, image_set, config.dataset.root_path,
                                      config.dataset.dataset_path,
                                      proposal=config.dataset.proposal, append_gt=False, flip=True,
                                      result_path=config.output_path,
                                      proposal_path=config.proposal_path)
                  for image_set in image_sets]

        roidb = merge_roidb(roidbs)
        # roidb = remove_small_boxes(roidb,max_scale=3,min_size=2)
        roidb = filter_roidb(roidb, config)
        bbox_means, bbox_stds = add_bbox_regression_targets(roidb, config)
    else:
        args.display = 20
        with open('/home/ubuntu/bigminival2014.pkl', 'rb') as file:
            roidb = cPickle.load(file)
        bbox_means, bbox_stds = add_bbox_regression_targets(roidb, config)

    print('Creating Iterator with {} Images'.format(len(roidb)))
    train_iter = MNIteratorChips(roidb=roidb, config=config, batch_size=batch_size, nGPUs=nGPUs, threads=32,
                                 pad_rois_to=400)
    print('The Iterator has {} samples!'.format(len(train_iter)))



    print('Initializing the model...')
    sym_inst = resnet_mx_101_e2e(n_proposals=400, momentum=args.momentum)
    sym = sym_inst.get_symbol_rcnn(config)

    # Creating the Logger
    logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)

    # get list of fixed parameters
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
    eval_metrics.add(rceval_metric)
    eval_metrics.add(rccls_metric)
    eval_metrics.add(rcbbox_metric)

    # eval_metrics.add(vis_metric)

    optimizer_params = get_optim_params(config, len(train_iter), batch_size)
    print ('Optimizer params: {}'.format(optimizer_params))

    # Checkpointing
    prefix = os.path.join(output_path, args.save_prefix)
    batch_end_callback = mx.callback.Speedometer(batch_size, args.display)
    epoch_end_callback = [mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True),
                          checkpoint_callback(sym_inst.get_bbox_param_names(), prefix, bbox_means, bbox_stds)]

    train_iter = PrefetchingIter(train_iter)
    mod.fit(train_iter, optimizer='sgd', optimizer_params=optimizer_params,
            eval_metric=eval_metrics, num_epoch=config.TRAIN.end_epoch, kvstore=config.default.kvstore,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback, arg_params=arg_params, aux_params=aux_params)
