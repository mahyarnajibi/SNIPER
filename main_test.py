import init
from iterators.MNIteratorTest import MNIteratorTest
from load_model import load_param
from symbols.faster.resnet_mx_101_e2e import resnet_mx_101_e2e, checkpoint_callback
from configs.faster.default_configs import config, update_config
from load_data import load_proposal_roidb, merge_roidb, filter_roidb, add_chip_data, remove_small_boxes
import mxnet as mx
from argparse import ArgumentParser
from general_utils import create_logger
from inference import Tester
import os

def parser():
    arg_parser = ArgumentParser('Faster R-CNN training module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/rpn_res101_mx_e2e.yml',type=str)
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
    batch_size = nGPUs * config.TEST.BATCH_IMAGES


    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # Create roidb
    roidb, imdb = load_proposal_roidb(config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path,
                                  config.dataset.dataset_path,
                                  proposal=config.dataset.proposal, only_gt=True, flip=False,
                                  result_path=config.output_path,
                                  proposal_path=config.proposal_path, get_imdb=True)
    # roidb = filter_roidb(roidb, config)


    print('Creating Iterator with {} Images'.format(len(roidb)))
    test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=batch_size, nGPUs=nGPUs, threads=32,
                                 pad_rois_to=400, crop_size=None)


    print('Initializing the model...')
    sym_inst = resnet_mx_101_e2e(n_proposals=400, momentum=args.momentum)
    sym = sym_inst.get_symbol_rcnn(config, is_train=False)

    # Creating the Logger
    logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)

    # Create the module
    shape_dict = dict(test_iter.provide_data_single)
    sym_inst.infer_shape(shape_dict)
    model_prefix = os.path.join(output_path, args.save_prefix)
    arg_params, aux_params = load_param(model_prefix, config.TEST.TEST_EPOCH, convert=True)
    mod = mx.mod.Module(symbol=sym,
                        context=context,
                        data_names=[k[0] for k in test_iter.provide_data_single],
                        label_names=None)

    mod.bind(test_iter.provide_data, test_iter.provide_label, for_training=False)
    mod.init_params(arg_params=arg_params,aux_params=aux_params)

    # Create Tester
    tester = Tester(mod, imdb, test_iter, cfg=config)
    tester.eval()


