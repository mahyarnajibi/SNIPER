# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Inference Module
# by Mahyar Najibi and Bharat Singh
# --------------------------------------------------------------
import init
import matplotlib
matplotlib.use('Agg')
from symbols.faster import *
from configs.faster.default_configs import config, update_config
from data_utils.load_data import load_proposal_roidb
import mxnet as mx
from argparse import ArgumentParser
from train_utils.utils import create_logger, load_param
from inference import imdb_detection_wrapper
from inference import imdb_proposal_extraction_wrapper
import os

def parser():
    arg_parser = ArgumentParser('Faster R-CNN training module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/sniper_res101_e2e.yml',type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--vis', dest='vis', help='Whether to visualize the detections',
                            action='store_true')
    return arg_parser.parse_args()



def main():
    args = parser()
    update_config(args.cfg)
    context = [mx.gpu(int(gpu)) for gpu in config.gpus.split(',')]

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # Create roidb
    roidb, imdb = load_proposal_roidb(config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path,
                                      config.dataset.dataset_path,
                                      proposal=config.dataset.proposal, only_gt=True, flip=False,
                                      result_path=config.output_path,
                                      proposal_path=config.proposal_path, get_imdb=True)

    # Creating the Logger
    logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    print(output_path)
    model_prefix = os.path.join(output_path, args.save_prefix)
    arg_params, aux_params = load_param(model_prefix, config.TEST.TEST_EPOCH,
                                        convert=True, process=True)

    sym_inst = eval('{}.{}'.format(config.symbol, config.symbol))
    if config.TEST.EXTRACT_PROPOSALS:
        imdb_proposal_extraction_wrapper(sym_inst, config, imdb, roidb, context, arg_params, aux_params, args.vis)
    else:
        imdb_detection_wrapper(sym_inst, config, imdb, roidb, context, arg_params, aux_params, args.vis)

if __name__ == '__main__':
    main()



