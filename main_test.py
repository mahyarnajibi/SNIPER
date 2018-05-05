import init
import matplotlib
import math
matplotlib.use('Agg')
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
import cPickle
import numpy as np
from multiprocessing import Pool
def parser():
    arg_parser = ArgumentParser('Faster R-CNN training module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/rpn_res101_mx_e2e.yml',type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='CRCNN', type=str)
    arg_parser.add_argument('--vis', dest='vis', help='Whether to visualize the detections',
                            action='store_true')
    return arg_parser.parse_args()

def detect_scale(arguments):
    [args, scale, nbatch, context, config,\
     roidb, imdb, arg_params, aux_params, vis] = arguments
    print('Performing detection for scale: {}'.format(scale))
    nGPUs= len(context)
    sym_inst = resnet_mx_101_e2e(n_proposals=400, test_nbatch=nbatch)
    sym = sym_inst.get_symbol_rcnn(config, is_train=False)
    test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=nGPUs * nbatch, nGPUs=nGPUs, threads=32,
                               pad_rois_to=400, crop_size=None, test_scale=scale)
    # Create the module
    shape_dict = dict(test_iter.provide_data_single)
    sym_inst.infer_shape(shape_dict)
    mod = mx.mod.Module(symbol=sym,
                        context=context,
                        data_names=[k[0] for k in test_iter.provide_data_single],
                        label_names=None)
    mod.bind(test_iter.provide_data, test_iter.provide_label, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)
    # Create Tester
    tester = Tester(mod, imdb, roidb, test_iter, cfg=config, batch_size=nbatch)
    return tester.get_detections(vis=vis,
                                    evaluate=(len(config.TEST.BATCH_IMAGES) == 1))

def main():
    args = parser()
    update_config(args.cfg)
    context = [mx.gpu(int(gpu)) for gpu in config.gpus.split(',')]
    nGPUs = len(context)

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

    if config.TEST.EXTRACT_PROPOSALS:
        proposals = []
        for nbatch, scale in zip(config.TEST.BATCH_IMAGES, config.TEST.SCALES):
            print('Extracting proposals for scale: {}'.format(scale))
            sym_inst = resnet_mx_101_e2e(n_proposals=400, momentum=args.momentum, test_nbatch=nbatch)
            sym = sym_inst.get_symbol_rpn(config, is_train=False)
            test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=nGPUs * nbatch, nGPUs=nGPUs, threads=32,
                                       pad_rois_to=400, crop_size=None, test_scale=scale)
            # Create the module
            shape_dict = dict(test_iter.provide_data_single)
            sym_inst.infer_shape(shape_dict)
            mod = mx.mod.Module(symbol=sym,
                                context=context,
                                data_names=[k[0] for k in test_iter.provide_data_single],
                                label_names=None)
            mod.bind(test_iter.provide_data, test_iter.provide_label, for_training=False)
            mod.init_params(arg_params=arg_params, aux_params=aux_params)
            # Create Tester
            tester = Tester(mod, imdb, roidb, test_iter, cfg=config, batch_size=nbatch)
            proposals.append(tester.extract_proposals(vis=args.vis))

        if not os.path.isdir(config.TEST.PROPOSAL_SAVE_PATH):
            os.makedirs(config.TEST.PROPOSAL_SAVE_PATH)

        final_proposals = proposals[0]

        if len(proposals) > 1:
            for i in range(len(proposals[0])):
                for j in range(1, len(proposals)):
                    final_proposals[i] = np.vstack((final_proposals[i], proposals[j][i]))
        save_path = os.path.join(config.TEST.PROPOSAL_SAVE_PATH, '{}_{}_rpn.pkl'.format(config.dataset.dataset,
                                                                                        config.dataset.test_image_set))
        with open(save_path, 'wb') as file:
            cPickle.dump(final_proposals, file)
    else:
        detections = []
        if config.TEST.CONCURRENT_JOBS==1:
            for nbatch, scale in zip(config.TEST.BATCH_IMAGES, config.TEST.SCALES):
                detections.append(detect_scale([args, scale, nbatch, context, config,
                    roidb, imdb, arg_params, aux_params, args.vis]))
        else:
            parallel_args = []
            im_per_job = int(math.ceil(float(len(roidb))/config.TEST.CONCURRENT_JOBS))
            roidbs = []
            pool = Pool(config.TEST.CONCURRENT_JOBS)
            for i in range(config.TEST.CONCURRENT_JOBS):
                roidbs.append([roidb[j] for j in range(im_per_job*i, min(im_per_job*(i+1), len(roidb)))])

            for i, (nbatch, scale) in enumerate(zip(config.TEST.BATCH_IMAGES, config.TEST.SCALES)):
                for j in range(config.TEST.CONCURRENT_JOBS):
                    parallel_args.append([args, scale, nbatch, context, config,
                        roidbs[j], imdb, arg_params, aux_params, args.vis])
                
                detection_list = pool.map(detect_scale, parallel_args)
                tmp_dets = []
                for det in detection_list:
                    tmp_dets.append(det)
                detections.append(tmp_dets)
            pool.close()
        if len(config.TEST.SCALES) > 1:
            tester = Tester(None, imdb, roidb, None, cfg=config, batch_size=nbatch)
            all_boxes = tester.aggregate(detections, vis=False) if len(config.TEST.SCALES) > 1 \
                                            else detections[0]
        print('Evaluating detections...')
        imdb.evaluate_detections(all_boxes)
        print('All done!')
            



if __name__ == '__main__':
    main()



