# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# SNIPER demo
# by Mahyar Najibi
# --------------------------------------------------------------
import init
import matplotlib
matplotlib.use('Agg')
from configs.faster.default_configs import config, update_config
import mxnet as mx
from argparse import ArgumentParser
from train_utils.utils import create_logger, load_param
import os
from PIL import Image
from iterators.MNIteratorTest import MNIteratorTest
from easydict import EasyDict
from inference import Tester
from multiprocessing import Pool
from symbols.faster import *
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
def parser():
    arg_parser = ArgumentParser('Faster R-CNN training module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/sniper_res101_e2e.yml',type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--model_weights', dest='weight_path', help='Path to the trained model weights', type=str,
                            default='output/sniper_res101_bn/sniper_res101_e2e/train2014_val2014/')
    arg_parser.add_argument('--im_path', dest='im_path', help='Path to the image', type=str,
                            default='data/demo/demo.jpg')
    arg_parser.add_argument('--vis', dest='vis', help='Whether to visualize the detections',
                            action='store_true')
    return arg_parser.parse_args()


# Create a model for a specific scale and
# perform detection in parallel for multi-scale settings
def scale_worker(arguments):
    [scale, context, config, sym_def, \
     roidb, imdb, arg_params, aux_params] = arguments

    # Create the module and initialize the weights
    sym_inst = sym_def(n_proposals=400, test_nbatch=1)
    sym = sym_inst.get_symbol_rcnn(config, is_train=False)
    test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=1, nGPUs=1, threads=1,
                               pad_rois_to=400, crop_size=None, test_scale=scale, num_classes = imdb.num_classes)
    # Create the module
    shape_dict = dict(test_iter.provide_data_single)
    sym_inst.infer_shape(shape_dict)
    mod = mx.mod.Module(symbol=sym,
                        context=context,
                        data_names=[k[0] for k in test_iter.provide_data_single],
                        label_names=None)
    mod.bind(test_iter.provide_data, test_iter.provide_label, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    tester = Tester(mod, imdb, roidb, test_iter, cfg=config, batch_size=1)
    return tester.get_detections(vis=False,
                                 evaluate=False,
                                 cache_name=None)

def main():
    args = parser()
    update_config(args.cfg)

    # Use just the first GPU for demo
    context = [mx.gpu(int(config.gpus[0]))]

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # Get image dimensions
    width, height = Image.open(args.im_path).size

    # Pack image info
    roidb = [{'image': args.im_path, 'width': width, 'height': height, 'flipped': False}]

    # Creating the Logger
    logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)

    # Create the model and initialize the weights
    model_prefix = os.path.join(output_path, args.save_prefix)
    arg_params, aux_params = load_param(model_prefix, config.TEST.TEST_EPOCH,
                                        convert=True, process=True)

    # Get the symbol definition
    sym_def = eval('{}.{}'.format(config.symbol, config.symbol))

    # Pack db info
    db_info = EasyDict()
    db_info.name = 'coco'
    db_info.result_path = 'data/demo'

    # Categories the detector trained for:
    db_info.classes = [u'BG', u'person', u'bicycle', u'car', u'motorcycle', u'airplane',
                       u'bus', u'train', u'truck', u'boat', u'traffic light', u'fire hydrant',
                       u'stop sign', u'parking meter', u'bench', u'bird', u'cat', u'dog', u'horse', u'sheep', u'cow',
                       u'elephant', u'bear', u'zebra', u'giraffe', u'backpack', u'umbrella', u'handbag', u'tie',
                       u'suitcase', u'frisbee', u'skis', u'snowboard', u'sports\nball', u'kite', u'baseball\nbat',
                       u'baseball glove', u'skateboard', u'surfboard', u'tennis racket', u'bottle', u'wine\nglass',
                       u'cup', u'fork', u'knife', u'spoon', u'bowl', u'banana', u'apple', u'sandwich', u'orange',
                       u'broccoli', u'carrot', u'hot dog', u'pizza', u'donut', u'cake', u'chair', u'couch',
                       u'potted plant', u'bed', u'dining table', u'toilet', u'tv', u'laptop', u'mouse', u'remote',
                       u'keyboard', u'cell phone', u'microwave', u'oven', u'toaster', u'sink', u'refrigerator', u'book',
                       u'clock', u'vase', u'scissors', u'teddy bear', u'hair\ndrier', u'toothbrush']
    db_info.num_classes = len(db_info.classes)

    # Perform detection for each scale in parallel
    p_args = []
    for s in config.TEST.SCALES:
        p_args.append([s, context, config, sym_def, roidb, db_info, arg_params, aux_params])
    pool = Pool(len(config.TEST.SCALES))
    all_detections = pool.map(scale_worker, p_args)

    tester = Tester(None, db_info, roidb, None, cfg=config, batch_size=1)
    all_detections = tester.aggregate(all_detections, vis=True, cache_name=None, vis_path='./data/demo/',
                                          vis_name='demo_detections')

if __name__ == '__main__':
    main()