import os
import argparse
import sys
import logging
import pprint
import cv2
sys.path.insert(0, 'lib')
from configs.faster.default_configs import config, update_config
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/configs/faster/res101_mx_3k.yml')

import mxnet as mx
from symbols import *


from bbox.bbox_transform import bbox_pred, clip_boxes
from demo.module import MutableModule
from demo.linear_classifier import train_model, classify_rois
from demo.vis_boxes import vis_boxes
from demo.image import resize, transform
from demo.load_model import load_param
from demo.tictoc import tic, toc
from demo.nms import nms
import pickle
from symbols.faster.resnet_mx_101_e2e_3k_demo import resnet_mx_101_e2e_3k_demo, checkpoint_callback

import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='R-FCN-3000 demo')
    parser.add_argument('--thresh', help='Output threshold', default=0.4)
    args = parser.parse_args()
    return args

args = parse_args()


def process_mul_scores(scores, xcls_scores):
    """
    Do multiplication of objectness score and classification score to obtain the final detection score.
    """
    final_scores = np.zeros((scores.shape[1], xcls_scores.shape[2]+1))
    final_scores[:, 1:] = xcls_scores[0][:, :] * scores[0][:, [1]]
    return final_scores


def main():
    # get symbol
    pprint.pprint(config)
    sym_inst = resnet_mx_101_e2e_3k_demo()
    sym = sym_inst.get_symbol_rcnn(config, is_train=False)

    # load demo data
    from os import listdir
    from os.path import isfile, join
    image_names = [f for f in listdir("./demo/image_3k") if isfile(join("./demo/image_3k", f))]

    data = []
    im_list = []
    im_info_list = []

    for im_name in image_names:
        assert os.path.exists(cur_path + '/demo/image_3k/' + im_name), ('%s does not exist'.format('/demo/image_3k/' + im_name))
        im = cv2.imread(cur_path + '/demo/image_3k/' + im_name, cv2.IMREAD_COLOR | 128)
        target_size = config.TEST.SCALES[0][0]
        max_size = config.TEST.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.RPN_FEAT_STRIDE)
        im_list.append(im)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        im_info_list.append(im_info)
        data.append({'data': im_tensor, 'im_info': im_info})


    # get module
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.TEST.SCALES]), max([v[1] for v in config.TEST.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    output_path = './output/chips_resnet101_3k/res101_mx_3k/fall11_whole/'
    model_prefix = os.path.join(output_path, 'CRCNN')
    arg_params, aux_params = load_param(model_prefix, config.TEST.TEST_EPOCH,
                                        convert=True, process=True)
    # set model
    mod = MutableModule(sym, data_names, label_names, context=[mx.gpu(0)], max_data_shapes=max_data_shape)
    mod.bind(provide_data, provide_label, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)   

    # load the [index] - [class name] matching file
    with open("./data/ILSVRC2014_devkit/data/3kcls_1C_words.txt",'rb') as f:
        index2words = pickle.load(f)

    # test
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])

        tic()
        mod.forward(data_batch)
        # get outputs from network
        scores = mod.get_outputs()[1].asnumpy()
        xcls_scores = mod.get_outputs()[3].asnumpy()
        roi_this = bbox_pred(mod.get_outputs()[0].asnumpy().reshape((-1, 5))[:, 1:], np.array([0.1, 0.1, 0.2, 0.2]) * mod.get_outputs()[2].asnumpy()[0])
        roi_this = clip_boxes(roi_this, im_info_list[idx][0][:2])
        boxes = roi_this / im_info_list[idx][0][2]

        mul_scores = process_mul_scores(scores, xcls_scores)
        boxes = boxes.astype('f')
        mul_scores = mul_scores.astype('f')
        dets_nms = []
        for j in range(1, mul_scores.shape[1]):
            cls_scores = mul_scores[:, j, np.newaxis]
            cls_boxes = boxes[:, 0:4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets, 0.5)
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > float(args.thresh), :]
            dets_nms.append(cls_dets)

        print 'testing {} {:.4f}s'.format(im_name, toc())
        # visualize
        im = cv2.cvtColor(im_list[idx].astype(np.uint8), cv2.COLOR_BGR2RGB)
        vis_boxes(im_name, im, dets_nms, im_info_list[idx][0][2], config, args.thresh, index2words)


    print 'done'

if __name__ == '__main__':
    main()
