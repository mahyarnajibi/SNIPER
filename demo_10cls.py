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
import pickle
from symbols.faster.resnet_mx_101_e2e_3k_demo import resnet_mx_101_e2e_3k_demo, checkpoint_callback


def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def parse_args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--thresh', help='Output threshold', default=0.5)
    args = parser.parse_args()
    return args

args = parse_args()

def process_tencls_scores(scores, xcls_scores):
    """
    Do multiplication of objectness score and classification score to obtain the final detection score.
    """
    final_scores = np.zeros((scores.shape[0], xcls_scores.shape[1]+1))
    final_scores[:, 1:] = xcls_scores[:, :] * scores[:, [1]]
    return final_scores

def main():
    # get symbol
    pprint.pprint(config)
    sym_inst = resnet_mx_101_e2e_3k_demo()
    sym = sym_inst.get_symbol_rcnn(config, is_train=False)

    # load data
    from os import listdir
    from os.path import isfile, isdir, join
    dir_names = [f for f in listdir("./demo/image/") if isdir(join("./demo/image", f))]
    image_names_full = []
    image_num_per_class = []
    for folder in dir_names:
        image_names = [folder + '/' + a for a in listdir("./demo/image/" + folder) if isfile(join("./demo/image", folder, a)) and a.split('.')[-1] == 'jpg']
        image_names_full = image_names_full + image_names
        image_num_per_class.append(len(image_names))

    data = []
    im_list = []
    im_info_list = []

    # load raw data(image); if pickled, load it
    if os.path.exists("./demo/cache/images.pkl"):
        print("Load data from cache.")
        f0 = open("./demo/cache/images.pkl", 'rb')
        im_list = pickle.load(f0)
        f0.close()
        f0 = open("./demo/cache/data.pkl", 'rb')
        data = pickle.load(f0)
        f0.close()
        for one in data:
            im_info_list.append(one['im_info'][0])
    else:
        for im_name in image_names_full:
            assert os.path.exists(cur_path + '/demo/image/' + im_name), ('%s does not exist'.format('./extract/' + im_name))
            im = cv2.imread(cur_path + '/demo/image/' + im_name, cv2.IMREAD_COLOR | 128)
            target_size = config.TEST.SCALES[0][0]
            max_size = config.TEST.SCALES[0][1]
            im, im_scale = resize(im, target_size, max_size, stride=config.network.RPN_FEAT_STRIDE)
            im_list.append(im)
            im_tensor = transform(im, config.network.PIXEL_MEANS)
            im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
            im_info_list.append(im_info)
            data.append({'data': im_tensor, 'im_info': im_info[0]}) #, 'im_ids': mx.nd.array([[1]])})

        f0 = open("./demo/cache/images.pkl", 'wb')
        pickle.dump(im_list, f0, protocol=2)
        f0.close()
        f0 = open("./demo/cache/data.pkl", 'wb')
        pickle.dump(data, f0, protocol=2)
        f0.close()


    # symbol preparation
    data_names = ['data', 'im_info'] #, 'im_ids']
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

    # extract feature extraction; if feature already pickled, skip
    if os.path.exists("./demo/cache/indices.pkl") and os.path.exists("./demo/cache/features.pkl"):
        print("Conv5 features extracted; skip.")
    else:
        features = []
        indices = []
        count = 0
        total = len(image_names_full)
        print("Extracting features of %d images..." %(total))
        tic()

        for idx, im_name in enumerate(image_names_full):
            count += 1
            data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                         provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                         provide_label=[None])
            mod.forward(data_batch)
            out = mod.get_outputs()[4].asnumpy()
            pooled_feat = mx.ndarray.Pooling(data=mx.ndarray.array(out), pool_type='avg', global_pool=True, kernel=(7, 7))
            features.append(pooled_feat.reshape((1, -1)).asnumpy())
            indices.append(im_name)

            if count % 100 == 0:
                print(str(count) + '/' + str(total) + ': {:.4f} seconds spent.'.format(toc()))

        # dump features and indices
        f1 = open("./demo/cache/indices.pkl", 'wb')
        pickle.dump(indices, f1, protocol=2)
        f1.close()
        f2 = open("./demo/cache/features.pkl", 'wb')
        pickle.dump(features, f2, protocol=2)
        f2.close()

    # train the linear classifier, get trained classifier and eval_list
    class_names = ()
    for one in dir_names:
        class_names = class_names + (one,)
    linear_classifier, eval_list = train_model(class_names, image_num_per_class, batch_size=100, learning_rate=0.001, momentum=0.9, num_epoch=250)
    eval_data = [data[i] for i in eval_list]
    eval_im_list = [im_list[i] for i in eval_list]
    im_info_list_eval = [im_info_list[i] for i in eval_list]
    image_names_eval = [image_names_full[i] for i in eval_list]

    # extract roi_pooled features for evaluation / visualization; if already pickled, skip
    rois = []
    objectness_scores = []
    roipooled_features = []

    if os.path.exists("./demo/cache/eval_roipooled_features.pkl") and \
        os.path.exists("./demo/cache/eval_rois.pkl") and \
        os.path.exists("./demo/cache/ob_scores.pkl"):

        print("Conv5 RoI Pooled features etc pickled, loading...")
        f3 = open("./demo/cache/eval_roipooled_features.pkl", 'rb')
        roipooled_features = pickle.load(f3)
        f3.close()
        f4 = open("./demo/cache/eval_rois.pkl", 'rb')
        rois = pickle.load(f4)
        f4.close()
        f5 = open("./demo/cache/ob_scores.pkl", 'rb')
        objectness_scores = pickle.load(f5)
        f5.close()        

    else:
        # get eval data based on the train val split
        count = 0
        total = len(eval_data)
        print("Extracting roipooled features of %d images..." %(total))
        tic()
        for idx in range(len(eval_data)):
            count += 1
            data_batch = mx.io.DataBatch(data=[eval_data[idx]], label=[], pad=0, index=idx,
                                         provide_data=[[(k, v.shape) for k, v in zip(data_names, eval_data[idx])]],
                                         provide_label=[None])
            mod.forward(data_batch)
            #can edit the weights and put the classifier in a fully convolutional way as well....change the network, for little bit speedup!
            roipooled_conv5_feat = mx.ndarray.ROIPooling(data=mod.get_outputs()[4], rois=mod.get_outputs()[0],
                                                             pooled_size=(7, 7), spatial_scale=0.0625)
            pooled_feat = mx.ndarray.Pooling(data=roipooled_conv5_feat, pool_type='avg', global_pool=True, kernel=(7, 7))

            roipooled_features.append(pooled_feat.reshape((pooled_feat.shape[0], -1)).asnumpy())
            roi_this = bbox_pred(mod.get_outputs()[0].asnumpy().reshape((-1, 5))[:, 1:], np.array([0.1, 0.1, 0.2, 0.2]) * mod.get_outputs()[2].asnumpy()[0])
            roi_this = clip_boxes(roi_this, im_info_list_eval[idx][:2])
            roi_this = roi_this / im_info_list_eval[idx][2]
            rois.append(roi_this)
            objectness_scores.append(mod.get_outputs()[1].asnumpy())

            if count % 100 == 0:
                print(str(count) + '/' + str(total) + ': {:.4f} seconds spent.'.format(toc()))
        
        # dump roi pooled features, rois and objectness scores for the eval images
        f3 = open("./demo/cache/eval_roipooled_features.pkl", 'wb')
        pickle.dump(roipooled_features, f3, protocol=2)
        f3.close()
        f4 = open("./demo/cache/eval_rois.pkl", 'wb')
        pickle.dump(rois, f4, protocol=2)
        f4.close()
        f5 = open("./demo/cache/ob_scores.pkl", 'wb')
        pickle.dump(objectness_scores, f5, protocol=2)
        f5.close()

    # classify the rois
    rois_cls = classify_rois(linear_classifier, roipooled_features)

    for idx in range(len(rois)):
        im_name = image_names_eval[idx]
        xcls_scores = process_tencls_scores(objectness_scores[idx][0], rois_cls[idx])
        boxes = rois[idx].astype('f')
        xcls_scores = xcls_scores.astype('f')
        dets_nms = []
        for j in range(1, xcls_scores.shape[1]):
            cls_scores = xcls_scores[:, j, np.newaxis]
            cls_boxes = boxes[:, 0:4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets, 0.45)
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > float(args.thresh), :]
            dets_nms.append(cls_dets)

        print 'testing {}'.format(im_name)
        # visualize
        im = cv2.cvtColor(eval_im_list[idx].astype(np.uint8), cv2.COLOR_BGR2RGB)
        vis_boxes(im_name, im, dets_nms, im_info_list_eval[idx][2], config, args.thresh, dir_names)

    print('Done')

if __name__ == '__main__':
    main()
