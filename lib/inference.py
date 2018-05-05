#------------------------------------------------------------------
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Inference module for performing detection and proposal extraction
# Written by Mahyar Najibi
# -----------------------------------------------------------------
import numpy as np
from bbox.bbox_transform import bbox_pred, clip_boxes
from iterators.PrefetchingIter import PrefetchingIter
import os
import time
import cPickle
from utils.data_workers import nms_worker
from nms.nms import py_nms_wrapper, soft_nms
from utils.visualization import visualize_dets
from multiprocessing import Pool
from tqdm import tqdm

class Tester(object):
    def __init__(self, module, imdb, roidb, test_iter, cfg, rcnn_output_names=None,rpn_output_names=None,
                 logger=None, batch_size=None):
        self.test_iter = test_iter
        self.scale = test_iter.test_scale
        # Make sure that iterator is instnace of Prefetching iterator
        if test_iter and not isinstance(test_iter, PrefetchingIter):
            self.test_iter = PrefetchingIter(self.test_iter)
        self.cfg = cfg
        self.module = module
        if test_iter:
            self.data_names = [k[0] for k in test_iter.provide_data_single]
        self.rcnn_output_names = rcnn_output_names
        if not self.rcnn_output_names:
            self.rcnn_output_names = {
                'cls' : 'cls_prob_reshape_output',
                'bbox': 'bbox_pred_reshape_output',
                'im_ids': 'im_ids'
            }
        self.rpn_output_names = rpn_output_names
        if not self.rpn_output_names:
            self.rpn_output_names = {
                'scores' : 'rois_score',
                'rois': 'rois_output',
                'im_ids': 'im_ids'
            }
        self.logger = logger
        self.result_path = imdb.result_path
        self.num_classes = imdb.num_classes
        self.class_names = imdb.classes
        self.num_images = len(roidb)
        self.imdb_name = imdb.name
        self.nms = py_nms_wrapper(cfg.TEST.NMS)
        self.batch_size = batch_size
        self.roidb = roidb

        if not self.batch_size:
            self.batch_size = self.cfg.TEST.BATCH_IMAGES

    def forward(self, batch):
        self.module.forward(batch, is_train=False)
        return [dict(zip(self.module.output_names, i))
                for i in zip(*self.module.get_outputs(merge_multi_context=False))]

    def get_proposals(self, batch, scales):
        data = dict(zip(self.data_names, batch.data))
        outputs = self.forward(batch)
        scores, rois = [], []
        im_shapes = np.array([im.shape[-2:] for im in data['data']]).reshape(-1, self.batch_size, 2)
        im_ids = np.array([], dtype=int)
        for i, (gpu_out, gpu_scales, gpu_shapes) in enumerate(zip(outputs, scales, im_shapes)):

            gpu_rois = gpu_out[self.rpn_output_names['rois']].asnumpy()
            # Reshape crois
            nper_gpu = gpu_rois.shape[0] / self.batch_size
            gpu_scores = gpu_out[self.rpn_output_names['scores']].asnumpy()
            im_ids = np.hstack((im_ids, gpu_out[self.rpn_output_names['im_ids']].asnumpy().astype(int)))
            for idx in range(self.batch_size):
                cids = np.where(gpu_rois[:, 0] == idx)[0]
                assert len(cids) == nper_gpu, 'The number of rois per GPU should be fixed!'
                crois = gpu_rois[cids, 1:]/gpu_scales[idx]
                cscores = gpu_scores[cids]
                # Store predictions
                scores.append(cscores)
                rois.append(crois)
        #print('Time to aggregate results in the batch: {}', time.time()-stime)
        return scores, rois, data, im_ids

    def detect(self, batch, scales):
        data = dict(zip(self.data_names, batch.data))
        outputs = self.forward(batch)
        scores, preds = [], []
        im_shapes = np.array([im.shape[-2:] for im in data['data']]).reshape(-1, self.batch_size, 2)
        im_ids = np.array([], dtype=int)

        for i, (gpu_out, gpu_scales, gpu_shapes) in enumerate(zip(outputs, scales, im_shapes)):
            gpu_rois = gpu_out[self.rpn_output_names['rois']].asnumpy()
            # Reshape crois
            nper_gpu = gpu_rois.shape[0] / self.batch_size
            gpu_scores = gpu_out[self.rcnn_output_names['cls']].asnumpy()
            gpu_deltas = gpu_out[self.rcnn_output_names['bbox']].asnumpy()
            im_ids = np.hstack((im_ids, gpu_out[self.rcnn_output_names['im_ids']].asnumpy().astype(int)))
            for idx in range(self.batch_size):
                cids = np.where(gpu_rois[:, 0] == idx)[0]
                assert len(cids) == nper_gpu, 'The number of rois per GPU should be fixed!'
                crois = gpu_rois[cids, 1:]
                cscores = gpu_scores[idx]
                cdeltas = gpu_deltas[idx]

                # Apply deltas and clip predictions
                cboxes = bbox_pred(crois, cdeltas)
                cboxes = clip_boxes(cboxes, gpu_shapes[idx])

                # Re-scale boxes
                cboxes = cboxes / gpu_scales[idx]

                # Store predictions
                scores.append(cscores)
                preds.append(cboxes)
        return scores, preds, data, im_ids

    def show_info(self, print_str):
        print(print_str)
        if self.logger: self.logger.info(print_str)



    def aggregate(self, scale_cls_dets, vis=False, cache_name= 'cache'):
        n_scales = len(scale_cls_dets)
        assert n_scales== len(self.cfg.TEST.VALID_RANGES), 'A valid range should be specified for each test scale'
        all_boxes = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)]
        nms_pool = Pool(32)
        self.show_info('Aggregating detections from multiple scales...')
        for i in tqdm(range(self.num_images)):
            parallel_nms_args = []
            for j in range(1, self.num_classes):
                agg_dets = np.empty((0,5),dtype=np.float32)
                for all_cls_dets, valid_range in zip(scale_cls_dets, self.cfg.TEST.VALID_RANGES):
                    cls_dets = all_cls_dets[j][i]
                    heights = cls_dets[:, 2] - cls_dets[:, 0]
                    widths = cls_dets[:, 3] - cls_dets[:, 1]
                    areas = widths * heights
                    lvalid_ids = np.where(areas > valid_range[0]*valid_range[0])[0] if valid_range[0] > 0 else \
                        np.arange(len(areas))
                    uvalid_ids = np.where(areas <= valid_range[1]*valid_range[1])[0] if valid_range[1] > 0 else \
                        np.arange(len(areas))
                    valid_ids = np.intersect1d(lvalid_ids,uvalid_ids)
                    cls_dets = cls_dets[valid_ids, :] if len(valid_ids) > 0 else cls_dets
                    agg_dets = np.vstack((agg_dets, cls_dets))
                parallel_nms_args.append([agg_dets, self.cfg.TEST.NMS_SIGMA])
            # Apply nms
            final_dets = nms_pool.map(nms_worker, parallel_nms_args)
            for j in range(1, self.num_classes):
                all_boxes[j][i] = final_dets[j-1]
            if self.cfg.TEST.MAX_PER_IMAGE > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, self.num_classes)])
                if len(image_scores) > self.cfg.TEST.MAX_PER_IMAGE:
                    image_thresh = np.sort(image_scores)[-self.cfg.TEST.MAX_PER_IMAGE]
                    for j in range(1, self.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]
            if vis:
                import cv2
                im = cv2.cvtColor(cv2.imread(self.roidb[i]['image']), cv2.COLOR_BGR2RGB)
                visualize_dets(im,
                               [[]] + [all_boxes[j][i] for j in range(1, self.num_classes)],
                               1.0,
                               self.cfg.network.PIXEL_MEANS, self.class_names, threshold=0.5,
                               save_path=os.path.join(self.cfg.TEST.VISUALIZATION_PATH, '{}.png'.format(i)), 
                               transform=False)

        nms_pool.close()
        cache_path = os.path.join(self.result_path, cache_name)
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        cache_path = os.path.join(cache_path, 'final_detections.pkl')
        self.show_info('Done! Saving detections into: {}'.format(cache_path))
        with open(cache_path, 'wb') as detfile:
            cPickle.dump(all_boxes, detfile)
        return all_boxes

    def get_detections(self, cls_thresh=1e-3, cache_name= 'cache', evaluate= False, vis=False):
        all_boxes = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)]
        data_counter = 0
        detect_time, post_time = 0, 0

        if vis and not os.path.isdir(self.cfg.TEST.VISUALIZATION_PATH):
            os.makedirs(self.cfg.TEST.VISUALIZATION_PATH)

        for batch in self.test_iter:
            im_info = batch.data[1].asnumpy()
            scales = im_info[:,2].reshape(-1,self.batch_size)
            # Run detection on the batch
            stime = time.time()
            scores, boxes, data, im_ids = self.detect(batch, scales)
            detect_time += time.time() - stime

            stime = time.time()
            for i, (cscores, cboxes, im_id) in enumerate(zip(scores, boxes, im_ids)):
                for j in range(1, self.num_classes):
                    # Apply the score threshold
                    inds = np.where(cscores[:, j] > cls_thresh)[0]
                    rem_scores = cscores[inds, j, np.newaxis]
                    rem_boxes = cboxes[inds, 0:4]
                    cls_dets = np.hstack((rem_boxes, rem_scores))

                    if evaluate or vis:
                        keep = self.nms(cls_dets)
                        cls_dets = cls_dets[keep, :]

                    all_boxes[j][im_id] = cls_dets

                # Filter boxes based on max_per_image if requested
                if evaluate and self.cfg.TEST.MAX_PER_IMAGE:
                    image_scores = np.hstack([all_boxes[j][im_id][:, -1]
                                              for j in range(1, self.num_classes)])
                    if len(image_scores) > self.cfg.TEST.MAX_PER_IMAGE:
                        image_thresh = np.sort(image_scores)[-self.cfg.TEST.MAX_PER_IMAGE]
                        for j in range(1, self.num_classes):
                            keep = np.where(all_boxes[j][im_id][:, -1] >= image_thresh)[0]
                            all_boxes[j][im_id] = all_boxes[j][im_id][keep, :]
                if vis:
                    visualize_dets(batch.data[0][i].asnumpy(),
                                   [[]]+[all_boxes[j][im_id] for j in range(1, self.num_classes)], im_info[i, 2],
                                   self.cfg.network.PIXEL_MEANS, self.class_names, threshold=0.5,
                                   save_path=os.path.join(self.cfg.TEST.VISUALIZATION_PATH,'{}.png'.format(im_id)))

            data_counter += self.test_iter.get_batch_size()
            post_time += time.time() - stime
            self.show_info('Tester: {}/{}, Detection: {:.4f}s, Post Processing: {:.4}s'.format(data_counter, self.num_images,
                                                                               detect_time / data_counter,
                                                                               post_time / data_counter ))

        cache_path = os.path.join(self.result_path, cache_name)
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        cache_path = os.path.join(cache_path, 'detections.pkl')
        self.show_info('Done! Saving detections into: {}'.format(cache_path))
        with open(cache_path, 'wb') as detfile:
            cPickle.dump(all_boxes, detfile)
        return all_boxes


    def extract_proposals(self, n_proposals=300, cache_name= 'cache', vis=False):
        all_boxes = [[] for _ in range(self.num_images)]
        data_counter = 0
        detect_time, post_time = 0, 0
        if vis and not os.path.isdir(self.cfg.TEST.VISUALIZATION_PATH):
            os.makedirs(self.cfg.TEST.VISUALIZATION_PATH)

        for batch in self.test_iter:
            im_info = batch.data[1].asnumpy()
            scales = im_info[:,2].reshape(-1,self.batch_size)
            # Run detection on the batch
            stime = time.time()
            scores, boxes, data, im_ids = self.get_proposals(batch, scales)
            detect_time += time.time() - stime

            stime = time.time()
            for i, (cscores, cboxes, im_id) in enumerate(zip(scores, boxes, im_ids)):
                # Keep the requested number of rois
                rem_scores = cscores[0:n_proposals, np.newaxis]
                rem_boxes = cboxes[0:n_proposals, 0:4]
                cls_dets = np.hstack((rem_boxes, rem_scores)).astype(np.float32)
                if vis:
                    visualize_dets(batch.data[0][i].asnumpy(),
                                   [[]]+[cls_dets], im_info[i, 2],
                                   self.cfg.network.PIXEL_MEANS, ['__background__','object'], threshold=0.5,
                                   save_path=os.path.join(self.cfg.TEST.VISUALIZATION_PATH,'{}.png'.format(im_id)))
                all_boxes[im_id] = cls_dets
            data_counter += self.test_iter.get_batch_size()
            post_time += time.time() - stime
            self.show_info('Tester: {}/{}, Detection: {:.4f}s, Post Processing: {:.4}s'.format(data_counter, self.num_images,
                                                                               detect_time / data_counter,
                                                                               post_time / data_counter ))
        cache_path = os.path.join(self.result_path, cache_name)
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        cache_path=os.path.join(cache_path,'proposals.pkl')
        self.show_info('Done! Saving detections into: {}'.format(cache_path))
        with open(cache_path, 'wb') as detfile:
            cPickle.dump(all_boxes, detfile)
        return all_boxes
