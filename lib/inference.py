import mxnet as mx
import numpy as np
from bbox.bbox_transform import bbox_pred, clip_boxes
from iterators.PrefetchingIter import PrefetchingIter
import os
import time
import cPickle
from nms.nms import py_nms_wrapper
from utils.visualization import visualize_dets
class Tester(object):
    def __init__(self, module, imdb, test_iter, cfg, output_names=None, logger=None):
        self.test_iter = test_iter
        # Make sure that iterator is instnace of Prefetching iterator
        if not isinstance(test_iter, PrefetchingIter):
            self.test_iter = PrefetchingIter(self.test_iter)
        self.cfg = cfg
        self.module = module
        self.data_names = [k[0] for k in test_iter.provide_data_single]
        self.output_names = output_names
        if not self.output_names:
            self.output_names = {
                'cls' : 'cls_prob_reshape_output',
                'bbox': 'bbox_pred_reshape_output'
            }
        self.logger = logger
        self.result_path = imdb.result_path
        self.num_classes = imdb.num_classes
        self.class_names = imdb.classes
        self.num_images = imdb.num_images
        self.imdb_name = imdb.name
        self.nms = py_nms_wrapper(cfg.TEST.NMS)

    def forward(self, batch):
        self.module.forward(batch)
        return [dict(zip(self.module.output_names, i))
                for i in zip(*self.module.get_outputs(merge_multi_context=False))]

    def detect(self, batch, scales):
        data = dict(zip(self.data_names, batch.data))
        outputs = self.forward(batch)
        scores, preds = [], []
        im_shapes = np.array([im.shape[-2:] for im in data['data']]).reshape(-1, self.cfg.TEST.BATCH_IMAGES, 2)
        im_ids = np.array([], dtype=int)

        for i, (gpu_out, gpu_scales, gpu_shapes) in enumerate(zip(outputs, scales, im_shapes)):
            gpu_rois = gpu_out['rois_output'].asnumpy()
            # Reshape crois
            nper_gpu = gpu_rois.shape[0] / self.cfg.TEST.BATCH_IMAGES
            gpu_scores = gpu_out[self.output_names['cls']].asnumpy()
            gpu_deltas = gpu_out[self.output_names['bbox']].asnumpy()
            im_ids = np.hstack((im_ids, gpu_out['im_ids'].asnumpy().astype(int)))
            for idx in range(self.cfg.TEST.BATCH_IMAGES):
                cids = np.where(gpu_rois[:, 0] == idx)[0]
                assert len(cids)==nper_gpu, 'The number of rois per GPU should be fixed!'
                crois = gpu_rois[cids, 1:]
                cscores = gpu_scores[idx]
                cdeltas = gpu_deltas[idx]

                # Apply deltas and clip predictions
                cboxes = bbox_pred(crois, cdeltas * np.array([2, 2, 0.5, 0.5]))
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

    def evaluate(self, cls_thresh=1e-3, cache_name= 'data/detections.pkl' , filter_detections= False, vis=False):
        all_boxes = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)]
        data_counter = 0
        detect_time, post_time = 0, 0

        if vis and not os.path.isdir(self.cfg.TEST.VISUALIZATION_PATH):
            os.makedirs(self.cfg.TEST.VISUALIZATION_PATH)

        for batch in self.test_iter:
            im_info = batch.data[1].asnumpy()
            scales = im_info[:,2].reshape(-1,self.cfg.TEST.BATCH_IMAGES)
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

                    if filter_detections or vis:
                        keep = self.nms(cls_dets)
                        cls_dets = cls_dets[keep, :]

                    all_boxes[j][im_id] = cls_dets

                # Filter boxes based on max_per_image if requested
                if filter_detections and self.cfg.TEST.max_per_image:
                    image_scores = np.hstack([all_boxes[j][im_id][:, -1]
                                              for j in range(1, self.num_classes)])
                    if len(image_scores) > self.cfg.TEST.max_per_image:
                        image_thresh = np.sort(image_scores)[-self.cfg.TEST.max_per_image]
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
        self.show_info('Done! Saving detections into: {}'.format(cache_path))
        with open(cache_path, 'wb') as detfile:
            cPickle.dump(all_boxes, detfile)
        return all_boxes


    #TODO(mahyar): Multi-crop inference to be implemented
    def eval_multi_crop(self):
        raise NotImplementedError
