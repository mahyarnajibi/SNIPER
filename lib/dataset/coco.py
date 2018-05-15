import cPickle
import cv2
import os
import json
import numpy as np

from imdb import IMDB

# coco api
from .pycocotools.coco import COCO
from .pycocotools.cocoeval import COCOeval
from .pycocotools import mask as COCOmask
from mask.mask_coco2voc import mask_coco2voc
from mask.mask_voc2coco import mask_voc2coco
from general_utils import tic, toc
from bbox.bbox_transform import clip_boxes, bbox_overlaps_py
import multiprocessing as mp


def coco_results_one_category_kernel(data_pack):
    cat_id = data_pack['cat_id']
    ann_type = data_pack['ann_type']
    binary_thresh = data_pack['binary_thresh']
    all_im_info = data_pack['all_im_info']
    boxes = data_pack['boxes']
    if ann_type == 'bbox':
        masks = []
    elif ann_type == 'segm':
        masks = data_pack['masks']
    else:
        print 'unimplemented ann_type: ' + ann_type
    cat_results = []
    for im_ind, im_info in enumerate(all_im_info):
        index = im_info['index']
        dets = boxes[im_ind].astype(np.float)
        if len(dets) == 0:
            continue
        scores = dets[:, -1]
        if ann_type == 'bbox':
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            result = [{'image_id': index,
                       'category_id': cat_id,
                       'bbox': [round(xs[k], 1), round(ys[k], 1), round(ws[k], 1), round(hs[k], 1)],
                       'score': round(scores[k], 8)} for k in xrange(dets.shape[0])]
        elif ann_type == 'segm':
            width = im_info['width']
            height = im_info['height']
            dets[:, :4] = clip_boxes(dets[:, :4], [height, width])
            mask_encode = mask_voc2coco(masks[im_ind], dets[:, :4], height, width, binary_thresh)
            result = [{'image_id': index,
                       'category_id': cat_id,
                       'segmentation': mask_encode[k],
                       'score': scores[k]} for k in xrange(len(mask_encode))]
        cat_results.extend(result)
    return cat_results


class coco(IMDB):
    def __init__(self, image_set, root_path, data_path, result_path=None, mask_size=-1, binary_thresh=None, load_mask=False):
        """
        fill basic information to initialize imdb
        :param image_set: train2014, val2014, test2015
        :param root_path: 'data', will write 'rpn_data', 'cache'
        :param data_path: 'data/coco'
        """
        super(coco, self).__init__('COCO', image_set, root_path, data_path, result_path)
        self.root_path = root_path
        self.data_path = data_path
        self.coco = COCO(self._get_ann_file())

        # deal with class names
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls], self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh
        self.load_mask = load_mask

        # deal with data name
        view_map = {'minival2014': 'val2014',
                    'sminival2014': 'val2014',
                    'valminusminival2014': 'val2014',
                    'test-dev2015': 'test2015'}

        self.data_name = view_map[image_set] if image_set in view_map else image_set

    def _get_ann_file(self):
        """ self.data_path / annotations / instances_train2014.json """
        prefix = 'instances' if 'test' not in self.image_set else 'image_info'
        return os.path.join(self.data_path, 'annotations',
                            prefix + '_' + self.image_set + '.json')

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def image_path_from_index(self, index):
        """ example: images / train2014 / COCO_train2014_000000119993.jpg """
        filename = 'COCO_%s_%012d.jpg' % (self.data_name, index)
        image_path = os.path.join(self.data_path, 'images', self.data_name, filename)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        index_file = os.path.join(self.cache_path, self.name + '_index_roidb.pkl')
        sindex_file = os.path.join(self.cache_path, self.name + '_sindex_roidb.pkl')
        if os.path.exists(cache_file) and os.path.exists(index_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            with open(index_file, 'rb') as fid:
                self.image_set_index = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = []
        valid_id = []
        vids = []
        ct = 0
        for index in self.image_set_index:
            roientry,flag = self._load_coco_annotation(index)
            if flag:
                gt_roidb.append(roientry)
                valid_id.append(index)
                vids.append(ct)
            ct = ct + 1
        self.image_set_index = valid_id

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        with open(index_file, 'wb') as fid:
            cPickle.dump(valid_id, fid, cPickle.HIGHEST_PROTOCOL)
        with open(sindex_file, 'wb') as fid:
            cPickle.dump(vids, fid, cPickle.HIGHEST_PROTOCOL)

        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def _load_coco_annotation(self, index):
        def _polys2boxes(polys):
            boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
            for i in range(len(polys)):
                poly = polys[i]
                x0 = min(min(p[::2]) for p in poly)
                x1 = max(max(p[::2]) for p in poly)
                y0 = min(min(p[1::2]) for p in poly)
                y1 = max(max(p[1::2]) for p in poly)
                boxes_from_polys[i, :] = [x0, y0, x1, y1]
            return boxes_from_polys
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: roidb entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=True)
        objsc = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)

        valid_objsc = []
        for obj in objsc:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objsc.append(obj)

        objs = valid_objs
        objc = valid_objsc
        num_objs = len(objs)
        num_objsc = len(objsc)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        boxesc = np.zeros((num_objsc, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        #for ix, obj in enumerate(objsc):
        #    boxesc[ix, :] = obj['clean_bbox']

        for ix, obj in enumerate(objs):
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            if obj['iscrowd']:
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0

        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]

        flag = True

        roi_rec = {'image': self.image_path_from_index(index),
                   'height': height,
                   'width': width,
                   'boxes': boxes,
                   'boxesc': boxesc,
                   'gt_classes': gt_classes,
                   'gt_overlaps': overlaps,
                   'max_classes': overlaps.argmax(axis=1),
                   'max_overlaps': overlaps.max(axis=1),
                   'flipped': False}
        if self.load_mask:
            # we only care about valid polygons

            segs = []
            for obj in objs:
                if not isinstance(obj['segmentation'], list):
                    # This is a crowd box
                    segs.append([])
                else:
                    segs.append([np.array(p) for p in obj['segmentation'] if len(p)>=6])
            
            roi_rec['gt_masks'] =  segs

            # Uncomment if you need to compute gts based on segmentation masks
            # seg_boxes = _polys2boxes(segs)
            # roi_rec['mask_boxes'] = seg_boxes
        return roi_rec, flag

    

    def evaluate_detections(self, detections, ann_type='bbox', all_masks=None, extra_path=''):
        """ detections_val2014_results.json """
        res_folder = os.path.join(self.result_path + extra_path, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(res_folder, 'detections_%s_results.json' % self.image_set)
        self._write_coco_results(detections, res_file, ann_type, all_masks)
        if 'test' not in self.image_set:
            info_str = self._do_python_eval(res_file, res_folder, ann_type)
            return info_str

    def evaluate_sds(self, all_boxes, all_masks):
        info_str = self.evaluate_detections(all_boxes, 'segm', all_masks)
        return info_str

    def _write_coco_results(self, all_boxes, res_file, ann_type, all_masks):
        """ example results
        [{"image_id": 42,
          "category_id": 18,
          "bbox": [258.15,41.29,348.26,243.78],
          "score": 0.236}, ...]
        """
        all_im_info = [{'index': index,
                        'height': self.coco.loadImgs(index)[0]['height'],
                        'width': self.coco.loadImgs(index)[0]['width']}
                        for index in self.image_set_index]

        if ann_type == 'bbox':
            data_pack = [{'cat_id': self._class_to_coco_ind[cls],
                          'cls_ind': cls_ind,
                          'cls': cls,
                          'ann_type': ann_type,
                          'binary_thresh': self.binary_thresh,
                          'all_im_info': all_im_info,
                          'boxes': all_boxes[cls_ind]}
                         for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']
        elif ann_type == 'segm':
            data_pack = [{'cat_id': self._class_to_coco_ind[cls],
                          'cls_ind': cls_ind,
                          'cls': cls,
                          'ann_type': ann_type,
                          'binary_thresh': self.binary_thresh,
                          'all_im_info': all_im_info,
                          'boxes': all_boxes[cls_ind],
                          'masks': all_masks[cls_ind]}
                         for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']
        else:
            print 'unimplemented ann_type: '+ann_type
        # results = coco_results_one_category_kernel(data_pack[1])
        # print results[0]
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(coco_results_one_category_kernel, data_pack)
        pool.close()
        pool.join()
        results = sum(results, [])
        print 'Writing results json to %s' % res_file
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _do_python_eval(self, res_file, res_folder, ann_type):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        info_str = self._print_detection_metrics(coco_eval)

        eval_file = os.path.join(res_folder, 'detections_%s_results.pkl' % self.image_set)
        with open(eval_file, 'w') as f:
            cPickle.dump(coco_eval, f, cPickle.HIGHEST_PROTOCOL)
        print 'coco eval results saved to %s' % eval_file
        info_str +=  'coco eval results saved to %s\n' % eval_file
        return info_str

    def _print_detection_metrics(self, coco_eval):
        info_str = ''
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)

        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print '~~~~ Mean and per-category AP @ IoU=%.2f,%.2f] ~~~~' % (IoU_lo_thresh, IoU_hi_thresh)
        info_str += '~~~~ Mean and per-category AP @ IoU=%.2f,%.2f] ~~~~\n' % (IoU_lo_thresh, IoU_hi_thresh)
        print '%-15s %5.1f' % ('all', 100 * ap_default)
        info_str += '%-15s %5.1f\n' % ('all', 100 * ap_default)
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print '%-15s %5.1f' % (cls, 100 * ap)
            info_str +=  '%-15s %5.1f\n' % (cls, 100 * ap)

        print '~~~~ Summary metrics ~~~~'
        coco_eval.summarize()

        return info_str
