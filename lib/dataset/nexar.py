# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os, sys

import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import subprocess

from imdb import IMDB

import pdb

class nexar(IMDB):
    def __init__(self, image_set, root_path, data_path, result_path=None):
        
        self.name = 'nexar'
        self._image_set = image_set
        self._data_path = data_path
        self.root_path = root_path
        self.num_classes = 2
        self._result_path = "./output/rfcn_dcn/nexar"
        # self.classes = ('__background__', 'car', 'van', 'pickup_truck', 'truck', 'bus')
        self.classes = ('__background__', 'vehicle')

        assert len(self.classes) == self.num_classes

        self._class_to_ind_image = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_ext = ['.jpg']

        self._image_index = self._load_image_set_index()
        self.image_set_index = self._image_index #for IMDB

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, self._image_set, index)
        assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    def imwidth(self, i):
        return self._sizes[i][0]

    def imheight(self, i):
        return self._sizes[i][1]

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt

        if self._image_set == 'train':

            image_set_file = os.path.join(self._data_path, 'verified_nexar_index.txt')
            # image_set_file = os.path.join(self._data_path, 'verified_nexar_train_index.txt')
            # image_set_file = os.path.join(self._data_path, 'verified_nexar_val_index.txt')

            image_index = []
            if os.path.exists(image_set_file):
                f = open(image_set_file, 'r')
                data = f.read().split()
                for lines in data:
                    if lines != '':
                        image_index.append(lines)
                f.close()
                return image_index

        else:
            if self._image_set == 'val': # val and train images are both in ./data/nexar/train/
                self._image_set == 'train'

            # image_set_file = os.path.join(self._data_path, 'verified_nexar_index.txt')
            # image_set_file = os.path.join(self._data_path, 'verified_nexar_val_index.txt')
            # image_set_file = os.path.join(self._data_path, 'verified_nexar_index.txt')

            image_set_file = os.path.join(self._data_path, 'nexar_test_index.txt')
            with open(image_set_file) as f:
                image_index = [x.strip().split(' ')[0] for x in f.readlines()]
        return image_index

    def get_csv(self, csvpath):
        """
        Return the gt_csv dict: image_index -> boxes
        """
        gt_csv = {}

        f = open(csvpath, 'r')
        lines = f.readlines()
        f.close()
        lines = lines[1:] # discard the first row (information row)
        for line in lines:
            line = line.split(',')
            box = [int(float(line[1])), int(float(line[2])), int(float(line[3])), int(float(line[4]))]
            image_name = line[0]
            if image_name not in gt_csv:
                gt_csv[image_name] = []
            gt_csv[image_name].append(box)

        return gt_csv

    def get_img_size(self, img_index_info_path):
        """
        Return dict: image_index -> image_size
        """
        img_size = {}

        f = open(img_index_info_path, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            size = [int(line[1]), int(line[2])] # width, height
            img_size[line[0].split('/')[1]] = size

        return img_size

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self._data_path, 'cache', self.name + '_gt_roidb.pkl')
        index_file = os.path.join(self._data_path, 'cache', self.name + '_index_roidb.pkl')
        
        # cache_file = os.path.join(self._data_path, 'cache', self.name + '_train_gt_roidb.pkl')
        # index_file = os.path.join(self._data_path, 'cache', self.name + '_train_index_roidb.pkl')
        # cache_file = os.path.join(self._data_path, 'cache', self.name + '_val_gt_roidb.pkl')
        # index_file = os.path.join(self._data_path, 'cache', self.name + '_val_index_roidb.pkl')

        # cache_file = os.path.join(self._data_path, 'cache', self.name + '_test_gt_roidb.pkl')
        # index_file = os.path.join(self._data_path, 'cache', self.name + '_test_index_roidb.pkl')

        if os.path.exists(cache_file) and os.path.exists(index_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            with open(index_file, 'rb') as fid:
                self._image_index = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            self.num_images = len(roidb)
            return roidb

        gt_roidb = []
        valid_index = []
        count = 0

        csv_path = os.path.join(self._data_path, "bbox.csv")
        self.gt_csv = self.get_csv(csv_path)

        # img_index_info_path = os.path.join(self._data_path, "nexar_test_index_imsize.txt")
        img_index_info_path = os.path.join(self._data_path, "nexar_index_imsize.txt")
        self.img_size = self.get_img_size(img_index_info_path)

        if self._image_set == 'nexar':
            for index in self._image_index:
                data = self._load_annotation(index)
                if len(data['boxes']) > 0:
                    gt_roidb.append(data)
                    valid_index.append(count)
                count = count + 1
        else:
            for index in self._image_index:
                data = self._load_annotation(index)
                gt_roidb.append(data)
                valid_index.append(count)
                count = count + 1

        self._image_index = [self._image_index[i] for i in valid_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        with open(index_file, 'wb') as fid:
            cPickle.dump(self._image_index, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        self.num_images = len(gt_roidb)
        return gt_roidb

    def _load_annotation(self, index):
        """
        Load image and bounding boxes info, etc.
        """

        # get image name from index (path)
        img_name = index.split('/')[1]

        width = self.img_size[img_name][0]
        height = self.img_size[img_name][1]

        objs = self.gt_csv[img_name]

        num_objs = len(objs)

        # boxes = np.zeros((num_objs, 4), dtype=np.int32)
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        ids = []
        for ix in range(len(objs)):
            obj = objs[ix]
            x1 = obj[0]
            y1 = obj[1]
            x2 = obj[2]
            y2 = obj[3]

            # ignore "bad boxes"
            if x2 > width or y2 > height:
                continue
            # can't be equal either
            if y2 <= y1 or x2 <= x1:
                continue

            cls = 1 # binary classification, 1 represents vehicle
            boxes[ix, :] = [x1, y1, x2, y2]
            
            gt_classes[ix] = cls

            overlaps[ix, cls] = 1.0
            ids.append(ix)

        boxes = boxes[ids,:]
        gt_classes = gt_classes[ids]
        overlaps = overlaps[ids, :]


        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False,
                        'width': width,
                        'height': height})

        return roi_rec

    # remains to be implemented 

    def _write_nexar_results_file(self, all_boxes):
        path = os.path.join(self._data_path, 'evaluation')
        filename = path + '/nexar_results.csv'
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self._image_index):
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s},{:.6f},{:.6f},{:.6f},{:.6f},{:s},{:.6f}\n'.
                                format(index, 
                                    dets[k, 0] + 1, dets[k, 1] + 1,
                                    dets[k, 2] + 1, dets[k, 3] + 1,
                                    'vehicle', dets[k, -1]))

    def evaluate_detections(self, all_boxes, output_dir=''):
        self._write_nexar_results_file(all_boxes)
        print "Detection results writen to evaluation folder."

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

