import os, sys

import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import subprocess

from imdb import IMDB

import pdb

class openimages(IMDB):
    def __init__(self, image_set, root_path, data_path, result_path=None):
        
        self.name = 'openimages'
        self._image_set = image_set
        self._data_path = data_path
        self.root_path = root_path
        self.num_classes = 602
        self._result_path = result_path
        # self.classes = ('__background__', 'car', 'van', 'pickup_truck', 'truck', 'bus')
        self.classes = ['__background__']
        f = open(os.path.join(data_path, 'annotations', 'class-descriptions-boxable.csv'))
        self.classIDMap = {}
        ct = 1
        for line in f:
            words = line.split(',')
            self.classes.append(words[0])
            self.classIDMap[words[0]] = (ct, words[1])
            ct = ct + 1
        
        self._class_to_ind_image = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_ext = ['.jpg']

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


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self._data_path, 'cache', self._image_set, self.name + '_gt_roidb.pkl')
        index_file = os.path.join(self._data_path, 'cache', self._image_set, self.name + '_index_roidb.pkl')
        
        if os.path.exists(cache_file) and os.path.exists(index_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            with open(index_file, 'rb') as fid:
                self._image_index = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            self.image_set_index = self._image_index
            self.num_images = len(roidb)
            return roidb

        gt_roidb = []
        valid_index = []
        count = 0

        csv_file = os.path.join(self._data_path, 'annotations', self._image_set + ".csv")
        f = open(csv_file, 'r')
        data_map = {}
        
        for lines in f:
            words = lines.split(',')
            imid = words[0]
            classid = self._class_to_ind_image[words[1]]
            x1 = float(words[2])
            y1 = float(words[3])
            x2 = float(words[4])
            y2 = float(words[5])
            crowd = int(words[6])
            height = float(words[7])            
            width = float(words[8])
            
            if imid in data_map:
                if crowd == 0:
                    data_map[imid]['boxes'].append([x1, y1, x2, y2, classid])
                else:
                    data_map[imid]['crowd'].append([x1, y1, x2, y2, classid])
            else:
                data_map[imid] = {}
                data_map[imid]['image'] = os.path.join(self._data_path, 'images', self._image_set, imid + '.jpg')
                data_map[imid]['height'] = height
                data_map[imid]['width'] = width
                if crowd == 0:
                    data_map[imid]['boxes'] = [[x1, y1, x2, y2, classid]]
                    data_map[imid]['crowd'] = []
                else:
                    data_map[imid]['crowd'] = [[x1, y1, x2, y2, classid]]
                    data_map[imid]['boxes'] = []
        f.close()
        ct = 0
        self._image_index = []
        gt_roidb = []
        keys = data_map.keys()
        for key in keys:
            boxes = np.array(data_map[key]['boxes'])
            group = np.array(data_map[key]['crowd'])
            if len(group) > 0:
                group = group[:, :4]
            else:
                group = np.zeros((0, 4))
            if len(boxes) == 0:
                continue
            gt_roidb.append({'boxes': boxes[:, :4],
                             'image': data_map[key]['image'],
                             'crowd': group,
                             'flipped': False,
                             'max_overlaps': np.ones(len(data_map[key]['boxes'])),
                             'max_classes': boxes[:, 4],
                             'gt_classes': boxes[:, 4],                             
                             'width': data_map[key]['width'],
                             'height': data_map[key]['height']})
            self._image_index.append(ct)
            ct = ct + 1
        self.image_set_index = self._image_index
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        with open(index_file, 'wb') as fid:
            cPickle.dump(self._image_index, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        self.num_images = len(gt_roidb)
        return gt_roidb

    def _write_openimages_results_file(self, all_boxes):
        path = os.path.join(self._data_path, 'evaluation')
        filename = path + '/openimages_results.csv'
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
                                    cls, dets[k, -1]))

    def evaluate_detections(self, all_boxes, output_dir=''):
        self._write_openimages_results_file(all_boxes)
        print "Detection results writen to evaluation folder."

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

