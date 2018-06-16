# --------------------------------------------------------
# Written by Hengduo Li, Bharat Singh
# --------------------------------------------------------

import os, sys

import xml.dom.minidom as minidom
import numpy as np
import cPickle
import pickle
from collections import defaultdict
import subprocess
from imdb import IMDB

import time

DEBUG = False

class imagenet(IMDB):
    def __init__(self, image_set, root_path='./', data_path='data/imagenet/', result_path=None, load_mask=False):

        self.name = 'imagenet'
        self._image_set = image_set
        self.image_set = image_set        
        self._data_path = data_path
        self.root_path = root_path
        self._devkit_path = "./data/imagenet/ILSVRC2014_devkit"
        self._result_path = "./output/imagenet"
        self.num_classes = 1 + 1
        self.num_sub_classes = 3130 + 1

        self._classes_image = ()
        self._wnid_image = ()
        self.cls_tag_is_noun = 0

        wnid_2_description = self.get_wnid_name_dict()
        self._sons, self._parents = self.get_cluster_info()
        assert len(self._sons) == len(self._parents), "Clustering information not matching"
        
        self._cluster_match = {}
        for i in range(len(self._sons)):
            self._cluster_match[self._sons[i]] = self._parents[i]

        num_of_subclasses = len(self._sons)

        if DEBUG:
            print("Number of subclasses: %d" %len(self._sons))

        for one in self._sons:
            self._wnid_image = self._wnid_image + (str(one),)
            self._classes_image = self._classes_image + (wnid_2_description[str(one)],)

        self.classes = self._classes_image

        self._wnid_to_ind_image = dict(zip(self._wnid_image, xrange(num_of_subclasses)))
        self._class_to_ind_image = dict(zip(self._classes_image, xrange(num_of_subclasses)))
        self._image_ext = ['.JPEG']

        self._image_index = self._load_image_set_index()

        self.image_set_index = self._image_index
        self.num_images = len(self._image_index)

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), 'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def get_wnid_name_dict(self):
        """
        :return: dict of {wnid: description}
        """
        path = self._data_path + "/ILSVRC2012_devkit_t12/data/wnid_name_dict.txt"
        assert os.path.exists(path)
        with open(path, 'rb') as f:
            wnid_name_dict = pickle.load(f)
        return wnid_name_dict

    def get_cluster_info(self):
        """
        return: dict of {wnid of subclass (fine-grained class): index of super class}
        """
        path = self._data_path + "/ILSVRC2012_devkit_t12/data/3kcls_cluster_result1.txt"

        assert os.path.exists(path)
        with open(path, 'rb') as f:
            result = defaultdict(dict)
            result = pickle.load(f)
        sons = []
        parents = []
        for i in result:
            length = len(result[i])
            for j in range(length):
                sons.append(result[i][j])
                parents.append(i)
        return sons, parents

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, self._image_set, index + self._image_ext[0])
        assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    def imwidth(self, i):
        return self._sizes[i][0]

    def imheight(self, i):
        return self._sizes[i][1]

    def _load_image_set_index(self):
        """
        Load the indices listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        if self._image_set == 'fall11_whole':             
            image_set_file = os.path.join(self._devkit_path, 'data/3kcls_index.txt')
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
            image_set_file = os.path.join(self._devkit_path, 'data', 'det_lists', 'small_val.txt')
            with open(image_set_file) as f:
                image_index = [x.strip().split(' ')[0] for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_path = './data/imagenet/cache/'
        if self._image_set == 'fall11_whole':
            cache_file = os.path.join(cache_path, self.name + '_3kcls_1C_loc_gt_roidb.pkl')
            index_file = os.path.join(cache_path, self.name + '_3kcls_1C_loc_index_roidb.pkl')
        else:
            cache_file = os.path.join(cache_path, self.name + '_det_new_val_gt_roidb.pkl')
            index_file = os.path.join(cache_path, self.name + '_det_new_val_index_roidb.pkl')

        if os.path.exists(cache_file) and os.path.exists(index_file):
            print ("found cache")
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            with open(index_file, 'rb') as fid:
                self._image_index = cPickle.load(fid)
            self.num_images = len(roidb)
            assert len(roidb) == len(self._image_index), "roidb and image index length not matching!"
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)

            return roidb

        gt_roidb = []
        valid_index = []
        count = 0
        t = time.time()
        for index in self._image_index:
            data = self._load_imagenet_annotation(index)
            if self._image_set == 'fall11_whole':
                if len(data['boxes']) > 0:
                    gt_roidb.append(data)
                    valid_index.append(count)
            else:
                gt_roidb.append(data)
                valid_index.append(count)
            count = count + 1
            if count % 1000 == 0:
                t1 = time.time() - t
                t = time.time()
                print str(count) + '/' + str(len(self._image_index)) + " time spent: %.2f" %(t1) 

        self._image_index = [self._image_index[i] for i in valid_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        with open(index_file, 'wb') as fid:
            cPickle.dump(self._image_index, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        filename = os.path.join(self._data_path, self._image_set + '_bbox', index + '.xml')

        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        imsize = data.getElementsByTagName('size')
        width = float(get_data_from_tag(imsize[0], 'width'))
        height = float(get_data_from_tag(imsize[0], 'height'))

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.float32)
        gt_subclasses = np.zeros((num_objs), dtype=np.float32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        ids = []

        for ix, obj in enumerate(objs):
            x1 = int(get_data_from_tag(obj, 'xmin'))
            y1 = int(get_data_from_tag(obj, 'ymin'))
            x2 = int(get_data_from_tag(obj, 'xmax'))
            y2 = int(get_data_from_tag(obj, 'ymax'))

            
            # ignore invalid boxes
            if x1 > 4000 or y1 > 4000 or x2 > 4000 or y2 > 4000 :
                continue
            if x2 > width or y2 > height:
                continue
            if y2 <= y1 or x2 <= x1:
                continue

            cls_tag = str(get_data_from_tag(obj, "name")).lower().strip()

            # discard images which includes unregistered object categories
            if not (cls_tag in self._wnid_to_ind_image):
                continue

            # correct class format is "nxxxxxxxx"; discard if not correct
            if cls_tag[0] != 'n' or (not cls_tag[1:].isdigit()):
                self.cls_tag_is_noun += 1
                continue

            cls_id = int(self._cluster_match[cls_tag]) + 1
            subcls_id = int(self._wnid_to_ind_image[cls_tag])

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls_id
            gt_subclasses[ix] = subcls_id

            overlaps[ix, cls_id] = 1.0
            ids.append(ix)

        boxes = boxes[ids,:]
        gt_classes = gt_classes[ids]
        gt_subclasses = gt_subclasses[ids]
        overlaps = overlaps[ids, :]

        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_subclasses,
                        'gt_subclasses': gt_subclasses,
                        'gt_overlaps': overlaps,
                        'max_classes': gt_subclasses,
                        'max_overlaps': np.ones((len(gt_subclasses), 1)),
                        'flipped': False,
                        'width': width,
                        'height': height})

        return roi_rec

    def _write_imagenet_results_file(self, all_boxes):
        eval_path = os.path.join(self._devkit_path, 'evaluation')
        filename = eval_path + '/3k_1C_pred/det_val_pred.txt'

        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self._image_index):
                for cls_ind, cls in enumerate(self._classes_image):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.
                                format(im_ind + 1, cls_ind, dets[k, -1],
                                    dets[k, 0] + 1, dets[k, 1] + 1,
                                    dets[k, 2] + 1, dets[k, 3] + 1))
    
    def _modify_scores(self):
        """
        The result file are indexed in a different way. For evaluation, map the indices back.
        """
        eval_path = os.path.join(self._devkit_path, 'evaluation')
        raw_path = eval_path + '/3k_1C_pred/det_val_pred.txt'
        out_path = eval_path + '/3k_1C_pred/modified_det_val_pred.txt'

        matching_file = open(eval_path + "/3k_1C_pred/3k_1C_matching.txt", 'rb')
        matching = pickle.load(matching_file)
        matching_file.close()

        f = open(raw_path, 'r')
        lines = f.readlines()
        f.close()

        output = open(out_path, 'w')

        print("Modifying the output...")

        for line in lines:
            line = line.split(' ')
            index = line[1]
            score = float(line[2])
            if score < 1e-3:
                continue
            new_ind = matching[int(index)]
            if new_ind > 201:
                continue
            line[1] = str(new_ind)
            new_line = line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + line[3] + ' ' + line[4] + ' ' + line[5] + ' ' + line[6]
            output.write(new_line)

        print("Done.")


    def _do_matlab_eval(self):
        eval_path = os.path.join(self._devkit_path, 'evaluation')
        
        cmd = 'cd {} && '.format(eval_path)
        cmd += '{:s} -nodisplay -nodesktop '.format('matlab')
        cmd += '-r "dbstop if error; '
        cmd += 'eval_det_3k_1C; quit;"'
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes):
        self._write_imagenet_results_file(all_boxes)
        self._modify_scores()
        self._do_matlab_eval()

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = imagenet_clsloc('val', '')
    res = d.roidb
    from IPython import embed; embed()
