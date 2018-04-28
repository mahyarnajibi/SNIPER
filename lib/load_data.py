import numpy as np
from dataset import *
import os
import cPickle
import gc

def load_gt_roidb(dataset_name, image_set_name, root_path, dataset_path, result_path=None,
                  flip=False):
    """ load ground truth roidb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)
    roidb = imdb.gt_roidb()
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def load_proposal_roidb(dataset_name, image_set_name, root_path, dataset_path, result_path=None,
                        proposal='rpn', append_gt=True, flip=False,proposal_path='proposals', only_gt=False,
                        get_imdb=False):
    """ load proposal roidb (append_gt when training) """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)

    roidb = imdb.gt_roidb()

    if not only_gt:
        roidb = eval('imdb.' + proposal + '_roidb')(roidb, append_gt,proposal_path=proposal_path)

    if flip:
        roidb = imdb.append_flipped_images(roidb)

    if get_imdb:
        return roidb, imdb

    return roidb


def merge_roidb(roidbs):
    """ roidb are list, concat them together """
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb

def add_chip_data(roidb,chip_meta_data_path=''):
    assert os.path.isfile(chip_meta_data_path),'Chip meta data does not exists!'
    print('Loading chip meta data from : {}'.format(chip_meta_data_path))
    file = open(chip_meta_data_path,'rb')
    gc.disable()
    chip_meta_data = cPickle.load(file)
    gc.enable()
    file.close()
    gc.collect()
    print('Done!')
    # print('Pre-computing valid proposals per chip....')
    for iroidb, imeta in zip(roidb,chip_meta_data):
        for k in imeta:
            iroidb[k] = imeta[k]
    assert len(chip_meta_data)==len(roidb), 'Length of chip meta data should be the same as roidb'
    # for iroidb, imeta in zip(roidb,chip_meta_data):
    #     for k in imeta:
    #         iroidb[k] = imeta[k]
    #     # Compute pid_2_chips
    #     n_boxes = iroidb['boxes'].shape[0]
    #     valid_in_chips = iroidb['valid_in_chips']
    #     pid2chips = [np.array([],dtype=np.uint16) for _ in range(n_boxes)]
    #     for chipid,valids in enumerate(valid_in_chips):
    #         for v in valids:
    #             pid2chips[v] = np.append(pid2chips[v],chipid)
    #     # Shuffle the chip orders
    #     for pid in range(n_boxes):
    #         pid2chips[pid] = np.random.permutation(pid2chips[pid])
    #     iroidb['pid2chips'] = pid2chips
    # print('Done!')


def remove_small_boxes(roidb,max_scale=3,min_size=10):
    remove_counter = 0
    total_counter = 0
    for iroidb in roidb:
        cboxes = iroidb['boxes']*max_scale
        widths = cboxes[:,2] - cboxes[:,0] + 1
        heights = cboxes[:,3] - cboxes[:,1] + 1
        max_sizes = np.maximum(widths,heights)
        valid_inds = np.where(max_sizes>=min_size)[0]
        total_counter += widths.shape[0]
        if valid_inds.shape[0]<widths.shape[0]:
            remove_counter+= (widths.shape[0] - valid_inds.shape[0])
            iroidb['gt_classes'] = iroidb['gt_classes'][valid_inds]
            iroidb['max_classes'] = iroidb['max_classes'][valid_inds]
            iroidb['max_overlaps'] = iroidb['max_overlaps'][valid_inds]
            iroidb['gt_overlaps'] = iroidb['gt_overlaps'][valid_inds,:]
            iroidb['boxes'] = iroidb['boxes'][valid_inds,:]
    print('Removed {} small boxes out of {} boxes!'.format(remove_counter,total_counter))
    return roidb


def filter_roidb(roidb, config):
    """ remove roidb entries without usable rois """

    def is_valid(entry):
        """ valid images have at least 1 fg or bg roi """
        overlaps = entry['max_overlaps']
        fg_inds = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
        bg_inds = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO + 0.0001))[0]
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'filtered %d roidb entries: %d -> %d' % (num - num_after, num, num_after)

    return filtered_roidb


def load_gt_segdb(dataset_name, image_set_name, root_path, dataset_path, result_path=None,
                  flip=False):
    """ load ground truth segdb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)
    segdb = imdb.gt_segdb()
    if flip:
        segdb = imdb.append_flipped_images_for_segmentation(segdb)
    return segdb


def merge_segdb(segdbs):
    """ segdb are list, concat them together """
    segdb = segdbs[0]
    for r in segdbs[1:]:
        segdb.extend(r)
    return segdb
